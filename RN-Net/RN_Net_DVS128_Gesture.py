from cmath import inf
from importlib.metadata import requires
from re import A
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import os
from numpy.core.numeric import indices

import sys
from datetime import datetime

from functools import partial

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd.grad_mode import no_grad
import torch.optim as optim

import tonic
from tonic import CachedDataset
import tonic.transforms as TTF

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.cuda.empty_cache

start_time = datetime.now()
current_time = start_time.strftime("%H:%M:%S")
print("Current Time =", current_time)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('cuda device:',device)
if torch.cuda.is_available():
    print('cuda device name:', torch.cuda.get_device_name(0))

class TE(nn.Module):
    def __init__(self, RR, PFRAC, GMAX, GMIN, TAU, spkrange):
        super(TE, self).__init__()
        self.RR = RR
        self.PFRAC = PFRAC
        self.GMAX = GMAX
        self.GMIN = GMIN
        self.TAU = TAU
        self.spkrange = spkrange

    def forward(self, event, time_trace, length):

        SN = event.shape[0]
        T2G = torch.zeros(SN,self.RR,2,128,128).cuda()
        event = event.long().cuda()
        time_trace = time_trace.cuda()
        length = length.cuda()
        
        with no_grad():
            ## R_in encoding using input events from a DVS-camera.
            
            ## potentiation
            T2G[range(SN), 0, event[range(SN), 0, 2],event[range(SN), 0, 0],event[range(SN), 0, 1]] += self.PFRAC * (self.GMAX-self.GMIN)
            for N in range(1, self.spkrange):
                ## relaxation
                MM = (torch.exp((event[:, N-1, 3]-event[:, N, 3])/self.TAU)).repeat(2,128,128,1).permute(3,0,1,2)
                T2G[range(SN), time_trace[range(SN), N]] = MM[range(SN)]*(T2G[range(SN), time_trace[range(SN), N-1]]-self.GMIN)+self.GMIN
                
                ## potentiation
                T2G[range(SN), time_trace[range(SN), N], event[range(SN), N, 2], event[range(SN), N, 0], event[range(SN), N, 1]]\
                    += (length[range(SN)] > N)*self.PFRAC*(self.GMAX - T2G[range(SN), time_trace[range(SN), N], event[range(SN), N, 2], event[range(SN), N, 0], event[range(SN), N, 1]])
                    
            del event
            del time_trace
            del length
            torch.cuda.empty_cache()
            
            ## The first 2D MaxPool
            MP= nn.MaxPool3d((1,2,2),(1,2,2))
            T2G = MP(T2G).clone()

        return T2G

def DataLoad_RC(dt, TAU, GMIN, GMAX, PFRAC):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    transform = TTF.NumpyAsType(int)

    ##############################################################
    ##################### Test dataset ###########################
    ##############################################################
    
    if os.path.exists('./DVSG_test_data_time_'+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC)):
        print("test dataset exists!!")
    else:
        dataset = tonic.datasets.DVSGesture(save_to='./data', transform=transform, train=False)
        
        ## check the length of clip for all the vidoe clips.
        length = torch.zeros(len(dataset)).long()
        for NUM in range(len(dataset)):
            length[NUM] = (torch.from_numpy(dataset[NUM][0])[:,3]>=int(1.5e6)).nonzero(as_tuple=True)[0][0]
            
        ## Max spike number = 1459889
        spkrange = length.max().item()

        event = torch.zeros((len(dataset),int(spkrange),4)).long()
        Y = torch.zeros(len(dataset))
        for NUM in range(len(dataset)):
            event[NUM,:length[NUM]] = torch.from_numpy(dataset[NUM][0])[:length[NUM]]
            event[NUM,length[NUM]:] = torch.from_numpy(dataset[NUM][0])[length[NUM]-1]
            Y[NUM] = dataset[NUM][1]
        torch.save(Y,"DVSG_Y_test")

        SN = len(length)

        ## Longest time range = 15591329
        print("event shape:", event.shape)
        time_trace = event[:,:,3].long().clone()
        print("Time trace max:", time_trace.max())
        time_trace = (time_trace//dt).long()

        last_times = event[:,-1,3].long().clone()

        for YY in range(SN):
            last_times[YY] = int(last_times[YY].item()//dt)

        print("Test last_time:", last_times.unique())
        print("Test last_time max:", last_times.max())
        print("Test last_time min:", last_times.min())
        print("Test last_time bin count:", torch.bincount(last_times))
        
        RR = int(last_times.max().item())+1

        print("SN:", SN)
        print("RR:", RR)
        print("dt:", dt)
        print("TAU0:", TAU)
        
        Temporal_encoding = TE(RR, PFRAC, GMAX, GMIN, TAU, int(spkrange))
        Temporal_encoding = nn.DataParallel(Temporal_encoding)
        Temporal_encoding = Temporal_encoding.to(device)

        ## State encoding of R_in using input events from a DVS-camera.
        T2G_out = Temporal_encoding(event, time_trace, length).cpu().clone()
        last_times = last_times.long().cpu().clone()

        torch.save(T2G_out,"DVSG_test_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
        torch.save(last_times, "last_times_test_"+str(dt))
    
    ##############################################################
    #################### Train dataset ###########################
    ##############################################################
    if os.path.exists('./DVSG_train_data_time_'+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC)):
        print("train dataset exists!!")
    else:
        dataset = tonic.datasets.DVSGesture(save_to='./data', transform=transform, train=True)
        
        ## check the length of clip for all the vidoe clips.
        length = torch.zeros(len(dataset)).long()
        for NUM in range(len(dataset)):
            length[NUM] = (torch.from_numpy(dataset[NUM][0])[:,3]>=int(1.5e6)).nonzero(as_tuple=True)[0][0]
            
        ## Max spike number = 1459889
        spkrange = length.max().item()
        print("spkrange:",spkrange)

        event = torch.zeros((len(dataset),int(spkrange),4)).long()
        Y = torch.zeros(len(dataset))
        for NUM in range(len(dataset)):
            event[NUM,:length[NUM]] = torch.from_numpy(dataset[NUM][0])[:length[NUM]]
            event[NUM,length[NUM]:] = torch.from_numpy(dataset[NUM][0])[length[NUM]-1]
            Y[NUM] = dataset[NUM][1]

        SN = len(length)

        ## Longest time range = 15591329
        print("event shape:", event.shape)
        time_trace = event[:,:,3].long().clone()
        print("Time trace max:", time_trace.max())
        time_trace = (time_trace//dt).long()

        last_times = event[:,-1,3].long().clone()

        for YY in range(SN):
            last_times[YY] = int(last_times[YY].item()//dt)

        print("Test last_time:", last_times.unique())
        print("Test last_time max:", last_times.max())
        print("Test last_time min:", last_times.min())
        print("Test last_time bin count:", torch.bincount(last_times))
        
        RR = int(last_times.max().item())+1

        print("SN:", SN)
        print("RR:", RR)
        print("dt:", dt)
        print("TAU0:", TAU)
        
        Temporal_encoding = TE(RR, PFRAC, GMAX, GMIN, TAU, spkrange)
        Temporal_encoding = nn.DataParallel(Temporal_encoding)
        Temporal_encoding = Temporal_encoding.to(device)

        ## State encoding of R_in using input events from a DVS-camera.
        T2G_out = Temporal_encoding(event, time_trace, length).cpu().clone()
        last_times = last_times.long().cpu().clone()

        train_data = []
        for NUM in range(SN):
            train_data.append([T2G_out[NUM], Y[NUM], last_times[NUM]])

        torch.save(train_data,"DVSG_train_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
    
    start_time = datetime.now()
    current_time = start_time.strftime("%H:%M:%S")
    print("Current Time =", current_time)

#############
ver_num = 0
##############
alpha = 0.9
beta = 0.85
spike_grad1 = surrogate.atan()

class Net(nn.Module):
    def __init__(self, TAU0, TAUFR, THFR, PFrac, ConCh, ConvK, F1K):
        super(Net, self).__init__()

        ## Define Convolution layers
        self.C1 = nn.Conv2d(in_channels=2,out_channels=ConCh[0], kernel_size=ConvK, stride = 1, bias=True)
        self.P1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN1 = nn.BatchNorm2d(self.C1.out_channels)

        self.C2 = nn.Conv2d(in_channels=self.C1.out_channels, out_channels=ConCh[1], kernel_size=3, stride = 1, padding = 1, bias=True)
        self.P2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN2 = nn.BatchNorm2d(self.C2.out_channels)

        self.C3 = nn.Conv2d(in_channels=self.C2.out_channels, out_channels=ConCh[2], kernel_size=3, stride = 1, bias=True)
        self.BN3 = nn.BatchNorm2d(self.C3.out_channels)
        
        self.C4 = nn.Conv2d(in_channels=self.C3.out_channels, out_channels=ConCh[3], kernel_size=3, stride = 1, padding = 1, bias=True)
        self.P4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN4 = nn.BatchNorm2d(self.C4.out_channels)
        
        self.C5 = nn.Conv2d(in_channels=self.C4.out_channels, out_channels=ConCh[3], kernel_size=3, stride = 1, bias=True)
        self.P5 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN5 = nn.BatchNorm2d(self.C5.out_channels)
        

        ## Calculate the output dimension of Convolution layers
        OutD1 = int((64-(self.C1.kernel_size[0]-1)-1)/self.P2.stride)
        OutD2 = int((OutD1-2+2-1)/self.P2.stride)
        OutD3 = int(OutD2-2)
        OutD4 = int((OutD3-2+2-1)/self.P4.stride)
        OutD5 = 1
        Size_Cout = self.C5.out_channels*OutD5*OutD5*5

        self.BN_RC = nn.BatchNorm1d(Size_Cout)

        ## Define Fully-connected layers
        self.F1 = nn.Linear(in_features=Size_Cout,out_features=F1K)
        self.BNF = nn.BatchNorm1d(self.F1.out_features)
        self.F2 = nn.Linear(in_features=self.F1.out_features,out_features=int(11))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.reluF1 = nn.ReLU()

        ## Hyperparameters
        self.Tau = TAU0
        self.TauFR = TAUFR
        self.FR_TH = THFR
        
        print(self.C1)
        print(self.C2)
        print(self.P2)
        print(self.C3)
        print("FR_TH:",self.FR_TH)
        print("FR_tau:","{:e}".format(self.TauFR))
        print(self.F1)
        print(self.F2)

        print("OutD1:",OutD1)
        print("OutD2:",OutD2)
        print("OutD3:",OutD3)
        self.Gmin = 0
        self.Gmax = 1
        self.PFrac = PFrac

        ## Just for convenience
        self.float_zero = torch.tensor(0).float().to(device)

    def RS_dyn(self, RR, G, TD, TAU, ltlt):
        D_multiplier = np.exp(-TD/TAU)

        ## R_f encoding using spikes generated by the Spike Conversion layer.

        ##Potentiation
        G[0:self.SN] = self.PFrac*(self.Gmax-self.Gmin)*G[0:self.SN]+self.Gmin

        for T in range(RR-1):
            ##Relaxation
            RELAX = (G[T*self.SN:(T+1)*self.SN].clone()-self.Gmin)*(D_multiplier - 1)*((T+1)<ltlt)[:, None, None, None] + G[T*self.SN:(T+1)*self.SN].clone()
            
            ##Potentiation
            G[(T+1)*self.SN:(T+2)*self.SN] = self.PFrac*(self.Gmax - RELAX)*G[(T+1)*self.SN:(T+2)*self.SN].clone()*((T+1)<ltlt)[:, None, None, None] + RELAX
            
        return G.clone()

    #def forward(self, T2G, TD, N_iter):
    def forward(self, T2G, TD, LTLT, noise_factor, testing):
        if testing == 0:
            #####################################
            ############# Training ##############
            #####################################
            self.SN = int(T2G.shape[0])
            T2G = T2G.permute(1,0,2,3,4)
            RR = T2G.shape[0]
            T2G = T2G.reshape(T2G.shape[0]*T2G.shape[1],T2G.shape[2],T2G.shape[3],T2G.shape[4])
            
            lif = snn.Leaky(beta=1, spike_grad=spike_grad1, reset_mechanism = 'none')

            ## Forward-passes of the convolution layers (C1-C4).
            G_L1 = self.relu1(self.BN1(self.P1(self.C1(T2G))))
            G_L2 = self.relu2(self.BN2(self.P2(self.C2(G_L1))))
            G_L3 = self.relu3(self.BN3(self.C3(G_L2)))
            G_L4 = self.relu4(self.BN4(self.P4(self.C4(G_L3))))
            
            ## Forward-pass of the fifth convolution layer and Activation of the Spike Conversion layer.
            G_last = lif(self.BN5(self.P5(self.C5(G_L4)))/self.FR_TH, 0)[0].clone()

            ## R_f encoding
            G_last = self.RS_dyn(RR, G_last, TD, self.TauFR, LTLT)

            G_last = G_last.reshape(RR, self.SN, G_last.shape[1]*G_last.shape[2]*G_last.shape[3]).clone()

            ## RN state retrieval of R_f Layer. States in 5 different time instants are retrieved. 5 x 512 states in total.
            G_VN = torch.cat((G_last[9],G_last[19],G_last[29],G_last[39],G_last[49]), 1).clone()
            
            ## Forward-passes of the Fully-connected layers (FC1-2)
            G_F1 = self.reluF1(self.BNF(self.F1(G_VN)))
            return self.F2(G_F1)
        else:
            #####################################
            ############## Testing ##############
            #####################################
            self.SN = int(T2G.shape[0])
            T2G = T2G.permute(1,0,2,3,4)
            RR = T2G.shape[0]
            D_multiplier = np.exp(-TD/self.TauFR)
            
            lif = snn.Leaky(beta=1, spike_grad=spike_grad1, reset_mechanism = 'none')

            for T in range(RR):

                ## Forward-passes of the convolution layers (C1-C4).
                G_L1 = self.relu1(self.BN1(self.P1(self.C1(T2G[T]))))
                G_L2 = self.relu2(self.BN2(self.P2(self.C2(G_L1))))
                G_L3 = self.relu3(self.BN3(self.C3(G_L2)))
                G_L4 = self.relu4(self.BN4(self.P4(self.C4(G_L3))))
                
                ## Forward-pass of the fifth convolution layer and Activation of the Spike Conversion layer.
                G_last = lif(self.BN5(self.P5(self.C5(G_L4)))/self.FR_TH, 0)[0].clone()

                ## R_f encoding
                ##Relaxation
                if T > 0:
                    RELAX = (G_RC-self.Gmin)*(D_multiplier - 1)*(T<LTLT)[:, None, None, None] + G_RC
                else:
                    RELAX = self.Gmin

                ##Potentiation
                G_RC = self.PFrac*(self.Gmax - RELAX)*G_last*(T<LTLT)[:, None, None, None] + RELAX

                ## RN state retrieval of R_f Layer. States in 5 different time instants are retrieved. 5 x 512 states in total.
                if T == 9:
                    G_VN = G_RC
                elif T%10==9:
                    G_VN = torch.cat((G_VN, G_RC), 1)

            ## Forward-passes of the Fully-connected layers (FC1-2)
            G_F1 = self.reluF1(self.BNF(self.F1(G_VN.flatten(1))))
            return self.F2(G_F1)

L1_epoch = 150

#torch.manual_seed(0)

log_softmax = nn.LogSoftmax(dim=0).to(device)
softmax = nn.Softmax(dim=0).to(device)
loss_function = nn.NLLLoss().to(device)

RRAM_Gmin = 0
RRAM_Gmax = 1

## Potentiation constant of the R_in layer
RRAM_PFrac1 = 0.5
## Potentiation constant of the R_f layer
RRAM_PFrac = 0.1
print("P_c of the R_in layer:",RRAM_PFrac1)
print("P_c of the R_in layer:",RRAM_PFrac)
dt = 3e4 ## state retrieval interval for the R_in layer
tau0 = 6e4 ## the decay time constant of the R_in layer

## For more efficient processing, R_in states encoded by the input events are locally saved and loaded for the following process.
DataLoad_RC(dt, tau0, RRAM_Gmin, RRAM_Gmax, RRAM_PFrac1)
train_data = torch.load("DVSG2_train_data_time_"+str(dt)+"_"+str(tau0)+"_"+str(RRAM_Gmin)+"_"+str(RRAM_Gmax)+"_"+str(RRAM_PFrac1))
T2Gt = torch.load("DVSG2_test_data_time_"+str(dt)+"_"+str(tau0)+"_"+str(RRAM_Gmin)+"_"+str(RRAM_Gmax)+"_"+str(RRAM_PFrac1)).to(device)
LTt = torch.load("last_times2_test_"+str(dt)).to(device)
Y_test = torch.load("DVSG_Y_test").to(device)

taufr = 2e6 ## the decay time constant of the R_f layer.
thfr = 0.3 ## threshold value of the Spike conversion layer.
CONV_CH = [64,128,256,512] ## Convolution layers' output feature
CONV_K = 3 ## The kernel size of the first convolution layer.
F1Ks = 512 ## The first FC layer's output dimension.
Last_RR = 50 ## R_in state retrieval times.

init_lr = 7e-3
WD = 0
AN = 0
Factor = 0.9
Patience = 2
scheduler_threshold=1e-5
batchsize = 32
sample_num = 1077

########################## MA1 Training & Inference #######################################
train_accuracy = torch.zeros(L1_epoch).to(device)
test_accuracy = torch.zeros(L1_epoch).to(device)

## RN-Net is defined using parameters.
model = Net(tau0, taufr, thfr, RRAM_PFrac, CONV_CH, CONV_K, F1Ks).to(device)
optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay = WD)
scheduler_in = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = Factor, patience = Patience, threshold= scheduler_threshold,\
                                            threshold_mode='rel', min_lr=1e-9, verbose=True)

## For monitoring the loss and the learning rate along training.
list_loss = []
list_lr = []
for EPOCH in range(L1_epoch):
    trainloader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    S = 0
    L = 0

    ## Training phase
    for nth in range(int(sample_num/batchsize)):
        event_tensor, target, LTtr = next(iter(trainloader))
        event_tensor = (event_tensor.to(device))
        LTtr = LTtr.to(device)
        target = target.long().to(device)
        
        model.train()
        optimizer.zero_grad()
        outpot = model(event_tensor[:,:Last_RR], dt, LTtr, AN, 0)
        log_prob = log_softmax(outpot)
        loss = loss_function(log_prob, target)
        loss.backward()
        outputsoftmax = torch.argmax(log_prob, dim=1)
        S += torch.where(outputsoftmax == target, 1,0).sum()
        optimizer.step()
        L += loss.data
    L = L/int(sample_num/batchsize)
    list_loss.append(L.cpu())
    list_lr.append(optimizer.param_groups[0]['lr'])
    scheduler_in.step(L)
    train_accuracy[EPOCH] = S/(int(sample_num/batchsize)*target.shape[0])

    ## Testing phase
    model.eval()
    with no_grad():
        outpot = model(T2Gt[:,:Last_RR], dt, LTt, 0, 1)
        outputsoftmax = torch.argmax(log_softmax(outpot), dim=1)
        correct1 = torch.where(outputsoftmax == Y_test, 1,0)
        test_accuracy[EPOCH] = correct1.sum()/(len(Y_test))

    ## generate the result plot every 10 epochs                                                        
    if EPOCH%10 == 9:
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(14,5))
        ax[0].plot(range(L1_epoch), train_accuracy.cpu().detach().numpy(), label="Train", color='red')
        ax[0].plot(range(L1_epoch), test_accuracy.cpu().detach().numpy(), label="Test", color='blue')
        ax[0].legend(loc='lower right')
        ax[0].set_ylim([0, 1])
        ax[0].text(0,0.9,"lr:"+"{:.1e}".format(init_lr), fontsize = 10)
        ax[0].text(0,0.85,"sche_TH:"+"{:.1e}".format(scheduler_threshold), fontsize = 10)
        ax[0].text(0,0.8,"WD:"+"{:.1e}".format(WD), fontsize = 10)
        ax[0].text(0,0.75,"Factor:"+"{:.1e}".format(Factor), fontsize = 10)
        ax[0].text(0,0.7,"Patience:"+"{:.1e}".format(Patience), fontsize = 10)
        ax[0].text(0,0.65,"Batch:"+"{:.1e}".format(batchsize), fontsize = 10)
        ax[0].text(0,0.60,"Noise:"+"{:.1e}".format(AN), fontsize = 10)
        ax[0].text(0,0.55,"1st C:"+"{:.1e}".format(CONV_K), fontsize = 10)
        ax[0].text(0,0.50,"ConvCH:"+str(CONV_CH[0])+"/"+str(CONV_CH[1])+"/"+str(CONV_CH[2]), fontsize = 10)
        ax[0].text(0,0.45,"F1K:"+str(F1Ks), fontsize = 10)
        ax[0].text(0,0.40,"ReLU", fontsize = 10)
        ax[0].set_title("{:.2e}".format(100*train_accuracy.max().item())+"/"+str(torch.argmax(train_accuracy).item())+"/"+"{:.2e}".format(100*test_accuracy.max().item())+"/"+str(torch.argmax(test_accuracy).item()))
        
        ax[1].plot(range(len(list_loss)), list_loss, color='red')
        ax[1].set_ylabel('Loss',color='red')
        
        ax2 = ax[1].twinx()

        ax2.plot(range(len(list_lr)), list_lr, color='blue')
        ax2.set_ylabel('Learning rate',color='blue')
        ax2.set_yscale('log')
        fig.savefig("Result.png")

print("Best train accuracy: ", train_accuracy.max())
print(torch.argmax(train_accuracy))
print("Best test accuracy: ", test_accuracy.max())
print(torch.argmax(test_accuracy))