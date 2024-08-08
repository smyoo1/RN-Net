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
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd.grad_mode import no_grad
import torch.optim as optim

import itertools
import tonic
from tonic import CachedDataset
import tonic.transforms as TTF


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

    def forward(self, event, time_trace, length, test):

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
                    += (length[range(SN)] >= N)*self.PFRAC*(self.GMAX - T2G[range(SN), time_trace[range(SN), N], event[range(SN), N, 2], event[range(SN), N, 0], event[range(SN), N, 1]])
                    
            del event
            del time_trace
            del length
            torch.cuda.empty_cache()

            ## Center-Crop
            print("##############################################################")
            print(T2G.shape)
            if test == 1:
                print("TEST!!, center cropped 76")
                CCrop = transforms.CenterCrop(76)
            else:
                print("Train!!, center cropped 96")
                CCrop = transforms.CenterCrop(96)
            T2G = CCrop(T2G).clone()
            print(T2G.shape)
            print("##############################################################")

        return T2G

def DataLoad_RC(dt, TAU, GMIN, GMAX, PFRAC):
    transform = TTF.NumpyAsType(int)
    ## spkrange is set for reformatting a tensor.
    ##############################################################
    ##################### Test dataset ###########################
    ##############################################################
    
    if os.path.exists('./DVSlip_test_data_time_'+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC)):
        print("test dataset exists!!")
    else:
        dataset = tonic.datasets.DVSLip(save_to='./data', transform=transform, train=False)

        spkrange = 75114

        event = torch.zeros((len(dataset),spkrange,4)).long()
        Y = torch.zeros(len(dataset))
        ## check the length of clip for all the vidoe clips.
        length = torch.zeros(len(dataset))
        for NUM in range(len(dataset)):
            if len(dataset[NUM][0]) > spkrange:
                event[NUM] = torch.from_numpy(dataset[NUM][0][:spkrange])
                length[NUM] = spkrange
            else:
                event[NUM,:len(dataset[NUM][0])] = torch.from_numpy(dataset[NUM][0])
                event[NUM,len(dataset[NUM][0]):] = torch.from_numpy(dataset[NUM][0][len(dataset[NUM][0])-1])
                length[NUM] = len(dataset[NUM][0])
            Y[NUM] = dataset[NUM][1]
        torch.save(Y,"DVSlip_Y_test")
        
        SN = len(length)

        time_trace = event[:,:,3].long().clone()
        time_trace = (time_trace//dt).long()

        last_times = event[:,-1,3].long().clone()

        for YY in range(SN):
            last_times[YY] = int(last_times[YY].item()//dt)

        print(last_times.unique())
        print(last_times.max())
        print(last_times.min())
        print(torch.bincount(last_times))
        RR = int(last_times.max().item())+1

        print("###################################")
        last_times = torch.where(last_times> 49, 49, last_times)
        time_trace = torch.where(time_trace> 49, 49, time_trace)
        print(last_times.unique())
        print(last_times.max())
        print(last_times.min())
        print(torch.bincount(last_times))
        RR = int(last_times.max().item())+1

        print("SN:", SN)
        print("RR:", RR)
        print("dt:", dt)
        print("TAU0:", TAU)
        
        Temporal_encoding = TE(RR, PFRAC, GMAX, GMIN, TAU, spkrange)
        Temporal_encoding = nn.DataParallel(Temporal_encoding)
        Temporal_encoding = Temporal_encoding.to(device)

        ## State encoding of R_in using input events from a DVS-camera.
        T2G_out = Temporal_encoding(event, time_trace, length, 1).cpu().clone()
        last_times = last_times.long().cpu().clone()

        torch.save(T2G_out,"DVSlip_test_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
        torch.save(last_times, "last_times_test_"+str(dt))
    
    ##############################################################
    #################### Train dataset ###########################
    ##############################################################
    if os.path.exists('./DVSlip_train_data_time_'+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC)):
        print("train dataset exists!!")
    else:
        dataset = tonic.datasets.DVSLip(save_to='./data', transform=transform, train=True)
        ## spkrange is set for reformatting a tensor.
        spkrange = 69281

        event = torch.zeros((len(dataset),spkrange,4))
        Y = torch.zeros(len(dataset))
        ## check the length of clip for all the vidoe clips.
        length = torch.zeros(len(dataset))
        for NUM in range(len(dataset)):
            if len(dataset[NUM][0]) > spkrange:
                event[NUM] = torch.from_numpy(dataset[NUM][0][:spkrange])
                length[NUM] = spkrange
            else:
                event[NUM,:len(dataset[NUM][0])] = torch.from_numpy(dataset[NUM][0])
                event[NUM,len(dataset[NUM][0]):] = torch.from_numpy(dataset[NUM][0][len(dataset[NUM][0])-1])
                length[NUM] = len(dataset[NUM][0])
            Y[NUM] = dataset[NUM][1]
        
        SN = len(length)

        time_trace = event[:,:,3].long().clone()
        time_trace = (time_trace//dt).long().clone()

        last_times = event[:,-1,3].long().clone()
        print(last_times.unique())
        print(last_times.max())
        print(last_times.min())

        for YY in range(SN):
            last_times[YY] = int(last_times[YY].item()//dt)

        print(last_times.unique())
        print(last_times.max())
        print(last_times.min())
        print(torch.bincount(last_times))

        print("###################################")
        last_times = torch.where(last_times> 49, 49, last_times)
        time_trace = torch.where(time_trace> 49, 49, time_trace)
        print(last_times.unique())
        print(last_times.max())
        print(last_times.min())
        print(torch.bincount(last_times))
        RR = int(last_times.max().item())+1

        print("SN:", SN)
        print("RR:", RR)
        print("dt:", dt)

        Temporal_encoding = TE(RR, PFRAC, GMAX, GMIN, TAU, spkrange)
        Temporal_encoding = nn.DataParallel(Temporal_encoding)
        Temporal_encoding = Temporal_encoding.to(device)

        train_data = []
        
        ## State encoding of R_in using input events from a DVS-camera.
        T2G_out = Temporal_encoding(event[:3724], time_trace[:3724], length[:3724], 0).detach().cpu().detach()

        for NUM in range(3724):
            train_data.append([T2G_out[NUM], Y[NUM], last_times[NUM]])

        print("1st Quarter!!")
        torch.save(train_data,"DVSlip_train_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
        del T2G_out
        torch.cuda.empty_cache()
            
        TDL = len(train_data)
        
        if TDL == 3724:
            T2G_out = Temporal_encoding(event[3724:7448], time_trace[3724:7448], length[3724:7448], 0).detach().cpu().detach()

            for NUM in range(3724):
                train_data.append([T2G_out[NUM], Y[NUM], last_times[NUM]])

            print("2nd Quarter!!")
            torch.save(train_data,"DVSlip_train_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
            del T2G_out
            torch.cuda.empty_cache()
        else:
            print("2nd Quater skipped!!")
            print("train_data current length:", TDL)
                
        TDL = len(train_data)

        if TDL == 7448:
            ## second half [7448:11172]

            T2G_out = Temporal_encoding(event[7448:11172], time_trace[7448:11172], length[7448:11172], 0).detach().cpu().detach()

            for NUM in range(3724):
                train_data.append([T2G_out[NUM], Y[NUM+7448], last_times[NUM+7448]])

            print("3rd Quater!!")
            torch.save(train_data,"DVSlip_train_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
            
            ## second half [11172:14896]
            del T2G_out
            torch.cuda.empty_cache()
        else:
            print("Third quater skipped!!")
            print("train_data current length:", TDL)
        
        TDL = len(train_data)

        if TDL == 11172:
            T2G_out = Temporal_encoding(event[11172:14896], time_trace[11172:14896], length[11172:14896], 0).detach().cpu().detach()

            for NUM in range(3724):
                train_data.append([T2G_out[NUM], Y[NUM+11172], last_times[NUM+11172]])

            torch.save(train_data,"DVSlip_train_data_time_"+str(dt)+"_"+str(TAU)+"_"+str(GMIN)+"_"+str(GMAX)+"_"+str(PFRAC))
        
    start_time = datetime.now()
    current_time = start_time.strftime("%H:%M:%S")
    print("Current Time =", current_time)

torch.manual_seed(17)
spike_grad1 = surrogate.atan()
print("##############fast_sigmoid##################")

class Net(nn.Module):
    def __init__(self, TAU0, TAUFR, THFR, K, PFrac, ConCh, ConvK, F1K, ACTIV):
        super(Net, self).__init__()
        BIAS = True
        self.BNi = nn.BatchNorm2d(2)

        ## Define Convolution layers
        self.C1 = nn.Conv2d(in_channels=2, out_channels=int(ConCh[0]*K), kernel_size=ConvK, stride = 2, bias=BIAS)
        self.BN1 = nn.BatchNorm2d(self.C1.out_channels)
        
        self.C2 = nn.Conv2d(in_channels=self.C1.out_channels, out_channels=int(ConCh[1]*K), kernel_size=3, stride = 1, padding = 1, bias=BIAS)
        self.P2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN2 = nn.BatchNorm2d(self.C2.out_channels)

        self.C3 = nn.Conv2d(in_channels=self.C2.out_channels, out_channels=int(ConCh[2]*K), kernel_size=3, stride = 1, padding =1, bias=BIAS)
        self.BN3 = nn.BatchNorm2d(self.C3.out_channels)

        self.C4 = nn.Conv2d(in_channels=self.C3.out_channels, out_channels=int(ConCh[3]*K), kernel_size=3, stride = 1, padding =1, bias=BIAS)
        self.P4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN4 = nn.BatchNorm2d(self.C4.out_channels)

        self.C5 = nn.Conv2d(in_channels=self.C4.out_channels, out_channels=int(ConCh[4]*K), kernel_size=3, stride = 1, padding =1, bias=BIAS)
        self.BN5 = nn.BatchNorm2d(self.C5.out_channels)

        self.C6 = nn.Conv2d(in_channels=self.C5.out_channels, out_channels=int(ConCh[5]*K), kernel_size=3, stride = 1, padding = 1, bias=BIAS)
        self.P6 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.BN6 = nn.BatchNorm2d(self.C6.out_channels)

        self.C7 = nn.Conv2d(in_channels=self.C6.out_channels, out_channels=int(ConCh[6]*K), kernel_size=3, stride = 1, bias=BIAS)
        self.BN7 = nn.BatchNorm2d(self.C7.out_channels)
        
        ## Calculate the output dimension of Convolution layers
        OutD1 = int((76-self.C1.kernel_size[0]+1)/2)#36
        OutD2 = int((OutD1-2-1+2)/self.P2.stride)#17 + Padding
        OutD3 = int(OutD2-2+2)#17 + Padding
        OutD4 = int((OutD3-2-1+2)/self.P4.stride)#8 + Padding
        OutD5 = int(OutD4-2+2)#8 + Padding
        OutD6 = int((OutD5-2+2-1)/self.P6.stride)#3
        OutD7 = int(OutD6-2)#1
        
        Size_Cout = self.C7.out_channels*OutD7*OutD7*5
        
        self.BN_VN = nn.BatchNorm1d(Size_Cout)

        ## Define Fully-connected layers
        self.F1 = nn.Linear(in_features=Size_Cout,out_features=int(F1K), bias = BIAS)
        self.BNF = nn.BatchNorm1d(self.F1.out_features)
        self.F2 = nn.Linear(in_features=self.F1.out_features,out_features=int(100), bias = BIAS)
        
        if ACTIV == 0:
            self.relu1 = nn.Tanh()
            self.relu2 = nn.Tanh()
            self.relu3 = nn.Tanh()
            self.relu4 = nn.Tanh()
            self.reluF1 = nn.Tanh()
        elif ACTIV == 1:
            self.relu1 = nn.Sigmoid()
            self.relu2 = nn.Sigmoid()
            self.relu3 = nn.Sigmoid()
            self.relu4 = nn.Sigmoid()
            self.reluF1 = nn.Sigmoid()
        elif ACTIV == 2:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.relu5 = nn.ReLU()
            self.relu6 = nn.ReLU()
            self.reluF1 = nn.ReLU()
        elif ACTIV == 3:
            self.relu1 = nn.LeakyReLU()
            self.relu2 = nn.LeakyReLU()
            self.relu3 = nn.LeakyReLU()
            self.relu4 = nn.LeakyReLU()
            self.reluF1 = nn.LeakyReLU()

        ## Hyperparameters
        self.Tau = TAU0
        self.TauFR = TAUFR
        self.FR_TH = THFR
        
        print(self.C1)
        print(self.C2)
        print(self.P2)
        print(self.C3)
        print(self.C4)
        print(self.P4)
        print(self.C5)
        print(self.C6)
        print(self.P6)
        print(self.C7)
        print("FR_TH:",self.FR_TH)
        print("FR_tau:","{:e}".format(self.TauFR))
        print(self.F1)

        print("OutD1:",OutD1)
        print("OutD2:",OutD2)
        print("OutD3:",OutD3)
        print("OutD4:",OutD4)
        print("OutD5:",OutD5)
        print("OutD6:",OutD6)
        print("OutD7:",OutD7)
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
            RELAX = (G[T*self.SN:(T+1)*self.SN].clone()-self.Gmin)*D_multiplier + self.Gmin
            
            ##Potentiation
            G[(T+1)*self.SN:(T+2)*self.SN] = self.PFrac*(self.Gmax - RELAX)*G[(T+1)*self.SN:(T+2)*self.SN].clone()*((T+1)<ltlt)[:, None, None, None] + RELAX
            
        return G.clone()

    def forward(self, T2G, TD, LTLT, noise_factor, testing):
        if testing == 0:
            ############################
            ######### Training #########
            ############################
            self.SN = int(T2G.shape[0])
            T2G = T2G.permute(1,0,2,3,4)
            RR = T2G.shape[0]
            T2G = T2G.reshape(T2G.shape[0]*T2G.shape[1],T2G.shape[2],T2G.shape[3],T2G.shape[4])
            
            lif = snn.Leaky(beta=1, spike_grad=spike_grad1, reset_mechanism = 'none')

            with no_grad():
                T2G = T2G + torch.randn_like(T2G) * noise_factor
                T2G = torch.clip(T2G,self.Gmin,self.Gmax)

            ## Forward-passes of the convolution layers (C1-C6).
            G_L1 = self.relu1(self.BN1(self.C1(T2G)))
            G_L2 = self.relu2(self.BN2(self.P2(self.C2(G_L1))))
            G_L3 = self.relu3(self.BN3(self.C3(G_L2)))
            G_L4 = self.relu4(self.BN4(self.P4(self.C4(G_L2+G_L3))))
            G_L5 = self.relu5(self.BN5(self.C5(G_L4)))
            G_L6 = self.relu6(self.BN6(self.P6(self.C6(G_L4+G_L5))))

            ## Forward-pass of the fifth convolution layer and Activation of the Spike Conversion layer.
            G_last = lif(self.BN7(self.C7(G_L6))/self.FR_TH, 0)[0].clone()
            
            ## R_f encoding
            G_last = self.RS_dyn(RR, G_last, TD, self.TauFR, LTLT)

            G_last = G_last.reshape(RR, self.SN, G_last.shape[1]*G_last.shape[2]*G_last.shape[3]).clone()
            
            ## RN state retrieval of R_f Layer. States in 5 different time instants are retrieved. 5 x 512 states in total.
            G_VN = torch.cat((G_last[9],G_last[19],G_last[29],G_last[39],G_last[49]), 1).clone()
            
            ## Forward-passes of the Fully-connected layers (FC1-2)
            G_F1 = self.reluF1(self.BNF(self.F1(G_VN)))
            return self.F2(G_F1)

        else:
            ###########################
            ######### Testing #########
            ###########################
            self.SN = int(T2G.shape[0])
            T2G = T2G.permute(1,0,2,3,4)
            RR = T2G.shape[0]
            D_multiplier = np.exp(-TD/self.TauFR)
            
            lif = snn.Leaky(beta=1, spike_grad=spike_grad1, reset_mechanism = 'none')

            for T in range(RR):

                ## Forward-passes of the convolution layers (C1-C6).
                G_L1 = self.relu1(self.BN1(self.C1(T2G[T])))
                G_L2 = self.relu2(self.BN2(self.P2(self.C2(G_L1))))
                G_L3 = self.relu3(self.BN3(self.C3(G_L2)))
                G_L4 = self.relu4(self.BN4(self.P4(self.C4(G_L2+G_L3))))
                G_L5 = self.relu5(self.BN5(self.C5(G_L4)))
                G_L6 = self.relu6(self.BN6(self.P6(self.C6(G_L4+G_L5))))
                
                ## Forward-pass of the fifth convolution layer and Activation of the Spike Conversion layer.
                G_last = lif(self.BN7(self.C7(G_L6))/self.FR_TH, 0)[0].clone()

                ## R_f encoding
                ##Relaxation
                if T >0:
                    RELAX = (G_RC-self.Gmin)*D_multiplier + self.Gmin
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
            return self.F2(G_F1)#, save

log_softmax = nn.LogSoftmax(dim=0).cuda()
loss_function = nn.NLLLoss().cuda()
Vflip = transforms.RandomVerticalFlip(p=0.5)
RCrop = transforms.RandomCrop(76)

L1_epoch = 150

RRAM_Gmin = 0
RRAM_Gmax = 1


## Potentiation constant of the R_in layer
RRAM_PFrac1 = 0.5
## Potentiation constant of the R_f layer
RRAM_PFrac = 0.3
print("P_c of the R_in layer:",RRAM_PFrac1)
print("P_c of the R_in layer:",RRAM_PFrac)
dt = 3e4
Tau0 = 2
tau0 = dt*Tau0
K = 1

## For more efficient processing, R_in states encoded by the input events are locally saved and loaded for the following process.
#DataLoad_RC(dt, tau0, RRAM_Gmin, RRAM_Gmax, RRAM_PFrac1)
train_data = torch.load("DVSlip_train_data_time_"+str(dt)+"_"+str(tau0)+"_"+str(RRAM_Gmin)+"_"+str(RRAM_Gmax)+"_"+str(RRAM_PFrac1))
T2Gt = torch.load("DVSlip_test_data_time_"+str(dt)+"_"+str(tau0)+"_"+str(RRAM_Gmin)+"_"+str(RRAM_Gmax)+"_"+str(RRAM_PFrac1))
LTt = torch.load("last_times_test_"+str(dt))
Y_test = torch.load("DVSlip_Y_test").long().cuda()

taufr = 2e6 ## the decay time constant of the R_f layer.
thfr = 0.3 ## threshold value of the Spike conversion layer.
CONV_channel = [64,128,128,256,256,512,512] ## Convolution layers' output feature
CONV_Ks = 5 ## The kernel size of the first convolution layer.
F1Ks = 512 ## The first FC layer's output dimension.

EM = 1e-4
init_lr=7e-5
WD = 0.005
AN = 5e-4
Factor = 0.9
Patience = 1
scheduler_threshold = 1e-6
batchsize = 32
## train sample number
sample_num = 14896
## test sample number
sample_num_test = 4975

## Active function
## 0: Tanh
## 1: Sigmoid
## 2: ReLU
## 3: LeakyReLU
activefunc = 2

########################## MA1 Training & Inference #######################################
train_accuracy = torch.zeros(L1_epoch).cuda()
test_accuracy = torch.zeros(L1_epoch).cuda()

## RN-Net is defined using parameters.
model = Net(tau0,taufr,thfr, K, RRAM_PFrac, CONV_channel, CONV_Ks, F1Ks, activefunc)
model = nn.DataParallel(model)
model = model.cuda()

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
    model.train()
    for nth in range(int(sample_num/batchsize)):
        event_tensor, target, LTtr = next(iter(trainloader))
        target = target.long().cuda()
        LTtr = LTtr.long()
        event_tensor = RCrop(event_tensor)
        
        ## Vertical flip
        outpot = model(Vflip(event_tensor), dt, LTtr, AN, 0)
        log_prob = log_softmax(outpot)
        ########################################
        loss = loss_function(log_prob, target)
        ARGM = torch.argmax(log_prob, dim = 1)
        S += torch.where(ARGM == target, 1, 0).sum()
        ########################################
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        L += loss.data
    L = L/int(sample_num/batchsize)
    list_loss.append(L.cpu().detach())
    list_lr.append(optimizer.param_groups[0]['lr'])
    scheduler_in.step(L)
    train_accuracy[EPOCH] = S/(batchsize*int(sample_num/batchsize))

    ## Testing phase
    model.eval()
    with no_grad():
        outpot = model(T2Gt, dt, LTt, 0, 1)
        log_prob = log_softmax(outpot)
        ARGM = torch.argmax(log_prob, dim = 1)
        correct_sum = torch.where(ARGM == Y_test, 1, 0).sum()
        test_accuracy[EPOCH] = correct_sum/sample_num_test

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
        ax[0].text(0,0.75,"Factor:"+str(Factor), fontsize = 10)
        ax[0].text(0,0.7,"Patience:"+str(Patience), fontsize = 10)
        ax[0].text(0,0.65,"Batch:"+str(batchsize), fontsize = 10)
        ax[0].text(0,0.60,"Noise:"+"{:.1e}".format(AN), fontsize = 10)
        ax[0].text(0,0.55,"1st C:"+str(CONV_Ks), fontsize = 10)
        ax[0].text(0,0.50,"ConvCH:"+str(CONV_channel[0])+"/"+str(CONV_channel[1])+"/"+str(CONV_channel[2])+"/"+str(CONV_channel[3])+"/"+str(CONV_channel[4]), fontsize = 10)
        ax[0].text(0,0.45,"F1K:"+str(F1Ks), fontsize = 10)
        if activefunc == 0:
            ax[0].text(0,0.40,"Tanh", fontsize = 10)
        elif activefunc == 1:
            ax[0].text(0,0.40,"Sigmoid", fontsize = 10)
        elif activefunc == 2:
            ax[0].text(0,0.40,"ReLU", fontsize = 10)
        elif activefunc == 3:
            ax[0].text(0,0.40,"LeakyReLU", fontsize = 10)
        ax[0].set_title("{:.2e}".format(100*train_accuracy.max().item())+"/"+str(torch.argmax(train_accuracy).item())+"/"+"{:.2e}".format(100*test_accuracy.max().item())+"/"+str(torch.argmax(test_accuracy).item()))
        
        ax[1].plot(range(len(list_loss)), list_loss, color='red')
        ax[1].set_ylabel('Loss',color='red')

        ax2 = ax[1].twinx()

        ax2.plot(range(len(list_lr)), list_lr, color='blue')
        ax2.set_ylabel('Learning rate',color='blue')
        ax2.set_yscale('log')
    
        fig.savefig("Results.png")