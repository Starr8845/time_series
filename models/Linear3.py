import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.use_mem = configs.use_mem
        
        self.Linear1 = nn.Linear(self.seq_len, 128)
        self.Linear2 = nn.Linear(128, self.pred_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        repres = self.Linear1(x.permute(0,2,1))
        repres = self.sigmoid(repres)
        result = self.Linear2(repres).permute(0,2,1) # [Batch, Output length, Channel]
        return result
    

