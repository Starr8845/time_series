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
        self.Linear1 = nn.Linear(self.seq_len, 128)
        self.projector = nn.Linear(128, 128)
        self.Linear2 = nn.Linear(256, self.pred_len)
        self.sigmoid = nn.Sigmoid()
        self.hid_size=128
        self.memory = nn.Parameter(torch.FloatTensor(size=(configs.mem_num, 128)))  
        nn.init.xavier_normal_(self.memory, gain=nn.init.calculate_gain("relu"))

    def memory_enhance(self, repres, get_attention=False):
        attention = torch.einsum("nd,fd->nf", repres, self.memory)
        m = nn.Softmax(dim=1)
        attention = m(attention)
        output = torch.einsum("nf,fd->nd", attention, self.memory)
        # 对应loss constraints的计算
        values, indices = attention.topk(2, dim=1, largest=True, sorted=True)
        largest = self.memory[indices[:,0].squeeze()]
        second_largest = self.memory[indices[:,1].squeeze()]
        distance1 = torch.linalg.vector_norm(repres-largest, dim=1, ord=2)/(self.hid_size)
        distance2 = torch.linalg.vector_norm(repres-second_largest, dim=1, ord=2)/(self.hid_size)
        loss_constraint = torch.mean(distance1)
        temp = (distance1-distance2+1e-3)<0
        if torch.sum(temp)>0:
            loss2 = torch.mean((distance1-distance2+1e-3)[temp])
            loss_constraint+=loss2
        loss_constraint += torch.linalg.matrix_norm(self.memory)
        repres = torch.cat((output, repres), dim=1) # [batch, 4*hidden_size]
        return repres, loss_constraint, None
        same_proto_mask = ((indices[:,0].reshape(-1,1) - indices[:,0].reshape(1,-1))==0)
        if get_attention:
            return repres, loss_constraint, same_proto_mask, attention
        return repres, loss_constraint, same_proto_mask

    def get_repre(self, x):
        # x: [Batch, Input length, Channel]
        batch_size, input_length, variable_num = x.shape[0], x.shape[1], x.shape[2]
        repres = self.Linear1(x.permute(0,2,1))  # [batch, channel, 128]
        repres = self.sigmoid(repres)
        repres = self.projector(repres)
        repres = repres.view(-1, repres.shape[2])
        repres, loss_constraint, same_proto_mask = self.memory_enhance(repres, False)
        repres = repres.view(batch_size, variable_num, -1)
        return repres, loss_constraint, same_proto_mask
    
    def predict(self, repres_enhanced):
        result = self.Linear2(repres_enhanced).permute(0,2,1) # [Batch, Output length, Channel]
        return result
    
    def forward(self, x):
        repres, _, _ = self.get_repre(x)
        result = self.predict(repres)
        return result
    

