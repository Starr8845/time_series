import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.Linear_Seasonal_1 = nn.Linear(self.seq_len, 128)
        self.Linear_Seasonal_2 = nn.Linear(256, self.pred_len)

        self.Linear_Trend_1 = nn.Linear(self.seq_len, 128)
        self.Linear_Trend_2 = nn.Linear(256, self.pred_len)

        self.sigmoid = nn.Sigmoid()
        self.hid_size=128

        self.memory_Seasonal = nn.Parameter(torch.FloatTensor(size=(configs.mem_num, 128)))
        self.memory_Trend = nn.Parameter(torch.FloatTensor(size=(configs.mem_num, 128)))
        nn.init.xavier_normal_(self.memory_Seasonal, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.memory_Trend, gain=nn.init.calculate_gain("relu"))


    def memory_enhance(self, repres, memory, get_attention=False):
        attention = torch.einsum("nd,fd->nf", repres, memory)
        m = nn.Softmax(dim=1)
        attention = m(attention)
        output = torch.einsum("nf,fd->nd", attention, memory)
        # 对应loss constraints的计算
        values, indices = attention.topk(2, dim=1, largest=True, sorted=True)
        largest = memory[indices[:,0].squeeze()]
        second_largest = memory[indices[:,1].squeeze()]
        distance1 = torch.linalg.vector_norm(repres-largest, dim=1, ord=2)/(self.hid_size)
        distance2 = torch.linalg.vector_norm(repres-second_largest, dim=1, ord=2)/(self.hid_size)
        loss_constraint = torch.mean(distance1)
        temp = (distance1-distance2+1e-3)<0
        if torch.sum(temp)>0:
            loss2 = torch.mean((distance1-distance2+1e-3)[temp])
            loss_constraint+=loss2
        loss_constraint += torch.linalg.matrix_norm(memory)
        repres = torch.cat((output, repres), dim=1) # [batch, 4*hidden_size]
        same_proto_mask = ((indices[:,0].reshape(-1,1) - indices[:,0].reshape(1,-1))==0)
        if get_attention:
            return repres, loss_constraint, same_proto_mask, attention
        return repres, loss_constraint, same_proto_mask

    def get_repre(self, x):
        # x: [Batch, Input length, Channel]
        batch_size, input_length, variable_num = x.shape[0], x.shape[1], x.shape[2]

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        seasonal_repres = self.Linear_Seasonal_1(seasonal_init)  # [batch, channel, 128]
        seasonal_repres = self.sigmoid(seasonal_repres)
        seasonal_repres = seasonal_repres.view(-1, seasonal_repres.shape[2])
        seasonal_repres, seasonal_loss_constraint, _ = self.memory_enhance(seasonal_repres, self.memory_Seasonal,False)
        seasonal_repres = seasonal_repres.view(batch_size, variable_num, -1)

        trend_repres = self.Linear_Trend_1(trend_init)  # [batch, channel, 128]
        trend_repres = self.sigmoid(trend_repres)
        trend_repres = trend_repres.view(-1, trend_repres.shape[2])
        trend_repres, trend_loss_constraint, _ = self.memory_enhance(trend_repres, self.memory_Trend, False)
        trend_repres = trend_repres.view(batch_size, variable_num, -1)

        return (seasonal_repres, trend_repres), seasonal_loss_constraint + trend_loss_constraint
    
    def predict(self, repres_enhanced):
        seasonal_repres, trend_repres = repres_enhanced
        seasonal_output = self.Linear_Seasonal_2(seasonal_repres).permute(0,2,1) # [Batch, Output length, Channel]
        trend_output = self.Linear_Trend_2(trend_repres).permute(0,2,1) # [Batch, Output length, Channel]
        return seasonal_output + trend_output
    
    def forward(self, x):
        repres, _ = self.get_repre(x)
        result = self.predict(repres)
        return result
    