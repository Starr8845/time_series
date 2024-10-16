import torch
import torch.nn as nn

# memory 和CL的逻辑也许可以写在这个😮类里，类对象是model
class MemoryModule(nn.Module):
    def __init__(self, hidden_size=128, prompt_num = 256):
        super().__init__()
        self.hid_size = hidden_size
        self.memory = nn.Parameter(torch.FloatTensor(size=(prompt_num, self.hid_size)))  
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
        same_proto_mask = ((indices[:,0].reshape(-1,1) - indices[:,0].reshape(1,-1))==0)
        if get_attention:
            return repres, loss_constraint, same_proto_mask, attention
        return repres, loss_constraint, same_proto_mask

