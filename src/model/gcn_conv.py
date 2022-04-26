import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
class BatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()



    def forward(self, x, adj):
        # x: [bs, N, in_features], adj: [N, N]
        input_x = torch.matmul(adj, x)              # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(input_x)             # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        if self.weight_self is not None:
            output += self.weight_self(x)               # [bs, N, out_features]
        return output

class Memory(nn.Module):
    def __init__(self, num_pattern, in_features,out_features, bias=True):
         #G*D
        super(Memory, self).__init__()
        self.num_pattern = num_pattern
        self.memory=nn.Parameter(torch.zeros(size=(num_pattern,  in_features), requires_grad=True))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        self.linear1=nn.Linear(in_features, out_features, bias=bias)
    def forward(self,x):
        v=self.linear1(x)
        attention = torch.matmul(v, self.memory.transpose(-1, -2))#B*N*G
        attention=F.softmax(attention,dim=-1)
        z=torch.matmul(attention,self.memory)#B*N*out
        out=x+z#B*N*(out)
        return out,attention


