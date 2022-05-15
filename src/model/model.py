import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv,Memory


class Basic_Model(nn.Module):
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["in_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.memory=Memory(args.memory["num_pattern"],args.gcn["out_channel"],args.memroy["channel"])
        self.activation = nn.GELU()
        #self.register_buffer('attention', torch.randn(98, 655, 64))
        self.args = args

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        batch=x.shape[0]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["in_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        z,attention= self.memory(x)#[bs,n,feature]
        z = x.reshape((-1, self.args.gcn["in_channel"]))+x          # [bs * N, feature]
        x = self.fc(self.activation(z))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x,attention


        


