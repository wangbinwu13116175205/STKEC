import numpy as np
import torch 
from torch_geometric.data import Data, Dataset
from torch.utils.data import TensorDataset
class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] # [T, Len, N]
            self.y = inputs[split+'_y'] # [T, Len, N]
        else:
            self.x = x
            self.y = y
    def __len__(self):
        
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        #x.requires_grad_()
        #y.requires_grad_()
        return Data(x=x, y=y) 

    def get_each_node(self,index):
        x= torch.tensor(self.x[:,:,index]).unsqueeze(-1)
        #x.requires_grad_()
        y = torch.Tensor(self.y[:,:,index]).unsqueeze(-1)
        #y.requires_grad_()
        return TensorDataset(x,y)

class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs # [T, Len, N]
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return Data(x=x)