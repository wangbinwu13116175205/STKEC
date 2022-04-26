import torch

import heapq
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
import numpy as np
from torch import nn, optim
import os
from sklearn.utils import shuffle
from itertools import combinations
from time import time
import os.path as osp
import networkx as nx
from torch.autograd import grad
from inverse_hvp import get_inverse_hvp
from src.trafficDataset import TrafficDataset
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

def compute_IF(args,model,
                          loss_fn,
                          train_loader,
                          test_loader,
                          lissa_params,
                          verbose=True):
    if verbose:
        print(">>> Begin computing IF values")
    
    params = list(model.parameters())


    train_grads = []
    for idx, (input, target) in enumerate(train_loader):
        loss =  loss_fn(model(input), target)
        grads = grad(loss, params)
        train_grads.append(grads)
        
    if verbose:
        print(">>> `train_grads` computed")

    s_test_values = []
    for s_test_idx, (input, target) in enumerate(test_loader):
        if verbose:
            start_time = time()
        
        input, target = loss_fn(input, target)
        loss =  loss_fn(model(input), target)
        test_grads = grad(loss, params)

        s_test = get_inverse_hvp(model,  loss_fn, train_loader, test_grads,
                                                approx_type='lissa',
                                                approx_params=lissa_params,
                                                preproc_data_fn=loss)
        s_test_values.append(s_test)
        
        if verbose and (s_test_idx+1) % 10 == 0:
            print(f">>> Completed {s_test_idx+1}/{len(test_loader)} test point, "
                     f"duration per point: {time() - start_time:.2f} seconds")
        
    if verbose:
        print(">>> `s_test_values` computed")
    IF_values = []
    for s_test_idx, s_test in enumerate(s_test_values):
            
        inf_up_loss = []
        for train_grad in train_grads:
            inf = 0
            for train_grad_p, s_test_p in zip(train_grad, s_test):
                assert train_grad_p.shape == s_test_p.shape
                inf += -torch.sum(train_grad_p * s_test_p)    # Parameter-wise dot product accumulation
            inf_up_loss.append(inf)
        IF_values.append(inf_up_loss)
        
        if verbose and (s_test_idx+1) % 10 == 0:
            print(f">>> Completed {s_test_idx+1}th test point")

    return torch.tensor(IF_values.sum())

def get_influence_node(args,inputs,model,loss_fn):
    cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, args.year+"_adj.npz"))["x"]).edges)).T)
    node_influence=[]
    for node in args.nodelist:
           
        subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph([node], num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
        #get last seven days
        train_data=TrafficDataset("", "", x=inputs["train_x"][7*24:-1, :, subgraph.numpy()], y=inputs["train_y"][7*24:-1, :,subgraph.numpy()],edge_index="", mode="subgraph")
        val_data=TrafficDataset("", "", x=inputs["val_x"][:, :, subgraph.numpy()], y=inputs["val_y"][:, :, subgraph.numpy()],edge_index="", mode="subgraph")
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.n_work)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.n_work) 
        nodeinf=compute_IF(args,model,train_loader,val_loader,loss_fn,args.lissa_params)
        node_influence.append(nodeinf)
    return heapq.nlargest(args.topk,node_influence)
    


