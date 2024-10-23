import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb
from Bio.Cluster import kcluster
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from utils.data_convert import generate_samples
from src.model.STKEC_model import Basic_Model
from src.model.stkec_ewc import EWC
from src.trafficDataset import TrafficDataset
from src.model import detect2
from src.model import replay

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = True
n_work = 16 
###################################################################################
                            #To Do
          #If you want to use this model on another dataset, aggregate the day vector of each node, and complete long_pattern=np.load('long_term_path'). 
          #select some nodes with stable patterns to evaluate the effectiveness of knowledge consolidation
          #select new nodes to to evaluate the effectiveness of knowledge expand
###################################################################################
def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year+1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))): 
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+args.time, str(args.year-1), loss[0]+".pkl")
    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    if 'tcn2.weight' in state_dict:
        del state_dict['tcn2.weight']
        del state_dict['tcn2.bias']
    model = Basic_Model(args)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model, loss[0]

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info

def long_term_pattern(args):
    inputs=inputs['train_x']
    T=inputs.shape[0]
    L=args.x_len

    N=inputs.shape[1]

    data=inputs[::12,:,:args.x_len]
    days=data.shape[0]//24
    L=24*days
    data23=data[:L,:,:].reshape(-1,288,N)
    data23=data23.sum(axis=0)
    long_pattern=data23.transpose(1,0)
  
    #long_pattern=np.load('long_term_path') anohter dataset if used.
  
    attention,_,_=kcluster(long_pattern,nclusters=args.cluster,dist='u')
    np_attention = np.zeros((len(attention),args.cluster))
    for i in attention:
        np_attention[i][attention[i]]=1.0
    return np_attention.astype(np.float32)

def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger
def get_max_columns(matrix):
    tensor = torch.tensor(matrix)
    max_columns, _ = torch.max(tensor, dim=1)
    return max_columns

def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(inputs, args):
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    if args.loss == "mse": lossfunc = func.mse_loss
    elif args.loss == "huber": lossfunc = func.smooth_l1_loss
    cluster_lossf=nn.CrossEntropyLoss()
    # Dataset Definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(TrafficDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work) 
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
    vars(args)["past_adj"]=args.sub_adj
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        if args.ewc:
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)
            ewc_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            model.register_ewc_params(ewc_loader, lossfunc, device)
        else:
            model = gnn_model
    else:
        gnn_model = Basic_Model(args).to(args.device)
        model = gnn_model

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    global_train_steps = len(train_loader) // args.batch_size +1

    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 8
    model.train()
    use_time = []
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(device, non_blocking=pin_memory)
            
            optimizer.zero_grad()
            pred,attention = model(data, args.sub_adj)
            batch_att=pred.shape[0]//args.sub_adj.shape[0]
            loss_cluster=0
            if args.year==args.begin_year:
                attention_label=torch.from_numpy(args.attention.repeat(batch_att,axis=0)).to(args.device)
                #print(attention.data.cpu().shape,attention_label.shape)
                loss_cluster = cluster_lossf(attention.data.cpu(),get_max_columns(attention_label).data.cpu().long())
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]
            loss = lossfunc(data.y, pred, reduction="mean")
            loss = loss+loss_cluster*0.1
            
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()
            training_loss += float(loss)
            loss.backward()
            optimizer.step()
            
            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn 
 
        # Validate Model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device,non_blocking=pin_memory)
                pred,_ = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        # Early Stop
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = Basic_Model(args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # Test Model
    test_model(best_model, args, test_loader, pin_memory)
    result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred,_ = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mae = metric(truth_, pred_, args)

        return loss

def metric(ground_truth, prediction, args):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae




def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        # Load Data 
        graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)

        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year)))) 

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        if year== args.begin_year:
            attention=long_term_pattern(args)
            vars(args)["attention"]=attention
            print(adj.shape[0])
            vars(args)["past_subgraph"]=np.array([i for i in range(adj.shape[0])])
        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            continue

        
        if year > args.begin_year and args.strategy == "incremental":
            
            model, _ = load_best_model(args)
            
            node_list = list()
            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))


            if args.detect:
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
                
                #vars(args)["infl_topk"] = int(args.influen_size*args.graph_size)
                old_adj=np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
                old_adj = old_adj/ (np.sum(old_adj, 1, keepdims=True) + 1e-6)
                old_adj= torch.from_numpy(old_adj).to(torch.float).to(args.device)

                if args.detect_infl_strategy:
                    influence_node_list=detect2.influencal_node_selection(args,old_adj,inputs)[:args.infl_topk]
                    node_list.extend(list(influence_node_list))
            
            node_list = list(set(node_list))
            if len(node_list) < int(0.2*args.graph_size):
                res=int(0.2*args.graph_size)-len(node_list)
                res_node = [a for a in range(cur_node_size) if a not in node_list]
                expand_node_list = random.sample(res_node, res)
                node_list.extend(list(expand_node_list))
            
            # Obtain subgraph of node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)

            vars(args)["past_subgraph"]=args.subgraph
        
            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            vars(args)["node_list"] = np.asarray(node_list)




        
        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        

        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
                test_model(model, args, test_loader, pin_memory=True)

    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            all12=0
            for year in range(args.begin_year, args.end_year+1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            all12=all12+result[i][j][year]
                            info+="{:.2f}\t".format(result[i][j][year])
            info+="{:.2f}\t".format(all12/7)
            logger.info("{}\t{}\t".format(i,j) + info)

    for year in range(args.begin_year, args.end_year+1):
        if year in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
            logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/test.json")
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    args = parser.parse_args()
    init(args)
    seed_set(13)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)
