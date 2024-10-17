from graph_encoder import GraphEncoder, GCLModel
from dataset import CFGDataset, get_split_index
from sampler import ImbalancedSampler

import pandas as pd
import numpy as np
import time
import os
import pickle
import random
import functools
import argparse
import concurrent.futures

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import dropout_adj
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", type=str, default="split1")
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--base_model", type=str, default="SAGEConv")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm", type=str, default="BatchNorm")
    parser.add_argument("--pooling", type=str, default="TopKPooling")
    parser.add_argument("--use_pool", type=bool, default=True)
    parser.add_argument("--use_skip", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=3)
    
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--biased", type=bool, default=True)
    parser.add_argument("--ablation", type=str, default="none")
    
    parser.add_argument("--pn", type=float, default=0.6)
    parser.add_argument("--pe", type=float, default=0.6)
    
    args = parser.parse_args()
    return args



def mask_node_features_unbiased(batch_data):
    pass



def drop_edges_unbiased(batch_data):
    pass



def cal_impt_scores(dataset, p_node, p_edge):
    p_node_dict = {}
    p_edge_dict = {}
    
    for k, data in enumerate(dataset):
        addr = data["address"]
        idx2impt = np.load(project_dir + "/data/facts/" + addr + "/idx2impt.npy", allow_pickle=True).item()  
        edge_index = data.edge_index
        
        S_node = list(idx2impt.values())
        S_edge = [
            (idx2impt[int(edge_index[0, i])] + idx2impt[int(edge_index[1, i])]) / 2  
            for i in range(edge_index.size(1))
        ]

        S_max, S_mean = np.max(S_node), np.mean(S_node)
        P_node = [
            min(0.5 * p_node, (S_max - S_node[i]) / (S_max - S_mean) * p_node)
            for i in range(len(S_node))
        ]

        if len(S_edge) == 0:
            P_edge = -1
        else:
            S_max, S_mean = np.max(S_edge), np.mean(S_edge)
            P_edge = [
                min(0.5 * p_edge, (S_max - S_edge[i]) / (S_max - S_mean) * p_edge)
                for i in range(len(S_edge))
            ]

        if all(p == P_node[0] for p in P_node):
            P_node = 0.5 * p_node
        if (P_edge != -1) and all(p == P_edge[0] for p in P_edge):
            P_edge = 0.5 * p_edge
            
        p_node_dict[addr] = P_node
        p_edge_dict[addr] = P_edge
        print(k, addr)
    
    print(len(p_node_dict))
    print(len(p_edge_dict))
    return p_node_dict, p_edge_dict



def MF_biased(k, batch_data_list, p_mask):
    data = batch_data_list[k]
    addr = data["address"]
    p = p_node_dict[addr]

    n = torch.empty((data.x.size(0),), dtype=torch.float32).uniform_(0, 1) < torch.tensor(p)
    n = n.to(device)
    mask = torch.empty((data.x.size(1),), dtype=torch.float32).uniform_(0, 1) < p_mask
    mask = mask.to(device)

    tmp = data.x[n, :]
    tmp[:, mask] = 0
    data.x[n, :] = tmp
    return data



def mask_node_features_biased(batch_data, p_node=0.6, p_mask=0.1):
    batch_data_list = batch_data.to_data_list()
    L = len(batch_data_list)

    func = functools.partial(MF_biased, batch_data_list=batch_data_list, p_mask=p_mask)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(func, list(range(L)))
        
    batch_data_list = [r for r in results]
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data



def DE_biased(k, batch_data_list):
    data = batch_data_list[k]
    addr = data["address"]
    p = p_edge_dict[addr]

    if p == -1: return data
    edge_index = data.edge_index

    if isinstance(p, list):
        p = torch.tensor(p, dtype=torch.float32)
        keep = torch.bernoulli(1 - p).to(torch.bool)  
        keep = keep.to(device)
        
        edge_index_deleted = edge_index[:, keep]
        data.edge_index = edge_index_deleted
    
    else:
        edge_index_deleted, _ = dropout_adj(edge_index, p=p)
        data.edge_index = edge_index_deleted
        
    return data
    


def drop_edges_biased(batch_data, p_edge=0.6):
    batch_data_list = batch_data.to_data_list()
    L = len(batch_data_list)

    func = functools.partial(DE_biased, batch_data_list=batch_data_list)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(func, list(range(L)))
    
    batch_data_list = [r for r in results]
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data



def pretrain(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_data in dataloader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)  
        
        
        if not args.biased:
            batch_data_1 = mask_node_features_unbiased(batch_data)
            batch_data_1 = drop_edges_unbiased(batch_data_1)
            batch_data_2 = mask_node_features_unbiased(batch_data)
            batch_data_2 = drop_edges_unbiased(batch_data_2)
        else:
            if args.ablation == "edge":
                batch_data_1 = mask_node_features_biased(batch_data, p_node=args.pn)
                batch_data_2 = mask_node_features_biased(batch_data, p_node=args.pn)

            elif args.ablation == "feature":
                batch_data_1 = drop_edges_biased(batch_data, p_edge=args.pe)
                batch_data_2 = drop_edges_biased(batch_data, p_edge=args.pe)

            else:
                batch_data_1 = mask_node_features_biased(batch_data, p_node=args.pn)
                batch_data_1 = drop_edges_biased(batch_data_1, p_edge=args.pe)
                batch_data_2 = mask_node_features_biased(batch_data, p_node=args.pn)
                batch_data_2 = drop_edges_biased(batch_data_2, p_edge=args.pe)
        
        z1 = model(batch_data_1.x, batch_data_1.edge_index, batch_data_1.batch)
        z2 = model(batch_data_2.x, batch_data_2.edge_index, batch_data_2.batch)
        loss = model.loss(z1, z2)
        
        loss.backward()
        optimizer.step()
        total_loss += loss
    
    total_loss = total_loss / len(dataloader)
    return total_loss



def pretrain_model(model, dataloader, optimizer, device):
    patience = args.patience
    patience_cur = 0
    best_validation_loss = np.inf
    
    for epoch in range(1, args.epochs + 1):
        loss = pretrain(model, dataloader, optimizer, device)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.8f}, patience: {patience_cur}")
        
        
        if loss < best_validation_loss:
            patience_cur = 0
            best_validation_loss = loss
            torch.save(model.state_dict(), model_file_dir + model_file_name)
        else:
            patience_cur += 1
            
        if patience_cur > patience:
            print("Early Stop!")
            break 



if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    model_file_dir = project_dir + "/algorithm/model_files/PonziHunter_20240307/"
    model_file_name = "sage_biased_pn{}_pe{}.pt".format(args.pn, args.pe)

    _, _, _, addr_list, label_list, _, _ = get_split_index(args.split)
    dataset = CFGDataset(
        root = project_dir + "/algorithm/dataset_files/dataset_{}dim/".format(args.dim),
        addr_list = addr_list, label_list = label_list
    )
    print(dataset)

    encoder = GraphEncoder(
        in_channels=args.dim, hidden_channels=args.dim * 2, out_channels=args.dim,
        base_model=args.base_model, activation=args.activation, norm=args.norm, pooling=args.pooling,
        use_pool=args.use_pool, use_skip=args.use_skip, num_layers=args.num_layers
    ).to(device)
    model = GCLModel(encoder=encoder, hidden_channels=args.dim, proj_channels=args.proj_dim, tau=args.tau).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    dataset = [data for data in dataset if data != None]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=28)  

    if args.biased:
        p_node_dict, p_edge_dict = cal_impt_scores(dataset, p_node=args.pn, p_edge=args.pe)

    pretrain_model(model, dataloader, optimizer, device)