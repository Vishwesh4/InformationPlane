import torch_geometric
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Tanh
from torch_geometric.nn import GINConv, SAGPooling, GCNConv

class GNN_Sup(nn.Module):
    """
    GNN with neural network for readout
    """
    def __init__(self, in_channel, hidden_channel, out_channel, num_gc_layers, batch_norm=False, **kwargs):
        super(GNN_Sup, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.is_batch_norm = batch_norm

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(hidden_channel, hidden_channel), ReLU(), Linear(hidden_channel, hidden_channel))
            else:
                nn = Sequential(Linear(in_channel, hidden_channel), ReLU(), Linear(hidden_channel, hidden_channel))
            conv = GINConv(nn, train_eps=True)
            bn = torch.nn.BatchNorm1d(hidden_channel)

            self.convs.append(conv)
            self.bns.append(bn)

        #Final layer applying network
        # self.embedder = Linear(hidden_channel*self.num_gc_layers,out_channel)
        self.embedder = Linear(hidden_channel,out_channel)


    def forward(self, x, edge_index, batch, return_rep=False):
        xs = []
        outputs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            if self.is_batch_norm:
                x = self.bns[i](x)
            xs.append(x)
            if return_rep:
                outputs.append(torch_geometric.nn.global_mean_pool(x,batch))
            # if i == 2:
                # feature_map = x2

        # xpool = [torch_geometric.nn.global_mean_pool(x, batch) for x in xs]
        # x = torch.cat(xpool, 1)
        x = torch_geometric.nn.global_mean_pool(x,batch)
        x = self.embedder(x)
        if return_rep:
            outputs.append(x)
            return x, outputs
        else:
            return x
