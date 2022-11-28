import math

import torch_geometric
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.models import GNN_Sup
from utils.edge_mi import EDGE


def train(dataloader,optimizer,loss_fun,metrics,device,epoch):
    loss_val = []
    model.train()
    for data in tqdm(dataloader,desc=f"Epoch {epoch}: Training..."):
        x, y, edge_index, batch = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.batch.to(device)
        optimizer.zero_grad()
        outputs = model(x,edge_index,batch)
        loss = loss_fun(outputs, y)
        loss.backward()
        optimizer.step()
        # metric calculation
        metrics(outputs, y)
        # Logging loss
        loss_val.append(loss.item())
    metrics_val = metrics.compute()
    metrics.reset()
    print(
        "Total Train loss: {}".format(
            np.mean(loss_val)
        )
    )
    print(f'Accuracy: {metrics_val["train_Accuracy"].item()}')
    return metrics_val["train_Accuracy"].item()

def test(dataloader,loss_fun,metrics,device,epoch):
    loss_val = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader,desc=f"Epoch {epoch}: Testing..."):
            x, y, edge_index, batch = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.batch.to(device)
            outputs = model(x,edge_index,batch)
            loss = loss_fun(outputs, y)
            # metric calculation
            metrics(outputs, y)
            # Logging loss
            loss_val.append(loss.item())
        metrics_val = metrics.compute()
    metrics.reset()
    print(
        "Total Test loss: {}".format(
            np.mean(loss_val)
        )
    )
    print(f'Accuracy: {metrics_val["test_Accuracy"].item()}')
    return metrics_val["test_Accuracy"].item()

def get_embeddings(dataloader,device):
    model.eval()
    X = []
    Y = []
    H = {}
    flag = 0
    with torch.no_grad():
        for data in tqdm(dataloader,desc="Getting Embeddings"):
            x_in, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            x_final,outputs = model(x_in,edge_index, batch, return_rep=True)
            x_input = torch_geometric.nn.global_mean_pool(data.x,data.batch)
            X.append(x_input)
            Y.append(data.y)
            for i,output in enumerate(outputs):
                if flag==0:
                    H[i] = []
                H[i].append(output.detach().cpu()) 
            flag = 1
    X = torch.cat(X)
    Y = torch.cat(Y)
    for key in H.keys():    
        H[key] = torch.cat(H[key])
        H[key] = H[key].numpy() 
    return X.numpy(),Y.numpy(),H

def calc_mi(X,Y,H):
    XH = []
    YH = []
    smoothness_vector_xt = np.array([1.0, 1.0, 1.2, 1.2])
    for i,key in enumerate(H.keys()):
        XH.append(EDGE(X,H[key],U=8, L_ensemble=10, epsilon_vector= 'range',gamma=[0.01,smoothness_vector_xt[i]]))
        YH.append(EDGE(Y,H[key],U=6, L_ensemble=10, epsilon_vector= 'range',gamma=[0.0001,smoothness_vector_xt[i]],epsilon=[0.2,0.2]))
    return XH, YH

# hyperparameters
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr            = 0.001
epochs        = 200
gamma         = 0.5
batch_size    = 128
momentum      = 0
weight_decay  = 0.0001


model = GNN_Sup(in_channel=89, hidden_channel=32,out_channel=2,num_gc_layers=3,batch_norm=True).to(device)

metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(), torchmetrics.ConfusionMatrix(num_classes=2)]
        )

trainmetrics = metricfun.clone(prefix="train_").to(device)
testmetrics = metricfun.clone(prefix="test_").to(device)

dataset = torch_geometric.datasets.TUDataset(root="./",name="DD")
trainset, testset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])
trainloader = torch_geometric.data.DataLoader(trainset, batch_size=batch_size)       
testloader = torch_geometric.data.DataLoader(testset, batch_size=batch_size)

# loss function
loss_function = nn.CrossEntropyLoss(reduction="mean")

# optimizer
optimizer = optim.Adam(model.parameters(),
                      lr=lr,
                      weight_decay=weight_decay)
XH = []
YH = []
train_acc_all = []
test_acc_all = []

for i in range(epochs):
    train_acc = train(trainloader,optimizer,loss_function,trainmetrics,device,i)
    test_acc = test(testloader,loss_function,testmetrics,device,i)
    X,Y,H = get_embeddings(trainloader, device)
    ep_xh,ep_yh = calc_mi(X, Y, H)
    XH.append(ep_xh)
    YH.append(ep_yh)
    train_acc_all.append(train_acc)
    test_acc_all.append(test_acc)


XH = np.array(XH)
YH = np.array(YH)
train_acc_all = np.array(train_acc_all)
test_acc_all = np.array(test_acc_all)

with open('./Results/mi_gnn_dd_3.npy', 'wb') as f:
    np.save(f, XH)
    np.save(f, YH)

with open('./Results/acc_val_dd.npy', 'wb') as f:
    np.save(f, train_acc_all)
    np.save(f, test_acc_all)
