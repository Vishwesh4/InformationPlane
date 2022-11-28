import math

import torch
import torchmetrics
import numpy as np
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

# from mine_calculate import cal_mi
from utils.edge_mi import EDGE
# from MI_renyi_alpha_entropy import Ent

epsilon = 10**(-8) 

def train(dataloader,optimizer,loss_fun,metrics,device,epoch):
    loss_val = []
    model.train()
    for data in tqdm(dataloader,desc=f"Epoch {epoch}: Training..."):
        image, label = data
        image = torch.flatten(image,start_dim=1,end_dim=-1)
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fun(outputs, label)
        loss.backward()
        optimizer.step()
        # metric calculation
        metrics(outputs, label)
        # Logging loss
        loss_val.append(loss.item())
    metrics_val = metrics.compute()
    metrics.reset()
    # print(
    #     "Total Train loss: {}".format(
    #         np.mean(loss_val)
    #     )
    # )
    # print(f'Accuracy: {metrics_val["train_Accuracy"].item()}')

def test(dataloader,loss_fun,metrics,device,epoch):
    loss_val = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader,desc=f"Epoch {epoch}: Testing..."):
            image, label = data
            image = torch.flatten(image,start_dim=1,end_dim=-1)
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = loss_fun(outputs, label)
            # metric calculation
            metrics(outputs, label)
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

def get_embeddings(dataloader,device):
    model.eval()
    X = []
    Y = []
    H = {}
    flag = 0
    with torch.no_grad():
        for data in tqdm(dataloader,desc="Getting Embeddings"):
            image,label = data
            image = torch.flatten(image,start_dim=1,end_dim=-1)
            image_in = image.to(device)
            outputs = model(image_in,return_rep=True)
            X.append(image)
            Y.append(label)
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

def calc_mi(X,Y,H,idx):
    XH = []
    YH = []
    smoothness_vector_xt = np.array([0.8, 1.0, 1.2, 1.2])
    for i,key in enumerate(H.keys()):
        XH.append(EDGE(X[idx,:],H[key][idx,:],U=15, L_ensemble=10, epsilon_vector= 'range',gamma=[0.2,smoothness_vector_xt[i]]))
        YH.append(EDGE(Y[idx],H[key][idx,:],U=6, L_ensemble=10, epsilon_vector= 'range',gamma=[0.0001,smoothness_vector_xt[i]],epsilon=[0.2,0.2]))
        # XH.append(MI(X[idx,:],H[key][idx,:],alpha=1.01,gamma=2)[0])
        # YH.append(MI(H[key][idx,:],Y[idx]/10,alpha=1.01,gamma=2)[0])
    return XH, YH

# def calc_mi_renyi(X,H,idx,gamma=2,alpha=1.01):
#     S_all = []
#     A_X, S_X = Ent(X[idx],gamma=gamma,alpha=alpha)
#     for i,key in enumerate(H.keys()):
#         A_Y, S_Y = Ent(H[key][idx,:],gamma=gamma,alpha=alpha)
#         A_XY = A_X*A_Y/np.trace(A_X*A_Y)
#         _, eigenv, _ = np.linalg.svd(A_XY)
#         S_XY = 1/(1-alpha)*np.log2(np.sum(eigenv**alpha)+epsilon).real  
#         S = S_X + S_Y - S_XY
#         S_all.append(S)
#     return S_all

# def calc_mi(X,Y,H,idx):
#     Y = np.reshape(Y,(-1,1))
#     XH = calc_mi_renyi(X, H, idx)
#     YH = calc_mi_renyi(Y, H, idx)
#     return XH,YH

# hyperparameters
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr            = 0.00001
epochs        = 250
gamma         = 0.5
batch_size    = 256
momentum      = 0
weight_decay  = 0.0001
np.random.seed(2022)

class MNIST_model(nn.Module):
    def __init__(self, input_size=784, hidden_size=[80,30,30,10], num_layers=4):
        super().__init__()
        self.n_layers = num_layers
        self.fc = torch.nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(self.n_layers-1):
            self.fc.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.fc.append(nn.Linear(hidden_size[i+1], 10))
        
    def forward(self, input,return_rep=False):
        outputs = []
        x = input
        for i in range(self.n_layers):
            x = F.tanh(self.fc[i](x))
            outputs.append(x)
        pred = self.fc[-1](x)
        if return_rep:
            return outputs
        else:
            return pred

model = MNIST_model().to(device)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(), torchmetrics.ConfusionMatrix(num_classes=10)]
        )

trainmetrics = metricfun.clone(prefix="train_").to(device)
testmetrics = metricfun.clone(prefix="test_").to(device)

trainset = torchvision.datasets.MNIST("/home/vishwesh/Projects/MAT1510-CourseProject/data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST("/home/vishwesh/Projects/MAT1510-CourseProject/data", train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

# loss function
loss_function = nn.CrossEntropyLoss(reduction="mean")

# optimizer
optimizer = optim.Adam(model.parameters(),
                      lr=lr,
                      weight_decay=weight_decay)
XH = []
YH = []
#Set of random points for calculation of MI
idx = np.random.permutation(len(trainset))[:10000]

for i in range(epochs):
    train(trainloader,optimizer,loss_function,trainmetrics,device,i)
    test(testloader,loss_function,testmetrics,device,i)
    # if i%10==0:
    X,Y,H = get_embeddings(trainloader, device)
    ep_xh,ep_yh = calc_mi(X, Y, H, idx)
    XH.append(ep_xh)
    YH.append(ep_yh)

XH = np.array(XH)
YH = np.array(YH)

with open('./results/mi_tanh_full.npy', 'wb') as f:
    np.save(f, XH)
    np.save(f, YH)

