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
    print(
        "Total Train loss: {}".format(
            np.mean(loss_val)
        )
    )
    print(metrics_val["train_Accuracy"])

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
    print(metrics_val["test_Accuracy"])

# hyperparameters
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr            = 0.00001
epochs        = 10
gamma         = 0.5
batch_size    = 256
momentum      = 0
weight_decay  = 0.0001

class MNIST_model(nn.Module):
    def __init__(self, input_size=784, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        
    def forward(self, input,return_rep=False):
        output1 = F.tanh(self.fc1(input))
        output2 = F.tanh(self.fc2(output1))
        output3 = self.fc3(output2)
        if return_rep:
            return output3,F.tanh(output3),output2,output1
        else:
            return output3

model = MNIST_model().to(device)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
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

for i in range(epochs):
    train(trainloader,optimizer,loss_function,trainmetrics,device,i)
    test(testloader,loss_function,testmetrics,device,i)