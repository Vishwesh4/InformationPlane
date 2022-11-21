import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class Image_embedding(nn.Module):
    def __init__(self, input_size=784, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class Representation_embedding(nn.Module):
    def __init__(self, input_size=784, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class Label_embedding(nn.Module):
    def __init__(self, input_size=10, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class MINE_XH(nn.Module):
    def __init__(self):
        x_rep = Image_embedding()
        h_rep = Representation_embedding()
    
    def forward(self,x,h):
        output_x = x_rep(x)
        output_h = h_rep(h)

        return torch.mm(output_x,output_h.t())

class MINE_YH(nn.Module):
    def __init__(self):
        y_rep = Label_embedding()
        h_rep = Representation_embedding()
    
    def forward(self,x,h):
        output_y = y_rep(x)
        output_h = h_rep(h)

        return torch.mm(output_y,output_h.t())

class MINE_dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return self.X[index,:], self.Y[index,:]


def mutual_information(data, mine_net):
    X,Y = data
    X,Y = X.to(device),Y.to(device)
    n = data[0].shape[0]
    joint_mask = torch.eye(n)
    marginal_mask = 1 - torch.eye(n)
    
    result_matrix = mine_net(X,Y)

    # t = mine_net(joint)
    # et = torch.exp(mine_net(marginal))
    # mi_lb = torch.mean(t) - torch.log(torch.mean(et))

    t = (joint_mask*result_matrix).sum()/joint_mask.sum()
    et = (marginal_mask*torch.exp(result_matrix)).sum()/marginal_mask.sum()
    mi_lb = t - torch.log(et)
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    mi_lb , t, et = mutual_information(batch, mine_net)
    # ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    ma_et = (1-ma_rate)*ma_et + ma_rate*et

    # unbiasing use moving average
    # loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    loss = -(t - (1/ma_et.mean()).detach()*et)
    # use biased estimator
#     loss = - mi_lb
    
    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et

# def train(data, mine_net,mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
#     # data is x or y
#     result = list()
#     ma_et = 1.
#     for i in range(iter_num):
#         batch = sample_batch(data,batch_size=batch_size)\
#         , sample_batch(data,batch_size=batch_size,sample_mode='marginal')
#         mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
#         result.append(mi_lb.detach().cpu().numpy())
#         if (i+1)%(log_freq)==0:
#             print(result[-1])
#     return result

def construct_dataloader()


def cal_mi(data, mine_net, mine_net_optim, batch_size=64, iter_num=1000, log_freq=50, verbose=False):
    # data is x or y
    result = []
    ma_et = 1.
    for i in range(iter_num):
        batch_result = []
        for data in dataloader:
            mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
            batch_result.append(mi_lb.detach().cpu().numpy())
        result.append(np.mean(batch_result))
        if ((i+1)%(log_freq)==0) and verbose:
            print(result[-1])
    return result


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]