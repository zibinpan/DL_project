from DTLZ1 import DTLZ1
import geatpy as ea

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt 
import numpy as np 
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# def train_transfer(pop.Phen, pop.ObjV):
# tensor_x = torch.Tensor(pop.Phen) # transform to torch tensor
# tensor_y = torch.Tensor(pop.ObjV)
# dataset = TensorDataset(tensor_x, tensor_y) # create your datset
# dataloader = DataLoader(dataset) # create your dataloader

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        # self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, data, target, N, epochs, optimizer, criterion):
    model.train()
    train_loss = []
    
    data = torch.as_tensor(data, dtype=torch.float32).to(device)
    target = torch.as_tensor(target, dtype=torch.float32).to(device)
    data_num = data.shape[0]
    Dim = data.shape[1]
    predicts_data = []
    train_time = []

    for epoch in range(1, epochs+1):
        print('epoch:', epoch)
        run_time = time.time()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        run_time = time.time() - run_time
        train_loss.append(loss.item())
        train_time.append(run_time)
    predicts_data.append(output.cpu().detach().numpy())
    predicts_data = np.hstack(predicts_data)

    return train_loss, train_time, predicts_data

    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device)
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     train_loss += loss.item()
        
    # train_time = time.time() - train_time
    # train_loss /= len(train_loader)
    # print('Training set: Average loss: {:.4f}'.format(train_loss))
    # return train_loss, train_time


# torch.save(model, 'output/MLP_epoch{}.pkl'.format(epoch))


if __name__ == "__main__":
    myProblem = DTLZ1(M=3)  # ?????????????????????
    N = 10  # ???????????????

    ## ???????????????????????????????????????????????????????????????????????????
    Encoding = 'RI'
    Field = ea.crtfld(Encoding, list(myProblem.varTypes), myProblem.ranges, myProblem.borders)
    pop = ea.Population(Encoding, Field, N)
    Vars = ea.crtpc(Encoding, pop.sizes, pop.Field)  # ?????????????????????????????????????????????
    pop.Phen = Vars
    myProblem.aimFunc(pop)  # ??????Objective Value
    ## ??????????????????????????????????????????

    # ?????????pop.ObjV ????????????????????????????????????pop.Phen??????????????????????????????
    print(pop.Phen)
    print(pop.ObjV)

    # tensor_x = torch.Tensor(pop.Phen) # transform to torch tensor
    # tensor_y = torch.Tensor(pop.ObjV)
    # dataset = TensorDataset(tensor_x, tensor_y) # create your datset
    # train_loader = DataLoader(dataset) # create your dataloader

    data = pop.Phen
    target = pop.ObjV
    epochs = 50
    model = Net(input_dim=data.shape[1]).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7, weight_decay=0.005)
    train(model, data, target, N, epochs, optimizer, criterion)
    
    