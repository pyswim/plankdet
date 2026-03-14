import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.utils import shuffle


class simpleCls(nn.Module):
    def __init__(self,hiddens):
        super().__init__()
        self.layers=[]
        self.layers.append(nn.Linear(4,hiddens[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hiddens)-1):
            self.layers.extend([nn.Linear(hiddens[i],hiddens[i+1]),nn.ReLU()])
        self.layers.extend([nn.Linear(hiddens[-1],3)])
        self.layers=nn.Sequential( *self.layers)

    def forward(self,x):
        return self.layers(x)

    def mytrain(self, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
        #model = model.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        cri=nn.CrossEntropyLoss()
        for epoch in range(1, epochs+1):
            #self.train()
            total_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                r= self(features)
                #target_probs = torch.zeros_like(path_probs)#path_probs[torch.arange(len(labels)), labels] + 1e-10
                #for i in range(labels.shape[0]):
                    #target_probs[i, build_path(labels[i])]=1
                #print('labels',labels)
                #print('path_probs',path_probs.shape)
                #print('target_probs',target_probs)
                labels=labels.long()
                loss =cri(r,labels)
                #torch.norm(r-labels).mean()#path_probs-target_probs,dim=1).mean()#-torch.log(target_probs).mean()
                #print('loss',loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(features)

            avg_loss = total_loss / len(train_loader.dataset)
            #val_acc = evaluate(self, val_loader, device)
            print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f}")# | Val Acc: {val_acc:.2f}%")
            scheduler.step()
    
sim=simpleCls([64])
ir=datasets.load_iris()
x,y=ir.data,ir.target
x,y=shuffle(x,y)

xi=torch.Tensor(x)[:100]
yi=torch.Tensor(y)[:100]
ds=TensorDataset(xi,yi)
dl=DataLoader(ds)
sim.mytrain(dl,dl,epochs=100)
