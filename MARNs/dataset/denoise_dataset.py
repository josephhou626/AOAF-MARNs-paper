import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
])



class imgdataset(Dataset) :
    def __init__(self,x,y = None,transform = None) :
        self.x = x 
        self.y = y
        self.transform = transform

    def __len__(self) :
        return len(self.x)
    
    def __getitem__(self,index) :
        X = self.x[index]
        if self.y is not None :
            Y = self.y[index]
        if self.transform is not None :
            X = self.transform(X)
            if self.y is not None :
                Y = self.transform(Y)

        if self.y is not None : 
            return X , Y
        else :
            return X 