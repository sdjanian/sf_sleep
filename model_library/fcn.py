"""
    Taken from this github https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
"""
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_shape = 1, nb_classes = 5, kernel_size=8):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size*2//3, padding=kernel_size*2//6)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size*4//9, padding=kernel_size*4//18)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#input_shape = (1920,1) # specify the input shape of your data
#nb_classes = 2 # specify the number of classes in your problem
#model = FCN(input_shape, nb_classes,32)
#print(model)
#import torchsummary
#torchsummary.summary(model,(1,1920))
