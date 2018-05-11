## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.batchn1 = nn.BatchNorm2d(32)
        #32x220x220
        self.maxpool1 = nn.MaxPool2d(2)
        
        #32x110x110
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.batchn2 = nn.BatchNorm2d(64)
        #64x107x107
        self.maxpool2 = nn.MaxPool2d(2)
        
        #64x54x54
        self.conv3 = nn.Conv2d(64, 32, 3) 
        #32x52x52
        self.batchn3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(2)
        
        #32x26x26
        #flatten
        #21,632
        self.lin1 = nn.Linear(20000, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 136)
        #134

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.maxpool1(self.batchn1(F.relu(self.conv1(x))))
        x = self.maxpool2(self.batchn2(F.relu(self.conv2(x))))
        x = self.maxpool3(self.batchn3(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.sigmoid(self.lin3(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
