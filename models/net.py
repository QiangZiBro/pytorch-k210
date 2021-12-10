# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pytorch-k210 -> lenet
@IDE    ：PyCharm
@Author ：QiangZiBro
@Date   ：2021/12/3 3:20 下午
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 32, 4)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 64, 1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64, 32, 1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(32, 10, 1)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.relu4(y)
        y = self.conv5(y)
        y = self.relu6(y)
        y = self.conv6(y)
        y = self.relu6(y)

        y = y.view(y.shape[0], -1)

        return y


if __name__ == '__main__':
    model = Net()
    x = torch.rand(4, 1, 28, 28)
    print(model(x).shape)
