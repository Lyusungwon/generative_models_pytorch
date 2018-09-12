import torch
from torch import nn
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 5, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 5, 2)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 5, 2)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv4.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv5.weight, 0, 0.02)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope = 0.02)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope = 0.02)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope = 0.02)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope = 0.02)
        x = self.conv5(x)
        return F.sigmoid(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1024, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, 5, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, 5, 1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 1, 5, 1)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv4.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv5.weight, 0, 0.02)

    def forward(self, x):
        x = x.view(-1, 1, 10, 10)
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope = 0.02)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope = 0.02)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope = 0.02)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope = 0.02)
        x = self.conv5(x)
        x = x.view(-1, 1, 28, 28)
        return F.tanh(x)
