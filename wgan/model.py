import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.s = 32
        self.conv1 = nn.Conv2d(1, self.s, 3, 1)
        # self.conv1_bn = nn.BatchNorm2d(self.s)
        self.conv2 = nn.Conv2d(self.s, self.s * 2, 3, 2)
        # self.conv2_bn = nn.BatchNorm2d(self.s * 2)
        self.conv3 = nn.Conv2d(self.s * 2, self.s * 4, 3, 2)
        # self.conv3_bn = nn.BatchNorm2d(self.s * 4)
        self.conv4 = nn.Conv2d(self.s * 4, 1, 4, 2)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv4.weight, 0, 0.02)

    def forward(self, x):
        # x = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope=0.02)
        # x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope=0.02)
        # x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope=0.02)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.02)
        x = self.conv4(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.s = 32
        self.conv1 = nn.ConvTranspose2d(1, self.s * 4, 4, 2)
        self.conv1_bn = nn.BatchNorm2d(self.s * 4)
        self.conv2 = nn.ConvTranspose2d(self.s * 4, self.s * 2, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(self.s * 2)
        self.conv3 = nn.ConvTranspose2d(self.s * 2, self.s, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(self.s)
        self.conv4 = nn.ConvTranspose2d(self.s, 1, 3, 1)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv4.weight, 0, 0.02)

    def forward(self, x):
        x = x.view(-1, 1, 10, 10)
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope=0.02)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope=0.02)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope=0.02)
        x = self.conv4(x)
        x = x.view(-1, 1, 28, 28)
        return F.tanh(x)

def gradient_penalty(input_data, fake, critic):
    e = torch.rand(len(input_data), 1, 1, 1)
    e = e.to(device)
    mix = e * input_data + (1 - e) * fake
    gradient = critic(mix)
    grad = torch.autograd.grad(outputs=gradient, inputs=mix,
                      grad_outputs= torch.ones_like(gradient).to(device),
                      create_graph=True, retain_graph=True)[0]
    grad = grad.view(len(input_data), -1)
    norm = grad.norm(2, dim=1)
    return ((norm - 1) **2).mean()

# def __init__(self, input_size, hidden_size):
#     super(Critic, self).__init__()
#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.fc1 = nn.Linear(input_size, hidden_size)
#     self.fc2 = nn.Linear(hidden_size, 512)
#     self.fc3 = nn.Linear(512, 256)
#     self.fc4 = nn.Linear(256, 1)
#     torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
#     torch.nn.init.normal_(self.fc2.weight, 0, 0.02)
#     torch.nn.init.normal_(self.fc3.weight, 0, 0.02)
#     torch.nn.init.normal_(self.fc4.weight, 0, 0.02)
#
# def forward(self, x):
#     x = x.view(-1, self.input_size)
#     x = F.leaky_relu(self.fc1(x), negative_slope = 0.02)
#     x = F.leaky_relu(self.fc2(x), negative_slope = 0.02)
#     x = F.leaky_relu(self.fc3(x), negative_slope = 0.02)
#     x = self.fc4(x)
#     return x
#
#
# def __init__(self):
#     super(Generator, self).__init__()
#     self.fc1 = nn.Linear(100, 256)
#     self.fc2 = nn.Linear(256, 512)
#     self.fc3 = nn.Linear(512, 1024)
#     self.fc4 = nn.Linear(1024, 784)
#     torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
#     torch.nn.init.normal_(self.fc2.weight, 0, 0.02)
#     torch.nn.init.normal_(self.fc3.weight, 0, 0.02)
#     torch.nn.init.normal_(self.fc4.weight, 0, 0.02)

# def forward(self, x):
#     x = x.view(-1, 100)
#     x = F.leaky_relu(self.fc1(x), negative_slope = 0.02)
#     x = F.leaky_relu(self.fc2(x), negative_slope = 0.02)
#     x = F.leaky_relu(self.fc3(x), negative_slope = 0.02)
#     x = F.tanh(self.fc4(x))
#     return x.view(-1, 1, 28, 28)

#


# import torch
# from torch import nn
# import torch.nn.functional as F

# is_cuda = torch.cuda.is_available()

# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.conv1 = nn.Conv2d(1, 128, 5, 1)
#         self.conv1_bn = nn.BatchNorm2d(128)
#         self.conv2 = nn.Conv2d(128, 256, 5, 1)
#         self.conv2_bn = nn.BatchNorm2d(256)
#         self.conv3 = nn.Conv2d(256, 512, 5, 1)
#         self.conv3_bn = nn.BatchNorm2d(512)
#         self.conv4 = nn.Conv2d(512, 1024, 5, 2)
#         self.conv4_bn = nn.BatchNorm2d(1024)
#         self.conv5 = nn.Conv2d(1024, 1, 5, 2)
#         torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv4.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv5.weight, 0, 0.02)

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope = 0.02)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope = 0.02)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope = 0.02)
#         x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope = 0.02)
#         x = self.conv5(x)
#         return x

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.conv1 = nn.ConvTranspose2d(1, 1024, 3, 1)
#         self.conv1_bn = nn.BatchNorm2d(1024)
#         self.conv2 = nn.ConvTranspose2d(1024, 512, 5, 1)
#         self.conv2_bn = nn.BatchNorm2d(512)
#         self.conv3 = nn.ConvTranspose2d(512, 256, 5, 1)
#         self.conv3_bn = nn.BatchNorm2d(256)
#         self.conv4 = nn.ConvTranspose2d(256, 128, 5, 1)
#         self.conv4_bn = nn.BatchNorm2d(128)
#         self.conv5 = nn.ConvTranspose2d(128, 1, 5, 1)
#         torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv4.weight, 0, 0.02)
#         torch.nn.init.normal_(self.conv5.weight, 0, 0.02)

#     def forward(self, x):
#         x = x.view(-1, 1, 10, 10)
#         x = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope = 0.02)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope = 0.02)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope = 0.02)
#         x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope = 0.02)
#         x = self.conv5(x)
#         x = x.view(-1, 1, 28, 28)
#         return F.tanh(x)
