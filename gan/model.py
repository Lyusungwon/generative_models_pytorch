import torch
from torch import nn
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        # torch.nn.init.xavier_uniform(self.conv3.weight)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.dropout(F.leaky_relu(self.fc1(x)))
        x = F.dropout(F.leaky_relu(self.fc2(x)))
        x = F.dropout(F.leaky_relu(self.fc3(x)))
        return F.sigmoid(self.fc4(x))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, x):
        x = x.view(-1, 100)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)

# class Discriminator(nn.Module):
#     def __init__(self, hidden_channel1, hidden_channel2):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_channel1, 3, padding = 1)
#         self.conv2 = nn.Conv2d(hidden_channel1, hidden_channel2, 3, padding = 1)
#         self.conv3 = nn.Conv2d(hidden_channel2, hidden_channel1, 3, padding = 1)
#         self.fc1 = nn.Linear(144, 12)
#         self.fc2 = nn.Linear(12, 1)
#         # torch.nn.init.xavier_uniform(self.conv1.weight)
#         # torch.nn.init.xavier_uniform(self.conv2.weight)
#         # torch.nn.init.xavier_uniform(self.conv3.weight)
#         # torch.nn.init.xavier_uniform(self.fc1.weight)
#         # torch.nn.init.xavier_uniform(self.fc2.weight)

#     def forward(self, x):
#         x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.leaky_relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 144)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.sigmoid(x.squeeze(1))