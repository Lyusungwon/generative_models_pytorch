import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.d_fc1 = nn.Linear(144, 12)
        self.d_fc2 = nn.Linear(12, 1)
        self.g_fc1 = nn.Linear(784, 1024)
        self.g_fc2 = nn.Linear(1024, 1024)
        self.g_fc3 = nn.Linear(1024, 784)
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        # torch.nn.init.xavier_uniform(self.conv3.weight)
        # torch.nn.init.xavier_uniform(self.d_fc1.weight)
        # torch.nn.init.xavier_uniform(self.d_fc2.weight)
        # torch.nn.init.xavier_uniform(self.g_fc1.weight)
        # torch.nn.init.xavier_uniform(self.g_fc2.weight)
        # torch.nn.init.xavier_uniform(self.g_fc3.weight)

    def discriminate(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 144)
        x = F.relu(self.d_fc1(x))
        x = self.d_fc2(x)
        return F.sigmoid(x.squeeze(1))

    def generate(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.g_fc1(x))
        x = F.relu(self.g_fc2(x))
        x = F.sigmoid(self.g_fc3(x))
        x = x.view(-1, 1, 28, 28)
        return x

    def loss_function(self, real_likelihood, fake_likelihood):
        gen_loss = -torch.sum(torch.log(fake_likelihood))
        dis_loss = -torch.sum(torch.log(real_likelihood) + torch.log(1 - fake_likelihood))
        return gen_loss, dis_loss

    def forward(self, x):
        noise = torch.rand(len(x), 1, 28, 28)
        noise = Variable(noise)
        if is_cuda:
            noise = noise.cuda()
        fake = self.generate(noise)
        real_likelihood = self.discriminate(x)
        fake_likelihood = self.discriminate(fake)
        return real_likelihood, fake_likelihood
        # real_est = torch.round(real_likelihood)
        # fake_est = torch.round(fake_likelihood)
        # accuracy = (torch.sum(real_est.data == 1) + torch.sum(fake_est.data == 0)) / float(batch_num * 2)