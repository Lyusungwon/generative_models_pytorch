import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from math import *
import pickle

is_cuda = torch.cuda.is_available()


class AAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=1000, latent_size = 20, prior_shape='gaussian_mixture'):
        super(AAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.prior_shape = prior_shape
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(input_size, hidden_size, latent_size)
        self.discriminator = Discriminator(2, 10)
        self.e = 1e-8
        # torch.nn.init.xavier_uniform(self.e_fc1.weight)
        # torch.nn.init.xavier_uniform(self.e_fc2.weight)
        # torch.nn.init.xavier_uniform(self.e_fc3.weight)
        # torch.nn.init.xavier_uniform(self.d_fc1.weight)
        # torch.nn.init.xavier_uniform(self.d_fc2.weight)
        # torch.nn.init.xavier_uniform(self.d_fc3.weight)

    # def encode(self, x):
    #     x = x.view(-1, 784)
    #     x = F.relu(self.e_fc1(x))
    #     x = F.relu(self.e_fc2(x))
    #     x = self.e_fc3(x)
    #     x = x.view(-1, self.latent_size)
    #     return x

    # def decode(self, x):
    #     x = F.relu(self.d_fc1(x))
    #     x = F.relu(self.d_fc2(x))
    #     x = F.sigmoid(self.d_fc3(x))
    #     return x.view(-1, 1, 28, 28)
    def sample(self, x, prior_shape):
        if prior_shape == 'gaussian_mixture':
            sample = self.gaussian_mixture(len(x), self.latent_size, 10)
        elif prior_shape == 'supervised_gaussian_mixture':
            sample = self.supervised_gaussian_mixture(len(x), self.latent_size, 10, 10)
        elif prior_shape == 'swiss_roll':
            sample = self.swiss_roll(len(x), self.latent_size, 10)
        else:
            sample = self.shape(len(x), self.latent_size)
        return torch.FloatTensor(sample)

    def forward(self, x):
        self.z = self.encoder(x)
        sample = self.sample(x, self.prior_shape)
        sample = Variable(sample)
        real_likelihood = self.discriminator(sample[:, :2])
        fake_likelihood = self.discriminator(self.z[:, :2])
        recon_batch = self.decoder(self.z)
        return recon_batch, real_likelihood, fake_likelihood

    def recon_loss(self, recon_batch, x):
        return F.binary_cross_entropy(recon_batch, x, size_average=True)

    def gen_loss(self, fake_likelihood):
        return -torch.mean(torch.log(fake_likelihood + self.e))

    def dis_loss(self, real_likelihood, fake_likelihood):
        dis_loss = - (torch.mean(torch.log(real_likelihood + self.e)) + torch.mean(torch.log(1 - fake_likelihood + self.e)))
        dis_loss /= 2
        return dis_loss

    def preprocess(self, data):
        noise = np.random.rand(data.shape[0], data.shape[1]) - 0.5
        noisy_data = data + noise
        r_data = noisy_data / data.max()
        return r_data

    def shape(self, batchsize, ndim):
        if ndim % 2 != 0:
            raise Exception("ndim must be a multiple of 2.")
        with open('datapoint.pkl', 'rb') as handle:
            data = pickle.load(handle)
        r_data = self.preprocess(data)
        padding = np.random.rand(len(r_data), ndim - 2)
        r_data = np.concatenate((r_data, padding), axis=1)
        np.random.shuffle(r_data)
        return r_data[:batchsize]

    def gaussian_mixture(self, batchsize, ndim, num_labels):
        if ndim % 2 != 0:
            raise Exception("ndim must be a multiple of 2.")

        def sample(x, y, label, num_labels):
            shift = 1.4
            r = 2.0 * np.pi / float(num_labels) * float(label)
            new_x = x * cos(r) - y * sin(r)
            new_y = x * sin(r) + y * cos(r)
            new_x += shift * cos(r)
            new_y += shift * sin(r)
            return np.array([new_x, new_y]).reshape((2,))

        x_var = 0.5
        y_var = 0.05
        x = np.random.normal(0, x_var, (batchsize, ndim // 2))
        y = np.random.normal(0, y_var, (batchsize, ndim // 2))
        z = np.empty((batchsize, ndim), dtype=np.float32)
        for batch in range(batchsize):
            for zi in range(ndim // 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1), num_labels)
        return z

    def supervised_gaussian_mixture(self, batchsize, ndim, label_indices, num_labels):
        if ndim % 2 != 0:
            raise Exception("ndim must be a multiple of 2.")

        def sample(x, y, label, num_labels):
            shift = 1.4
            r = 2.0 * np.pi / float(num_labels) * float(label)
            new_x = x * cos(r) - y * sin(r)
            new_y = x * sin(r) + y * cos(r)
            new_x += shift * cos(r)
            new_y += shift * sin(r)
            return np.array([new_x, new_y]).reshape((2,))

        x_var = 0.5
        y_var = 0.05
        x = np.random.normal(0, x_var, (batchsize, ndim // 2))
        y = np.random.normal(0, y_var, (batchsize, ndim // 2))
        z = np.empty((batchsize, ndim), dtype=np.float32)
        for batch in range(batchsize):
            for zi in range(ndim // 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], num_labels)
        return z

    def swiss_roll(self, batchsize, ndim, num_labels):
        def sample(label, num_labels):
            uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
            r = sqrt(uni) * 3.0
            rad = np.pi * 4.0 * sqrt(uni)
            x = r * cos(rad)
            y = r * sin(rad)
            return np.array([x, y]).reshape((2,))

        z = np.zeros((batchsize, ndim), dtype=np.float32)
        for batch in range(batchsize):
            for zi in range(ndim // 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(random.randint(0, num_labels - 1), num_labels)
        return z


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, latent_size)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        # torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.latent_size)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        # torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, latent_size, hidden_size):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        # torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, self.latent_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)
