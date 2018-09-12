import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        return x

class VQVAE(nn.Module):
    def __init__(self, hidden_size, K, D, beta):
        super(VQVAE, self).__init__()
        self.hidden_size = hidden_size
        self.K = K
        self.D = D
        self.beta = beta
        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size = 4, stride = 2, padding = 1)
        self.res_en1 = ResidualBlock(hidden_size)
        self.res_en2 = ResidualBlock(hidden_size)
        self.end_conv = nn.Conv2d(hidden_size, D, kernel_size = 1, stride = 1, padding = 0)

        self.start_conv = nn.Conv2d(D, hidden_size, kernel_size = 1, stride = 1, padding = 0)
        self.res_de1 = ResidualBlock(hidden_size)
        self.res_de2 = ResidualBlock(hidden_size)
        self.upconv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size = 4, stride = 2, padding = 1)
        self.upconv2 = nn.ConvTranspose2d(hidden_size, 3, kernel_size = 4, stride = 2, padding = 1)
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(K, D)
        self.ze = None
        self.zq = None
        self.st_grad = None

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.res_en1(x)
        x = self.res_en2(x)
        x = self.end_conv(x)
        return x

    def distance(self, x, y):
        return ((x - y) ** 2)

    def nearest_vector(self, x):
        x = x.view(-1, 64, self.D)
        embedding_weight = self.embedding.weight
        maxi = self.distance(x.unsqueeze(1), embedding_weight.unsqueeze(1)).sum(3).min(1)[1]
        return maxi

    def decode(self, x):
        x = self.start_conv(x)
        x = self.res_de1(x)
        x = self.res_de2(x)
        x = self.relu(self.upconv1(x))
        x = self.sigmoid(self.upconv2(x))
        return x

    def loss_function(self, recon_x, x):
        # bce_loss = F.binary_cross_entropy(recon_x, x, size_average=True)
        # vq_loss = F.mse_loss(self.embed, self.ze.detach(), size_average=True)
        # commitment_loss =  F.mse_loss(self.ze, self.zq.detach(), size_average=True)
        bce_loss = F.binary_cross_entropy(recon_x, x, size_average=False)
        vq_loss = F.mse_loss(self.embed, self.ze.detach(), size_average=False)
        commitment_loss =  F.mse_loss(self.ze, self.zq.detach(), size_average=False)
        return bce_loss + vq_loss + self.beta * commitment_loss, bce_loss, vq_loss, commitment_loss

    def hook(self, grad):
        self.st_grad = grad
        return grad * 0

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        ze = self.encode(x)
        self.ze = ze.permute(0, 2, 3, 1).contiguous()
        maxi = self.nearest_vector(self.ze.detach())
        self.embed = self.embedding(maxi)
        zq = self.embedding(maxi)
        self.zq = zq.view(-1, 8, 8, 10).permute(0, 3, 1, 2).contiguous()
        recon_batch = self.decode(self.zq)
        self.zq.register_hook(self.hook)
        return recon_batch

    def st_bwd(self):
        self.ze.backward(self.st_grad)