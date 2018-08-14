import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_h = 28, input_w = 28, hidden_size = 400, latent_size = 10, h = 5):
        super(Encoder, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.h = h
        self.encoder = nn.Sequential(
            nn.Linear(self.input_h * self.input_w, self.hidden_size),
            nn.ReLU(inplace= True),
            nn.Linear(self.hidden_size, self.h + self.latent_size * 2 )
            )

    def forward(self, x):
        x = x.view(-1, self.input_h * self.input_w)
        x = self.encoder(x)
        x = x[:, self.h:].view(-1, 2, self.latent_size)
        h = x[:, :self.h]
        return x, h

class Decoder(nn.Module):
    def __init__(self, output_h = 28, output_w = 28, hidden_size = 400, latent_size = 10):
        super(Decoder, self).__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(inplace= True),
            nn.Linear(hidden_size, output_h * output_w),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, 1, self.output_h, self.output_w)

class IAF(nn.Module):
    def __init__(self, T, latent_size, layer_num, hidden_size, h):
        super(IAF, self).__init__()
        self.T = T
        self.latent_size = latent_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.iaf = nn.ModuleList([Made(self.latent_size, self.layer_num, self.hidden_size) for i in range(self.T)])

    def forward(self, x, h):
        log_det_sum = 0
        for made in self.iaf: 
            x, log_det = made(x, h)
            log_det_sum += log_det
        return x, log_det_sum

class Made(nn.Module):
    def __init__(self, latent_size, layer_num, hidden_size = 100):
        super(Made, self).__init__()
        self.latent_size = latent_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        # self.mask_num = mask_num
        self.m = torch.Tensor([i for i in range(self.latent_size)])
        # self.m = {}
        # for i in range(self.mask_num):
        #     self.m[i] = torch.randperm(self.input_h * self.input_w)
        self.architecture = [self.latent_size + self.h]
        for j in range(self.layer_num):
            self.architecture.append(self.hidden_size)
        self.architecture.append(self.latent_size)
        self.m_net = []
        self.s_net = []
        for n1, n2 in zip(self.architecture, self.architecture[1:]):
            self.m_net.append(MadeLayer(n1, n2))
            self.m_net.append(nn.ReLU())
            self.s_net.append(MadeLayer(n1, n2))
            self.s_net.append(nn.ReLU())
        self.m_net = nn.ModuleList(self.m_net[:-1])
        self.m_net = nn.Sequential(*self.m_net)
        self.s_net = nn.ModuleList(self.s_net[:-1])
        self.s_net = nn.Sequential(*self.s_net)
        self.set_masks()

    def set_masks(self):
        m = {}
        m[0] = self.m
        for l in range(1, self.layer_num +1):
            m[l] = torch.randint(m[l-1].min().int().item(), self.latent_size - 1, size = (self.hidden_size,))
        self.mask = [m[l-1][:, None].long() <= m[l][None, :].long() for l in range(1, self.layer_num + 1)]
        self.mask.append(m[self.layer_num][:, None].long() < m[0][None, :].long())
        m_layers = [l for l in self.m_net.modules() if isinstance(l, MadeLayer)]
        s_layers = [l for l in self.s_net.modules() if isinstance(l, MadeLayer)]
        for ml, sl, mask in zip(m_layers, s_layers, self.mask):
            ml.set_mask(mask)
            sl.set_mask(mask)

    def forward(self, x, h):
        x = torch.cat([x, h], 1)
        m = self.s_net(x)
        s = self.m_net(x)
        z = x * s.exp() + m
        log_det = s.sum(1)
        return z, log_det

class MadeLayer(nn.Module):
    def __init__(self, input_l, output_l):
        super(MadeLayer, self).__init__()
        self.input_l = input_l
        self.output_l = output_l
        self.weight = nn.Parameter(torch.Tensor(output_l, input_l).normal_(0, 0.02))
        self.bias = nn.Parameter(torch.Tensor(output_l).normal_(0, 0.02))
        self.register_buffer('mask', torch.ones(output_l, input_l))
    def set_mask(self, mask):
        self.mask.data.copy_(mask.float().t())

    def forward(self, x):
        x = x.view(-1, self.input_l)
        x = F.linear(x, self.mask * self.weight, self.bias)
        return x