import torch
from torch import nn
import torch.nn.functional as F

class Made(nn.Module):
    def __init__(self, input_h = 28, input_w = 28, hidden_size = 1000, layer_size = 2, random_order = False, mask_num = 1):
        super(Made, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.random_order = random_order
        self.mask_num = mask_num
        self.order = {}
        if self.random_order:
            for i in range(self.mask_num):
                self.order[i] = torch.randperm(self.input_h * self.input_w)
        else:
            self.order[0] = torch.Tensor([i for i in range(self.input_h * self.input_w)])

        self.layer_num = [self.input_h * self.input_w]
        for j in range(self.layer_size):
            self.layer_num.append(self.hidden_size)
        self.layer_num.append(self.input_h * self.input_w)
        self.net = []
        for n1, n2 in zip(self.layer_num, self.layer_num[1:]):
            self.net.append(MadeLayer(n1, n2))
            self.net.append(nn.ReLU())
        self.net = nn.ModuleList(self.net[:-1])
        self.net = nn.Sequential(*self.net)
        self.update_mask()
        self.weight = nn.Parameter(torch.Tensor(self.input_h * self.input_w, self.input_h * self.input_w).normal_())
        self.register_buffer('maska', torch.tril(torch.ones(self.input_h * self.input_w, self.input_h * self.input_w), -1))
    def update_mask(self):
        m = {}
        if self.random_order:
            m[0] = self.order[torch.randint(0, self.mask_num, (1,)).item()]
        else:
            m[0] = self.order[0]
        for l in range(1, self.layer_size +1):
            m[l] = torch.randint(m[l-1].min().int().item(), self.input_h * self.input_w - 1, size = (self.hidden_size,))
        self.mask = [m[l-1][:, None].long() <= m[l][None, :].long() for l in range(1, self.layer_size + 1)]
        self.mask.append(m[self.layer_size][:, None].long() < m[0][None, :].long())
        layers = [l for l in self.net.modules() if isinstance(l, MadeLayer)]
        for l, m in zip(layers, self.mask):
            l.set_mask(m)

    def forward(self, x):
        out = self.net(x)
        a = F.linear(x.view(-1, self.input_h * self.input_w), self.weight * self.maska)
        x = F.sigmoid(out + a)
        x = x.view(-1, 1, self.input_h, self.input_w)
        return x

class MadeLayer(nn.Module):
    def __init__(self, input_l, output_l):
        super(MadeLayer, self).__init__()
        self.input_l = input_l
        self.output_l = output_l
        self.weight = nn.Parameter(torch.Tensor(output_l, input_l).normal_())
        self.bias = nn.Parameter(torch.Tensor(output_l).normal_())
        self.register_buffer('mask', torch.ones(output_l, input_l))
    def set_mask(self, mask):
        self.mask.data.copy_(mask.float().t())

    def forward(self, x):
        x = x.view(-1, self.input_l)
        x = F.linear(x, self.mask * self.weight, self.bias)
        return x
