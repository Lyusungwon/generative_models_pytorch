import torch
from torch import nn
import torch.nn.functional as F

class Made(nn.Module):
    def __init__(self, input_h = 28, input_w = 28, hidden_size = 1000, layer_size = 3):
        super(Made, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.layer_num = [self.input_h * self.input_w]
        for i in range(self.layer_size):
            self.layer_num.append(self.hidden_size)
        self.layer_num.append(self.input_h * self.input_w)
        self.setup_mask()
        self.net = []
        for n1, n2, m in zip(self.layer_num, self.layer_num[1:], self.mask):
            # print(n1, n2, m.size())
            self.net.append(MadeLayer(n1, n2, m.float()))
            self.net.append(nn.ReLU())
        self.net = nn.ModuleList(self.net[:-1])
        self.net = nn.Sequential(*self.net)

    def setup_mask(self):
        m = {}
        m[0] = torch.randperm(self.input_h * self.input_w)
        for l in range(1, self.layer_size +1):
            m[l] = torch.randint(m[l-1].min().int().item(), self.input_h * self.input_w - 1, size = (self.hidden_size,))
        self.mask = [m[l-1][None, :].long()<= m[l][:, None].long() for l in range(1, self.layer_size+1)]
        self.mask.append(m[self.layer_size][None,:].long() < m[0][:, None].long())
    def forward(self, x):
        x = self.net(x)
        x = F.sigmoid(x)
        x = x.view(-1, 1, self.input_h, self.input_w)
        return x

class MadeLayer(nn.Module):
    def __init__(self, input_l, output_l, mask):
        super(MadeLayer, self).__init__()
        self.input_l = input_l
        self.output_l = output_l
        self.mask = nn.Parameter(mask)
        self.linear = nn.Linear(input_l, output_l)
        self.linear.weight.data *= self.mask
        self.handle = self.linear.register_backward_hook(self.zero_grad)
    def zero_grad(self, module, grad_input, grad_output):
        grad_input = (grad_input[0], grad_input[1], grad_input[2] * self.mask.t())
        return grad_input

    def forward(self, x):
        x = x.view(-1, self.input_l)
        x = self.linear(x)
        return x
