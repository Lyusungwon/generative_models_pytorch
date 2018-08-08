import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_h = 28, input_w = 28, hidden_size = 400, latent_size = 10):
        super(Encoder, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_h * self.input_w, self.hidden_size),
            nn.ReLU(inplace= True),
            nn.Linear(self.hidden_size, self.latent_size * 2)
            )

    def forward(self, x):
        x = x.view(-1, self.input_h * self.input_w)
        x = self.encoder(x)
        x = x.view(-1, 2, self.latent_size)
        return x

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

class NormalizingFlow(nn.Module):
    def __init__(self, latent_size = 10, K = 10):
        super(NormalizingFlow, self).__init__()
        self.latent_size = latent_size
        self.K = K
        self.flows = nn.ModuleList([PlanarFlow(self.latent_size) for i in range(self.K)])

    def forward(self, z):
        log_abs_det_jacobian_sum = 0
        for flow in self.flows:
            z, log_abs_det_jacobian = flow(z)
            log_abs_det_jacobian_sum += log_abs_det_jacobian
        return z, log_abs_det_jacobian_sum

class PlanarFlow(nn.Module):
    def __init__(self, latent_size = 10):
        super(PlanarFlow, self).__init__()
        self.latent_size = latent_size
        self.w = nn.Parameter(torch.Tensor(self.latent_size).normal_())
        self.b = nn.Parameter(torch.Tensor(1).normal_())
        self.u = nn.Parameter(torch.Tensor(self.latent_size).normal_())

    def forward(self, z):
        f = F.tanh(F.linear(z, self.w, self.b))
        f_z = z + self.u * f
        psi_z = (1 - f**2) * self.w
        log_abs_det_jacobian = torch.log((1 + torch.mm(psi_z, self.u.t() )).abs())
        return f_z, log_abs_det_jacobian