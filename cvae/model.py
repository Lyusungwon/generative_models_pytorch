from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, channel_size = 1, input_h = 28, input_w = 28, filter_num = 16, latent_size = 16):
        super(Encoder, self).__init__()
        self.channel_size = channel_size
        self.input_h = input_h
        self.input_w = input_w
        self.filter_num = filter_num
        self.latent_size = latent_size
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel_size, self.filter_num, 3, 1),
            nn.Conv2d(self.filter_num, self.filter_num, 3, 1),
            nn.Conv2d(self.filter_num, self.filter_num, 3, 1),
            nn.Conv2d(self.filter_num, self.filter_num*2, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(self.filter_num*2, self.filter_num*2, 3, 1),
            nn.Conv2d(self.filter_num*2, self.filter_num*2, 3, 1),
            nn.Conv2d(self.filter_num*2, 2, 3, 1)
            )

    def forward(self, x):
        x = x.view(-1, self.channel_size, self.input_h, self.input_w)
        x = self.conv(x)
        x = x.view(-1, 2, self.latent_size)
        return x

class Decoder(nn.Module):
    def __init__(self, channel_size = 1, output_h = 28, output_w = 28, filter_num = 16, latent_size = 16):
        super(Decoder, self).__init__()
        self.channel_size = channel_size
        self.output_h = output_h
        self.output_w = output_w
        self.filter_num = filter_num
        self.latent_size = latent_size
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(1, self.filter_num*2, 3, 1),
            nn.ConvTranspose2d(self.filter_num*2, self.filter_num*2, 3, 1),
            nn.ConvTranspose2d(self.filter_num*2, self.filter_num*2, 3, 1),
            nn.ConvTranspose2d(self.filter_num*2, self.filter_num*2, 3, 1),
            nn.ConvTranspose2d(self.filter_num*2, self.filter_num, 5, 1),
            nn.ConvTranspose2d(self.filter_num, self.filter_num, 5, 1),
            nn.ConvTranspose2d(self.filter_num, self.filter_num, 5, 1),
            nn.ConvTranspose2d(self.filter_num, 1, 5, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = x.view(-1, 1, 4, 4)
        x = self.convt(x)
        x = x.view(-1, self.channel_size, self.output_h, self.output_w)
        return x