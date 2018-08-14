import sys
sys.path.append('../utils/')
import argparser
import dataloader
import model
import time
import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--input-h', type=int, default=28, metavar='N')
parser.add_argument('--input-w', type=int, default=28, metavar='N')
parser.add_argument('--hidden-size', type=int, default=400, metavar='N')
parser.add_argument('--flow-hidden-size', type=int, default=20, metavar='N')
parser.add_argument('--latent-size', type=int, default=10, metavar='N')
parser.add_argument('--layer-num', type=int, default=3, metavar='N')
parser.add_argument('--T', type=int, default=3, metavar='N')
parser.add_argument('--h', type=int, default=2, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.device))
	torch.cuda.set_device(args.device)

config_list = [args.epochs, args.batch_size, args.lr, 
				args.input_h, args.input_w, 
				args.hidden_size, args.flow_hidden_size,
				args.latent_size, args.layer_num,
				args.T, args.h]
if args.sample:
	config_list.append('sample')				
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('mnist', args.data_directory, args.batch_size)

encoder = model.Encoder(args.input_h, args.input_w, args.hidden_size, args.latent_size, args.h).to(device)
decoder = model.Decoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)
iaf = model.IAF(args.T, args.latent_size, args.layer_num, args.flow_hidden_size, args.h).to(device)

if args.load_model != '000000000000':
	encoder.load_state_dict(torch.load(args.log_directory  + '/' + args.load_model + '/vae_iaf_encoder.pt')) 
	decoder.load_state_dict(torch.load(args.log_directory  + '/' + args.load_model + '/vae_iaf_decoder.pt'))
	iaf.load_state_dict(torch.load(args.log_directory  + '/' + args.load_model + '/vae_iaf.pt'))
	args.time_stamep = args.load_model[:12]

log = args.log_directory + 'vae_iaf/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters())+list(iaf.parameters()), lr = args.lr)

def binarize(data):
	data = data > 0.5
	return data.float()

def train(epoch):
	epoch_start_time = time.time()
	train_loss = 0
	r_loss= 0
	k_loss = 0
	d_loss = 0
	log_abs_det_jacobian_sum = 0
	encoder.train()
	decoder.train()
	iaf.train()
	for batch_idx, (input_data, label) in enumerate(train_loader):
		start_time = time.time()
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		optimizer.zero_grad()
		input_data = binarize(input_data).to(device)
		z_params, h = encoder(input_data)
		z_mu = z_params[:, 0]
		z_logvar = z_params[:, 1]
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		z = q.rsample().to(device)
		zs, log_det = iaf(z, h)
		output_data = decoder(zs)
		reconstruction_loss = F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
		kld_loss = D.kl_divergence(q, prior).sum()
		loss = reconstruction_loss + kld_loss - log_det.sum()
		loss.backward()
		r_loss += reconstruction_loss.item()
		k_loss += kld_loss.item()
		d_loss += log_det.sum().item()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.6f}'.format(
				epoch, batch_idx * len(input_data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item() / len(input_data), time.time() - start_time))
	print('====> Epoch: {} Average loss: {:.4f}\tTime: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset), time.time() - epoch_start_time))	
	writer.add_scalars('Train loss', {'Reconstruction loss': r_loss / len(train_loader.dataset),
											'KL divergence': k_loss / len(train_loader.dataset),
											'Determinant': - d_loss / len(train_loader.dataset),
											'Train loss': train_loss / len(train_loader.dataset)}, epoch)

def test(epoch):
	test_loss = 0
	r_loss= 0
	k_loss = 0
	d_loss = 0
	encoder.eval()
	decoder.eval()
	iaf.eval()
	for i, (input_data, label) in enumerate(test_loader):
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		input_data = binarize(input_data).to(device)
		z_params = encoder(input_data)
		z_mu = z_params[:, 0]
		z_logvar = z_params[:, 1]
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		z = q.rsample().to(device)
		zs, log_det = iaf(z)
		output_data = decoder(zs)
		reconstruction_loss = F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
		kld_loss = D.kl_divergence(q, prior).sum()
		loss = reconstruction_loss + kld_loss - log_det.sum()
		r_loss += reconstruction_loss.item()
		k_loss += kld_loss.item()
		d_loss += log_det.sum().item()
		test_loss += loss.item()
		if i == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  output_data[:n]])
			writer.add_image('Reconstruction Image', comparison, epoch)
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	writer.add_scalars('Test loss', {'Reconstruction loss': r_loss / len(train_loader.dataset),
											'KL divergence': k_loss / len(train_loader.dataset),
											'Determinant': - d_loss / len(train_loader.dataset),
											'Test loss': train_loss / len(train_loader.dataset)}, epoch)


def sample(epoch):
	sample = D.Normal(torch.zeros(args.latent_size).to(device), torch.ones(args.latent_size).to(device))
	sample_t, log_det = iaf(sample.sample(torch.Size([64])))
	output = decoder(sample_t)
	writer.add_image('Sample Image', output, epoch)

for epoch in range(args.epochs):
	if not args.sample:
		train(epoch)
		test(epoch)
	sample(epoch)

if not args.sample:
	torch.save(encoder.state_dict(), log + 'vae_iaf_encoder.pt')
	torch.save(decoder.state_dict(), log + 'vae_iaf_decoder.pt')
	torch.save(iaf.state_dict(), log + 'vae_iaf.pt')
	print('Model saved in ', log + 'vae_iaf_encoder.pt')
	print('Model saved in ', log + 'vae_iaf_decoder.pt')
	print('Model saved in ', log + 'vae_iaf.pt')
writer.close()