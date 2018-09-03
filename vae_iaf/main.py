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
parser.add_argument('--name', type=str, default='vae_iaf',
                        help='name of the model')
parser.add_argument('--input-h', type=int, default=28,
                        help='height of the input (default: 28)')
parser.add_argument('--input-w', type=int, default=28,
                        help='width of the input (default: 28)')
parser.add_argument('--hidden-size', type=int, default=400,
                        help='the number of hidden nodes of encoder and decoder (default: 400)')
parser.add_argument('--flow-hidden-size', type=int, default=10,
                        help='the number of hidden nodes of autoregressive model (default: 10)')
parser.add_argument('--latent-size', type=int, default=10,
                        help='the number of latent variables (default: 10)')
parser.add_argument('--layer-num', type=int, default=2,
                        help='the number of layers of autoregressive model (default: 2)')
parser.add_argument('--L', type=int, default=1,
                        help='the number of sampling (default: 1)')
parser.add_argument('--T', type=int, default=4,
                        help='the number of autoregressive flow (default: 4)')
parser.add_argument('--h', type=int, default=2,
                        help='the number of direct code (default: 2)')
parser.add_argument('--binarize', action='store_true', default=False,
					help='binarize input')
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
				args.L, args.T, args.h, arg.binarize]
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
    encoder.load_state_dict(torch.load(args.log_directory + '/' + args.load_model+ '/{}_encoder.pt'.format(args.name)))
    decoder.load_state_dict(torch.load(args.log_directory + '/' + args.load_model + '/{}_decoder.pt'.format(args.name)))
	iaf.load_state_dict(torch.load(args.log_directory  + '/' + args.load_model + '/{}.pt'.format(args.name)))
	args.time_stamep = args.load_model[:12]

log = args.log_directory + args.name + '/' + args.time_stamp + config + '/'
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
		if args.binarize:
			input_data = binarize(input_data)
        input_data = input_data.to(device)
		z_params, h = encoder(input_data)
		z_mu = z_params[:, 0]
		z_logvar = z_params[:, 1]
		q_0 = D.Normal(z_mu, (z_logvar/ 2).exp())
		recon_loss = 0
		kld_loss = 0
		for j in range(args.L):
			z_0 = q_0.rsample().to(device)
			z_t, log_abs_det_jacobian = iaf(z_0, h)
			output_data = decoder(z_t)
			recon_loss += F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
			kld_loss += q_0.log_prob(z_0).sum() - log_abs_det_jacobian.sum() - prior.log_prob(z_t).sum()
		recon_loss /= args.L
		kld_loss /= args.L
		r_loss += recon_loss.item()
		k_loss += kld_loss.item()
		loss = kld_loss + recon_loss
        loss.backward()
        r_loss += log_pxz.item()
        k_loss += log_qk_zk.item()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.6f}'.format(
				epoch, batch_idx * len(input_data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item() / len(input_data), time.time() - start_time))
	print('====> Epoch: {} Average loss: {:.4f}\tTime: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset), time.time() - epoch_start_time))	
	wrtier.add_scalar('Train Reconstruction loss',  r_loss / len(train_loader.dataset), epoch)
	wrtier.add_scalar('Train KL divergence',  k_loss / len(train_loader.dataset), epoch)
	wrtier.add_scalar('Train loss',  train_loss / len(train_loader.dataset), epoch)

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
		if args.binarize:
			input_data = binarize(input_data)
		z_params, h = encoder(input_data)
		z_mu = z_params[:, 0]
		z_logvar = z_params[:, 1]
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		recon_loss = 0
		kld_loss = 0
		for j in range(args.L):
			z_0 = q_0.rsample().to(device)
			z_t, log_abs_det_jacobian = iaf(z_0, h)
			output_data = decoder(z_t)
			recon_loss += F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
			kld_loss += q_0.log_prob(z_0).sum() - log_abs_det_jacobian.sum() - prior.log_prob(z_t).sum()
		recon_loss /= args.L
		kld_loss /= args.L
		r_loss += recon_loss.item()
		k_loss += kld_loss.item()
		loss = recon_loss + kld_loss
		test_loss += loss.item()
		if i == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  output_data[:n]])
			writer.add_image('Reconstruction Image', comparison, epoch)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	wrtier.add_scalar('Test Reconstruction loss',  r_loss / len(test_loader.dataset), epoch)
	wrtier.add_scalar('Test KL divergence',  k_loss / len(test_loader.dataset), epoch)
	wrtier.add_scalar('Test loss',  test_loss / len(test_loader.dataset), epoch)

def sample(epoch):
	std_n = D.Normal(torch.zeros(args.latent_size + args.h).to(device), torch.ones(args.latent_size + args.h).to(device))
	sample = std_n.sample(torch.Size([64]))
	output = decoder(sample)
	writer.add_image('Sample Image', output, epoch)

for epoch in range(args.epochs):
	if not args.sample:
		train(epoch)
		test(epoch)
	sample(epoch)

if not args.sample:
    torch.save(encoder.state_dict(), log + + '{}_encoder.pt'.format(args.name))
    torch.save(decoder.state_dict(), log + '{}_decoder.pt'.format(args.name))
    torch.save(nflow.state_dict(), log + '{}.pt'.format(args.name))
    print('Model saved in ', log + '{}_encoder.pt'.format(args.name))
    print('Model saved in ', log + '{}_decoder.pt'.format(args.name))
    print('Model saved in ', log + '{}.pt'.format(args.name))
writer.close()