import sys
sys.path.append('../utils/')
import argparser
import dataloader
import model
import time
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--name', type=str, default='vae', metavar='N')
parser.add_argument('--input-h', type=int, default=28, metavar='N')
parser.add_argument('--input-w', type=int, default=28, metavar='N')
parser.add_argument('--hidden-size', type=int, default=400, metavar='N')
parser.add_argument('--latent-size', type=int, default=10, metavar='N')
parser.add_argument('--L', type=int, default=1, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.device))
	torch.cuda.set_device(args.device)

config_list = [args.name, args.epochs, args.batch_size, args.lr, 
				args.input_h, args.input_w, 
				args.hidden_size, args.latent_size,
				args.L]
if args.sample:
	config_list.append('sample')
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('mnist', args.data_directory, args.batch_size)

encoder = model.Encoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)
decoder = model.Decoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)
if args.load_model != '000000000000':
	encoder.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model+ '/{}_encoder.pt'.format(args.name)))
	decoder.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/{}_decoder.pt'.format(args.name)))
	args.time_stamp = args.load_model[:12]

log = args.log_directory + args.name + '/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = args.lr)

def train(epoch):
	epoch_start_time = time.time()
	train_loss = 0
	r_loss= 0
	k_loss = 0
	encoder.train()
	decoder.train()
	for batch_idx, (input_data, label) in enumerate(train_loader):
		start_time = time.time()
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		optimizer.zero_grad()
		input_data = input_data.to(device)
		params = encoder(input_data)
		z_mu = params[:, 0]
		z_logvar = params[:, 1]
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		reconstruction_loss = 0
		kld_loss = 0
		for j in range(args.L):
			z = q.rsample().to(device)
			output_data = decoder(z)
			reconstruction_loss += F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
			kld_loss += q.log_prob(z).sum() - prior.log_prob(z).sum()
		reconstruction_loss /= args.L
		kld_loss /= args.L
		loss = reconstruction_loss + kld_loss
		loss.backward()
		r_loss += reconstruction_loss.item() 
		k_loss += kld_loss.item()
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
											'Train loss': train_loss / len(train_loader.dataset)}, epoch)

def test(epoch):
	encoder.eval()
	decoder.eval()
	r_loss= 0
	k_loss = 0
	test_loss = 0
	for i, (input_data, label) in enumerate(test_loader):
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		input_data = input_data.to(device)
		params = encoder(input_data)
		z_mu = params[:, 0]
		z_logvar = params[:, 1]
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		z = q.rsample().to(device)
		output_data = decoder(z)
		reconstruction_loss = F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
		kld_loss = q.log_prob(z).sum() - prior.log_prob(z).sum()
		loss = reconstruction_loss + kld_loss
		r_loss += reconstruction_loss.item() 
		k_loss += kld_loss.item()
		test_loss += loss.item()
		if i == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  output_data[:n]])
			writer.add_image('Reconstruction Image', comparison, epoch)
	print('====> Test set loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))
	writer.add_scalars('Test loss', {'Reconstruction loss': r_loss / len(test_loader.dataset),
											'KL divergence': k_loss / len(test_loader.dataset),
											'Test loss': test_loss / len(test_loader.dataset)}, epoch)

def sample(epoch):
	sample = D.Normal(torch.zeros(args.latent_size).to(device), torch.ones(args.latent_size).to(device))
	output = decoder(sample.sample(torch.Size([64])))
	writer.add_image('Sample Image', output, epoch)

for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
	if not args.sample:
		train(epoch)
		test(epoch)
	sample(epoch)

if not args.sample:
	torch.save(encoder.state_dict(), log + '{}_encoder.pt'.format(args.name))
	torch.save(decoder.state_dict(), log + '{}_decoder.pt'.format(args.name))
	print('Model saved in ', log + '{}_encoder.pt'.format(args.name))
	print('Model saved in ', log + '{}_decoder.pt'.format(args.name))
writer.close()