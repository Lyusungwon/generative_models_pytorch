# import sys
# sys.path.append('../utils/')
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
parser.add_argument('--input-h', type=list, default=28, metavar='N')
parser.add_argument('--input-w', type=list, default=28, metavar='N')
parser.add_argument('--hidden-size', type=list, default=400, metavar='N')
parser.add_argument('--latent-size', type=list, default=10, metavar='N')
parser.add_argument('--L', type=list, default=10, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.device))
	torch.cuda.set_device(args.device)

config_list = [args.epochs, args.batch_size, args.lr, 
				args.input_h, args.input_w, 
				args.hidden_size, args.latent_size,
				args.L, args.device]
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('mnist', args.data_directory, args.batch_size)

if args.load_model != '000000000000':
	encoder = torch.load(args.log_directory  + '/' + args.load_model + '/vae_encoder.pt')
	decoder = torch.load(args.log_directory  + '/' + args.load_model + '/vae_decoder.pt')
	args.time_stamep = args.load_mode[:12]
else:
	encoder = model.Encoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)
	decoder = model.Decoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)

log = args.log_directory + 'vae/' + args.time_stamp + config + '/'
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
		reconstruction_loss = 0
		for j in range(args.L):
			epsilon = prior.sample().to(device)
			z = z_mu + epsilon * (z_logvar / 2).exp()
			output_data = decoder(z)
			reconstruction_loss += F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
		reconstruction_loss /= args.L
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		kld_loss = D.kl_divergence(q, prior).sum()
		r_loss += reconstruction_loss.item() 
		k_loss += kld_loss.item()
		loss = (reconstruction_loss + kld_loss)
		loss.backward()
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
	test_loss = 0
	for i, (input_data, label) in enumerate(test_loader):
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		input_data = input_data.to(device)
		params = encoder(input_data)
		z_mu = params[:, 0]
		z_logvar = params[:, 1]
		output_data = decoder(z_mu)
		reconstruction_loss = F.binary_cross_entropy(output_data, input_data, size_average=False)
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		kld_loss = D.kl_divergence(q, prior).sum()
		loss = reconstruction_loss + kld_loss
		test_loss += loss.item()
		if i == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  output_data[:n]])
			writer.add_image('Reconstruction Image', comparison, epoch)
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	writer.add_scalar('Test loss', test_loss, epoch)

for epoch in range(args.epochs):
	batch = 0
	train(epoch)
	test(epoch)
	sample = D.Normal(torch.zeros(10).to(device), torch.ones(10).to(device))
	output = decoder(sample.sample(torch.Size([64])))
	if not os.path.exists(log + 'results'):
		os.mkdir(log + 'results')
	save_image(output,
			   log + 'results/sample_' + str(epoch) + '.png')
	writer.add_image('Sample Image', output, epoch)

torch.save(encoder, log + 'vae_encoder.pt')
torch.save(decoder, log + 'vae_decoder.pt')
print('Model saved in ', log + 'vae_encoder.pt')
print('Model saved in ', log + 'vae_decoder.pt')
writer.close()