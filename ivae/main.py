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
parser.add_argument('--input-h', type=int, default=28, metavar='N')
parser.add_argument('--input-w', type=int, default=28, metavar='N')
parser.add_argument('--hidden-size', type=int, default=400, metavar='N')
parser.add_argument('--latent-size', type=int, default=10, metavar='N')
parser.add_argument('--m', type=float, default=25, metavar='N')
parser.add_argument('--alpha', type=float, default=0.5, metavar='N')
parser.add_argument('--beta', type=float, default=0.5, metavar='N')
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
				args.m, args.alpha, args.beta]
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
	encoder.load_state_dict(torch.load(args.log_directory + 'ivae/' + args.load_model + '/vae_encoder.pt'))
	decoder.load_state_dict(torch.load(args.log_directory + 'ivae/' + args.load_model + '/vae_decoder.pt'))
	args.time_stamep = args.load_model[:12]

log = args.log_directory + 'ivae/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

encoder_optimizer = optim.Adam(encoder.parameters(), lr = args.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = args.lr)

def train(epoch):
	epoch_start_time = time.time()
	encoder.train()
	decoder.train()
	d_loss = 0
	r_loss= 0
	g_loss= 0
	enc_loss= 0
	dec_loss= 0
	train_loss = 0
	for batch_idx, (input_data, label) in enumerate(train_loader):
		start_time = time.time()
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		input_data = input_data.to(device)
		## train encoder
		encoder_optimizer.zero_grad()
		z_ = encoder(input_data)
		q_z = D.Normal(z_[:, 0], (z_[:, 1]/ 2).exp())
		z = q_z.rsample().to(device)
		# epsilon = prior.sample().to(device)
		# z = z_mu + epsilon * (z_logvar / 2).exp()

		output_data = decoder(z)
		reconstruction_loss = F.mse_loss(output_data, input_data, size_average=False) / 2

		fake_z = prior.rsample().to(device)
		fake_data = decoder(fake_z)

		z_r_ = encoder(output_data.detach())
		q_z_r = D.Normal(z_r_[:, 0], (z_r_[:, 1]/ 2).exp())
		# z_r = q_z_r.sample().to(device)
		z_pp_ = encoder(fake_data.detach())
		q_z_pp = D.Normal(z_pp_[:, 0], (z_pp_[:, 1]/ 2).exp())
		# z_pp = q_z_pp.sample().to(device)

		z_kld = D.kl_divergence(q_z, prior).sum()
		z_r_kld = D.kl_divergence(q_z_r, prior).sum()
		z_pp_kld = D.kl_divergence(q_z_pp, prior).sum()

		discriminator_loss = z_kld + args.alpha * (max(0, args.m - z_r_kld) + max(0, args.m - z_pp_kld))
		encoder_loss = discriminator_loss + args.beta * reconstruction_loss
		encoder_loss.backward(retain_graph = True)
		encoder_optimizer.step()


		## train decoder
		decoder_optimizer.zero_grad()
		z_r_ = encoder(output_data)
		q_z_r = D.Normal(z_r_[:, 0], (z_r_[:, 1]/ 2).exp())
		z_pp_ = encoder(fake_data)
		q_z_pp = D.Normal(z_pp_[:, 0], (z_pp_[:, 1]/ 2).exp())

		z_r_kld = D.kl_divergence(q_z_r, prior).sum()
		z_pp_kld = D.kl_divergence(q_z_pp, prior).sum()
		generator_loss =  z_r_kld + z_pp_kld

		decoder_loss = args.alpha *generator_loss + args.beta * reconstruction_loss
		decoder_loss.backward()
		decoder_optimizer.step()

		loss = encoder_loss + decoder_loss

		d_loss += discriminator_loss.item()
		r_loss += reconstruction_loss.item() 
		g_loss += generator_loss.item() 
		enc_loss += encoder_loss.item()
		dec_loss += decoder_loss.item()
		train_loss += loss.item()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.6f}'.format(
				epoch, batch_idx * len(input_data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item() / len(input_data), time.time() - start_time))
	print('====> Epoch: {} Average loss: {:.4f}\tTime: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset), time.time() - epoch_start_time))	
	writer.add_scalars('Train loss', {'Reconstruction loss': r_loss / len(train_loader.dataset),
											'Generator loss': g_loss / len(train_loader.dataset),
											'Discriminator loss': d_loss / len(train_loader.dataset),
											'Encoder': enc_loss / len(train_loader.dataset),
											'Decoder': dec_loss / len(train_loader.dataset),
											'Train loss': train_loss / len(train_loader.dataset)}, epoch)

def test(epoch):
	encoder.eval()
	decoder.eval()
	d_loss = 0
	r_loss= 0
	g_loss= 0
	enc_loss= 0
	dec_loss= 0
	test_loss = 0
	for batch_idx, (input_data, label) in enumerate(test_loader):
		start_time = time.time()
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		input_data = input_data.to(device)
		z_ = encoder(input_data)
		q_z = D.Normal(z_[:, 0], (z_[:, 1]/ 2).exp())
		z = q_z.rsample().to(device)
		output_data = decoder(z)
		reconstruction_loss = F.mse_loss(output_data, input_data, size_average=False) / 2
		fake_z = prior.rsample().to(device)
		fake_data = decoder(fake_z)
		z_r_ = encoder(output_data.detach())
		q_z_r = D.Normal(z_r_[:, 0], (z_r_[:, 1]/ 2).exp())
		z_pp_ = encoder(fake_data.detach())
		q_z_pp = D.Normal(z_pp_[:, 0], (z_pp_[:, 1]/ 2).exp())
		z_kld = D.kl_divergence(q_z, prior).sum()
		z_r_kld = D.kl_divergence(q_z_r, prior).sum()
		z_pp_kld = D.kl_divergence(q_z_pp, prior).sum()
		discriminator_loss = z_kld + args.alpha * (max(0, args.m - z_r_kld) + max(0, args.m - z_pp_kld))
		encoder_loss = discriminator_loss + args.beta * reconstruction_loss

		z_r_ = encoder(output_data)
		q_z_r = D.Normal(z_r_[:, 0], (z_r_[:, 1]/ 2).exp())
		z_pp_ = encoder(fake_data)
		q_z_pp = D.Normal(z_pp_[:, 0], (z_pp_[:, 1]/ 2).exp())
		z_r_kld = D.kl_divergence(q_z_r, prior).sum()
		z_pp_kld = D.kl_divergence(q_z_pp, prior).sum()
		generator_loss =  z_r_kld + z_pp_kld
		decoder_loss = args.alpha * generator_loss + args.beta * reconstruction_loss

		loss = encoder_loss + decoder_loss

		d_loss += discriminator_loss.item()
		r_loss += reconstruction_loss.item() 
		g_loss += generator_loss.item() 
		enc_loss += encoder_loss.item()
		dec_loss += decoder_loss.item()
		test_loss += loss.item()
		if batch_idx == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  output_data[:n],
								  fake_data[:n]])
			writer.add_image('Reconstruction Image', comparison, epoch)
	print('====> Test set loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))
	writer.add_scalars('Test loss', {'Reconstruction loss': r_loss / len(test_loader.dataset),
											'Generator loss': g_loss / len(test_loader.dataset),
											'Discriminator loss': d_loss / len(test_loader.dataset),
											'Encoder': enc_loss / len(test_loader.dataset),
											'Decoder': dec_loss / len(test_loader.dataset),
											'Test loss': test_loss / len(test_loader.dataset)}, epoch)

def sample(epoch):
	sample = D.Normal(torch.zeros(args.latent_size).to(device), torch.ones(args.latent_size).to(device))
	output = decoder(sample.sample(torch.Size([64])))
	writer.add_image('Sample Image', output, epoch)

for epoch in range(args.epochs):
	if not args.sample:
		train(epoch)
		test(epoch)
	sample(epoch)

if not args.sample:
	torch.save(encoder.state_dict(), log + 'ivae_encoder.pt')
	torch.save(decoder.state_dict(), log + 'ivae_decoder.pt')
	print('Model saved in ', log + 'ivae_encoder.pt')
	print('Model saved in ', log + 'ivae_decoder.pt')
writer.close()