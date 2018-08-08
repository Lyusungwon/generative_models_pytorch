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
parser.add_argument('--latent-size', type=int, default=40, metavar='N')
parser.add_argument('--K', type=int, default=80, metavar='N')
parser.add_argument('--L', type=int, default=1, metavar='N')
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
				args.K, args.L]
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('mnist', args.data_directory, args.batch_size)

if args.load_model != '000000000000':
	encoder = torch.load(args.log_directory  + '/' + args.load_model + '/vae_nf_encoder.pt')
	decoder = torch.load(args.log_directory  + '/' + args.load_model + '/vae_nf_decoder.pt')
	nflow = torch.load(args.log_directory  + '/' + args.load_model + '/vae_nf_flow.pt')
	args.time_stamep = args.load_mode[:12]
else:
	encoder = model.Encoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)
	decoder = model.Decoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(device)
	nflow = model.NormalizingFlow(args.latent_size, args.K).to(device)

log = args.log_directory + 'vae_nf/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters())+list(nflow.parameters()), lr = args.lr)

def binarize(data):
	data = data > 0.5
	return data.float()

def train(epoch):
	epoch_start_time = time.time()
	train_loss = 0
	r_loss= 0
	k_loss = 0
	log_abs_det_jacobian_sum = 0
	encoder.train()
	decoder.train()
	for batch_idx, (input_data, label) in enumerate(train_loader):
		start_time = time.time()
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		optimizer.zero_grad()
		input_data = binarize(input_data)
		input_data = input_data.to(device)
		z_params = encoder(input_data)
		z_mu = z_params[:, 0]
		z_logvar = z_params[:, 1]
		reconstruction_loss = 0
		log_abs_det_jacobian = 0
		for j in range(args.L):
			epsilon = prior.sample().to(device)
			z = z_mu + epsilon * (z_logvar / 2).exp()
			z_t, log_abs_det_jacobian_t = nflow(z)
			log_abs_det_jacobian += log_abs_det_jacobian_t
			output_data = decoder(z_t)
			reconstruction_loss += F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
		log_abs_det_jacobian /= args.L 
		reconstruction_loss /= args.L
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		kld_loss = D.kl_divergence(prior, q).sum()
		if args.K != 0:
			log_abs_det_jacobian = log_abs_det_jacobian.sum()
			log_abs_det_jacobian_sum += log_abs_det_jacobian
		r_loss += reconstruction_loss.item() 
		k_loss += kld_loss.item()
		loss = (reconstruction_loss + kld_loss - log_abs_det_jacobian)
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
											'Determinant': - log_abs_det_jacobian_sum / len(train_loader.dataset),
											'Train loss': train_loss / len(train_loader.dataset)}, epoch)

def test(epoch):
	encoder.eval()
	decoder.eval()
	test_loss = 0
	for i, (input_data, label) in enumerate(test_loader):
		batch_size = input_data.size()[0]
		prior = D.Normal(torch.zeros(batch_size, args.latent_size).to(device), torch.ones(batch_size, args.latent_size).to(device))
		input_data = binarize(input_data)
		input_data = input_data.to(device)
		z_params = encoder(input_data)
		z_mu = z_params[:, 0]
		z_logvar = z_params[:, 1]
		z_t, log_abs_det_jacobian = nflow(z_mu)
		output_data = decoder(z_t)
		reconstruction_loss = F.binary_cross_entropy(output_data, input_data.detach(), size_average=False)
		q = D.Normal(z_mu, (z_logvar/ 2).exp())
		kld_loss = D.kl_divergence(prior, q).sum()
		if args.K != 0:
			kld_loss -= log_abs_det_jacobian.sum()
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
	sample = D.Normal(torch.zeros(args.latent_size).to(device), torch.ones(args.latent_size).to(device))
	sample_t, log_abs_det_jacobian = nflow(sample.sample(torch.Size([64])))
	output = decoder(sample_t)
	# if not os.path.exists(log + 'results'):
	# 	os.mkdir(log + 'results')
	# save_image(output,
	# 		   log + 'results/sample_' + str(epoch) + '.png')
	writer.add_image('Sample Image', output, epoch)

torch.save(encoder, log + 'vae_nf_encoder.pt')
torch.save(decoder, log + 'vae_nf_decoder.pt')
torch.save(nflow, log + 'vae_nf_flow.pt')
print('Model saved in ', log + 'vae_nf_encoder.pt')
print('Model saved in ', log + 'vae_nf_decoder.pt')
print('Model saved in ', log + 'vae_nf_flow.pt')
writer.close()