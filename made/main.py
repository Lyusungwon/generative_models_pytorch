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
parser.add_argument('--name', type=str, default='made', metavar='N')
parser.add_argument('--input-h', type=int, default=28, metavar='N')
parser.add_argument('--input-w', type=int, default=28, metavar='N')
parser.add_argument('--hidden-size', type=int, default=1000, metavar='N')
parser.add_argument('--layer-size', type=int, default=2, metavar='N')
parser.add_argument('--mask-num', type=int, default=1, metavar='N')
parser.add_argument('--start-sample', type=int, default=394, metavar='N')
parser.add_argument('--random-order', action='store_true', default=False)
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.device))
	torch.cuda.set_device(args.device)

config_list = [args.name, args.epochs, args.batch_size, args.lr, 
				args.input_h, args.input_w, 
				args.hidden_size, args.layer_size, 
				args.mask_num, args.random_order]
if args.sample:
	config_list.append('sample')
	config_list.append(args.start_sample)
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('mnist', args.data_directory, args.batch_size)

made = model.Made(args.input_h, args.input_w, args.hidden_size, args.layer_size, args.random_order, args.mask_num).to(device)
if args.load_model != '000000000000':
	made.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + '/{}.pt'.format(args.name)))
	args.time_stamp = args.load_model[:12]

log = args.log_directory + args.name + '/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)
optimizer = optim.Adam(made.parameters(), lr = args.lr)
def binarize(data):
	data = data > 0.5
	return data.float()

def train(epoch):
	epoch_start_time = time.time()
	train_loss = 0
	made.train()
	for batch_idx, (input_data, label) in enumerate(train_loader):
		start_time = time.time()
		optimizer.zero_grad()
		batch_size = input_data.size()[0]
		input_data = binarize(input_data)
		input_data = input_data.to(args.device)
		made.update_mask()
		recon = made(input_data)
		loss = F.binary_cross_entropy(recon, input_data, size_average=False)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.6f}'.format(
				epoch, batch_idx * len(input_data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item() / len(input_data), time.time() - start_time))
	print('====> Epoch: {} Average loss: {:.4f}\tTime: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset), time.time() - epoch_start_time))	
	writer.add_scalar('Train loss', train_loss / len(train_loader.dataset), epoch)

def test(epoch):
	made.eval()
	test_loss = 0
	for i, (input_data, label) in enumerate(test_loader):
		batch_size = input_data.size()[0]
		input_data = binarize(input_data)
		input_data = input_data.to(device)
		recon = made(input_data)
		loss = F.binary_cross_entropy(recon, input_data, size_average=False)
		test_loss += loss.item()
		if i == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  recon[:n]])
			if args.sample:
				inputs, outputs = sample(input_data[:n])
				comparison = torch.cat([comparison,
					  inputs, outputs])

			writer.add_image('Reconstruction Image', comparison, epoch)
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	writer.add_scalar('Test loss', test_loss, epoch)

def sample(inputs):
	made.eval()
	order = made.order[0]
	mask = (order < args.start_sample).float().view(1, 1, args.input_h, args.input_w).to(device)
	inputs = inputs * mask
	outputs = inputs.clone()
	# sample = torch.randn(1, 1, args.input_h, args.input_w).to(device)
	for i in range(args.start_sample, args.input_h * args.input_w):
		samples = made(inputs)
		# sample_add = torch.bernoulli(output.view(1, 1, args.input_h * args.input_w)* nmask).view(1, 1, args.input_h, args.input_w)
		nmask = (mask == i).float().to(device)
		# sample_add = torch.bernoulli(samples.view(len(inputs), 1, args.input_h * args.input_w)* nmask).view(len(inputs), 1, args.input_h, args.input_w)
		print(sample.size())
		print(nmask.size())
		sample_add = (samples.view(len(inputs), 1, args.input_h * args.input_w)* nmask).view(len(inputs), 1, args.input_h, args.input_w)
		outputs += sample_add
		# writer.add_image('Sample Image', inputs, epoch)
	return inputs, outputs
	# if not os.path.exists(log + 'results'):
	# 	os.mkdir(log + 'results')
	# save_image(output,
	# 		   log + 'results/sample_' + str(epoch) + '.png')


for epoch in range(args.epochs):
	if not args.sample:
		train(epoch)
	test(epoch)

if not args.sample:
	torch.save(made.state_dict(), log + '{}.pt'.format(args.name))
	print('Model saved in ', log + '{}.pt'.format(args.name))
writer.close()