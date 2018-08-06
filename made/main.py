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
parser.add_argument('--hidden-size', type=int, default=1000, metavar='N')
parser.add_argument('--layer-size', type=int, default=3, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.device))
	torch.cuda.set_device(args.device)

config_list = [args.epochs, args.batch_size, args.lr, 
				args.input_h, args.input_w, 
				args.hidden_size, args.layer_size]
config = ""
for i in map(str, config_list):
	config = config + '_' + i
print("Config:", config)

train_loader = dataloader.train_loader('fashionmnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('fashionmnist', args.data_directory, args.batch_size)

if args.load_model != '000000000000':
	made = torch.load(args.log_directory  + '/' + args.load_model + '/made.pt')
	args.time_stamep = args.load_mode[:12]
else:
	made = model.Made(args.input_h, args.input_w, args.hidden_size, args.layer_size).to(device)

log = args.log_directory + 'made/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(made.parameters(), lr = args.lr)

def train(epoch):
	epoch_start_time = time.time()
	train_loss = 0
	made.train()
	for batch_idx, (input_data, label) in enumerate(train_loader):
		start_time = time.time()
		optimizer.zero_grad()
		input_data = input_data.to(args.device)
		batch_size = input_data.size()[0]
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
		input_data = input_data.to(device)
		recon = made(input_data)
		loss = F.binary_cross_entropy(recon, input_data, size_average=False)
		test_loss += loss.item()
		if i == 0:
			n = min(batch_size, 8)
			comparison = torch.cat([input_data[:n],
								  recon[:n]])
			writer.add_image('Reconstruction Image', comparison, epoch)
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	writer.add_scalar('Test loss', test_loss, epoch)

for epoch in range(args.epochs):
	batch = 0
	train(epoch)
	test(epoch)
	# sample = D.Normal(torch.zeros(10).to(device), torch.ones(10).to(device))
	# sample_t, log_abs_det_jacobian = nflow(sample.sample(torch.Size([64])))
	# output = decoder(sample_t)
	# if not os.path.exists(log + 'results'):
	# 	os.mkdir(log + 'results')
	# save_image(output,
	# 		   log + 'results/sample_' + str(epoch) + '.png')
	# writer.add_image('Sample Image', output, epoch)

torch.save(made, log + 'made.pt')
print('Model saved in ', log + 'made.pt')
writer.close()