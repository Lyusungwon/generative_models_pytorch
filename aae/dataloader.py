import torch
from torchvision import datasets, transforms

is_cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

def train_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128):
	if data == 'mnist':
		train_dataloader = torch.utils.data.DataLoader(
					datasets.MNIST(data_directory + '/' + data, train=True, download=True, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)
	if data == 'cifar10':
		train_dataloader = torch.utils.data.DataLoader(
					datasets.CIFAR10(data_directory + '/' + data, train=True, download=True, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)
	return train_dataloader

def test_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128):
	if data == 'mnist':
		test_dataloader = torch.utils.data.DataLoader(
					datasets.MNIST(data_directory + '/' + data, train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)
	if data == 'cifar10':
		test_dataloader = torch.utils.data.DataLoader(
					datasets.CIFAR10(data_directory + '/' + data, train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)

	return test_dataloader
