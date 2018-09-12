import argparser
import dataloader
import model
import torch.optim as optim
from torch.autograd import Variable
import torch.distributions
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--data', type=str, default='cifar10', metavar='N',
                    help='data')
parser.add_argument('--log-directory', type=str, default='/home/sungwonlyu/experiment/vqvae', metavar='N',
                    help='log directory')
parser.add_argument('--parameters', type=list, default=[256, 512, 10, 0.25], metavar='N',
                    help='vqvae parameters [hidden_size, K, D, beta]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

writer = SummaryWriter(args.log_directory + '/' + args.time_stamp + '/')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = dataloader.train_loader(args.data, args.data_directory, args.batch_size)
test_loader = dataloader.test_loader(args.data, args.data_directory, args.batch_size)

hidden_size, K, D, beta = args.parameters

if args.load_model != '000000':
    vqvae = torch.load(args.log_directory  + '/' + args.load_model + '/vqvae.pt')
else:
    vqvae = model.VQVAE(hidden_size, K, D, beta)
if args.cuda:
    vqvae.cuda()

optimizer = optim.Adam(vqvae.parameters(), lr = args.lr)

def train(epoch):
    vqvae.train()
    train_loss = 0
    cnt = 0
    for batch_idx, (input_data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        input_data = Variable(input_data)
        if args.cuda:
            input_data = input_data.cuda()
        recon_batch = vqvae(input_data)
        total_loss, recon_loss, emb_loss, com_loss = vqvae.loss_function(recon_batch, input_data)
        total_loss.backward(retain_graph=True)
        vqvae.st_bwd()
        train_loss += total_loss.data[0]
        optimizer.step()
        writer.add_scalar('recon loss', recon_loss / len(input_data), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('embed loss', emb_loss / len(input_data), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('com loss', com_loss / len(input_data), epoch * len(train_loader) + batch_idx)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.data[0]/len(input_data)))
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    # writer.add_scalar('Train loss', train_loss, epoch)


def test(epoch):
    vqvae.eval()
    test_loss = 0
    for i, (input_data, label) in enumerate(test_loader):
        input_data = Variable(input_data, volatile=True)
        if args.cuda:
            input_data = input_data.cuda()
        recon_batch = vqvae(input_data)
        total_loss, recon_loss, emb_loss, com_loss = vqvae.loss_function(recon_batch, input_data)
        test_loss += total_loss.data[0]
        if i == 0:
            n = min(input_data.size(0), 8)
            comparison = torch.cat([input_data[:n],
                                  recon_batch[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            writer.add_image('Reconstruction Image', comparison.data, epoch)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('test loss', test_loss, epoch)

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

torch.save(vqvae, args.log_directory + '/' + args.time_stamp + '/vqvae.pt')
print('Model saved in ', args.log_directory + '/' + args.time_stamp + '/vqvae.pt')
writer.close()