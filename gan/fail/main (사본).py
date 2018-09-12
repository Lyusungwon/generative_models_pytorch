import argparser
import dataloader
import model2
import torch.optim as optim
from torch.autograd import Variable
import torch.distributions
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--log-directory', type=str, default='/home/sungwonlyu/experiment/gan', metavar='N',
                    help='log directory')
parser.add_argument('--parameters', type=list, default=[784, 400, 10, 10], metavar='N',
                    help='gan parameters [input_size, hidden_size, latent_size, L]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

writer = SummaryWriter(args.log_directory + '/' + args.time_stamp + '/')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)

input_size, hidden_size, latent_size, L = args.parameters

if args.load_model != '000000':
    gan = torch.load(args.log_directory  + '/' + args.load_model + '/gan.pt')
else:
    gan = model2.GAN()

if args.cuda:
    gan.cuda()

optimizer = optim.Adam(gan.parameters(), lr = args.lr)

def train(epoch):
    train_loss = 0
    gan.train()
    for batch_idx, (input_data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        input_data = Variable(input_data)
        if args.cuda:
            input_data = input_data.cuda()
        real_likelihood, fake_likelihood = gan(input_data)
        gen_loss, dis_loss = gan.loss_function(real_likelihood, fake_likelihood)
        writer.add_scalar('gen loss', gen_loss/len(input_data), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('dis loss', dis_loss/len(input_data), epoch * len(train_loader) + batch_idx)
        gen_loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()
        dis_loss.backward()
        optimizer.step()

        loss = (gen_loss + dis_loss)
        # loss.backward()
        train_loss += loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(input_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
    epoch, train_loss / len(train_loader.dataset)))


def sample():
    noise = torch.rand(8, 1, 28, 28)
    noise = Variable(noise, volatile=True)
    if args.cuda:
        noise = noise.cuda()
    sample = gan.generate(noise)
    save_image(sample.data, 'results/sample_' + str(epoch) + '.png')
    writer.add_image('Sample Image', sample.data, epoch)

for epoch in range(args.epochs):
    train(epoch)
    sample()

torch.save(gan, args.log_directory + '/' + args.time_stamp + '/gan.pt')
print('Model saved in ', args.log_directory + '/' + args.time_stamp + '/gan.pt')
writer.close()