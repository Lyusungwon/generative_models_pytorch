import sys
sys.path.append('../utils/')
import argparser
import dataloader
import getpass
import model
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--data-directory', type=str, default='/home/' + getpass.getuser() + '/data', metavar='N',
                    help='directory of data')
parser.add_argument('--log-directory', type=str, default='/home/' + getpass.getuser() + '/experiment/bvae', metavar='N',
                    help='log directory')
parser.add_argument('--input-h', type=int, default=140, metavar='N')
parser.add_argument('--input-w', type=int, default=210, metavar='N')
parser.add_argument('--hidden-size', type=int, default=1024, metavar='N')
parser.add_argument('--latent-size', type=int, default=24, metavar='N')
parser.add_argument('--L', type=int, default=1, metavar='N')
parser.add_argument('--beta', type=int, default=1, metavar='N')

args = parser.parse_args()
args.device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
config_list = [args.batch_size, args.epochs, args.lr, args.input_h, args.input_w, args.hidden_size, args.latent_size, args.L, args.beta]
config = ""
for i in map(str, config_list):
    config = config + '_' + i
print("Config:", config)
torch.manual_seed(args.seed)

train_loader = dataloader.train_loader('alphachu', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('alphachu', args.data_directory, args.batch_size)

if args.load_model != '000000':
    encoder = torch.load(args.log_directory + '/' + args.load_model + config + '/bvae_encoder.pt')
    decoder = torch.load(args.log_directory + '/' + args.load_model + config + '/bvae_decoder.pt')
    args.time_stamp = args.load_model
    print('Model loaded from ', args.log_directory + '/' + args.time_stamp + config + '/bvae_encoder.pt')
    print('Model loaded from ', args.log_directory + '/' + args.time_stamp + config + '/bvae_decoder.pt')
else:
    encoder = model.Encoder(args.input_h, args.input_w, args.hidden_size, args.latent_size)
    decoder = model.Decoder(args.input_h, args.input_w, args.hidden_size, args.latent_size)
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

writer = SummaryWriter(args.log_directory + '/' + args.time_stamp + config + '/')


def train(epoch):
    train_loss = 0
    encoder.train()
    decoder.train()
    for batch_idx, input_data in enumerate(train_loader):
        batch_num = input_data.size()[0]
        optimizer.zero_grad()
        input_data = input_data.to(args.device)
        params = encoder(input_data)
        z_mu = params[:, 0]
        z_logvar = params[:, 1]
        epsilon_mu = torch.zeros(batch_num, args.latent_size)
        epsilon_sigma = torch.ones(batch_num, args.latent_size)
        reconstruction_loss = 0
        for j in range(args.L):
            epsilons = torch.distributions.Normal(epsilon_mu, epsilon_sigma).sample()
            epsilons = epsilons.to(args.device)
            z = z_mu + epsilons * z_logvar.mul(0.5).exp()
            output_data = decoder(z)
            reconstruction_loss += F.binary_cross_entropy(output_data, input_data, size_average=False)
        reconstruction_loss /= args.L
        kl_divergence = - 0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), 0)
        summary = {'reconstruction loss': reconstruction_loss / batch_num,
                   'KLDivergence loss': kl_divergence.sum() / batch_num}
        for i in range(args.latent_size):
            summary['KLDivergence loss (detail) {}'.format(i)] = kl_divergence[i] / batch_num
        writer.add_scalars('scalar groups', summary, epoch * len(train_loader) + batch_idx)
        kl_divergence = kl_divergence.sum()
        loss = reconstruction_loss + args.beta * kl_divergence
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(input_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    writer.add_scalar('train loss', train_loss, epoch)


def test(epoch):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    for i, input_data in enumerate(test_loader):
        input_data = input_data.to(args.device)
        params = encoder(input_data)
        z_mu = params[:, 0]
        z_logvar = params[:, 1]
        output_data = decoder(z_mu)
        reconstruction_loss = F.binary_cross_entropy(output_data, input_data, size_average=False)
        kl_divergence = - 0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum()
        loss = reconstruction_loss + args.beta * kl_divergence
        test_loss += loss.data
        if i == 0:
            n = min(input_data.size(0), 8)
            comparison = torch.cat([input_data[:n],
                                    output_data[:n]])
            writer.add_image('Reconstruction Image', comparison.data, epoch)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {}'.format(test_loss))
    writer.add_scalar('test loss', test_loss, epoch)


for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    torch.save(encoder, args.log_directory + '/' + args.time_stamp + config + '/bvae_encoder.pt')
    torch.save(decoder, args.log_directory + '/' + args.time_stamp + config + '/bvae_decoder.pt')
    print('Model saved in ', args.log_directory + '/' + args.time_stamp + config + '/bvae_encoder.pt')
    print('Model saved in ', args.log_directory + '/' + args.time_stamp + config + '/bvae_decoder.pt')
writer.close()
