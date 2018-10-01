import sys
sys.path.append('../utils/')
import argparser
import dataloader
import model
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import torch.distributions
from tensorboardX import SummaryWriter
import time 

parser = argparser.default_parser()
parser.add_argument('--name', type=str, default='bvae')
parser.add_argument('--dataset', type=str, default='clevr')
parser.add_argument('--channel-size', type=int, default=3)
parser.add_argument('--input-h', type=int, default=128, metavar='N')
parser.add_argument('--input-w', type=int, default=128, metavar='N')
parser.add_argument('--filter-size', type=int, default=256, metavar='N')
parser.add_argument('--kernel-size', type=int, default=3, metavar='N')
parser.add_argument('--stride-size', type=int, default=2)
parser.add_argument('--layer-size', type=int, default=5)
# parser.add_argument('--hidden-size', type=int, default=1024, metavar='N')
parser.add_argument('--latent-size', type=int, default=10, metavar='N')
parser.add_argument('--L', type=int, default=1, metavar='N')
parser.add_argument('--beta', type=int, default=1, metavar='N')

args = parser.parse_args()

torch.manual_seed(args.seed)

if args.device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.device))
    torch.cuda.set_device(args.device)

config_list = [args.name, args.dataset, args.batch_size, args.epochs, args.lr, args.device, 
                args.channel_size, args.input_h, args.input_w, 
                args.filter_size, args.kernel_size, args.stride_size, args.layer_size, args.latent_size, 
                # args.hidden_size, 
                args.L, args.beta]
config = '_'.join(map(str, config_list))
print("Config:", config)

train_loader = dataloader.train_loader(args.dataset, args.data_directory, args.batch_size, args.input_h, args.input_w, args.cpu_num)
test_loader = dataloader.test_loader(args.dataset, args.data_directory, args.batch_size, args.input_h, args.input_w, args.cpu_num)

# encoder = model.Encoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(args.device)
# decoder = model.Decoder(args.input_h, args.input_w, args.hidden_size, args.latent_size).to(args.device)
encoder = model.Encoder(args.channel_size, args.filter_size, args.kernel_size, args.stride_size, args.layer_size, args.latent_size).to(args.device)
decoder = model.Decoder(args.channel_size, args.filter_size, args.kernel_size, args.stride_size, args.layer_size, args.latent_size).to(args.device)

if args.load_model != '000000000000':
    encoder.load_state_dict(torch.load(args.log_directory + args.name + '/' + args.load_model + 'bvae_encoder.pt')).to(args.device)
    decoder.load_state_dict(torch.load(args.log_director + args.name + '/' + args.load_model + '/bvae_decoder.pt')).to(args.device)
    args.time_stamp = args.load_model[:12]
    print('Model {} loaded.'.format(args.load_model))

log = args.log_directory + args.name + '/' + args.time_stamp + config + '/'
writer = SummaryWriter(log)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

def train(epoch):
    epoch_start_time = time.time()
    start_time = time.time()
    train_loss = 0
    r_loss= 0
    k_loss = 0
    encoder.train()
    decoder.train()
    for batch_idx, input_data in enumerate(train_loader):
        batch_size = input_data.size()[0]
        optimizer.zero_grad()
        input_data = input_data.to(args.device)
        params = encoder(input_data)
        z_mu = params[:, 0].squeeze(1)
        z_logvar = params[:, 1].squeeze(1)
        q = D.Normal(z_mu, (z_logvar / 2).exp())
        recon_loss = 0
        for j in range(args.L):
            z = q.rsample().to(device)
            output_data = decoder(z)
            recon_loss += F.binary_cross_entropy(output_data, input_data, size_average=False)
        recon_loss /= args.L
        prior = D.Normal(torch.zeros_like(z_mu).to(device), torch.ones_like(z_mu).to(device))
        kld_loss = D.kl_divergence(q, prior)
        loss = recon_loss + args.beta * kld_loss.sum()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        r_loss += recon_loss
        k_loss += kld_loss.sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} / Time: {:.4f}'.format(
                epoch,
                batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / batch_size,
                time.time() - start_time))

            idx = epoch * len(train_loader) // args.log_interval + batch_idx // args.log_interval
            writer.add_scalar('Batch loss', loss / batch_size, idx)
            writer.add_scalar('Batch Reconstruction Loss', recon_loss / batch_size, idx)
            writer.add_scalar('Batch KL Divergence Loss', kld_loss.sum() / batch_size, idx)
            summary = {}
            kl_divergence = kld_loss.sum(0).view(-1)
            for i in range(kl_divergence.size()[0]):
                summary['Latent variable {}'.format(i)] = kl_divergence[i] / batch_size
            writer.add_scalars('Batch KL Divergence Loss(detail)', summary, idx)
            writer.add_scalar('Batch time', time.time() - start_time, idx)
            start_time = time.time()

    print('====> Epoch: {} Average loss: {:.4f} / Time: {:.4f}'.format(
        epoch,
        train_loss / len(train_loader.dataset),
        time.time() - epoch_start_time))
    writer.add_scalar('Train Loss', train_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('Train Reconstruction Loss', r_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('Train KL Divergence Loss', k_loss / len(train_loader.dataset), epoch)

def test(epoch):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    r_loss= 0
    k_loss = 0    
    for batch_idx, input_data in enumerate(test_loader):
        input_data = input_data.to(args.device)
        params = encoder(input_data)
        z_mu = params[:, 0].squeeze(1)
        z_logvar = params[:, 1].squeeze(1)
        q = D.Normal(z_mu, (z_logvar / 2).exp())
        output_data = decoder(z_mu)
        recon_loss = F.binary_cross_entropy(output_data, input_data, size_average=False)
        prior = D.Normal(torch.zeros_like(z_mu).to(device), torch.ones_like(z_mu).to(device))
        kld_loss = D.kl_divergence(q, prior).sum()
        loss = recon_loss + args.beta * kld_loss
        test_loss += loss.item()
        r_loss += recon_loss.item()
        k_loss += kld_loss.item()
        if batch_idx == 0:
            n = min(input_data.size(0), 8)
            comparison = torch.cat([input_data[:n],
                                    output_data[:n]])
            writer.add_image('Reconstruction Image', comparison.data, epoch)
    
    print('====> Test set loss: {}'.format(test_loss / len(test_loader.dataset)))
    writer.add_scalar('Test loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test Reconstruction Loss', r_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test KL Divergence Loss', k_loss / len(test_loader.dataset), epoch)

for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    torch.save(encoder.state_dict(), log + 'bvae_encoder.pt')
    torch.save(decoder.state_dict(), log + 'bvae_decoder.pt')
    print('Model saved in ', log)
writer.close()
