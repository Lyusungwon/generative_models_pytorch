import argparser
import dataloader
import model
import torch.optim as optim
from torch.autograd import Variable
import torch.distributions
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparser.default_parser()
parser.add_argument('--log-directory', type=str, default='/home/sungwonlyu/experiment/aae', metavar='N',
                    help='log directory')
parser.add_argument('--parameters', type=list, default=[784, 1000, 20, 'gaussian_mixture'], metavar='N',
                    help='aae parameters [input_size, hidden_size, latent_size, prior shape]')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

writer = SummaryWriter(args.log_directory + '/' + args.time_stamp + '/')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)
test_loader = dataloader.test_loader('mnist', args.data_directory, args.batch_size)

input_size, hidden_size, latent_size, prior_shape = args.parameters

if args.load_model != '000000000000':
    aae = torch.load(args.log_directory + '/' + args.load_model + '/aae.pt')
else:
    aae = model.AAE(input_size, hidden_size, latent_size, prior_shape)
if args.cuda:
    aae.cuda()

encoder_optimizer = optim.Adam(aae.encoder.parameters(), lr=args.lr)
decoder_optimizer = optim.Adam(aae.decoder.parameters(), lr=args.lr)
dis_optimizer = optim.Adam(aae.discriminator.parameters(), lr=args.lr)


def train(epoch):
    train_loss = 0
    aae.train()
    for batch_idx, (input_data, label) in enumerate(train_loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        dis_optimizer.zero_grad()
        input_data = Variable(input_data)
        if args.cuda:
            input_data = input_data.cuda()
        output_data, real_likelihood, fake_likelihood = aae(input_data)
        # recon phase
        recon_loss = aae.recon_loss(output_data, input_data)
        recon_loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()

        # adversarial phase
        dis_loss = aae.dis_loss(real_likelihood, fake_likelihood)
        dis_loss.backward(retain_graph=True)
        dis_optimizer.step()

        gen_loss = aae.gen_loss(fake_likelihood)
        gen_loss.backward()
        encoder_optimizer.step()

        writer.add_scalar('recon loss', recon_loss, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('dis loss', dis_loss, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('gen loss', gen_loss, epoch * len(train_loader) + batch_idx)

        loss = (recon_loss + gen_loss + dis_loss)
        train_loss += loss.data
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))


def test(epoch):
    test_loss = 0
    aae.eval()
    plot = torch.zeros(0, 2)
    for i, (input_data, label) in enumerate(test_loader):
        input_data = Variable(input_data, volatile=True)
        if args.cuda:
            input_data = input_data.cuda()
        output_data, real_likelihood, fake_likelihood = aae(input_data)
        plot = torch.cat([plot, aae.z.data], 0)
        recon_loss = aae.recon_loss(output_data, input_data)
        dis_loss = aae.dis_loss(real_likelihood, fake_likelihood)
        gen_loss = aae.gen_loss(fake_likelihood)
        loss = (recon_loss + gen_loss + dis_loss)
        test_loss += loss.data[0]
        if i == 0:
            n = min(input_data.size(0), 8)
            comparison = torch.cat([input_data[:n], output_data[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            writer.add_image('Reconstruction Image', comparison.data, epoch)
    plt.figure()
    plt.scatter(plot[:, 0], plot[:, 1], s=2, color='red')
    plt.savefig('results/plot_' + str(epoch) + '.png')
    writer.add_scalar('test loss', test_loss, epoch)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def visualize(latent_size):
    embeddings = torch.zeros(0, latent_size)
    embedding_labels = torch.zeros(0)
    embedding_images = torch.zeros(0, 1, 28, 28)
    for i, (input_data, label) in enumerate(test_loader):
        input_data = Variable(input_data, volatile=True)
        if args.cuda:
            input_data = input_data.cuda()
        aae(input_data)
        embeddings = torch.cat([embeddings, aae.z.data], 0)
        embedding_labels = torch.cat([embedding_labels, label.float()], 0)
        embedding_images = torch.cat([embedding_images, input_data.data], 0)
    writer.add_embedding(embeddings, metadata=embedding_labels, label_img=embedding_images)


for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
visualize(latent_size)

#     sample = Variable(torch.randn(64, 10))
#     if args.cuda:
#         sample = sample.cuda()
#     sample = decoder(sample)
#     save_image(sample.data,
#                'results/sample_' + str(epoch) + '.png')
#     writer.add_image('Sample Image', sample.data, epoch)
#
torch.save(aae, args.log_directory + '/' + args.time_stamp + '/aae.pt')
print('Model saved in ', args.log_directory + '/' + args.time_stamp + '/aae.pt')
writer.close()
