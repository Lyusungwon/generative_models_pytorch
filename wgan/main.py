import argparser
import dataloader
import model
import torch.optim as optim
import torch.distributions
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

parser = argparser.default_parser()
parser.add_argument('--log-directory', type=str, default='/home/sungwonlyu/experiment/wgan', metavar='N',
                    help='log directory')
parser.add_argument('--parameters', type=list, default=[784, 1024, 100, 5, 10], metavar='N',
                    help='wgan parameters [input_size, hidden_size, latent_size, k, lambda]')
args = parser.parse_args()
args.device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)

train_loader = dataloader.train_loader('mnist', args.data_directory, args.batch_size)

input_size, hidden_size, latent_size, k, l = args.parameters

if args.load_model != '000000000000':
    critic = torch.load(args.log_directory + '/' + args.load_model + '/critic.pt')
    generator = torch.load(args.log_directory + '/' + args.load_model + '/generator.pt')
    args.time_stamp = args.load_model
else:
    critic = model.Critic()
    generator = model.Generator()
    critic = critic.to(args.device)
    generator = generator.to(args.device)

writer = SummaryWriter(args.log_directory + '/' + args.time_stamp + '/')

critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr, betas=(0, 0.9))
generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0, 0.9))


def train(epoch):
    critic.train()
    generator.train()
    for batch_idx, (input_data, label) in enumerate(train_loader):
        if batch_idx > k:
            break
        critic_optimizer.zero_grad()
        input_data = ((input_data - 0.5) * 2).to(args.device)
        noise = torch.rand(len(input_data), latent_size)
        noise = noise.to(args.device)
        fake = generator(noise)
        g_loss = l * model.gradient_penalty(input_data, fake, critic)
        f_loss = critic(fake.detach()).mean()
        t_loss = - critic(input_data).mean()
        critic_loss = f_loss + t_loss + g_loss
        critic_loss.backward()
        critic_optimizer.step()
    writer.add_scalars('critic_loss_detail', {'g loss': g_loss,
                                             'f loss': f_loss,
                                             't loss': t_loss}, epoch)
    writer.add_scalar('critic loss', critic_loss, epoch)
    generator_optimizer.zero_grad()
    noise = torch.rand(args.batch_size, latent_size)
    noise = noise.to(args.device)
    fake = generator(noise)
    generator_loss = - critic(fake).mean()
    generator_loss.backward()
    generator_optimizer.step()
    writer.add_scalar('gen loss', generator_loss, epoch)
    return critic_loss, generator_loss

def sample():
    generator.eval()
    noise = torch.rand(8, latent_size)
    noise = noise.to(args.device)
    sample = generator(noise)
    sample = sample / 2 + 0.5
    save_image(sample.data, 'results/sample_' + str(epoch) + '.png')
    writer.add_image('Sample Image', sample.data, epoch)


for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    critic_loss, generator_loss = train(epoch)
    if epoch % args.log_interval == 0:
        print('====> Step: {} critic loss: {:.4f} generator loss: {:.4f}'.format(
            epoch, critic_loss.item(), generator_loss.item()))
        sample()
        torch.save(critic, args.log_directory + '/' + args.time_stamp + '/critic.pt')
        torch.save(generator, args.log_directory + '/' + args.time_stamp + '/generator.pt')
        print('Model saved in ', args.log_directory + '/' + args.time_stamp + '/critic.pt')
        print('Model saved in ', args.log_directory + '/' + args.time_stamp + '/generator.pt')

writer.close()
