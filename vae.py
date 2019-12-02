# Adapted from Pytorch examples
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from utils import Binarize


parser = argparse.ArgumentParser(description='VAE for MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--binary', action='store_true',
                    help='Whether to binarize the input data')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
transform = [transforms.ToTensor()]
if args.binary:
    transform.append(Binarize(0.5))
transform = transforms.Compose(transform)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/datasets', train=True, download=True,
                   transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/datasets', train=False, transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(True), nn.Linear(512, 256), nn.ReLU(True))
        self.fce1 = nn.Linear(256, 20)
        self.fce2 = nn.Linear(256, 20)
        self.dec = nn.Sequential(nn.Linear(20, 256), nn.ReLU(True), nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, 784))

    def encode(self, x):
        h1 = self.enc(x)
        return self.fce1(h1), self.fce2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = z.view(z.size(0), 20)
        h3 = self.dec(z)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), 784))
        z = self.reparameterize(mu, logvar)
        q0 = torch.distributions.normal.Normal(mu, (0.5 * logvar).exp())
        prior = torch.distributions.normal.Normal(0., 1.)
        log_prior_z = prior.log_prob(z).sum(-1)
        log_q_z = q0.log_prob(z).sum(-1)
        return self.decode(z), log_q_z - log_prior_z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, KL):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE + KL


def train(epoch):
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, _) in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, kl = model(data)
        vaeloss = loss_function(recon_batch, data.view(data.size(0), 784), kl.sum())
        loss = vaeloss / len(data)
        loss.backward()
        train_loss += vaeloss.item()
        optimizer.step()
        pbar.update()
        if batch_idx % args.log_interval == 0:
            desc = 'Ep: {}, VAELoss: {:.6f}'.format(epoch, loss.item())
            pbar.set_description(desc)

    print('====> Train set loss: {:.4f}'.format(train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, kl = model(data)
            test_loss += loss_function(recon_batch, data.view(data.size(0), 784), kl.sum()).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/recons_vae_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_vae_' + str(epoch) + '.png')
