import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, h_dim):
        super(UnFlatten, self).__init__()
        self.h_dim = h_dim

    def forward(self, inputs):
        return inputs.view(inputs.size(0), self.h_dim, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=5, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1),
            # nn.ReLU(),
            Flatten()
        )

        # used for encoder
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_log_var = nn.Linear(h_dim, z_dim)

        # used for decoder
        self.fc_hidden = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(h_dim=h_dim),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z):
        z = self.fc_hidden(z)
        z = self.decoder(z)
        return z

    def forward(self, x, test=False):
        if test:
            h = self.encoder(x)
            mu = self.fc_mu(h)
            z = self.decode(mu)
            return z, None, None

        else:
            z, mu, log_var = self.encode(x)
            z = self.decode(z)
            return z, mu, log_var