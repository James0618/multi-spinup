import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def get_graph(matrix, data):
    x = data
    edge = matrix.nonzero().transpose(0, 1)

    return x, edge


class UnFlatten(nn.Module):
    def __init__(self, h_dim):
        super(UnFlatten, self).__init__()
        self.h_dim = h_dim

    def forward(self, inputs):
        return inputs.view(inputs.size(0), self.h_dim, 1, 1)


class NeighNet(torch.nn.Module):
    def __init__(self, data_shape, h_dim, output_shape):
        super(NeighNet, self).__init__()
        self.num_features = data_shape
        self.nhid = h_dim
        self.output_shape = output_shape
        self.pooling_ratio = 0.2

        self.conv = GCNConv(self.num_features, self.nhid)

        self.fc = nn.Sequential(
            nn.Linear(self.nhid, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, self.output_shape),
        )

    def forward(self, data, matrix):
        x, edge_index = get_graph(matrix, data)
        x = F.relu(self.conv(x, edge_index))
        x = self.fc(x)

        return x


class DecoderPos(nn.Module):
    def __init__(self, observation_shape, h_dim=256, z_dim=64):
        super(DecoderPos, self).__init__()
        self.observation_shape, self.h_dim, self.z_dim = observation_shape, h_dim, z_dim
        self.decode_net = nn.Sequential(
            nn.Linear(z_dim + 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, observation_shape),
            nn.Sigmoid(),
        )

    def forward(self, latent_state, positions):
        inputs = torch.cat((latent_state, positions), dim=-1)
        reconstructed_observations = self.decode_net(inputs)

        return reconstructed_observations


class Decoder(nn.Module):
    def __init__(self, observation_shape, h_dim=256, z_dim=64):
        super(Decoder, self).__init__()
        self.observation_shape, self.h_dim, self.z_dim = observation_shape, h_dim, z_dim

        self.decoder_hidden = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            UnFlatten(h_dim=h_dim),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 5, kernel_size=5, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, latent_state):
        latent_state = self.decoder_hidden(latent_state)
        reconstructed_observations = self.decoder(latent_state)

        return reconstructed_observations


class Encoder(nn.Module):
    def __init__(self, obs_shape, hid_shape, h_dim):
        super(Encoder, self).__init__()
        self.transition_net = nn.GRUCell(obs_shape, hid_shape)
        self.obs_net = NeighNet(obs_shape, h_dim, obs_shape)
        self.hid_net = NeighNet(hid_shape, h_dim, hid_shape)
        self.encoder = nn.Linear(hid_shape, h_dim)

    def forward(self, obs, hidden_states, matrix):
        # hidden_states, obs: Tensor[n_agents, data_shape]
        phi = self.obs_net(obs, matrix)
        psi = self.hid_net(hidden_states, matrix)
        next_hid = self.transition_net(phi, psi)
        latent_state = self.encoder(next_hid)

        return latent_state, next_hid


class VAE(nn.Module):
    def __init__(self, obs_shape=96, hid_shape=128, h_dim=256, z_dim=64):
        super(VAE, self).__init__()
        self.observation_shape, self.hid_shape, self.h_dim, self.z_dim = obs_shape, hid_shape, h_dim, z_dim

        self.encoder = Encoder(obs_shape=obs_shape, hid_shape=hid_shape, h_dim=h_dim)

        # used for encoder
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_log_var = nn.Linear(h_dim, z_dim)

        # self.decoder = Decoder(observation_shape=observation_shape, h_dim=h_dim, z_dim=z_dim)
        self.decoder = DecoderPos(observation_shape=obs_shape, h_dim=h_dim, z_dim=z_dim)

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

    def encode(self, observations, hidden_states, matrix):
        latent_h, next_hid = self.encoder(observations, hidden_states, matrix)
        z, mu, log_var = self.bottleneck(latent_h)
        return z, next_hid, mu, log_var

    def decode(self, z, pos):
        z = self.decoder(z, pos)
        return z

    def forward(self, obs, matrix, hid, pos):
        z, next_hid, mu, log_var = self.encode(obs, hid, matrix)
        z = self.decode(z, pos)
        return z, next_hid, mu, log_var
