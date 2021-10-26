import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class UnFlatten(nn.Module):
    def __init__(self, h_dim):
        super(UnFlatten, self).__init__()
        self.h_dim = h_dim

    def forward(self, inputs):
        return inputs.view(inputs.size(0), self.h_dim, 1, 1)


class Encoder(torch.nn.Module):
    def __init__(self, observation_shape, h_dim):
        super(Encoder, self).__init__()
        self.num_features = observation_shape
        self.nhid = h_dim
        self.pooling_ratio = 0.3

        self.conv1 = GCNConv(self.num_features, self.nhid // 4)
        self.conv2 = GCNConv(self.nhid // 4, self.nhid)
        # self.conv3 = GCNConv(self.nhid // 2, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(self.nhid * 2, self.nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nhid, self.nhid)
        )

    def forward(self, obs, is_alive):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
            is_alive = is_alive.unsqueeze(0)

        numbers = int(is_alive.sum())

        x = torch.zeros(numbers + obs.shape[0], obs.shape[2])
        batch = torch.zeros(numbers + obs.shape[0], dtype=torch.long)
        edge_index = torch.zeros(2, numbers * 2, dtype=torch.long)

        batch_ptr, edge_ptr = 0, 0
        for i in range(obs.shape[0]):
            agent_number = int(is_alive[i].sum())
            fill_slice = slice(batch_ptr + 1, batch_ptr + agent_number + 1)

            x[batch_ptr] = torch.zeros(self.num_features)
            x[fill_slice] = obs[i][is_alive[i] == 1]
            batch[batch_ptr] = i
            batch[fill_slice] = torch.ones(agent_number) * i

            edge_index[1, edge_ptr * 2: edge_ptr * 2 + agent_number] = torch.arange(agent_number) + 1
            edge_index[0, edge_ptr * 2 + agent_number: (edge_ptr + agent_number) * 2] = torch.arange(agent_number) + 1
            edge_index[:, edge_ptr * 2: (edge_ptr + agent_number) * 2] += batch_ptr

            batch_ptr += agent_number + 1
            edge_ptr += agent_number

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1
        x = self.lin1(x)

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
        agent_num = positions.shape[1]
        latent_state = latent_state.expand(
            torch.Size([agent_num, latent_state.shape[0], latent_state.shape[1]])).transpose(0, 1)

        inputs = torch.cat((latent_state, positions / 20), dim=-1)
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


class VAE(nn.Module):
    def __init__(self, observation_shape=96, h_dim=256, z_dim=64):
        super(VAE, self).__init__()
        self.observation_shape, self.h_dim, self.z_dim = observation_shape, h_dim, z_dim

        self.encoder = Encoder(observation_shape=observation_shape + 2, h_dim=h_dim)

        # used for encoder
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_log_var = nn.Linear(h_dim, z_dim)

        # self.decoder = Decoder(observation_shape=observation_shape, h_dim=h_dim, z_dim=z_dim)
        self.decoder = DecoderPos(observation_shape=observation_shape, h_dim=h_dim, z_dim=z_dim)

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

    def encode(self, observations, is_alive, positions):
        observations = torch.cat((observations, positions / 20), dim=-1)
        h = self.encoder(observations, is_alive)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z, pos):
        z = self.decoder(z, pos)
        return z

    def forward(self, obs, is_alive, pos):
        z, mu, log_var = self.encode(obs, is_alive, pos)
        z = self.decode(z, pos)
        return z, mu, log_var
