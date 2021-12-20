import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def process_matrix(matrix, min_neigh):
    n_agents = matrix.shape[0]
    temp = torch.arange(n_agents).tolist()

    min_matrix = torch.zeros_like(matrix)
    for i in range(min_neigh):
        min_index = matrix.argmin(dim=-1).tolist()
        min_matrix[temp, min_index] = 1
        matrix[temp, min_index] += 1000
        min_matrix[min_index, temp] = 1
        matrix[min_index, temp] += 1000

    return min_matrix


def get_neigh(matrix, data, min_neigh=2):
    # matrix = process_matrix(matrix, min_neigh)

    n_agents, data_shape = matrix.shape[-1], data.shape[-1]
    result = []

    n_neigh, ptr = matrix.sum(-1).cpu(), 0
    neighs = matrix.nonzero().transpose(0, 1).tolist()[1]
    for i in range(n_agents):
        index = neighs[ptr: ptr + n_neigh[i]]
        neigh_data = data[index]
        all_data = torch.cat((data[i].unsqueeze(0), neigh_data))

        result.append(all_data)
        ptr += n_neigh[i]

    x = torch.cat(result)
    batch = torch.zeros(x.shape[0]).type(torch.long)
    edge_index = torch.zeros(2, sum(n_neigh) * 2).type(torch.long)

    batch_ptr, edge_ptr = 0, 0
    for i in range(n_agents):
        batch[batch_ptr: batch_ptr + n_neigh[i] + 1] = i

        edge_index[1, edge_ptr: edge_ptr + n_neigh[i]] = torch.arange(n_neigh[i]) + 1
        edge_index[0, edge_ptr + n_neigh[i]: edge_ptr + 2 * n_neigh[i]] = torch.arange(n_neigh[i]) + 1
        edge_index[:, edge_ptr: edge_ptr + 2 * n_neigh[i]] += int(edge_ptr / 2 + i)

        edge_ptr += 2 * n_neigh[i]
        batch_ptr += n_neigh[i] + 1

    return x, batch, edge_index


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

        self.conv1 = GCNConv(self.num_features, self.nhid // 4)
        self.conv2 = GCNConv(self.nhid // 4, self.nhid)
        # self.conv3 = GCNConv(self.nhid // 2, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = nn.Sequential(
            nn.Linear(self.nhid * 2, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, self.output_shape),
        )

    def forward(self, data, matrix):
        x, batch, edge_index = get_neigh(matrix, data)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.lin1(x1)

        return x


# class NeighNet(nn.Module):
#     def __init__(self, data_shape, h_dim, output_shape):
#         super(NeighNet, self).__init__()
#         self.num_features = data_shape
#         self.nhid = h_dim
#         self.output_shape = output_shape
#         self.pooling_ratio = 0.3
#
#         self.conv1 = GCNConv(self.num_features, self.nhid // 4)
#         self.conv2 = GCNConv(self.nhid // 4, self.nhid)
#         # self.conv3 = GCNConv(self.nhid // 2, self.nhid)
#         self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
#
#         self.lin1 = torch.nn.Sequential(
#             nn.Linear(self.nhid * 2, self.nhid),
#             nn.ReLU(),
#             nn.Linear(self.nhid, self.output_shape)
#         )
#
#     def forward(self, data, matrix):
#         x, batch, edge_index = get_neigh(matrix, data)
#
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         # x = F.relu(self.conv3(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
#         x = x1
#         x = self.lin1(x)
#
#         return x


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
        # self.fc_mu = nn.Sequential(
        #     nn.Linear(h_dim, z_dim),
        #     nn.Tanh()
        # )
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
        # mu = torch.tanh(mu)
        return z, next_hid, mu, log_var

    def decode(self, z, pos):
        z = self.decoder(z, pos)
        return z

    def forward(self, obs, matrix, hid, pos):
        z, next_hid, mu, log_var = self.encode(obs, hid, matrix)
        z = self.decode(z, pos)
        return z, next_hid, mu, log_var
