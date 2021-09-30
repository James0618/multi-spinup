import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class VAEActor(Actor):
    def __init__(self, args):
        super(VAEActor, self).__init__()
        self.args = args
        self.logits_net = nn.Sequential(
            nn.Linear(args.vae_observation_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
        )

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CentralizedCritic(nn.Module):
    def __init__(self, args, agents):
        super(CentralizedCritic, self).__init__()
        self.args = args
        self.possible_agents = agents
        # self.feature_net, self.cnn_output_size = cnn(args)
        self.v_net = nn.Sequential(
            nn.Linear(args.nhid, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1),
        )
        self.state_net = EstimationNet(args=args)

    def forward(self, obs, is_alive=None):
        if type(obs) == dict:
            # the type of obs is dict
            agent_number = len(self.possible_agents)
            obs_tensor = torch.zeros(agent_number, self.args.vae_observation_dim)
            for i, agent in enumerate(self.possible_agents):
                if agent in obs.keys():
                    obs_tensor[i] = obs[agent]
            is_alive = torch.ones(agent_number)
            state = self.state_net(obs_tensor, is_alive).squeeze()

            return self.v_net(state)

        else:
            state = self.state_net(obs, is_alive).squeeze()

            return self.v_net(state)


class VAEActorCritic(nn.Module):
    def __init__(self, args, agents):
        super(VAEActorCritic, self).__init__()
        self.args = args
        self.possible_agents = agents
        self.pi = VAEActor(args=args)
        self.v = CentralizedCritic(args=args, agents=agents)

    def step(self, obs):
        a, logp_a = {}, {}
        with torch.no_grad():
            for agent in obs.keys():
                pi = self.pi._distribution(obs[agent])
                a[agent] = pi.sample()
                logp_a[agent] = self.pi._log_prob_from_distribution(pi, a[agent])

            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]


class EstimationNet(torch.nn.Module):
    def __init__(self, args):
        super(EstimationNet, self).__init__()
        self.args = args
        self.num_features = args.vae_observation_dim
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

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

            x[batch_ptr] = torch.zeros(self.args.vae_observation_dim)
            x[fill_slice] = obs[i][is_alive[i] == 1]
            batch[batch_ptr] = i
            batch[fill_slice] = torch.ones(agent_number) * i

            edge_index[1, edge_ptr * 2: edge_ptr * 2 + agent_number] = torch.arange(agent_number) + 1
            edge_index[0, edge_ptr * 2 + agent_number: (edge_ptr + agent_number) * 2] = torch.arange(agent_number) + 1

            batch_ptr += agent_number + 1
            edge_ptr += agent_number

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)

        return x
