import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, obs, action, reward, new_obs, matrix, next_matrix, done):
        experience = (obs, action, reward, new_obs, matrix, next_matrix, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, d_out):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_out)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fc_out(out))
        return out


class QNet(nn.Module):
    def __init__(self, hidden_dim, d_out):
        super(QNet, self).__init__()
        self.fc = nn.Linear(hidden_dim, d_out)

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions):
        super(DGN, self).__init__()

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = QNet(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask)
        q = self.q_net(h3)
        return q


class Policy:
    def __init__(self, args, model, agents):
        self.args = args
        self.model = model
        self.agents = agents

    def choose_action(self, observations, adj_matrix, epsilon):
        actions = {}

        n_agent = len(observations)
        obs_tensor = torch.zeros(n_agent, self.args.obs_shape).cuda()
        adj_tensor = torch.from_numpy(adj_matrix).cuda()

        for key in observations.keys():
            obs_tensor[self.agents.index(key)] = torch.from_numpy(observations[key]).cuda()

        q = self.model(obs_tensor, adj_tensor)[0]

        for key in observations.keys():
            index = self.agents.index(key)
            if np.random.rand() < epsilon:
                action = np.random.randint(self.args.n_actions)
            else:
                action = q[index].argmax().item()

            actions[key] = int(action)

        return actions