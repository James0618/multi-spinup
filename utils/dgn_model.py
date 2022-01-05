import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size, agents, obs_shape):
        self.buffer_size = buffer_size
        self.agents = agents
        self.obs_shape = obs_shape
        self.obs, self.next_obs, self.action, self.reward, self.done = None, None, None, None, None
        self.matrix, self.next_matrix, self.is_alive, self.next_is_alive = None, None, None, None
        self.index, self.num_experiences = 0, 0
        self.reset()

    def get_batch(self, batch_size):
        sample_index = random.sample(np.arange(self.num_experiences).tolist(), batch_size)
        batch = [self.obs[sample_index], self.action[sample_index], self.reward[sample_index],
                 self.next_obs[sample_index], self.matrix[sample_index], self.next_matrix[sample_index],
                 self.done[sample_index], self.is_alive[sample_index], self.next_is_alive[sample_index]]

        return batch

    def add(self, obs, action, reward, new_obs, matrix, next_matrix, done, is_alive, next_is_alive):
        obs_tensor = torch.zeros(len(self.agents), self.obs_shape)
        next_obs_tensor = torch.zeros(len(self.agents), self.obs_shape)
        reward_tensor = torch.zeros(len(self.agents))
        action_tensor = torch.zeros(len(self.agents)).type(torch.long)
        for key in obs.keys():
            obs_tensor[self.agents.index(key)] = obs[key]

        for key in new_obs.keys():
            next_obs_tensor[self.agents.index(key)] = new_obs[key]
            reward_tensor[self.agents.index(key)] = reward[key]
            action_tensor[self.agents.index(key)] = int(action[key])

        self.obs[self.index] = obs_tensor
        self.action[self.index] = action_tensor
        self.reward[self.index] = reward_tensor
        self.next_obs[self.index] = next_obs_tensor
        self.matrix[self.index] = torch.tensor(matrix).type(torch.long)
        self.next_matrix[self.index] = torch.tensor(next_matrix).type(torch.long)
        self.done[self.index] = done
        self.is_alive[self.index] = torch.tensor(is_alive).type(torch.long)
        self.next_is_alive[self.index] = torch.tensor(next_is_alive).type(torch.long)

        if self.num_experiences < self.buffer_size:
            self.num_experiences += 1

        self.index += 1
        if self.index == self.buffer_size:
            self.index = 0

    def reset(self):
        self.obs = torch.zeros(self.buffer_size, len(self.agents), self.obs_shape)
        self.next_obs = torch.zeros(self.buffer_size, len(self.agents), self.obs_shape)
        self.reward = torch.zeros(self.buffer_size, len(self.agents))
        self.action = torch.zeros(self.buffer_size, len(self.agents)).type(torch.long)
        self.matrix = torch.zeros(self.buffer_size, len(self.agents), len(self.agents)).type(torch.long)
        self.next_matrix = torch.zeros(self.buffer_size, len(self.agents), len(self.agents)).type(torch.long)
        self.done = torch.zeros(self.buffer_size).type(torch.long)
        self.is_alive = torch.zeros(self.buffer_size, len(self.agents)).type(torch.long)
        self.next_is_alive = torch.zeros(self.buffer_size, len(self.agents)).type(torch.long)


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
    def __init__(self, args, obs_shape, model, agents):
        self.args = args
        self.obs_shape = obs_shape
        self.model = model
        self.agents = agents

    def choose_action(self, observations, adj_matrix, epsilon):
        actions = {}

        n_agent = len(self.agents)
        obs_tensor = torch.zeros(n_agent, self.obs_shape).cuda()
        adj_tensor = torch.from_numpy(adj_matrix).cuda()

        for key in observations.keys():
            obs_tensor[self.agents.index(key)] = observations[key].cuda()

        obs_tensor = obs_tensor.unsqueeze(0)

        q = self.model(obs_tensor, adj_tensor)[0]

        for key in observations.keys():
            index = self.agents.index(key)
            if np.random.rand() < epsilon:
                action = np.random.randint(self.args.n_actions)
            else:
                action = q[index].argmax().item()

            actions[key] = int(action)

        return actions
