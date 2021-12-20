import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, copy
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from DGN import DGN
from buffer import ReplayBuffer
from surviving import Surviving
from config import *


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
    def __init__(self, hidden_dim, dout):
        super(QNet, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

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

def dgn(env_fn, args, seed=0, steps_per_epoch=32000, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=4e-4, vf_lr=8e-4,
        train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=1000, actor_critic=core.CNNActorCritic,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):


    USE_CUDA = torch.cuda.is_available()

    env = Surviving(n_agent=100)
    n_ant = env.n_agent
    observation_space = env.len_obs
    n_actions = env.n_action

    buff = ReplayBuffer(capacity)
    model = DGN(n_ant, observation_space, hidden_dim, n_actions)
    model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions)
    model = model.cuda()
    model_tar = model_tar.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    O = np.ones((batch_size, n_ant, observation_space))
    Next_O = np.ones((batch_size, n_ant, observation_space))
    Matrix = np.ones((batch_size, n_ant, n_ant))
    Next_Matrix = np.ones((batch_size, n_ant, n_ant))

    f = open('r.txt', 'w')
    while i_episode < n_episode:

        if i_episode > 100:
            epsilon -= 0.0004
            if epsilon < 0.1:
                epsilon = 0.1
        i_episode += 1
        steps = 0
        obs, adj = env.reset()

        while steps < max_step:
            steps += 1
            action = []
            q = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(adj).cuda())[0]
            for i in range(n_ant):
                if np.random.rand() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = q[i].argmax().item()
                action.append(a)

            next_obs, next_adj, reward, terminated = env.step(action)

            buff.add(np.array(obs), action, reward, np.array(next_obs), adj, next_adj, terminated)
            obs = next_obs
            adj = next_adj
            score += sum(reward)

        if i_episode % 20 == 0:
            print(score / 2000)
            f.write(str(score / 2000) + '\n')
            score = 0

        if i_episode < 100:
            continue

        for e in range(n_epoch):

            batch = buff.getBatch(batch_size)
            for j in range(batch_size):
                sample = batch[j]
                O[j] = sample[0]
                Next_O[j] = sample[3]
                Matrix[j] = sample[4]
                Next_Matrix[j] = sample[5]

            q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
            target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda()).max(dim=2)[0]
            target_q_values = np.array(target_q_values.cpu().data)
            expected_q = np.array(q_values.cpu().data)

            for j in range(batch_size):
                sample = batch[j]
                for i in range(n_ant):
                    expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * GAMMA * target_q_values[j][i]

            loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i_episode % 5 == 0:
            model_tar.load_state_dict(model.state_dict())
