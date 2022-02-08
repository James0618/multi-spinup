import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.load_config import load_config, load_default_config
from envs.gather import GatherEnv
import argparse


class ReplayBuffer(object):
    def __init__(self, buffer_size, agents, args):
        self.buffer_size = buffer_size
        self.agents = agents
        self.args = args
        self.obs_shape = args.vae_observation_dim
        self.n_actions = args.n_actions
        self.obs, self.next_obs, self.action, self.reward = None, None, None, None
        self.matrix, self.next_matrix, self.mask, self.terminated = None, None, None, None
        self.index, self.num_experiences = 0, 0
        self.reset()

    def get_batch(self, batch_size):
        sample_index = random.sample(np.arange(self.num_experiences).tolist(), batch_size)
        batch = [self.obs[sample_index], self.action[sample_index], self.reward[sample_index],
                 self.next_obs[sample_index], self.matrix[sample_index], self.next_matrix[sample_index],
                 self.mask[sample_index], self.terminated[sample_index]]

        return batch

    def add(self, obs, action, reward, new_obs, matrix, next_matrix, done):
        obs_tensor = torch.zeros(len(self.agents), self.obs_shape)
        next_obs_tensor = torch.zeros(len(self.agents), self.obs_shape)
        reward_tensor = torch.zeros(len(self.agents))
        action_tensor = torch.zeros(len(self.agents)).type(torch.long)
        mask_tensor = torch.zeros(len(self.agents))
        terminated_tensor = torch.zeros(len(self.agents))

        for key in reward.keys():
            obs_tensor[self.agents.index(key)] = obs[key]
            action_tensor[self.agents.index(key)] = int(action[key])
            reward_tensor[self.agents.index(key)] = reward[key]
            next_obs_tensor[self.agents.index(key)] = new_obs[key]
            terminated_tensor[self.agents.index(key)] = done[key]
            mask_tensor[self.agents.index(key)] = 1

        self.obs[self.index] = obs_tensor
        self.action[self.index] = action_tensor
        self.reward[self.index] = reward_tensor
        self.next_obs[self.index] = next_obs_tensor
        self.matrix[self.index] = torch.tensor(matrix).type(torch.long)
        self.next_matrix[self.index] = torch.tensor(next_matrix).type(torch.long)
        self.mask[self.index] = mask_tensor
        self.terminated[self.index] = terminated_tensor

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
        self.mask = torch.zeros(self.buffer_size, len(self.agents)).type(torch.long)
        self.terminated = torch.zeros(self.buffer_size, len(self.agents)).type(torch.long)


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
    def __init__(self, args, n_agent):
        super(DGN, self).__init__()
        self.args = args
        obs_shape = args.vae_observation_dim

        self.encoder = Encoder(obs_shape, args.hidden_dim)
        self.att_1 = AttModel(n_agent, args.hidden_dim, args.hidden_dim, args.hidden_dim)
        self.att_2 = AttModel(n_agent, args.hidden_dim, args.hidden_dim, args.hidden_dim)
        self.q_net = QNet(args.hidden_dim, args.n_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask)
        q = self.q_net(h3)
        return q

    def update_param(self, target_model):
        for param, target_param in zip(self.parameters(), target_model.parameters()):
            param.data.copy_(self.args.tau * target_param.data + (1 - self.args.tau) * param.data)


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


def load_model(args):
    model_path = args.path + '/pyt_save/model.pt'
    actor_critic = torch.load(model_path)

    return actor_critic


def is_terminated(terminated):
    groups = ['omnivore']
    results = []
    for group in groups:
        results.append(np.array([terminated[key] for key in terminated.keys() if group in key]).prod())

    return bool(np.array(results).sum())


def test_policy(experiment, config_name, test_num=10, render=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/drl/PycharmProjects/multi-spinup/results/{}'.format(
        experiment))
    parser.add_argument('--test_episode', type=int, default=10)
    args = parser.parse_args()

    model = load_model(args=args)
    env_args = load_default_config(config_name)
    env = GatherEnv(args=env_args)
    run_args = load_config(config_name, env, output_path='/home/drl/PycharmProjects/multi-spinup/results/',
                           exp_name=experiment)
    env.preprocessor.args = run_args

    agents = env.possible_agents
    policy = Policy(args=run_args, model=model, agents=agents, obs_shape=run_args.vae_observation_dim)

    i_episode = 0
    test_ret = 0

    while i_episode < test_num:
        i_episode += 1

        steps, ep_ret = 0, 0
        obs = env.reset()
        adj = env.graph_builder.adjacency_matrix

        while True:
            steps += 1
            actions = policy.choose_action(observations=obs, adj_matrix=adj, epsilon=0.0)

            next_obs, rewards, done, next_positions = env.step(actions)
            next_adj = env.graph_builder.adjacency_matrix

            if render:
                env.render()

            terminal = is_terminated(terminated=done)

            obs = next_obs
            adj = next_adj

            if run_args.global_reward:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()]) / len(rewards)
            else:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()])

            if terminal:
                break

        test_ret += ep_ret

    return test_ret / test_num
