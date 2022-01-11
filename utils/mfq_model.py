import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class MFQ(nn.Module):
    def __init__(self, args):
        super(MFQ, self).__init__()
        self.args = args
        self.obs_net = nn.Sequential(
            nn.Linear(args.args.vae_observation_dim, args.hidden_dim),
            nn.ReLU(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(args.n_actions, args.hidden_dim),
            nn.ReLU(),
        )

        self.q_net = nn.Sequential(
            nn.Linear(2 * args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
        )

    def forward(self, obs, mean_actions):
        obs_feature = self.obs_net(obs)
        act_feature = self.action_net(mean_actions)
        feature = torch.cat((obs_feature, act_feature), dim=-1)

        q = self.q_net(feature)

        return q


class Policy:
    def __init__(self, args, obs_shape, model, agents):
        self.args = args
        self.obs_shape = obs_shape
        self.model = model
        self.agents = agents

    def choose_action(self, observations, mean_actions, epsilon=1):
        actions = {}

        n_agent = len(observations.keys())
        obs_tensor = torch.zeros(n_agent, self.obs_shape).cuda()
        act_tensor = torch.from_numpy(mean_actions).cuda()

        for key in observations.keys():
            obs_tensor[self.agents.index(key)] = observations[key].cuda()

        obs_tensor = obs_tensor.unsqueeze(0)

        q = self.model(obs_tensor, act_tensor)[0]
        q = F.softmax(q / epsilon, dim=-1)

        # for key in observations.keys():
        #     index = self.agents.index(key)
        #     if np.random.rand() < epsilon:
        #         action = np.random.randint(self.args.n_actions)
        #     else:
        #         action = q[index].argmax().item()
        #
        #     actions[key] = int(action)

        for key in observations.keys():
            index = self.agents.index(key)
            action = q[index].argmax()

            actions[key] = int(action)

        return actions
