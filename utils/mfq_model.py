import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, agents, args):
        self.buffer_size = buffer_size
        self.agents = agents
        self.n_agents = len(agents)
        self.args = args
        self.obs_shape = args.vae_observation_shape
        self.obs, self.next_obs, self.action, self.reward = None, None, None, None
        self.prob, self.next_prob, self.mask = None, None, None
        self.index, self.num_experiences = 0, 0
        self.reset()

    def get_batch(self, batch_size):
        sample_index = random.sample(np.arange(self.num_experiences).tolist(), batch_size)
        batch = [self.obs[sample_index], self.action[sample_index], self.reward[sample_index],
                 self.next_obs[sample_index], self.matrix[sample_index], self.next_matrix[sample_index],
                 self.done[sample_index], self.is_alive[sample_index], self.next_is_alive[sample_index]]

        return batch

    def add(self, obs, action, prob, reward, new_obs, next_prob):
        obs_tensor = torch.zeros(self.n_agents, self.obs_shape)
        next_obs_tensor = torch.zeros(self.n_agents, self.obs_shape)
        reward_tensor = torch.zeros(self.n_agents)
        action_tensor = torch.zeros(self.n_agents).type(torch.long)
        prob_tensor = torch.from_numpy(prob)
        next_prob_tensor = torch.from_numpy(next_prob)
        mask_tensor = torch.zeros(self.n_agents)
        
        for key in reward.keys():
            obs_tensor[self.agents.index(key)] = obs[key]
            action_tensor[self.agents.index(key)] = int(action[key])
            reward_tensor[self.agents.index(key)] = reward[key]
            next_obs_tensor[self.agents.index(key)] = new_obs[key]
            mask_tensor[self.agents.index(key)] = 1

        self.obs[self.index] = obs_tensor
        self.action[self.index] = action_tensor
        self.reward[self.index] = reward_tensor
        self.next_obs[self.index] = next_obs_tensor
        self.prob[self.index] = prob_tensor
        self.next_prob[self.index] = next_prob_tensor
        self.mask[self.index] = mask_tensor

        if self.num_experiences < self.buffer_size:
            self.num_experiences += 1

        self.index += 1
        if self.index == self.buffer_size:
            self.index = 0

    def reset(self):
        self.obs = torch.zeros(self.buffer_size, self.n_agents, self.obs_shape)
        self.next_obs = torch.zeros(self.buffer_size, self.n_agents, self.obs_shape)
        self.reward = torch.zeros(self.buffer_size, self.n_agents)
        self.action = torch.zeros(self.buffer_size, self.n_agents).type(torch.long)
        self.prob = torch.zeros(self.buffer_size, self.n_agents, self.args.n_actions)
        self.next_prob = torch.zeros(self.buffer_size, self.n_agents, self.args.n_actions)
        self.mask = torch.zeros(self.buffer_size, self.n_agents).type(torch.long)


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
    def __init__(self, args, model, agents):
        self.args = args
        self.obs_shape = args.obs_shape
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
