import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs.load_config import load_config, load_default_config
from envs.gather import GatherEnv
import argparse
import time


class ReplayBuffer(object):
    def __init__(self, buffer_size, agents, args):
        self.buffer_size = buffer_size
        self.agents = agents
        self.n_agents = len(agents)
        self.args = args
        if args.with_state:
            self.obs_shape = args.vae_observation_dim + args.latent_state_shape
        else:
            self.obs_shape = args.vae_observation_dim

        self.obs, self.next_obs, self.action, self.reward = None, None, None, None
        self.prob, self.next_prob, self.mask, self.terminated = None, None, None, None
        self.index, self.num_experiences = 0, 0
        self.reset()

    def get_batch(self, batch_size):
        # sample_index = random.sample(np.arange(self.num_experiences).tolist(), batch_size)
        batch_dict = {'obs': [], 'next_obs': [], 'action': [], 'reward': [], 'prob': [], 'next_prob': [],
                      'terminated': []}

        batch_num = 0
        while True:
            i = random.randint(0, self.num_experiences - 1)
            j = random.randint(0, self.n_agents - 1)
            if self.mask[i, j] == 1:
                batch_dict['obs'].append(self.obs[i, j])
                batch_dict['next_obs'].append(self.next_obs[i, j])
                batch_dict['action'].append(self.action[i, j])
                batch_dict['reward'].append(self.reward[i, j])
                batch_dict['prob'].append(self.prob[i, j])
                batch_dict['next_prob'].append(self.next_prob[i, j])
                batch_dict['terminated'].append(self.terminated[i, j])
                batch_num += 1

            if batch_num == batch_size:
                break

        # batch = [self.obs[sample_index], self.next_obs[sample_index], self.action[sample_index],
        #          self.reward[sample_index], self.prob[sample_index], self.next_prob[sample_index],
        #          self.mask[sample_index], self.terminated[sample_index]]

        batch = {key: torch.stack(batch_dict[key]) for key in batch_dict.keys()}

        return batch

    def add(self, obs, action, prob, reward, new_obs, next_prob, done):
        if not self.args.vae_model:
            obs_tensor = torch.zeros(self.n_agents, *self.args.input_shape[0])
            next_obs_tensor = torch.zeros(self.n_agents, *self.args.input_shape[0])
        else:
            obs_tensor = torch.zeros(self.n_agents, self.obs_shape)
            next_obs_tensor = torch.zeros(self.n_agents, self.obs_shape)

        reward_tensor = torch.zeros(self.n_agents)
        action_tensor = torch.zeros(self.n_agents).type(torch.long)
        prob_tensor = torch.from_numpy(prob)
        next_prob_tensor = torch.from_numpy(next_prob)
        mask_tensor = torch.zeros(self.n_agents)
        terminated_tensor = torch.zeros(self.n_agents)
        
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
        self.prob[self.index] = prob_tensor
        self.next_prob[self.index] = next_prob_tensor
        self.mask[self.index] = mask_tensor
        self.terminated[self.index] = terminated_tensor

        if self.num_experiences < self.buffer_size:
            self.num_experiences += 1

        self.index += 1
        if self.index == self.buffer_size:
            self.index = 0

    def reset(self):
        if not self.args.vae_model:
            self.obs = torch.zeros(self.buffer_size, self.n_agents, *self.args.input_shape[0])
            self.next_obs = torch.zeros(self.buffer_size, self.n_agents, *self.args.input_shape[0])
        else:
            self.obs = torch.zeros(self.buffer_size, self.n_agents, self.obs_shape)
            self.next_obs = torch.zeros(self.buffer_size, self.n_agents, self.obs_shape)

        self.reward = torch.zeros(self.buffer_size, self.n_agents)
        self.action = torch.zeros(self.buffer_size, self.n_agents).type(torch.long)
        self.prob = torch.zeros(self.buffer_size, self.n_agents, self.args.n_actions)
        self.next_prob = torch.zeros(self.buffer_size, self.n_agents, self.args.n_actions)
        self.mask = torch.zeros(self.buffer_size, self.n_agents).type(torch.long)
        self.terminated = torch.zeros(self.buffer_size, self.n_agents).type(torch.long)


class NVIFDQN(nn.Module):
    def __init__(self, args):
        super(NVIFDQN, self).__init__()
        self.args = args
        self.obs_shape = args.vae_observation_dim + args.latent_state_shape

        if not args.vae_model:
            self.encoder = nn.Sequential(
                nn.Conv2d(7, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            self.obs_shape = 576

        self.q_net = nn.Sequential(
            nn.Linear(self.obs_shape, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions),
        )

    def forward(self, obs):
        if not self.args.vae_model:
            obs = self.encoder(obs).view(obs.shape[0], -1)

        q = self.q_net(obs)

        return q

    def update_param(self, target_model):
        for param, target_param in zip(self.parameters(), target_model.parameters()):
            param.data.copy_(self.args.tau * target_param.data + (1 - self.args.tau) * param.data)


class Policy:
    def __init__(self, args, model, agents):
        self.args = args
        self.obs_shape = args.vae_observation_dim + args.latent_state_shape
        self.model = model
        self.agents = agents

    def choose_action(self, observations, epsilon=1.0):
        actions = {}

        n_agent = len(self.agents)
        if not self.args.vae_model:
            obs_tensor = torch.zeros(n_agent, *self.args.input_shape[0]).cuda()
        else:
            obs_tensor = torch.zeros(n_agent, self.obs_shape).cuda()

        for key in observations.keys():
            obs_tensor[self.agents.index(key)] = observations[key].cuda()

        q = self.model(obs_tensor)

        for key in observations.keys():
            index = self.agents.index(key)
            if np.random.rand() < epsilon:
                action = np.random.randint(self.args.n_actions)
            else:
                action = q[index].argmax().item()

            actions[key] = int(action)

        return actions


def is_terminated(terminated):
    groups = ['omnivore']
    results = []
    for group in groups:
        results.append(np.array([terminated[key] for key in terminated.keys() if group in key]).prod())

    return bool(np.array(results).sum())


def load_model(args):
    model_path = args.path + '/pyt_save/model.pt'
    actor_critic = torch.load(model_path)

    return actor_critic


def get_prob(actions, n_actions):
    prob = np.zeros(n_actions)
    n_agents = len(actions.keys())
    for key in actions.keys():
        prob += np.eye(n_actions)[actions[key]]

    return prob / n_agents


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
    policy = Policy(args=run_args, model=model, agents=agents)

    i_episode = 0
    test_ret = 0

    while i_episode < test_num:
        i_episode += 1

        steps, ep_ret = 0, 0
        obs = env.reset()
        prob = np.zeros(run_args.n_actions)

        while True:
            steps += 1
            actions = policy.choose_action(observations=obs, mean_actions=prob, epsilon=0.0)
            next_prob = get_prob(actions=actions, n_actions=run_args.n_actions)

            if render:
                env.render()

            next_obs, rewards, done, next_positions = env.step(actions)
            terminal = is_terminated(terminated=done)

            obs = next_obs
            prob = next_prob

            if run_args.global_reward:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()]) / len(rewards)
            else:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()])

            if terminal:
                break

        test_ret += ep_ret

    return test_ret / test_num
