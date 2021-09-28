import numpy as np
import torch
from utils import core
from spinup.utils.mpi_tools import mpi_statistics_scalar


class PPOCentralizedBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_shape, n_actions, max_agents, max_cycle, buffer_size, agents, gamma=0.99, lam=0.95):
        self.possible_agents = agents
        self.obs_shape, self.n_actions, self.max_agents, self.max_cycle = obs_shape, n_actions, max_agents, max_cycle
        self.buffer_size, self.gamma, self.lam = buffer_size, gamma, lam
        self.obs_buf, self.act_buf, self.adv_buf, self.val_buf = None, None, None, None
        self.logp_buf, self.ret_buf, self.rew_buf, self.alive_buf = None, None, None, None

        self.ptr, self.path_start_idx = 0, 0

    def reset(self):
        self.obs_buf = torch.zeros(self.buffer_size, self.max_agents, *self.obs_shape)
        self.act_buf = torch.zeros(self.buffer_size, self.max_agents, dtype=torch.long)
        self.logp_buf = torch.zeros(self.buffer_size, self.max_agents)
        self.alive_buf = torch.zeros(self.buffer_size, self.max_agents)

        self.adv_buf = torch.zeros(self.buffer_size)
        self.rew_buf = torch.zeros(self.buffer_size)
        self.ret_buf = torch.zeros(self.buffer_size)
        self.val_buf = torch.zeros(self.buffer_size)

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.buffer_size  # buffer has to have room so you can store
        global_reward = 0
        for i, agent in enumerate(self.possible_agents):
            if agent in rew.keys():
                try:
                    self.obs_buf[self.ptr, i] = obs[agent]
                    self.act_buf[self.ptr, i] = act[agent]
                    self.logp_buf[self.ptr, i] = logp[agent]
                    self.alive_buf[self.ptr, i] = 1
                    global_reward += rew[agent]

                except:
                    raise ValueError('Wrong Index')

        self.rew_buf[self.ptr] = global_reward
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
