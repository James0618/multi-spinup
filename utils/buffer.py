import numpy as np
import torch
from utils import core
from spinup.utils.mpi_tools import mpi_statistics_scalar


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_shape, n_actions, max_agents, max_cycle, buffer_size, agents, gamma=0.99, lam=0.95):
        self.possible_agents = agents
        self.obs_shape, self.n_actions, self.max_agents, self.max_cycle = obs_shape, n_actions, max_agents, max_cycle
        self.obs_buf, self.act_buf, self.adv_buf, self.rew_buf = None, None, None, None
        self.ret_buf, self.val_buf, self.logp_buf = None, None, None

        self.extra_obs_buf, self.adj_buf, self.pos_buf, self.is_alive_buf = None, None, None, None
        self.terminated_buf = None
        self.dsin_ptr = 0

        self.obs_temp_buf, self.act_temp_buf, self.adv_temp_buf, self.rew_temp_buf = None, None, None, None
        self.ret_temp_buf, self.val_temp_buf, self.logp_temp_buf = None, None, None

        self.gamma, self.lam = gamma, lam
        self.ptr, self.steps_in_buffer, self.buffer_size = np.zeros(max_agents).astype(np.int), 0, buffer_size
        self.buffer_ptr = 0

        self.reset()
        self.reset_temp_buffer()

    def reset(self):
        self.obs_buf = torch.zeros(self.buffer_size, *self.obs_shape)
        self.act_buf = torch.zeros(self.buffer_size).type(torch.long)
        self.adv_buf = torch.zeros(self.buffer_size)
        self.rew_buf = torch.zeros(self.buffer_size)
        self.ret_buf = torch.zeros(self.buffer_size)
        self.val_buf = torch.zeros(self.buffer_size)
        self.logp_buf = torch.zeros(self.buffer_size)

        self.adj_buf = torch.zeros(self.buffer_size, self.max_agents, self.max_agents)
        self.pos_buf = torch.zeros(self.buffer_size, self.max_agents, 2)
        self.is_alive_buf = torch.zeros(self.buffer_size, self.max_agents)
        self.terminated_buf = torch.zeros(self.buffer_size)

        self.steps_in_buffer = 0
        self.buffer_ptr = 0
        self.dsin_ptr = 0

    def reset_temp_buffer(self):
        self.obs_temp_buf = torch.zeros(self.max_cycle, self.max_agents, *self.obs_shape)
        self.act_temp_buf = torch.zeros(self.max_cycle, self.max_agents).type(torch.long)
        self.adv_temp_buf = torch.zeros(self.max_cycle, self.max_agents)
        self.rew_temp_buf = torch.zeros(self.max_cycle, self.max_agents)
        self.ret_temp_buf = torch.zeros(self.max_cycle, self.max_agents)
        self.val_temp_buf = torch.zeros(self.max_cycle, self.max_agents)
        self.logp_temp_buf = torch.zeros(self.max_cycle, self.max_agents)

        self.ptr = np.zeros(self.max_agents).astype(np.int)

    def store(self, obs, act, rew, val, logp, state=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        for i, agent in enumerate(self.possible_agents):
            if agent in rew.keys():
                try:
                    self.obs_temp_buf[self.ptr[i], i] = obs[agent]
                    self.act_temp_buf[self.ptr[i], i] = act[agent]
                    self.rew_temp_buf[self.ptr[i], i] = rew[agent]
                    self.val_temp_buf[self.ptr[i], i] = val[agent]
                    self.logp_temp_buf[self.ptr[i], i] = logp[agent]
                    self.ptr[i] += 1
                except:
                    print(rew)

        self.steps_in_buffer += len(list(rew.keys()))

    def store_extra(self, matrix, obs, pos, agents, terminal):
        self.adj_buf[self.dsin_ptr] = torch.from_numpy(matrix).type(torch.float)

        for agent in agents:
            self.extra_obs_buf[self.dsin_ptr, self.possible_agents.index(agent)] = obs[agent]
            self.pos_buf[self.dsin_ptr, self.possible_agents.index(agent)] = torch.from_numpy(pos[agent])
            self.is_alive_buf[self.dsin_ptr, self.possible_agents.index(agent)] = 1

        self.terminated_buf[self.dsin_ptr] = 1 if terminal else 0
        self.pos_buf[self.dsin_ptr] = pos

        self.dsin_ptr += 1

    def finish_path(self, last_val):
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
        for agent in self.possible_agents:
            self.finish_path_agent(agent, last_val[agent])

        for i, agent in enumerate(self.possible_agents):
            epoch_full = self.buffer_ptr + self.ptr[i] >= self.buffer_size
            agent_path_slice = slice(self.buffer_ptr, min(self.buffer_ptr + self.ptr[i], self.buffer_size))
            agent_ptr = min(self.buffer_ptr + self.ptr[i], self.buffer_size) - self.buffer_ptr
            self.obs_buf[agent_path_slice] = self.obs_temp_buf[0: agent_ptr, i]
            self.act_buf[agent_path_slice] = self.act_temp_buf[0: agent_ptr, i]
            self.rew_buf[agent_path_slice] = self.rew_temp_buf[0: agent_ptr, i]
            self.adv_buf[agent_path_slice] = self.adv_temp_buf[0: agent_ptr, i]
            self.ret_buf[agent_path_slice] = self.ret_temp_buf[0: agent_ptr, i]
            self.val_buf[agent_path_slice] = self.val_temp_buf[0: agent_ptr, i]
            self.logp_buf[agent_path_slice] = self.logp_temp_buf[0: agent_ptr, i]

            self.buffer_ptr += agent_ptr

            if epoch_full:
                break

        self.reset_temp_buffer()

    def finish_path_agent(self, agent, last_val=0):
        agent_id = self.possible_agents.index(agent)
        path_slice = slice(0, self.ptr[agent_id])
        rews = np.append(self.rew_temp_buf[path_slice, agent_id], last_val)
        vals = np.append(self.val_temp_buf[path_slice, agent_id], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_temp_buf[path_slice, agent_id] = torch.from_numpy(np.ascontiguousarray(
            core.discount_cumsum(deltas, self.gamma * self.lam)).squeeze())

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_temp_buf[path_slice, agent_id] = torch.from_numpy(np.ascontiguousarray(
            core.discount_cumsum(rews, self.gamma)[:-1]).squeeze())

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.buffer_ptr == self.buffer_size  # buffer has to be full before you can get
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def get_extra(self):
        data = dict(matrix=self.adj_buf, obs=self.extra_obs_buf, pos=self.pos_buf,
                    is_alive=self.is_alive_buf, terminated=self.terminated_buf)
        return {k: torch.as_tensor(v, dtype=torch.float) for k, v in data.items()}
