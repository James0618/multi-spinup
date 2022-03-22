import copy
import torch
import numpy as np
import matplotlib.pyplot as plt


class Preprocessor:
    def __init__(self, args):
        self.args = args
        if self.args.vae_model:
            self.vae_model = torch.load('envs/vae_model/gather_vae_model.pth').cpu()

    def preprocess(self, observations):
        # observation: dict
        obs, positions = {}, {}
        images = {}
        for key, value in observations.items():
            if not self.args.vae_model:
                obs[key] = torch.tensor(value[:, :, [0, 1, 2, 4, 5, 7, 8]].transpose(2, 0, 1), dtype=torch.float)
            else:
                images[key] = value[:, :, [0, 1, 2, 4, 5, 7, 8]].transpose(2, 0, 1)

            positions[key] = (value[0, 0, 7:] * self.args.map_size).astype(np.int)

        if self.args.vae_model:
            obs = self._vae_observation(images)

        return obs, positions

    def add_state(self, env, obs, hid_states=None, matrix=None, state=None):
        obs_with_state, next_hidden_states = {}, {}
        if self.args.gcn:
            obs_tensor = torch.zeros(len(env.possible_agents), self.args.vae_observation_dim)
            hid_states_tensor = torch.zeros(len(env.possible_agents), self.args.hid_shape)
            for key in obs.keys():
                obs_tensor[env.possible_agents.index(key)] = obs[key]
                hid_states_tensor[env.possible_agents.index(key)] = hid_states[key]

            _, next_hid, latent_state, _ = env.state_net.encode(obs_tensor, hid_states_tensor, torch.from_numpy(matrix))
            state = latent_state.detach()
            env.state = state
            for key, value in obs.items():
                obs_with_state[key] = torch.cat((value, state[env.possible_agents.index(key)]), dim=-1)
                next_hidden_states[key] = next_hid[env.possible_agents.index(key)]

        else:
            _, temp, _ = env.state_net.encode(torch.from_numpy(state).type(torch.float).unsqueeze(0))
            state = temp.squeeze().detach()
            env.state = state
            for key, value in obs.items():
                obs_with_state[key] = torch.cat((value, state))

        return obs_with_state, next_hidden_states

    def add_zero_state(self, env, obs):
        obs_with_zero_state = {}
        state = torch.zeros(self.args.latent_state_shape)
        env.state = state
        for key, value in obs.items():
            obs_with_zero_state[key] = torch.cat((value, state))

        return obs_with_zero_state

    def add_mean_state(self, env, obs):
        obs_with_mean_state = {}
        mean_state = torch.zeros(self.args.vae_observation_dim)
        for key, value in obs.items():
            mean_state += value

        mean_state /= len(obs)

        env.state = mean_state
        for key, value in obs.items():
            obs_with_mean_state[key] = torch.cat((value, mean_state))

        return obs_with_mean_state

    def _vae_observation(self, images):
        image_tensor = torch.zeros(len(images), *self.args.observation_shape)
        for i, key in enumerate(images.keys()):
            image_tensor[i] = torch.from_numpy(images[key])

        result = self.vae_model.encoder(image_tensor)
        z, mu, log_var = self.vae_model.bottleneck(result)

        vae_obs = {}
        for i, key in enumerate(images.keys()):
            vae_obs[key] = (mu[i].detach().squeeze() + 1) / 2

        return vae_obs


class GraphBuilder:
    def __init__(self, args, agents):
        self.args = args
        self.possible_agents = agents
        self.min_neigh = args.min_neigh
        self.num_agents = len(agents)
        self.adjacency_matrix, self.min_matrix, self.distance_matrix, self.accessible_agents = None, None, None, None
        self.all_direction_matrix = None

    def reset(self):
        self.adjacency_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.int)
        self.distance_matrix = np.ones((self.num_agents, self.num_agents), dtype=np.int) * 1000
        self.all_direction_matrix = np.ones((4, self.num_agents, self.num_agents), dtype=np.int) * 1000
        self.accessible_agents = {}

    def change_side(self, agents):
        self.possible_agents = agents

    def build_graph(self, positions):
        self.distance_matrix = np.ones((self.num_agents, self.num_agents), dtype=np.int) * 1000
        self.all_direction_matrix = np.ones((4, self.num_agents, self.num_agents), dtype=np.int) * 1000

        for agent in positions.keys():
            agent_id = self.possible_agents.index(agent)
            agent_position = positions[agent]
            for other_agent in positions.keys():
                if other_agent is not agent:
                    other_agent_id = self.possible_agents.index(other_agent)
                    other_agent_position = positions[other_agent]
                    distance = max(abs(agent_position - other_agent_position))
                    self.distance_matrix[agent_id, other_agent_id] = distance
                    self.distance_matrix[other_agent_id, agent_id] = distance

        for agent in positions.keys():
            agent_id = self.possible_agents.index(agent)
            agent_position = positions[agent]
            for other_agent in positions.keys():
                if other_agent is not agent:
                    other_agent_id = self.possible_agents.index(other_agent)
                    other_agent_position = positions[other_agent]
                    distance = max(abs(agent_position - other_agent_position))
                    if other_agent_position[0] < agent_position[0]:
                        self.all_direction_matrix[0, agent_id, other_agent_id] = distance
                    if other_agent_position[0] > agent_position[0]:
                        self.all_direction_matrix[1, agent_id, other_agent_id] = distance
                    if other_agent_position[1] < agent_position[1]:
                        self.all_direction_matrix[2, agent_id, other_agent_id] = distance
                    if other_agent_position[1] > agent_position[1]:
                        self.all_direction_matrix[3, agent_id, other_agent_id] = distance

        self._get_adjacency_matrix()
        self._get_accessible_agents(agents=list(positions.keys()))

    def get_communication_topology(self, state, positions, controlled_group):
        for agent in self.accessible_agents.keys():
            agent_position = positions[agent]
            for accessible_agent in self.accessible_agents[agent]:
                accessible_agent_position = positions[accessible_agent]
                plt.plot([agent_position[0], accessible_agent_position[0]],
                         [agent_position[1], accessible_agent_position[1]], 'r')

        if controlled_group == 'red':
            plt.spy(state[:, :, 1].transpose())
            plt.show()
        else:
            plt.spy(state[:, :, 3].transpose())
            plt.show()

    def _get_adjacency_matrix(self):
        n_agents = self.distance_matrix.shape[0]
        temp = torch.arange(n_agents).tolist()
        temp_matrix = copy.deepcopy(self.all_direction_matrix)
        all_min_matrix = np.zeros((n_agents, n_agents))

        for i in range(4):
            min_index = temp_matrix[i].argmin(-1).tolist()
            invalid_temp = (temp_matrix[i, temp, min_index] > self.args.map_size).nonzero()[0].tolist()
            invalid_index = np.array(min_index)[invalid_temp].tolist()
            all_min_matrix[temp, min_index] = 1
            all_min_matrix[min_index, temp] = 1
            all_min_matrix[invalid_temp, invalid_index] = 0
            all_min_matrix[invalid_index, invalid_temp] = 0

        self.adjacency_matrix = all_min_matrix.astype(np.int)

    def _get_accessible_agents(self, agents):
        for agent in agents:
            agent_id = self.possible_agents.index(agent)
            accessible_agents_id = self.adjacency_matrix[agent_id].nonzero()[0].tolist()
            self.accessible_agents[agent] = [self.possible_agents[i] for i in accessible_agents_id]
