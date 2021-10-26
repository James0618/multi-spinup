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
                obs[key] = value[:, :, [0, 1, 2, 4, 5, 7, 8]].transpose(2, 0, 1)
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

    # def _vae_observation(self, observation):
    #     result = self.vae_model.encoder(torch.from_numpy(observation).unsqueeze(0))[0]
    #     z, mu, log_var = self.vae_model.bottleneck(result)
    #     return mu.detach().squeeze()

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

    def reset(self):
        self.adjacency_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.int)
        self.distance_matrix = np.ones((self.num_agents, self.num_agents),
                                       dtype=np.int) * 1000
        self.accessible_agents = {}

    def change_side(self, agents):
        self.possible_agents = agents

    def build_graph(self, positions):
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

        self._get_adjacency_matrix()
        # self._get_accessible_agents(agents=list(positions.keys()))

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
        self.adjacency_matrix = (self.distance_matrix <= self.args.communication_range).astype(np.int)

        temp = torch.arange(self.num_agents).tolist()
        self.min_matrix = np.zeros_like(self.adjacency_matrix)
        for i in range(self.min_neigh):
            min_index = self.distance_matrix.argmin(axis=-1)
            self.min_matrix[temp, min_index.tolist()] = 1
            self.distance_matrix[temp, min_index.tolist()] += 1000

    def _get_accessible_agents(self, agents):
        for agent in agents:
            agent_id = self.possible_agents.index(agent)
            accessible_agents_id = self.adjacency_matrix[agent_id].nonzero()[0].tolist()
            self.accessible_agents[agent] = [self.possible_agents[i] for i in accessible_agents_id]
