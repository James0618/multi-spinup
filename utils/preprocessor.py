import torch
import numpy as np


class Preprocessor:
    def __init__(self, args):
        self.args = args

    def preprocess(self, observations):
        # observation: dict
        result, positions = {}, {}
        for key, value in observations.items():
            image = value.transpose(2, 0, 1)
            position = value[0, 0, 7:]

            result[key] = torch.from_numpy(image)
            positions[key] = (position * self.args.map_size).astype(np.int)

        return result, positions


class GraphBuilder:
    def __init__(self, args, agents):
        self.args = args
        self.possible_agents = agents
        self.num_agents = len(agents)
        self.adjacency_matrix, self.distance_matrix, self.accessible_agents = None, None, None

    def reset(self):
        self.adjacency_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.int)
        self.distance_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.int)
        self.accessible_agents = {}

    def build_graph(self, positions):
        for agent in self.possible_agents:
            agent_id = self.possible_agents.index(agent)
            agent_position = positions[agent]
            for other_agent in positions.keys():
                if other_agent is not agent:
                    other_agent_id = self.possible_agents.index(other_agent)
                    other_agent_position = positions[other_agent]
                    distance = max(abs(agent_position - other_agent_position))
                    self.distance_matrix[agent_id, other_agent_id] = distance
                    self.distance_matrix[other_agent_id, agent_id] = distance
                else:
                    self.distance_matrix[agent_id, agent_id] = self.args.communication_range + 1

        self._get_adjacency_matrix()
        self._get_accessible_agents(agents=list(positions.keys()))

    def _get_adjacency_matrix(self):
        self.adjacency_matrix = (self.distance_matrix <= self.args.communication_range).astype(np.int)

    def _get_accessible_agents(self, agents):
        for agent in agents:
            agent_id = self.possible_agents.index(agent)
            accessible_agents_id = self.adjacency_matrix[agent_id].nonzero()[0].tolist()
            self.accessible_agents[agent] = [self.possible_agents[i] for i in accessible_agents_id]
