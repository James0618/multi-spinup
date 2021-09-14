import numpy as np
from envs import magent_gather
from utils.preprocessor import Preprocessor, GraphBuilder


class GatherEnv:
    def __init__(self, args):
        self.args = args
        self.env = magent_gather.parallel_env(map_size=args.map_size, minimap_mode=True, step_reward=args.step_reward,
                                              dead_penalty=args.dead_penalty, attack_penalty=args.attack_penalty,
                                              attack_food_reward=args.attack_food_reward,
                                              max_cycles=args.max_cycle, extra_features=False,
                                              view_range=args.view_range)
        self.preprocessor = Preprocessor(args=args)
        self.possible_agents = self.env.possible_agents

        self.graph_builder = GraphBuilder(args=args, agents=self.possible_agents)
        self.graph_builder.reset()
        self.message = {agent: np.random.rand(args.message_size) for agent in self.possible_agents}
        # self.message = {agent: np.exp(self.message[agent]) / sum(np.exp(self.message[agent]))
        #                 for agent in self.message.keys()}

        self.action_space = self.env.action_spaces[self.possible_agents[0]]
        self.observation_space = self.env.observation_spaces[self.possible_agents[0]]
        self.state_space = self.env.state_space
        self.controlled_group = 'omnivore'

    def reset(self):
        observations = self.env.reset()
        observations, positions = self.preprocessor.preprocess(observations=observations)

        self.graph_builder.reset()
        self.graph_builder.build_graph(positions=positions)
        self.message = {agent: np.random.rand(self.args.message_size) for agent in self.possible_agents}

        if self.args.plot_topology:
            self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions,
                                                          controlled_group=self.controlled_group)

        return observations

    def step(self, actions):
        assert type(actions) is dict

        observations, rewards, done, infos = self.env.step(actions)
        observations, positions = self.preprocessor.preprocess(observations=observations)

        self.graph_builder.reset()
        self.graph_builder.build_graph(positions=positions)

        if self.args.plot_topology:
            self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions,
                                                          controlled_group=self.controlled_group)

        if self.args.communicate:
            for i in range(100):
                self.communicate([agent for agent in rewards.keys()])

        return observations, rewards, done, infos

    def render(self):
        for i in range(3):
            self.env.render()

    def close(self):
        self.env.close()

    def communicate(self, agents):
        accessible_agents = self.graph_builder.accessible_agents
        temp = {agent: np.zeros(self.args.message_size) for agent in agents}
        for agent in agents:
            accessible_agent = accessible_agents[agent]
            for other_agent in accessible_agent:
                temp[agent] += self.message[other_agent]

            temp[agent] -= np.matmul(temp[agent], self.message[agent])
            temp[agent] -= self.message[agent] * sum(self.graph_builder.adjacency_matrix[
                                                         self.possible_agents.index(agent)])

        for agent in agents:
            self.message[agent] -= temp[agent] * 0.05
            # self.message[agent] = np.exp(self.message[agent]) / sum(np.exp(self.message[agent]))

        show_array = np.zeros((len(agents), self.args.message_size))
        for i, agent in enumerate(agents):
            show_array[i] = self.message[agent]

        print('End Communication!')
