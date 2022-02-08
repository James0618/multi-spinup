import numpy as np
import torch
from envs import magent_gather
from utils.preprocessor import Preprocessor, GraphBuilder


class GatherEnv:
    def __init__(self, args):
        self.args = args
        self.env = magent_gather.parallel_env(map_size=args.map_size, minimap_mode=True, step_reward=args.step_reward,
                                              dead_penalty=args.dead_penalty, attack_penalty=args.attack_penalty,
                                              attack_food_reward=args.attack_food_reward,
                                              max_cycles=args.max_cycle, extra_features=False,
                                              view_range=args.view_range, if_random=args.random)
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
        self.state_net = torch.load('envs/vae_model/state-vae.pth').cpu()
        self.state = None
        self.hidden_states = None
        self.with_state = args.with_state

    def reset(self):
        observations = self.env.reset()
        state = self.env.state().transpose(2, 0, 1)
        observations, positions = self.preprocessor.preprocess(observations=observations)

        self.graph_builder.reset()
        self.graph_builder.build_graph(positions=positions)

        if self.args.with_state:
            self.hidden_states = {agent: torch.zeros(self.args.hid_shape) for agent in self.possible_agents}

            if self.args.zero_state:
                observations = self.preprocessor.add_zero_state(env=self, obs=observations)
            else:
                observations, self.hidden_states = self.preprocessor.add_state(
                    env=self, obs=observations, hid_states=self.hidden_states,
                    matrix=self.graph_builder.adjacency_matrix, state=state)

        if self.args.plot_topology:
            self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions,
                                                          controlled_group=self.controlled_group)

        return observations

    def step(self, actions):
        assert type(actions) is dict

        observations, rewards, done, infos = self.env.step(actions)
        if self.args.global_reward:
            total_reward = 0
            for key in rewards.keys():
                total_reward += rewards[key]

            for key in rewards.keys():
                rewards[key] = total_reward

        state = self.env.state().transpose(2, 0, 1)
        observations, positions = self.preprocessor.preprocess(observations=observations)

        self.graph_builder.reset()
        self.graph_builder.build_graph(positions=positions)

        if self.args.with_state:
            if self.args.zero_state:
                observations = self.preprocessor.add_zero_state(env=self, obs=observations)
            else:
                observations, self.hidden_states = self.preprocessor.add_state(
                    env=self, obs=observations, hid_states=self.hidden_states,
                    matrix=self.graph_builder.adjacency_matrix, state=state)

        if self.args.plot_topology:
            self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions,
                                                          controlled_group=self.controlled_group)

        return observations, rewards, done, positions

    def render(self):
        for i in range(3):
            self.env.render()

    def close(self):
        self.env.close()
