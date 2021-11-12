import numpy as np
from envs import magent_battle
from utils.preprocessor import Preprocessor, GraphBuilder


class BattleEnv:
    def __init__(self, args):
        self.args = args
        self.env = magent_battle.parallel_env(map_size=args.map_size, minimap_mode=True, step_reward=args.step_reward,
                                              dead_penalty=args.dead_penalty, attack_penalty=args.attack_penalty,
                                              attack_opponent_reward=args.attack_opponent_reward,
                                              max_cycles=args.max_cycle, extra_features=False,
                                              view_range=args.view_range)
        self.preprocessor = Preprocessor(args=args)
        self.possible_agents = self.env.possible_agents

        self.graph_builder = GraphBuilder(args=args, agents=[agent for agent in self.possible_agents if 'red' in agent])
        self.graph_builder.reset()
        self.message = {agent: np.random.rand(args.message_size) for agent in self.possible_agents if 'red' in agent}
        # self.message = {agent: np.exp(self.message[agent]) / sum(np.exp(self.message[agent]))
        #                 for agent in self.message.keys()}

        self.action_space = self.env.action_spaces[self.possible_agents[0]]
        self.observation_space = self.env.observation_spaces[self.possible_agents[0]]
        self.state_space = self.env.state_space
        self.controlled_group = 'red'

    def reset(self, first_time=False):
        if first_time:
            observations = self.env.reset()
        else:
            _ = self.env.reset()
            observations = self.env.reset()

        observations, positions = self.preprocessor.preprocess(observations=observations)

        if self.controlled_group == 'blue':
            self.graph_builder.reset()
            self.graph_builder.build_graph(positions={key: positions[key] for key in positions.keys() if 'blue' in key})
            self.message = {agent: np.random.rand(self.args.message_size) for agent in self.possible_agents
                            if 'blue' in agent}

            if self.args.plot_topology:
                self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions)

            return self._change_side(observations)

        self.graph_builder.reset()
        self.graph_builder.build_graph(positions={key: positions[key] for key in positions.keys() if 'red' in key})
        self.message = {agent: np.random.rand(self.args.message_size) for agent in self.possible_agents
                        if 'red' in agent}

        if self.args.plot_topology:
            self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions)

        return observations

    def step(self, actions):
        assert type(actions) is dict
        if self.controlled_group == 'blue':
            actions = self._change_side(actions)

        observations, rewards, done, infos = self.env.step(actions)
        observations, positions = self.preprocessor.preprocess(observations=observations)

        if self.controlled_group == 'blue':
            if self.args.plot_topology:
                self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions,
                                                              controlled_group=self.controlled_group)

            if self.args.communicate:
                for i in range(100):
                    self.communicate([agent for agent in rewards.keys() if 'blue' in agent])

            self.graph_builder.reset()
            self.graph_builder.build_graph(positions={key: positions[key] for key in positions.keys() if 'blue' in key})
            return self._change_side(observations), self._change_side(rewards), self._change_side(done), \
                   self._change_side(infos)

        self.graph_builder.reset()
        self.graph_builder.build_graph(positions={key: positions[key] for key in positions.keys() if 'red' in key})

        if self.args.plot_topology:
            self.graph_builder.get_communication_topology(state=self.env.state(), positions=positions,
                                                              controlled_group=self.controlled_group)

        if self.args.communicate:
            for i in range(100):
                self.communicate([agent for agent in rewards.keys() if 'red' in agent])

        return observations, rewards, done, infos

    def render(self):
        self.env.render()

    def change_side(self):
        if self.controlled_group == 'red':
            self.controlled_group = 'blue'
        else:
            self.controlled_group = 'red'

        if self.controlled_group == 'blue':
            self.graph_builder.change_side(agents=[agent for agent in self.possible_agents if 'blue' in agent])
        else:
            self.graph_builder.change_side(agents=[agent for agent in self.possible_agents if 'red' in agent])

    def _change_side(self, variables):
        results = {}
        for key in variables.keys():
            if 'red' in key:
                new_key = key.replace('red', 'blue')
            else:
                new_key = key.replace('blue', 'red')

            results[new_key] = variables[key]

        return results

    def close(self):
        self.env.close()
