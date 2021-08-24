from pettingzoo.magent import battle_v3
from utils.preprocessor import Preprocessor


class BattleEnv:
    def __init__(self, args):
        self.args = args
        self.env = battle_v3.parallel_env(map_size=args.map_size, minimap_mode=True, step_reward=args.step_reward,
                                          dead_penalty=args.dead_penalty, attack_penalty=args.attack_penalty,
                                          attack_opponent_reward=args.attack_opponent_reward,
                                          max_cycles=args.max_cycle, extra_features=False)
        self.preprocessor = Preprocessor(args=args)
        self.possible_agents = self.env.possible_agents
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
            return self._change_side(observations)

        return observations

    def step(self, actions):
        assert type(actions) is dict
        if self.controlled_group == 'blue':
            actions = self._change_side(actions)

        observations, rewards, done, infos = self.env.step(actions)
        observations, positions = self.preprocessor.preprocess(observations=observations)

        if self.controlled_group == 'blue':
            return self._change_side(observations), self._change_side(rewards), self._change_side(done), \
                   self._change_side(infos)

        return observations, rewards, done, infos

    def render(self):
        self.env.render()

    def change_side(self):
        if self.controlled_group == 'red':
            self.controlled_group = 'blue'
        else:
            self.controlled_group = 'red'

    def _change_side(self, variables):
        results = {}
        for key in variables.keys():
            if 'red' in key:
                new_key = key.replace('red', 'blue')
            else:
                new_key = key.replace('blue', 'red')

            results[new_key] = variables[key]

        return results