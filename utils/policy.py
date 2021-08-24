import torch
import numpy as np


class Policy:
    def __init__(self, args, actor_critic):
        self.args = args
        self.actor_critic = actor_critic
        if actor_critic is None:
            self.character = 'enemy'
        else:
            self.character = 'ally'

    def choose_action(self, observations):
        actions, values, log_probs = {}, {}, {}
        if self.character == 'enemy':
            for key in observations.keys():
                random_action = self.random_actions()
                actions[key] = random_action
        else:
            for key in observations.keys():
                action, value, log_prob = self.ppo_action(observation=observations[key])
                actions[key], values[key], log_probs[key] = action, value, log_prob

        return actions, values, log_probs

    def ppo_action(self, observation):
        return self.actor_critic.step(observation)

    def random_actions(self):
        action = np.random.randint(0, self.args.n_actions)
        return action
