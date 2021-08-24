import numpy as np
import torch
import argparse
from torch.optim import Adam
import time
from utils import core, policy, buffer
from envs.battle import BattleEnv
from configs.load_config import load_config, load_default_config
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_avg, proc_id, mpi_statistics_scalar, num_procs


def is_terminated(groups, terminated):
    results = []
    for group in groups:
        results.append(np.array([terminated[key] for key in terminated.keys() if group in key]).prod())

    return bool(np.array(results).sum())


def load_model(args):
    model_path = args.path + '/pyt_save/model.pt'
    actor_critic = torch.load(model_path)

    return actor_critic


def test_policy():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/drl/PycharmProjects/multi-spinup/results/ppo')
    parser.add_argument('--test_episode', type=int, default=10)
    args = parser.parse_args()

    actor_critic = load_model(args=args)
    env_args = load_default_config('test')
    env = BattleEnv(args=env_args)
    run_args = load_config('test', env)

    groups = ['red', 'blue']
    controlled_group = 0

    ally_policy = policy.Policy(args=run_args, actor_critic=actor_critic)
    enemy_policy = policy.Policy(args=run_args, actor_critic=None)

    # Main loop: collect experience in env and update/log each epoch
    obs = env.reset(first_time=True)
    for episode in range(args.test_episode):
        terminal = False
        ep_ret, ep_len = 0, 0
        env.change_side()
        obs = env.reset()
        while not terminal:
            ally_actions, values, log_probs = ally_policy.choose_action({
                key: obs[key] for key in obs.keys() if groups[controlled_group] in key})
            enemy_actions, _, _ = enemy_policy.choose_action({
                key: obs[key] for key in obs.keys() if groups[1 - controlled_group] in key})

            actions = {**ally_actions, **enemy_actions}

            env.render()
            time.sleep(0.05)
            next_obs, rewards, done, _ = env.step(actions)
            ep_ret += sum([rewards[agent] for agent in rewards.keys() if groups[controlled_group] in agent])
            ep_len += 1

            # Update obs (critical!)
            obs = next_obs

            terminal = is_terminated(groups=groups, terminated=done)

        print('Test episode {}: return {:.2f}'.format(episode, ep_ret))


if __name__ == '__main__':
    test_policy()
