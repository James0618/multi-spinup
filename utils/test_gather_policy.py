import numpy as np
import torch
import argparse
from torch.optim import Adam
import time
from utils import core, policy, buffer
from envs.gather import GatherEnv
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


def test_policy(experiment, config_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/drl/PycharmProjects/multi-spinup/results/{}'.format(
        experiment))
    parser.add_argument('--test_episode', type=int, default=10)
    args = parser.parse_args()

    actor_critic = load_model(args=args)
    env_args = load_default_config(config_name)
    env = GatherEnv(args=env_args)
    run_args = load_config(config_name, env)

    groups = ['omnivore']

    ally_policy = policy.Policy(args=run_args, actor_critic=actor_critic)

    # Main loop: collect experience in env and update/log each epoch
    obs = env.reset()
    for episode in range(args.test_episode):
        terminal = False
        ep_ret, ep_len = 0, 0
        obs = env.reset()
        while not terminal:
            actions, values, log_probs = ally_policy.choose_action(obs)

            env.render()
            time.sleep(0.05)
            next_obs, rewards, done, _ = env.step(actions)
            ep_ret += sum([rewards[agent] for agent in rewards.keys()])
            ep_len += 1

            # Update obs (critical!)
            obs = next_obs

            terminal = is_terminated(groups=groups, terminated=done)

        print('Test episode {}: return {:.2f}'.format(episode, ep_ret))
        env.close()
