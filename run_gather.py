import gym
import time
from spinup.utils.mpi_tools import mpi_fork
from algos.ppo_gather import ppo
from algos.centralized_ppo_gather import ppo as centralized_ppo
from utils import core, centralized_core
from configs.load_config import load_config, load_default_config
from envs.gather import GatherEnv
from utils.test_gather_policy import test_policy


def run(exp_name='ppo', config_name='gather'):
    default_args = load_default_config(config_name)
    default_env = GatherEnv(args=default_args)
    args = load_config(name=config_name, env=default_env)

    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = {
        'exp_name': exp_name,
        'output_dir': '/home/drl/PycharmProjects/multi-spinup/results/{}'.format(exp_name)
    }
    if args.vae_model:
        if args.centralized:
            actor_critic = centralized_core.StateVAEActorCritic
        else:
            actor_critic = core.VAEActorCritic

    else:
        actor_critic = core.CNNActorCritic

    if args.centralized:
        centralized_ppo(lambda: GatherEnv(args=args), args=args, gamma=args.gamma, seed=args.seed,
                        steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, logger_kwargs=logger_kwargs,
                        actor_critic=actor_critic)
    else:
        ppo(lambda: GatherEnv(args=args), args=args, gamma=args.gamma, seed=args.seed,
            steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, logger_kwargs=logger_kwargs,
            actor_critic=actor_critic)


if __name__ == '__main__':
    test, centralized = False, False
    t = time.localtime(time.time())
    if centralized:
        experiment = 'gather-centralized-ppo-{}-{}'.format(t.tm_mon, t.tm_mday)
        config_name = 'centralized_gather'
    else:
        experiment = 'gather-ppo-{}-{}'.format(t.tm_mon, t.tm_mday)
        config_name = 'gather'
    # experiment = 'gather-ppo-{}-{}'.format(9, 23)
    if test:
        test_policy(experiment=experiment, config_name=config_name)
    else:
        run(exp_name=experiment, config_name=config_name)
