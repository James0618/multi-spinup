import gym
import time
from spinup.utils.mpi_tools import mpi_fork
from algos.ppo_gather import ppo
from configs.load_config import load_config, load_default_config
from envs.gather import GatherEnv
from utils.test_gather_policy import test_policy


def run(exp_name='ppo'):
    default_args = load_default_config('gather')
    default_env = GatherEnv(args=default_args)
    args = load_config(name='gather', env=default_env)

    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = {
        'exp_name': exp_name,
        'output_dir': '/home/drl/PycharmProjects/multi-spinup/results/{}'.format(exp_name)
    }

    ppo(lambda: GatherEnv(args=args), args=args, gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    test = False
    t = time.localtime(time.time())
    experiment = 'gather-ppo-{}-{}'.format(t.tm_mon, t.tm_mday)
    if test:
        test_policy(experiment=experiment)
    else:
        run(exp_name=experiment)
