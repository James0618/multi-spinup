import gym
from spinup.utils.mpi_tools import mpi_fork
from algos.ppo import ppo
from configs.load_config import load_config, load_default_config
from spinup.utils.run_utils import setup_logger_kwargs
from envs.battle import BattleEnv
from utils.test_policy import test_policy


def run():
    default_args = load_default_config('test')
    default_env = BattleEnv(args=default_args)
    args = load_config(name='test', env=default_env)

    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = {
        'exp_name': 'ppo',
        'output_dir': '/home/drl/PycharmProjects/multi-spinup/results/ppo'
    }

    ppo(lambda: BattleEnv(args=args), args=args, gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps,
        epochs=args.epochs, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    test = True
    if test:
        test_policy()
    else:
        run()
