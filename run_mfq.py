import os
import time
from algos.mfq import mfq
from configs.load_config import load_config, load_default_config
from envs.gather import GatherEnv
from utils.mfq_model import test_policy


def run(exp_name='mfq', config_name='mfq'):
    output_path = '/home/drl/PycharmProjects/multi-spinup/results/'
    default_args = load_default_config(config_name)
    default_env = GatherEnv(args=default_args)
    args = load_config(name=config_name, env=default_env, output_path=output_path, exp_name=exp_name)

    logger_kwargs = {
        'exp_name': exp_name,
        'output_dir': output_path + '{}'.format(exp_name)
    }

    mfq(lambda: GatherEnv(args=args), args=args, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    test = False

    device_id = 0
    experiment_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(device_id)

    t = time.localtime(time.time())
    experiment = 'gather-mfq-{}-{}/{}'.format(t.tm_mon, t.tm_mday, experiment_id)
    config_name = 'mfq'

    if test:
        test_policy(experiment=experiment, config_name=config_name)
    else:
        run(exp_name=experiment, config_name=config_name)
