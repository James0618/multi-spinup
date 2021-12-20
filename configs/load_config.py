import os
import torch
import numpy as np
import yaml
import json
from types import SimpleNamespace


def load_default_config(name):
    with open('configs/default.yaml') as f:
        default_config = yaml.safe_load(f)

    with open('configs/{}.yaml'.format(name)) as f:
        specialized_config = yaml.safe_load(f)

    config = default_config
    for key, value in specialized_config.items():
        config[key] = value

    args = SimpleNamespace(**config)

    return args


def load_config(name, env, output_path='/home/drl/PycharmProjects/multi-spinup/results/', exp_name='ppo'):
    with open('configs/default.yaml') as f:
        default_config = yaml.safe_load(f)

    with open('configs/{}.yaml'.format(name)) as f:
        specialized_config = yaml.safe_load(f)

    config = default_config
    for key, value in specialized_config.items():
        config[key] = value

    observation_shape = env.observation_space.shape
    state_shape = env.state_space.shape
    temp = torch.zeros(3).type(torch.int)
    temp[0], temp[1], temp[2] = int(state_shape[2]) - 2, int(state_shape[0]), int(state_shape[1])
    state_shape = torch.Size(temp)
    n_actions = int(env.action_space.n)

    temp = torch.zeros(3).type(torch.int)
    temp[0], temp[1], temp[2] = int(observation_shape[2] - 2), int(observation_shape[0]), int(observation_shape[1])
    image_shape = torch.Size(temp)
    embed_shape = torch.Size([2 + n_actions])
    input_shape = [image_shape, embed_shape]

    config['input_shape'] = input_shape
    config['observation_shape'] = image_shape
    config['state_shape'] = state_shape
    config['n_actions'] = n_actions

    if not os.path.exists(output_path + exp_name):
        os.makedirs(output_path + exp_name)

    with open(output_path + exp_name + '/full-config.json', 'w') as f:
        saved_config = {}
        for key in config.keys():
            if (type(config[key]) is np.int) or (type(config[key]) is torch.int):
                saved_config[key] = int(config[key])
            elif type(config[key]) is torch.Size:
                saved_config[key] = str(config[key])
            elif type(config[key]) is list:
                saved_config[key] = [str(element) for element in config[key]]
            else:
                saved_config[key] = config[key]
        f.write(json.dumps(saved_config, indent=4))

    args = SimpleNamespace(**config)

    return args
