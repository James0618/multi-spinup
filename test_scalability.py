import numpy as np
from utils.test_gather_policy import test_policy as gather_test_policy
from utils.mfq_model import test_policy as mfq_test_policy
from utils.dgn_model import test_policy as dgn_test_policy


def test_method(method, model_size, model_scenario, test_size, test_scenario):
    config = 'scalability/{}/{}-{}'.format(test_scenario, method, test_size)
    experiment = 'scalability/{}/{}/{}'.format(model_scenario, model_size, method)

    if method == 'ippo' or method == 'dsin':
        result = gather_test_policy(experiment=experiment, config_name=config, render=False)
    elif method == 'mfq':
        result = mfq_test_policy(experiment=experiment, config_name=config, render=False)
    elif method == 'dgn':
        result = dgn_test_policy(experiment=experiment, config_name=config, render=False)
    else:
        raise ValueError('Wrong test method!')

    return result


def test_scalability():
    scenarios = ['random', 'normal']
    sizes = [24, 48, 96]
    methods = ['mfq']

    results = {
        'dsin': np.zeros((6, 6)),
        'mfq': np.zeros((6, 6))
    }

    for method in methods:
        for model_i, model_scenario in enumerate(scenarios):
            for model_j, model_size in enumerate(sizes):
                for test_i, test_scenario in enumerate(scenarios):
                    for test_j, test_size in enumerate(sizes):
                        x = model_i * 3 + model_j
                        y = test_i * 3 + test_j
                        result = test_method(method=method, model_size=model_size, model_scenario=model_scenario,
                                             test_size=test_size, test_scenario=test_scenario)
                        results[method][x, y] = result

    return results


if __name__ == '__main__':
    scalability = test_scalability()
    for key in scalability.keys():
        print(key + ': ', scalability[key])
        np.save('results/scalability/origin-{}.npy'.format(key), scalability[key])
