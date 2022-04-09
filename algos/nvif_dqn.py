import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.mfq_model import ReplayBuffer, MFQ, Policy
from envs.gather import GatherEnv
from spinup.utils.logx import EpochLogger


def is_terminated(terminated):
    groups = ['omnivore']
    results = []
    for group in groups:
        results.append(np.array([terminated[key] for key in terminated.keys() if group in key]).prod())

    return bool(np.array(results).sum())


def get_prob(actions, n_actions):
    prob = np.zeros(n_actions)
    n_agents = len(actions.keys())
    for key in actions.keys():
        prob += np.eye(n_actions)[actions[key]]

    return prob / n_agents


def test(model, env, args, test_num=10):
    agents = env.possible_agents
    policy = Policy(args=args, model=model, agents=agents)

    i_episode = 0
    test_ret = 0
    test_steps = 0

    while i_episode < test_num:
        i_episode += 1

        steps, ep_ret = 0, 0
        obs = env.reset()
        prob = np.zeros(args.n_actions)

        while True:
            steps += 1
            actions = policy.choose_action(observations=obs, mean_actions=prob, epsilon=0.0)
            next_prob = get_prob(actions=actions, n_actions=args.n_actions)

            next_obs, rewards, done, next_positions = env.step(actions)
            terminal = is_terminated(terminated=done)

            obs = next_obs
            prob = next_prob

            if args.global_reward:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()]) / len(rewards)
            else:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()])

            if terminal:
                break

        test_ret += ep_ret
        test_steps += steps

    return test_ret / test_num, test_steps / test_num


def nvif_dqn(env_fn, args, logger_kwargs=None):
    if logger_kwargs is None:
        logger_kwargs = {}

    logger = EpochLogger(**logger_kwargs)
    temp = locals()
    inputs = {key: temp[key] for key in temp if type(temp[key]) is float or type(temp[key]) is int}
    logger.save_config(inputs)

    env: GatherEnv = env_fn()
    agents = env.possible_agents
    n_agent = len(agents)

    n_actions = args.n_actions

    buff = ReplayBuffer(args.capacity, agents=agents, args=args)

    model = MFQ(args=args).cuda()
    model_tar = MFQ(args=args).cuda()
    policy = Policy(args=args, model=model, agents=agents)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    logger.setup_pytorch_saver(model)

    i_episode, epsilon = 0, 0.9
    origin_time = time.time()

    while i_episode < args.n_episode:
        i_episode += 1

        steps, ep_ret = 0, 0
        obs = env.reset()
        prob = np.zeros(args.n_actions)

        while True:
            if i_episode > args.random_episode:
                epsilon -= 0.0004
                if epsilon < 0.05:
                    epsilon = 0.05

            steps += 1
            actions = policy.choose_action(observations=obs, mean_actions=prob, epsilon=epsilon)
            next_prob = get_prob(actions=actions, n_actions=args.n_actions)

            next_obs, rewards, done, next_positions = env.step(actions)
            terminal = is_terminated(terminated=done)

            buff.add(obs, actions, prob, rewards, next_obs, next_prob, done)
            obs = next_obs
            prob = next_prob

            if args.global_reward:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()]) / len(rewards)
            else:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()])

            if terminal:
                logger.store(EpRet=ep_ret, EpLen=steps)
                break

        if i_episode >= args.random_episode:
            for e in range(args.n_epoch):
                train(buff, args, model, model_tar, optimizer)
                model_tar.update_param(model)

        if i_episode % args.save_interval == 0:
            avg_ret, avg_steps = test(model, env, args, test_num=10)
            logger.log_tabular('Episode', i_episode)
            logger.log_tabular('TestRet', avg_ret)
            logger.log_tabular('TestSteps', avg_steps)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('Time', cal_time(int(time.time() - origin_time)))
            logger.dump_tabular()

            logger.save_state({'env': env}, None)


def train(buff, args, model, model_tar, optimizer):
    batch = buff.get_batch(args.batch_size)

    observations = batch['obs'].cuda()
    next_observations = batch['next_obs'].cuda()
    action = batch['action'].cuda()
    reward = batch['reward'].cuda()
    prob = batch['prob'].cuda()
    next_prob = batch['next_prob'].cuda()
    terminated = batch['terminated'].cuda()

    q_values = model(observations, prob)
    action_onehot = F.one_hot(action, args.n_actions)
    q_values = (q_values * action_onehot).sum(-1)

    target_q_values = model_tar(next_observations, next_prob).max(dim=-1)[0].detach()
    targets = reward + args.gamma * (1 - terminated) * target_q_values

    delta = (targets - q_values) ** 2

    loss = delta.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu())


def cal_time(t):
    assert type(t) == int
    if t < 60:
        second = t
        return '{} sec'.format(second)

    elif t < 3600:
        minute = t // 60
        second = t % 60
        return '{} min {} sec'.format(minute, second)

    else:
        hour = t // 3600
        minute = (t - hour * 3600) // 60
        return '{} h {} min'.format(hour, minute)
