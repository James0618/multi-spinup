import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.dgn_model import ReplayBuffer, DGN, Policy
from envs.gather import GatherEnv
from spinup.utils.logx import EpochLogger


def is_terminated(terminated):
    groups = ['omnivore']
    results = []
    for group in groups:
        results.append(np.array([terminated[key] for key in terminated.keys() if group in key]).prod())

    return bool(np.array(results).sum())


def test(model, env, args, test_num=10):
    agents = env.possible_agents

    if args.vae_model:
        if args.with_state:
            obs_shape = args.vae_observation_dim + args.latent_state_shape
        else:
            obs_shape = args.vae_observation_dim
    else:
        obs_shape = args.observation_shape

    buff = ReplayBuffer(args.capacity, agents=agents, args=args)
    policy = Policy(args=args, obs_shape=obs_shape, model=model, agents=agents)

    i_episode, update_time, epsilon = 0, 0, 0.9
    test_ret, test_steps = 0, 0

    while i_episode < test_num:
        i_episode += 1
        steps, ep_ret = 0, 0
        obs = env.reset()
        adj = env.graph_builder.adjacency_matrix
        while True:
            steps += 1
            actions = policy.choose_action(observations=obs, adj_matrix=adj, epsilon=epsilon)

            next_obs, rewards, done, next_positions = env.step(actions)
            next_adj = env.graph_builder.adjacency_matrix
            terminal = is_terminated(terminated=done)

            buff.add(obs, actions, rewards, next_obs, adj, next_adj, done)
            obs = next_obs
            adj = next_adj

            if args.global_reward:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()]) / len(rewards)
            else:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()])

            if terminal:
                break

        test_ret += ep_ret
        test_steps += steps

    return test_ret / test_num, test_steps / test_num


def dgn(env_fn, args, logger_kwargs=None):
    if logger_kwargs is None:
        logger_kwargs = {}

    logger = EpochLogger(**logger_kwargs)
    temp = locals()
    inputs = {key: temp[key] for key in temp if type(temp[key]) is float or type(temp[key]) is int}
    logger.save_config(inputs)

    env: GatherEnv = env_fn()
    agents = env.possible_agents
    n_agent = len(agents)

    if args.vae_model:
        if args.with_state:
            obs_shape = args.vae_observation_dim + args.latent_state_shape
        else:
            obs_shape = args.vae_observation_dim
    else:
        obs_shape = args.observation_shape

    n_actions = args.n_actions

    buff = ReplayBuffer(args.capacity, agents=agents, args=args)

    model = DGN(args, n_agent).cuda()
    model_tar = DGN(args, n_agent).cuda()
    policy = Policy(args=args, obs_shape=obs_shape, model=model, agents=agents)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    logger.setup_pytorch_saver(model)

    i_episode, update_time, epsilon = 0, 0, 0.9
    origin_time = time.time()

    while i_episode < args.n_episode:
        if i_episode > args.random_episode:
            epsilon -= 0.0004
            if epsilon < 0.05:
                epsilon = 0.05

        i_episode += 1

        steps, ep_ret = 0, 0
        obs = env.reset()
        adj = env.graph_builder.adjacency_matrix
        while True:
            steps += 1
            actions = policy.choose_action(observations=obs, adj_matrix=adj, epsilon=epsilon)

            next_obs, rewards, done, next_positions = env.step(actions)
            next_adj = env.graph_builder.adjacency_matrix
            terminal = is_terminated(terminated=done)

            buff.add(obs, actions, rewards, next_obs, adj, next_adj, done)
            obs = next_obs
            adj = next_adj

            if args.global_reward:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()]) / len(rewards)
            else:
                ep_ret += sum([rewards[agent] for agent in rewards.keys()])

            if terminal:
                logger.store(EpRet=ep_ret, EpLen=steps)
                break

        if i_episode >= args.random_episode:
            for e in range(args.n_epoch):
                train(buff, args, model, model_tar, n_agent, optimizer)
                model_tar.update_param(model)
                # update_time += 1
                # if update_time == args.update_interval:
                #     update_time = 0
                #     model_tar.load_state_dict(model.state_dict())

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


def train(buff, args, model, model_tar, n_agent, optimizer):
    batch = buff.get_batch(args.batch_size)
    observations = batch[0]
    action = batch[1].cuda()
    reward = batch[2].cuda()
    next_observations = batch[3]
    matrix = batch[4]
    next_matrix = batch[5]
    mask = batch[6].cuda()
    terminated = batch[7].cuda()

    q_values = model(observations.cuda(), matrix.cuda())
    action_onehot = F.one_hot(action, args.n_actions)
    q_values = (q_values * action_onehot).sum(-1)

    target_q_values = model_tar(next_observations.cuda(), next_matrix.cuda()).max(dim=2)[0].detach()
    targets = reward + args.gamma * (1 - terminated) * target_q_values

    delta = ((targets - q_values) ** 2) * mask
    loss = delta.sum() / mask.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss)


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
        second = t % 60
        return '{} h {} min {} sec'.format(hour, minute, second)
