import time
import numpy as np
import torch
import torch.nn.functional
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


def mfq(env_fn, args, logger_kwargs=None):
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

    model = MFQ(args=args).cuda()
    model_tar = MFQ(args=args).cuda()
    policy = Policy(args=args, model=model, agents=agents)

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
        prob = np.zeros(args.n_actions)

        while True:
            steps += 1
            actions = policy.choose_action(observations=obs, mean_actions=prob)
            next_prob = get_prob(actions=actions, n_actions=args.n_actions)

            next_obs, rewards, done, next_positions = env.step(actions)
            terminal = is_terminated(terminated=done)

            buff.add(obs, actions, prob, reward, new_obs, next_prob)
            obs = next_obs

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
                update_time += 1
                if update_time == args.update_interval:
                    update_time = 0
                    model_tar.load_state_dict(model.state_dict())

        if i_episode % args.save_interval == 0:
            logger.log_tabular('Episode', i_episode)
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

    q_values = model(observations.cuda(), matrix.cuda())
    target_q_values = model_tar(next_observations.cuda(), next_matrix.cuda()).max(dim=2)[0].detach()
    expected_q = q_values.detach().clone()

    mesh_grid = torch.meshgrid(torch.arange(args.batch_size), torch.arange(n_agent))
    x, y, z = mesh_grid[0].reshape(-1).tolist(), mesh_grid[1].reshape(-1).tolist(), action.reshape(-1).tolist()
    expected_q[x, y, z] = reward[x, y] + (1 - mask[x, y]) * args.gamma * target_q_values[x, y]

    loss = (q_values[x, y, z] - expected_q[x, y, z]).pow(2).mean()

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
        return '{} h {} min'.format(hour, minute)