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

    i_episode, update_time = 0, 0
    origin_time = time.time()

    while i_episode < args.n_episode:
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


def train(buff, args, model, model_tar, optimizer):
    batch = buff.get_batch(args.batch_size)

    observations = batch[0].cuda()
    next_observations = batch[1].cuda()
    action = batch[2].cuda()
    reward = batch[3].cuda()
    prob = batch[4].cuda()
    next_prob = batch[5].cuda()
    mask = batch[6].cuda()
    terminated = batch[7].cuda()

    q_values = model(observations, prob)
    action_onehot = F.one_hot(action, args.n_actions)
    q_values = (q_values * action_onehot).sum(-1)

    target_q_values = model_tar(next_observations, next_prob).max(dim=-1)[0].detach()
    targets = reward + args.gamma * (1 - terminated) * target_q_values

    delta = (targets - q_values) ** 2

    loss = (delta * mask).sum() / mask.sum()

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
