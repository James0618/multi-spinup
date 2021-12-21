import numpy as np
import torch
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


def dgn(env_fn, args, gamma=0.99, logger_kwargs=None):
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
            obs_shape = torch.Size([args.vae_observation_dim + args.latent_state_shape])
        else:
            obs_shape = torch.Size([args.vae_observation_dim])
    else:
        obs_shape = args.observation_shape

    n_actions = args.n_actions

    buff = ReplayBuffer(args.capacity)

    model = DGN(n_agent, obs_shape, args.hidden_dim, n_actions).cuda()
    model_tar = DGN(n_agent, obs_shape, args.hidden_dim, n_actions).cuda()
    policy = Policy(args=args, model=model, agents=agents)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    logger.setup_pytorch_saver(model)

    obs = np.ones((args.batch_size, n_agent, obs_shape))
    next_obs = np.ones((args.batch_size, n_agent, obs_shape))
    matrix = np.ones((args.batch_size, n_agent, n_agent))
    next_matrix = np.ones((args.batch_size, n_agent, n_agent))

    i_episode, score, epsilon = 0, 0, 0.9

    while i_episode < args.n_episode:
        if i_episode > 100:
            epsilon -= 0.0004
            if epsilon < 0.1:
                epsilon = 0.1
        i_episode += 1
        steps = 0
        obs = env.reset()
        adj = env.graph_builder.adjacency_matrix

        while steps < args.max_step:
            steps += 1
            actions = policy.choose_action(observations=obs, adj_matrix=adj, epsilon=epsilon)

            next_obs, rewards, done, next_positions = env.step(actions)
            next_adj = env.graph_builder.adjacency_matrix
            terminal = is_terminated(terminated=done)

            buff.add(np.array(obs), actions, rewards, np.array(next_obs), adj, next_adj, terminal)
            obs = next_obs
            adj = next_adj
            score += sum(rewards)

        if i_episode < 100:
            continue

        for e in range(args.n_epoch):

            batch = buff.get_batch(args.batch_size)
            for j in range(args.batch_size):
                sample = batch[j]
                obs[j] = sample[0]
                next_obs[j] = sample[3]
                matrix[j] = sample[4]
                next_matrix[j] = sample[5]

            q_values = model(torch.Tensor(obs).cuda(), torch.Tensor(matrix).cuda())
            target_q_values = model_tar(torch.Tensor(next_obs).cuda(), torch.Tensor(next_matrix).cuda()).max(dim=2)[0]
            target_q_values = np.array(target_q_values.cpu().data)
            expected_q = np.array(q_values.cpu().data)

            for j in range(args.batch_size):
                sample = batch[j]
                for i in range(n_agent):
                    expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * gamma * target_q_values[j][i]

            loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i_episode % 5 == 0:
            model_tar.load_state_dict(model.state_dict())
