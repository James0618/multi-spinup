import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


# DEVICE = 'cuda:0'
DEVICE = 'cpu'


def parse_dataset(min_neigh=None):
    path = 'data/full-dataset/processed/'
    if min_neigh is not None:
        matrix = torch.load(path + 'min_matrix.pth')
    else:
        matrix = torch.from_numpy(torch.load(path + 'matrix.pth')).type(torch.long)

    is_alive_buf = torch.load(path + 'is_alive.pth')
    obs_buf = torch.load(path + 'obs.pth')
    pos_buf = torch.load(path + 'pos.pth')
    terminated_buf = torch.load(path + 'terminated.pth')

    terminals = terminated_buf.nonzero().squeeze().tolist()
    matrix_dataset, is_alive_dataset, obs_dataset, pos_dataset = [], [], [], []
    last_terminal = 0

    for terminal in terminals:
        temp_slice = slice(last_terminal, terminal + 1)
        matrix_dataset.append(matrix[temp_slice])
        is_alive_dataset.append(is_alive_buf[temp_slice])
        obs_dataset.append(obs_buf[temp_slice])
        pos_dataset.append(pos_buf[temp_slice])

        last_terminal = terminal + 1

    dataset = dict(matrix=matrix_dataset, is_alive=is_alive_dataset, obs=obs_dataset, pos=pos_dataset,
                   size=len(terminals))

    get_batch(dataset, 0)
    get_neigh(matrix[0], obs_buf[0])

    return dataset


def shuffle_dataset(dataset):
    shuffled_dataset = {'size': dataset['size']}
    size = dataset['size']
    shuffled_index: list = np.arange(size).tolist()
    random.shuffle(shuffled_index)

    for key, value in dataset.items():
        if type(value) is not int:
            temp = []
            for i in shuffled_index:
                temp.append(value[i])

            shuffled_dataset[key] = temp

    return shuffled_dataset


def get_neigh(matrix, data):
    n_agents, data_shape = matrix.shape[-1], data.shape[-1]
    result = []

    n_neigh, ptr = matrix.sum(-1).cpu(), 0
    neighs = matrix.nonzero().transpose(0, 1).tolist()[1]
    for i in range(n_agents):
        index = neighs[ptr: ptr + n_neigh[i]]
        neigh_data = data[index]
        all_data = torch.cat((data[i].unsqueeze(0), neigh_data))

        result.append(all_data)
        ptr += n_neigh[i]

    x = torch.cat(result).to(DEVICE)
    batch = torch.zeros(x.shape[0]).type(torch.long).to(DEVICE)
    edge_index = torch.zeros(2, sum(n_neigh) * 2).type(torch.long).to(DEVICE)

    batch_ptr, edge_ptr = 0, 0
    for i in range(n_agents):
        batch[batch_ptr: batch_ptr + n_neigh[i] + 1] = i

        edge_index[1, edge_ptr: edge_ptr + n_neigh[i]] = torch.arange(n_neigh[i]) + 1
        edge_index[0, edge_ptr + n_neigh[i]: edge_ptr + 2 * n_neigh[i]] = torch.arange(n_neigh[i]) + 1
        edge_index[:, edge_ptr: edge_ptr + 2 * n_neigh[i]] += int(edge_ptr / 2 + i)

        edge_ptr += 2 * n_neigh[i]
        batch_ptr += n_neigh[i] + 1

    return x, batch, edge_index


def get_graph(matrix, data):
    x = data.to(DEVICE)
    edge = matrix.nonzero().transpose(0, 1).to(DEVICE)

    return x, edge


def get_batch(dataset, idx):
    matrix, is_alive = dataset['matrix'][idx], dataset['is_alive'][idx]
    obs, pos = dataset['obs'][idx], dataset['pos'][idx]

    return dict(matrix=matrix, is_alive=is_alive, obs=obs, pos=pos)


def test_matrix(matrix, pos):
    accessible_agents = []
    for i in range(38):
        accessible_agents.append(matrix[i].nonzero()[0].tolist())

    for i in range(38):
        agent_position = pos[i]
        for accessible_agent in accessible_agents[i]:
            accessible_agent_position = pos[accessible_agent]
            plt.plot([agent_position[0], accessible_agent_position[0]],
                     [agent_position[1], accessible_agent_position[1]], 'r')

    state = np.zeros((38, 38))
    for i in range(38):
        state[int(pos[i, 1]), int(pos[i, 0])] = 1

    plt.spy(state)
    plt.savefig('test.png')
