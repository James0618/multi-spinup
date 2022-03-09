import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def process_dataset(data):
    terminated_buf = data['terminated']
    last_terminal = terminated_buf.nonzero().max()

    matrix_buf = data['matrix'][: (last_terminal + 1)]
    is_alive_buf = data['is_alive'][: (last_terminal + 1)]
    obs_buf = data['obs'][: (last_terminal + 1)]
    pos_buf = data['pos'][: (last_terminal + 1)]

    n_agents = matrix_buf.shape[-1]

    temp = torch.arange(n_agents).tolist()
    temp_matrix = copy.deepcopy(matrix_buf)
    all_min_matrix = np.zeros((matrix_buf.shape[0], n_agents, n_agents))

    for i in range(matrix_buf.shape[0]):
        for j in range(4):
            min_index = temp_matrix[i, j].argmin(-1).tolist()
            invalid_temp = (temp_matrix[i, j, temp, min_index] > 100).nonzero()[0].tolist()
            invalid_index = np.array(min_index)[invalid_temp].tolist()
            all_min_matrix[i, temp, min_index] = 1
            all_min_matrix[i, min_index, temp] = 1
            all_min_matrix[i, invalid_temp, invalid_index] = 0
            all_min_matrix[i, invalid_index, invalid_temp] = 0

    data = dict(matrix=all_min_matrix, obs=obs_buf, pos=pos_buf,
                is_alive=is_alive_buf, terminated=terminated_buf)

    return {k: torch.as_tensor(v, dtype=torch.float) for k, v in data.items()}


def parse_dataset(data):
    data = process_dataset(data)

    matrix = data['matrix'].type(torch.long)
    is_alive_buf = data['is_alive']
    obs_buf = data['obs']
    pos_buf = data['pos']
    terminated_buf = data['terminated']

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

    n_neigh, ptr = matrix.sum(-1), 0
    neighs = matrix.nonzero().transpose(0, 1).tolist()[1]
    for i in range(n_agents):
        index = neighs[ptr: ptr + n_neigh[i]]
        neigh_data = data[index]
        all_data = torch.cat((data[i].unsqueeze(0), neigh_data))

        result.append(all_data)
        ptr += n_neigh[i]

    x = torch.cat(result)
    batch = torch.zeros(x.shape[0]).type(torch.long)
    edge_index = torch.zeros(2, sum(n_neigh) * 2).type(torch.long)

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
    x = data
    edge = matrix.nonzero().transpose(0, 1)

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
