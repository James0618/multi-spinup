import torch
import torch.utils.data
import torch.nn.functional as functional
from utils.parse_dataset import shuffle_dataset, get_batch
from spinup.utils.mpi_pytorch import mpi_avg_grads, mpi_avg_grads_cuda
from spinup.utils.mpi_tools import mpi_min, mpi_avg, proc_id


def loss_function(reconstructed_inputs, original_inputs, mus, log_vars):
    bce_loss = functional.binary_cross_entropy(reconstructed_inputs, original_inputs, reduction='sum')

    kl_loss = - 0.5 * torch.mean(1 + log_vars - mus.pow(2) - log_vars.exp())
    loss = bce_loss + kl_loss

    mean_recons = reconstructed_inputs.mean(dim=1, keepdims=True)
    cons_dif = (reconstructed_inputs - mean_recons.detach()) ** 2
    cons_loss = cons_dif.mean()

    loss += cons_loss

    return loss


def test_model(dataset, vae_model):
    mean_difference = 0

    for idx in range(int(mpi_min(dataset['size']))):
        batch = get_batch(dataset=dataset, idx=idx)
        matrix, is_alive, obs, pos = batch['matrix'], batch['is_alive'], batch['obs'], batch['pos']
        ep_len = matrix.shape[0]
        hidden_states = torch.zeros(matrix.shape[1], vae_model.hid_shape)
        recon_states, batch_mus, batch_log_vars = [], [], []
        for i in range(ep_len):
            recon_state, next_hid, mus, log_vars = vae_model(obs[i], matrix[i], hidden_states, pos[i])
            recon_states.append(recon_state.clamp(0, 1))
            batch_mus.append(mus)
            batch_log_vars.append(log_vars)
            hidden_states = next_hid.detach()

        recon_states = torch.stack(recon_states) * is_alive.unsqueeze(-1).to(vae_model.device)
        target_obs = obs * is_alive.unsqueeze(-1)

        mean_difference += float(torch.abs(target_obs - recon_states.cpu().detach()).mean())

    mean_difference /= dataset['size']
    mean_difference = mpi_avg(mean_difference)

    if proc_id() == 0:
        print('Evaluate Difference: {:.4f}'.format(mean_difference))

    return mean_difference


def train_vae(dataset, vae_model, optimizer, threshold=0.07):
    evaluate_difference = test_model(dataset=dataset, vae_model=vae_model)
    if evaluate_difference < threshold:
        return 0

    mean_difference = 1
    train_iter = 0
    while mean_difference > threshold:
        dataset = shuffle_dataset(dataset)
        mean_difference = 0

        for idx in range(int(mpi_min(dataset['size']))):
            batch = get_batch(dataset=dataset, idx=idx)
            matrix, is_alive, obs, pos = batch['matrix'], batch['is_alive'], batch['obs'], batch['pos']
            ep_len = matrix.shape[0]
            hidden_states = torch.zeros(matrix.shape[1], vae_model.hid_shape)
            recon_states, batch_mus, batch_log_vars = [], [], []
            for i in range(ep_len):
                recon_state, next_hid, mus, log_vars = vae_model(obs[i], matrix[i], hidden_states, pos[i])
                recon_states.append(recon_state.clamp(0, 1))
                batch_mus.append(mus)
                batch_log_vars.append(log_vars)
                hidden_states = next_hid.detach()

            recon_states = torch.stack(recon_states) * is_alive.unsqueeze(-1).to(vae_model.device)
            target_obs = obs * is_alive.unsqueeze(-1)
            batch_mus = torch.stack(batch_mus) * is_alive.unsqueeze(-1).to(vae_model.device)
            batch_log_vars = torch.stack(batch_log_vars) * is_alive.unsqueeze(-1).to(vae_model.device)

            loss = loss_function(recon_states, target_obs.to(vae_model.device), batch_mus, batch_log_vars)
            optimizer.zero_grad()
            loss.backward()
            # mpi_avg_grads(vae_model)
            mpi_avg_grads_cuda(vae_model)
            optimizer.step()

            mean_difference += float(torch.abs(target_obs - recon_states.cpu().detach()).mean())

        mean_difference /= dataset['size']
        mean_difference = mpi_avg(mean_difference)
        if proc_id() == 0:
            print('Train Difference: {:.4f}'.format(mean_difference))

        train_iter += 1

    return train_iter
