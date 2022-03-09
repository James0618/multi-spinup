import tqdm
import torch
import torch.utils.data
import torch.nn.functional as functional
from utils.parse_dataset import shuffle_dataset, get_batch


def loss_function(reconstructed_inputs, original_inputs, mus, log_vars):
    bce_loss = functional.binary_cross_entropy(reconstructed_inputs, original_inputs, reduction='sum')

    kl_loss = - 0.5 * torch.mean(1 + log_vars - mus.pow(2) - log_vars.exp())
    loss = bce_loss + kl_loss

    mean_recons = reconstructed_inputs.mean(dim=1, keepdims=True)
    cons_dif = (reconstructed_inputs - mean_recons.detach()) ** 2
    cons_loss = cons_dif.mean()

    loss += cons_loss

    return loss


def train_vae(dataset, vae_model, optimizer, max_epochs=100):
    for epoch in range(max_epochs):
        dataset = shuffle_dataset(dataset)
        mean_difference, max_difference = 0, 0
        print('Epoch {}/{}: '.format(epoch + 1, max_epochs))

        for idx in tqdm.tqdm(range(dataset['size'])):
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

            recon_states = torch.stack(recon_states) * is_alive.unsqueeze(-1)
            target_obs = obs * is_alive.unsqueeze(-1)
            batch_mus = torch.stack(batch_mus) * is_alive.unsqueeze(-1)
            batch_log_vars = torch.stack(batch_log_vars) * is_alive.unsqueeze(-1)

            loss = loss_function(recon_states, target_obs, batch_mus, batch_log_vars)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_difference += float(torch.abs(target_obs - recon_states.cpu().detach()).mean())
            max_difference += float(torch.abs(target_obs - recon_states.cpu().detach()).max())

        mean_difference /= dataset['size']
        max_difference /= dataset['size']

        print('\r    Epoch {}/{} Mean difference: {:.3f}, Max difference: {:.3f}'.format(
            epoch + 1, max_epochs, mean_difference, max_difference))


