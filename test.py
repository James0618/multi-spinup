def train_vae(graph_dataset, graph_vae_model, optimizer, max_epochs=100):
    graph_mean_difference, graph_max_difference = 0, 0
    for graph_epoch in range(max_epochs):
        graph_dataset = shuffle_dataset(graph_dataset)
        graph_mean_difference, graph_max_difference = 0, 0

        for idx in range(dataset['size']):
            batch = get_batch(dataset=graph_dataset, idx=idx)
            matrix, is_alive, graph_obs, pos = batch['matrix'], batch['is_alive'], batch['obs'], batch['pos']
            graph_ep_len = matrix.shape[0]
            hidden_states = torch.zeros(matrix.shape[1], graph_vae_model.hid_shape)
            recon_states, batch_mus, batch_log_vars = [], [], []
            for i in range(graph_ep_len):
                recon_state, next_hid, mus, log_vars = graph_vae_model(graph_obs[i], matrix[i], hidden_states, pos[i])
                recon_states.append(recon_state.clamp(0, 1))
                batch_mus.append(mus)
                batch_log_vars.append(log_vars)
                hidden_states = next_hid.detach()

            recon_states = torch.stack(recon_states) * is_alive.unsqueeze(-1)
            target_obs = graph_obs * is_alive.unsqueeze(-1)
            batch_mus = torch.stack(batch_mus) * is_alive.unsqueeze(-1)
            batch_log_vars = torch.stack(batch_log_vars) * is_alive.unsqueeze(-1)

            loss = loss_function(recon_states, target_obs, batch_mus, batch_log_vars)
            optimizer.zero_grad()
            loss.backward()
            # mpi_avg_grads(vae_model)
            optimizer.step()

            graph_mean_difference += float(torch.abs(target_obs - recon_states.cpu().detach()).mean())
            graph_max_difference += float(torch.abs(target_obs - recon_states.cpu().detach()).max())

        graph_mean_difference /= graph_dataset['size']
        graph_max_difference /= graph_dataset['size']

    return graph_mean_difference, graph_max_difference