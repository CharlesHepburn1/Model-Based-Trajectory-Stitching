import random
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim = 750, device='cpu'):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(2*state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim) #latent instead of action
        self.log_std = nn.Linear(hidden_dim, latent_dim) #latent instead of action

        self.d1 = nn.Linear(2*state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.device = device

    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], dim=1)
        z = F.relu(self.e1(x))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, next_state,  z)

        return u, mean, std

    def decode(self, state, next_state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        obs = torch.cat([state,next_state, z], dim=1)
        a = F.relu(self.d1(obs))
        a = F.relu(self.d2(a))
        return torch.tensor(self.max_action) * torch.tanh(self.d3(a))



class VAE_pretrain(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action,hidden_dim = 750, batch_size = 100, device='cpu', lr = 0.0001):
        super(VAE_pretrain, self).__init__()
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=hidden_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
    def train(self, replay_buffer, iterations):
        vae_tot = 0
        for it in range(iterations):
            #sample state, action from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)

            a_tilde, mean, std = self.vae.forward(state, action, next_state)
            vae_loss1 = F.mse_loss(action, a_tilde)
            vae_loss2 = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = 0.5 * (vae_loss1 + vae_loss2)
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            vae_tot += vae_loss.item()
        return vae_tot /iterations
