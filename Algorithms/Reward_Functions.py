import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

########################################################################################################################
############################                     WGAN                                  #################################
########################################################################################################################

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim = 256):
        super(Generator, self).__init__()
        self.input_dim = state_dim + action_dim + state_dim + latent_dim
        self.l1 = nn.Linear(self.input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, next_state, z):
        r = F.relu(self.l1(torch.cat([z, state, action, next_state], dim=-1)))
        r = F.relu(self.l2(r))
        rew = self.l3(r)

        return rew


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = state_dim +state_dim +action_dim + 1
        self.l1 = nn.Linear(self.input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
    def forward(self, state, action, next_state, reward):
        p = F.relu(self.l1(torch.cat([state, next_state, action, reward], dim=-1)))
        p = F.relu(self.l2(p))
        p = self.l3(p)
        return p

class GAN(object):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=512, l2_reg_D = 0.0,
                 batch_size=256, lr=1e-4, device="cpu"):
        # WGAN values from paper
        self.b1 = 0.5
        self.b2 = 0.999

        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.b1, self.b2))

        self.generator = Generator(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(self.b1, self.b2))


        self.latent_dim = latent_dim
        self.device = device
        self.batch_size = batch_size

        self.discriminator_loss_history = []
        self.generator_loss_history = []

        self.total_it = 0
        self.critic_iter = 5
        self.l2_reg_D = l2_reg_D

    def train(self, replay_buffer, iterations=1):
        indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
        state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
        action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
        reward = torch.index_select(replay_buffer[2], 0, indices).reshape(self.batch_size, 1).to(self.device)
        next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)

        for g_iter in range(iterations):
            # Requires grad, Generator requires_grad = False
            # for p in self.discriminator.parameters():
            #     p.requires_grad = True;
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):

                # Train discriminator
                self.discriminator_optimizer.zero_grad()
                # WGAN - Training discriminator more iterations than generator
                # Train with real next_states and actions
                d_loss_real = self.discriminator(state, action,next_state, reward)
                # Train with fake actions
                z = torch.randn(self.batch_size, self.latent_dim, dtype = torch.float32, device = self.device)

                fake_reward = self.generator(state, action, next_state, z)
                fake_reward = fake_reward.reshape(self.batch_size,1)
                d_loss_fake = self.discriminator(state, action,next_state, fake_reward)

                d_loss = d_loss_real.mean() - d_loss_fake.mean()
                #Grad Penalty
                loss_D_reg = 0
                for par in self.discriminator.parameters():
                    loss_D_reg += torch.dot(par.view(-1), par.view(-1))
                loss_D_reg = self.l2_reg_D * loss_D_reg
                D_loss_tot = d_loss+loss_D_reg

                self.discriminator_loss_history.append(D_loss_tot.item())

                D_loss_tot.backward()
                self.discriminator_optimizer.step()

            # Generator update

            # train generator
            self.generator_optimizer.zero_grad()
            # compute loss with fake images
            z2 = torch.randn(self.batch_size, self.latent_dim, dtype = torch.float32, device = self.device)
            fake_reward = self.generator(state, action, next_state, z2)
            d_fake = self.discriminator(state, action, next_state, fake_reward)
            g_loss = d_fake.mean()
            g_loss.backward()

            self.generator_loss_history.append(g_loss.item())
            self.generator_optimizer.step()


########################################################################################################################

########################################################################################################################
############################                     VAE                                   #################################
########################################################################################################################


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim = 750, device='cpu'):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(2*state_dim + action_dim +1, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim) #latent instead of action
        self.log_std = nn.Linear(hidden_dim, latent_dim) #latent instead of action

        self.d1 = nn.Linear(2*state_dim + +action_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, 1)

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.device = device

    def forward(self, state, action, reward, next_state):
        x = torch.cat([state, action, reward, next_state], dim=1)
        z = F.relu(self.e1(x))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state,action, next_state,  z)

        return u, mean, std

    def decode(self, state, action, next_state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        obs = torch.cat([state,action,next_state, z], dim=1)
        r = F.relu(self.d1(obs))
        r = F.relu(self.d2(r))
        return self.d3(r)



class VAE_pretrain(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim,hidden_dim = 750, batch_size = 100, device='cpu', lr = 0.0001):
        super(VAE_pretrain, self).__init__()
        self.vae = VAE(state_dim, action_dim, latent_dim, hidden_dim=hidden_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
        self.vae_loss_history = []
    def train(self, replay_buffer, iterations):
        vae_tot = 0
        for it in range(iterations):
            #sample state, action from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)

            r_tilde, mean, std = self.vae.forward(state, action, reward.reshape(self.batch_size,1), next_state)
            vae_loss1 = F.mse_loss(reward, r_tilde.reshape(self.batch_size,))
            vae_loss2 = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = 0.5 * (vae_loss1 + vae_loss2)
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            vae_tot += vae_loss.item()
            self.vae_loss_history.append(vae_loss.item())
        return vae_tot /iterations

########################################################################################################################

########################################################################################################################
############################                     MLP                                   #################################
########################################################################################################################

#MSE Loss?
class Reward_MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Reward_MLP, self).__init__()
        self.input_dim = state_dim +action_dim + state_dim
        self.l1 = nn.Linear(self.input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
    def forward(self, state, action, next_state):
        p = F.relu(self.l1(torch.cat([state, action, next_state], dim=-1)))
        p = F.relu(self.l2(p))
        p = self.l3(p)
        return p

class MLP_train(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim = 750, batch_size = 100, device='cpu', lr = 0.0001):
        super(MLP_train, self).__init__()
        self.Rew = Reward_MLP(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.rew_optimizer = torch.optim.Adam(self.Rew.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
        self.mlp_loss_history =[]
    def train(self, replay_buffer, iterations):
        rew_tot = 0
        for it in range(iterations):
            #sample state, action from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)

            r_tilde = self.Rew.forward(state, action,  next_state).reshape(self.batch_size,)
            rew_loss = F.mse_loss(reward, r_tilde)
            self.rew_optimizer.zero_grad()
            rew_loss.backward()
            self.rew_optimizer.step()
            rew_tot += rew_loss.item()
            self.mlp_loss_history.append(rew_loss.item())
        return rew_tot /iterations

########################################################################################################################

########################################################################################################################
############################                     Gaussian                              #################################
########################################################################################################################

class Gaussian_Model(nn.Module):
    def __init__(self, state_dim,action_dim, hidden_dim=200):
        super(Gaussian_Model, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim+state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        # self.l3 = nn.Linear(hidden_dim, hidden_dim)
        # self.l4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, next_state):
        r = F.relu(self.l1(torch.cat([state, action, next_state], dim=-1)))
        r = F.relu(self.l2(r))
        # r = F.relu(self.l3(r))
        # r = F.relu(self.l4(r))

        mu = self.mean(r)
        log_std = self.log_std(r).clamp(min=-20, max=2.0)
        std = log_std.exp()

        return mu, std

    def distribution(self, state,  action, next_state):
        mu, std = self.forward(state, action, next_state)

        distribution = Independent(Normal(mu, std), 1)

        return distribution

    def samples(self, state, action, next_state):
        dist = self.distribution(state, action, next_state)

        reward = dist.sample()

        return reward


class Gaussian_train(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim = 750, batch_size = 100, device='cpu', lr = 0.0001):
        super(Gaussian_train, self).__init__()
        self.Rew = Gaussian_Model(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.rew_optimizer = torch.optim.Adam(self.Rew.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
        self.gauss_loss_history = []
    def train(self, replay_buffer, iterations):
        rew_tot = 0
        for it in range(iterations):
            #sample state, action from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)

            dist = self.Rew.distribution(state, action, next_state)
            rew_loss = -dist.log_prob(reward).mean()

            self.rew_optimizer.zero_grad()
            rew_loss.backward()
            self.rew_optimizer.step()
            rew_tot += rew_loss.item()
            self.gauss_loss_history.append(rew_loss.item())
        return rew_tot /iterations

#
# def train_model(model, model_optimizer, replay_buffer_train, replay_buffer_val, device):
#     # Training #
#     training_loss = 0
#     for i, batch in enumerate(replay_buffer_train):
#         state_batch = batch[0].float().to(device)
#         action_batch = batch[1].float().to(device)
#         reward_batch = batch[2].float().to(device)
#         next_state_batch = batch[3].float().to(device)
#
#         dist = model.distribution(state_batch, action_batch, next_state_batch)
#         loss = -dist.log_prob(reward_batch).mean()
#
#         model_optimizer.zero_grad()
#         loss.backward()
#         model_optimizer.step()
#
#         training_loss += loss.item()
#
#     # Validation #
#     validation_loss = 0
#     with torch.no_grad():
#         for i, batch in enumerate(replay_buffer_val):
#             state_batch = batch[0].float().to(device)
#             action_batch = batch[1].float().to(device)
#             reward_batch = batch[2].float().to(device)
#             next_state_batch = batch[3].float().to(device)
#
#             dist = model.distribution(state_batch, action_batch, next_state_batch)
#             loss = -dist.log_prob(reward_batch).mean()
#
#             validation_loss += loss.item()
#
#     training_loss /= len(replay_buffer_train)
#     validation_loss /= len(replay_buffer_val)
#
#     return training_loss, validation_loss
