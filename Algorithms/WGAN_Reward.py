import random
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

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
