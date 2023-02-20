import random
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

class RewardNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256, device='cpu'):
        super(RewardNet, self).__init__()
        self.l1 = nn.Linear(2*state_dim +action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

        self.state_dim = state_dim
        self.device = device

    def forward(self, state, action, next_state):
        r = torch.cat([state, action, next_state], dim=1)
        r = F.relu(self.l1(r))
        r = F.relu(self.l2(r))
        reward = self.l4(r)

        return reward


class Reward_train(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim = 256, batch_size = 256, device='cpu', lr = 0.00001):
        super(Reward_train, self).__init__()
        self.RewardNet = RewardNet(state_dim, action_dim, hidden_dim=hidden_dim, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.RewardNet.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
    def train(self, replay_buffer, iterations):
        loss_num = 0
        for it in range(iterations):
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2],0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)

            fake_reward = self.RewardNet.forward(state, action, next_state)
            loss = F.mse_loss(reward, fake_reward)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_num += loss.item()
        return loss_num /iterations
