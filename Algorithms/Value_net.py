# Imports
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Value network
class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(Value, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.l4 = nn.Linear(state_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(state))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1)

class Agent(object):
    def __init__(self, state_dim,hidden_dim = 256, batch_size=256, gamma=0.99, tau=0.005, lr=3e-4, device="cpu"):

        self.value = Value(state_dim,hidden_dim).to(device)
        self.value_target = copy.deepcopy(self.value)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        self.device = device
        self.batch_size = batch_size

        self.value_loss_history = []
        self.lmbda_history = []

        self.total_it = 0


    def train(self, replay_buffer, iterations=1):

        for it in range(iterations):
            self.total_it += 1
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,)).to(self.device)
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

                # Value #
            with torch.no_grad():
                targetV1, targetV2 = self.value_target(next_state)
                targetV = reward + (1 - done) * self.gamma * torch.min(targetV1, targetV2)

            currentV1, currentV2 = self.value(state)

            value_loss = F.mse_loss(currentV1, targetV) + F.mse_loss(currentV2, targetV)
            self.value_loss_history.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()


            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
