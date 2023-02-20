# Import modules
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

# Dynamics model
class Dynamics_Model(nn.Module):
    def __init__(self, state_dim, hidden_dim=200):
        super(Dynamics_Model, self).__init__()
        self.l1 = nn.Linear(state_dim , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, state_dim)
        self.log_std = nn.Linear(hidden_dim, state_dim)

    def forward(self, state):
        ns = F.relu(self.l1(state))
        ns = F.relu(self.l2(ns))
        ns = F.relu(self.l3(ns))
        ns = F.relu(self.l4(ns))

        mu = self.mean(ns)
        log_std = self.log_std(ns).clamp(min=-20, max=2.0)
        std = log_std.exp()

        return mu, std

    def distribution(self, state):
        mu, std = self.forward(state)

        distribution = Independent(Normal(mu, std), 1)

        return distribution

    def samples(self, state):
        dist = self.distribution(state)

        next_state = dist.sample()

        return next_state


def train_model(model, model_optimizer, replay_buffer_train, replay_buffer_val, device):

    # Training #
    training_loss = 0
    for i, batch in enumerate(replay_buffer_train):
        state_batch = batch[0].float().to(device)
        next_state_batch = batch[3].float().to(device)

        dist = model.distribution(state_batch)
        loss = -dist.log_prob(next_state_batch).mean()

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        training_loss += loss.item()

    # Validation #
    validation_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(replay_buffer_val):
            state_batch = batch[0].float().to(device)
            next_state_batch = batch[3].float().to(device)

            dist = model.distribution(state_batch)
            loss = -dist.log_prob(next_state_batch).mean()

            validation_loss += loss.item()

    training_loss /= len(replay_buffer_train)
    validation_loss /= len(replay_buffer_val)

    return training_loss, validation_loss