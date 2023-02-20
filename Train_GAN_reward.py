# Imports
import gym
import random
import numpy as np
import copy
import pickle
import os
import torch
import d4rl
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Algorithms import WGAN_Reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "Hopper")    # OpenAI gym environment name to save file
    parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--diff", default = "Med")              # D4RL difficulty
    parser.add_argument("--seed", default=42, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--latent_dim", default = 64, type = int)   #laten dim for gan
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--lr", default = 3e-4)            # learning rate of models
    parser.add_argument("--epochs", default = 300000, type =int)   #Number of epochs
    parser.add_argument("--iterations", default = 1, type = int)   #Number of iterations
    parser.add_argument("--l2_reg_D", default = 0.0001)
    args = parser.parse_args()
    if not os.path.exists("./PlannerModels"):
        os.makedirs("./PlannerModels")

    print("Inputs:", args.env_name, args.env, args.diff)
    if args.diff == 'Rand':
        file_name = 'Random'
    elif args.diff =='MedRep':
        file_name = 'MediumReplay'
    elif args.diff =='MedExp':
        file_name ='MediumExpert'
    elif args.diff == 'Med':
        file_name = 'Medium'
    else:
        file_name = args.diff

    environment = args.env_name
    env0 = args.env
    env = gym.make(env0)
    dataset = d4rl.qlearning_dataset(env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    seed = args.seed
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # Convert D4RL to replay buffer
    print("Converting data...")
    states = torch.Tensor(dataset["observations"])
    actions = torch.Tensor(dataset["actions"])
    rewards = torch.Tensor(dataset["rewards"])
    next_states = torch.Tensor(dataset["next_observations"])
    dones = torch.Tensor(dataset["terminals"])

    replay_buffer = [states, actions, rewards, next_states, dones]
    print("...data conversion complete")

    # Hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    hidden_dim = 512
    latent_dim = 2
    epochs = 100000
    iterations = 1

    agent = WGAN_Reward.GAN(state_dim, action_dim, latent_dim,hidden_dim = hidden_dim,  l2_reg_D=0.0001, device=device)

    grad_steps = 0

    discriminator_test_loss = []
    generator_test_loss = []
    gen_accuracy = []
    for epoch in range(epochs):
        # Training #
        agent.train(replay_buffer, iterations)
        grad_steps += iterations

        if epoch % 10000 == 0 :
            print("Epoch ", epoch, " out of ", epochs, "Generator loss = ",
                  np.round(agent.generator_loss_history[epoch],6), "Discriminator loss = ",
                  np.round(agent.discriminator_loss_history[epoch],6)) # "Test Accuracy = ", np.round(test_acc,6),


    gen_loss = agent.generator_loss_history
    dis_loss = agent.discriminator_loss_history

    torch.save(agent.discriminator.state_dict(), f"PlannerModels/{environment}/{file_name}-Rewards-Discriminator")
    torch.save(agent.generator.state_dict(), f"PlannerModels/{environment}/{file_name}-Rewards-Generator")

