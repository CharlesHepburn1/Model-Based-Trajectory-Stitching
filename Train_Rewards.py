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
from Algorithms import Reward_Functions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "Hopper")    # OpenAI gym environment name to save file
    parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--diff", default = "Med")              # D4RL difficulty
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--lr", default = 3e-4)            # learning rate of models
    parser.add_argument("--epochs", default = 500000, type =int)   #Number of epochs
    parser.add_argument("--iterations", default = 1, type = int)   #Number of iterations
    parser.add_argument("--hidden_dim", default = 512, type = int)
    args = parser.parse_args()
    if not os.path.exists("./PlannerModels"):
        os.makedirs("./PlannerModels")
    if args.diff == 'Rand':
        file_name = 'Random'
    elif args.diff == 'MedRep':
        file_name = 'MediumReplay'
    elif args.diff == 'MedExp':
        file_name = 'MediumExpert'
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Convert D4RL to replay buffer
    print("Converting data...")
    states = torch.Tensor(dataset["observations"])
    actions = torch.Tensor(dataset["actions"])
    rewards = torch.Tensor(dataset["rewards"])
    next_states = torch.Tensor(dataset["next_observations"])
    dones = torch.Tensor(dataset["terminals"])

    num = int(0.95*len(states))
    list_full = np.linspace(0, len(states)-1, len(states), dtype = int)
    idx_train = np.random.choice(len(states)-1, num, replace=False)
    idx_test = list(set(list_full) - set(idx_train))
    idx_val = np.random.choice(idx_train, int(len(idx_test)), replace = False)

    states_train = np.take(states, idx_train, axis = 0)
    actions_train = np.take(actions, idx_train, axis = 0)
    rewards_train = np.take(rewards, idx_train, axis = 0)
    next_states_train = np.take(next_states, idx_train, axis = 0)
    dones_train = np.take(dones, idx_train, axis = 0)

    states_test = torch.Tensor(np.take(states, idx_test, axis = 0)).to(device)
    actions_test = torch.Tensor(np.take(actions, idx_test, axis = 0)).to(device)
    rewards_test= torch.Tensor(np.take(rewards, idx_test, axis = 0).reshape(len(states_test),1)).to(device)
    next_states_test=torch.Tensor(np.take(next_states, idx_test, axis = 0)).to(device)
    dones_test= torch.Tensor(np.take(dones, idx_test, axis = 0)).to(device)

    states_val = torch.Tensor(np.take(states, idx_val, axis = 0)).to(device)
    actions_val = torch.Tensor(np.take(actions, idx_val, axis = 0)).to(device)
    rewards_val= torch.Tensor(np.take(rewards, idx_val, axis = 0).reshape(len(states_val),1)).to(device)
    next_states_val= torch.Tensor(np.take(next_states, idx_val, axis = 0)).to(device)
    dones_val= torch.Tensor(np.take(dones, idx_val, axis = 0)).to(device)

    replay_buffer_full = [states,actions,rewards, next_states, dones]
    replay_buffer_train = [states_train, actions_train, rewards_train, next_states_train, dones_train]
    print("...data conversion complete")
    # Hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    epochs = args.epochs
    iterations = 1
    agent_mlp = Reward_Functions.MLP_train(state_dim, action_dim,hidden_dim = hidden_dim, batch_size=batch_size,
                                   device=device)
    grad_steps = 0
    agent_type ="MLP" #"GAUSS" #"VAE" #"GAN"  #
    agent = agent_mlp
    seed = 0
    env = gym.make(env0)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_accuracy = []
    test_accuracy = []
    for epoch in range(epochs):
        # Training #
        agent.train(replay_buffer_train, iterations)
        grad_steps += iterations

        if epoch % 1000 == 0 :
            fake_reward = agent.Rew.forward(states_test, actions_test, next_states_test)
            test_acc = (torch.sum((fake_reward - rewards_test) ** 2).detach().cpu().item()) / len(states_test)

            fake_reward_tr = agent.Rew.forward(states_val, actions_val, next_states_val)
            train_acc = (torch.sum((fake_reward_tr - rewards_val) ** 2).detach().cpu().item()) / len(states_val)

            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            print("Epoch ", epoch, " out of ", epochs, "Train_acc = ", np.round(train_acc,5) , "Test_acc = ",np.round(test_acc,5))#,

    rewards_model = agent.Rew.state_dict()
    reward_loss = agent.mlp_loss_history

    #torch.save(train_accuracy, f"Reward_Functions/Data/TrainAcc_{agent_type}_hd{hidden_dim}")
    #torch.save(test_accuracy, f"Reward_Functions/Data/TestAcc_{agent_type}_hd{hidden_dim}")
    #torch.save(reward_loss, f"Reward_Functions/Data/Loss_{agent_type}_hd{hidden_dim}")
    #torch.save(rewards_model, f"Reward_Functions/Models/{agent_type}_hd{hidden_dim}")

    torch.save(rewards_model, f"PlannerModels/{environment}/Rewards_{agent_type}_{file_name}")
