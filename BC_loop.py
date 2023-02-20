import numpy as np
import d4rl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

import gym
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=(256,256)):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))

        return a

def train_policy(replay_buffer, model, loss_function=nn.MSELoss(), iterations=1, batch_size = 256, device = "cuda:0"):
    total_it = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = 0
    for it in range(iterations):
        total_it += 1
        indices = torch.randint(0, len(replay_buffer[0]), size=(batch_size,)).to(device)
        state = torch.index_select(replay_buffer[0], 0, indices).to(device)
        action = torch.index_select(replay_buffer[1], 0, indices).to(device)


        optimizer.zero_grad()
        # Actor #
        policy_actions = model.forward(state)


        loss = loss_function(policy_actions, action)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(replay_buffer)
    return train_loss

k = 3 #This is the 5th iteration of TS 

Mean_BC = []
STD_BC = []
ENV = []
# Load environment
Environment =['Hopper', 'Walker2d','Halfcheetah' ]## #
Diffs = ['Expert', 'MedRep', 'MedExp', 'Med']
for en in range(len(Environment)):
    environment = Environment[en]
    if environment == 'Hopper':
        env_name = 'hopper'
    elif environment == 'Walker2d':
        env_name = 'walker2d'
    elif environment == 'Halfcheetah':
        env_name = 'halfcheetah'


    for di in range(len(Diffs)):
        Diff = Diffs[di]
        if Diff == 'Expert':
            diff_env = 'expert'
            Diff_pi = 'Expert'
        elif Diff == 'MedRep':
            diff_env = 'medium-replay'
            Diff_pi = 'MediumReplay'
        elif Diff == 'MedExp':
            diff_env = 'medium-expert'
            Diff_pi = 'MediumExpert'
        elif Diff == 'Med':
            diff_env = 'medium'
            Diff_pi = 'Medium'

        device = "cuda:0"
        BC_scores = []
        seeds = [51, 118, 2026, 19876,  717]
        env0 = f'{env_name}-{diff_env}-v2'
        for i in range(3):
            seed_data = i + 1
            for j in range(len(seeds)):
                torch.cuda.empty_cache()
                seed = seeds[j]
                env = gym.make(env0)
                # dataset = d4rl.qlearning_dataset(env)
                env.seed(seed)
                env.action_space.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                if seed == 51:
                    s_idx = 1
                elif seed == 118:
                    s_idx = 2
                elif seed == 2026:
                    s_idx = 3
                elif seed == 19876:
                    s_idx = 4
                elif seed == 717:
                    s_idx = 5

                if Diff == 'Med':
                    dataset_new = torch.load(
                        f"Data/NewMethod/{environment}/state{Diff}_new_data_iter{k}-S{seed_data}.pt")
                else:
                    dataset_new = torch.load(f"Data/NewMethod/{environment}/{Diff}_iter{k}-S{seed_data}.pt")

                states = torch.Tensor(dataset_new[0]).to(device)
                actions = torch.Tensor(dataset_new[1]).to(device)
                states_np = states.detach().cpu().numpy()
                actions_np = actions.detach().cpu().numpy()

                num = int(0.95*len(states))
                num_else = int(len(states)-num)
                list_full = np.linspace(0, len(states)-1, len(states), dtype = int)
                idx_train = np.random.choice(len(states)-1, num, replace=False)
                idx_test = list(set(list_full) - set(idx_train))
                idx_train_small = np.random.choice(idx_train, num_else, replace=False)

                states_train = np.take(states_np, idx_train, axis = 0)
                states_train = torch.Tensor(states_train).to(device)
                actions_train = np.take(actions_np, idx_train, axis = 0)
                actions_train = torch.Tensor(actions_train).to(device)

                states_test = np.take(states_np, idx_test, axis = 0)
                states_test = torch.Tensor(states_test).to(device)
                actions_test = np.take(actions_np, idx_test, axis = 0)
                actions_test = torch.Tensor(actions_test).to(device)

                states_tr_small = np.take(states_np, idx_train_small, axis = 0)
                states_tr_small = torch.Tensor(states_tr_small).to(device)
                actions_tr_small = np.take(actions_np, idx_train_small, axis = 0)
                actions_tr_small = torch.Tensor(actions_tr_small).to(device)

                replay_buffer_new = [states_train, actions_train]
                # print("...data conversion complete")

                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                max_action = env.action_space.high[0]
                '''
                Train BC Model
                '''
                epochs = 101
                grad_steps = 0
                iters = 1000
                evals = 10
                model = Policy(state_dim, action_dim, max_action).to(device)
                best_score = 0
                test_acc = []
                train_acc = []
                Score = []
                Train_loss = []
                for epoch in range(epochs):
                    # Training #
                    t_loss = train_policy(replay_buffer_new, model = model,iterations = iters)
                    Train_loss.append(t_loss)
                    grad_steps += iters
                    with torch.no_grad():
                        acts_test = model.forward(states_test)
                        accuracy_test = torch.sum((acts_test - actions_test)**2)/num_else
                        test_acc.append(accuracy_test.detach().cpu().item())
                        acts_tr = model.forward(states_tr_small)
                        accuracy_tr_small = torch.sum((acts_tr - actions_tr_small) ** 2)/num_else
                        train_acc.append(accuracy_tr_small.detach().cpu().item())
                    # if epoch %10 == 0:
                    #     print("Epoch", epoch, "Grad Steps", grad_steps,"Training loss % .4f" % t_loss,
                    #           "Train Accuracy % .4f" % accuracy_tr_small,"Test Accuracy % .4f" % accuracy_test)
                    # Evaluation #
                    if epoch % 10 ==0:
                        env_eval = gym.make(env0)
                        env_eval.seed(seed)
                        env_eval.action_space.seed(seed)
                        scores = []
                        scores_norm = []
                        for eval in range(evals):
                            done = False
                            state = env_eval.reset()
                            score = 0
                            while not done:
                                with torch.no_grad():
                                    st = torch.Tensor([state]).to(device)
                                    act = model.forward(st)
                                    action = act.detach().cpu().numpy()[0]
                                    state, reward, done, info = env_eval.step(action)
                                    score += reward
                            score_norm = env_eval.get_normalized_score(score)*100
                            scores.append(score)
                            scores_norm.append(score_norm)
                        # print("Epoch", epoch, "Grad Steps", grad_steps, "Score Norm %.4f" % np.mean(scores_norm), "Std %.4f" % np.std(scores_norm))
                        Score.append(np.mean(scores_norm))
                        if epoch>35 and np.mean(scores_norm)>best_score:
                            best_score = np.mean(scores_norm)
                            print("Epoch", epoch, "Grad Steps", grad_steps, "Score Norm %.4f" % np.mean(scores_norm),
                                  "Std %.4f" % np.std(scores_norm))
                            torch.save(model.state_dict(),
                                      f"Models/Policy/{environment}_{Diff}_iter{k}-S_{seed_data}_{s_idx}.pt")
                            print("...... Saving Policy .......")
                BC_scores.append(best_score)
        print(env0)
        print('Mean of BC', np.mean(BC_scores), 'std', np.std(BC_scores))
        ENV.append(env0)
        Mean_BC.append(np.round(np.mean(BC_scores), 4))
        STD_BC.append(np.round(np.std(BC_scores), 5))

print('K = ', k)
print(ENV)
print(Mean_BC)
print(STD_BC)
