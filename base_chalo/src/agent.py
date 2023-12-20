import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import os
import time
import sys

class model(nn.Module):
    def __init__(self, input_length, n_actions):
        super(model, self).__init__()
        slope = 0.1
        self.network = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.LeakyReLU(slope),

            nn.Linear(128, 128),
            nn.LeakyReLU(slope),

            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.network(x)

class actions():
    linear_vel_range = (-0.5, 2)
    linear_vel_buckets = 5
    angular_vel_range = (-1, 1)
    angular_vel_buckets = 5
    activity = dict()

    def update():
        actions.activity = dict()
        t = 0
        V = np.linspace(actions.linear_vel_range[0], actions.linear_vel_range[1], actions.linear_vel_buckets)
        W = np.linspace(actions.angular_vel_range[0], actions.angular_vel_range[1], actions.angular_vel_buckets)
        for v in V:
            for w in W:
                actions.activity[t] = (v, w)
                t = t + 1

    def get(n):
        return actions.activity[n]

class memory():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.length = 0
        self.data = []

    def push(self, state, new_state, action, reward, done):
        self.data.append([state, new_state, action, reward, done])
        self.length += 1

    def sample(self):
        size = min(self.batch_size, self.length)
        batch = random.sample(self.data, size)
        return zip(*batch)

    def __len__(self):
        return self.length

class agent():
    def __init__(self, 
            input_length, 
            n_actions, 
            batch_size=64,
            alpha=0.001,
            gamma=0.99,
            epsilon=1,
            reuse="move.pth"):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.input_length = input_length
        self.n_actions = n_actions
        
        self.main_model = model(input_length, n_actions)
        self.temp_model = model(input_length, n_actions)
        if (reuse):
            self.main_model.load_state_dict(torch.load(reuse))
            self.temp_model.load_state_dict(torch.load(reuse))
        else:
            self.temp_model.load_state_dict(self.main_model.state_dict())

        self.optimizer = optim.Adam(self.temp_model.parameters(), lr=self.alpha)
        self.loss_criteria = nn.MSELoss()
        self.memory = memory(batch_size)

    def choose_action(self, state, echo=False):
        if (torch.rand(1).item() < self.epsilon):
            return torch.randint(0, self.n_actions, (1, )).item()

        state = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.temp_model(state)
            action = q_values.argmax().item()

        if (echo):
            print(q_values)

        return action
            
    def store(self, *transition):
        self.memory.push(*transition)

    def train(self, state, new_state, action, reward, done):
        state = torch.tensor([state], dtype=torch.float32)
        new_state = torch.tensor([new_state], dtype=torch.float32)
        action = int(action)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        curr_q = self.temp_model(state)

        with torch.no_grad():
            next_q = self.main_model(new_state)
            max_future_q = next_q.max()
            target_q = curr_q.clone()
            target_q[0, action] = reward + (1 - done) * self.gamma * max_future_q

        loss = self.loss_criteria(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        length = min(self.memory.batch_size, len(self.memory))
        states, new_states, actions, rewards, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        curr_q = self.temp_model(states)

        with torch.no_grad():
            next_q = self.main_model(new_states)
            max_future_q = next_q.max(dim=1)[0]
            target_q = curr_q.clone()
            for t in range(length):
                action = actions[t].item()
                target_q[:, action] = rewards[t] + (1 - dones[t]) * self.gamma * max_future_q[t]

        loss = self.loss_criteria(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_main_model(self):
        self.main_model.load_state_dict(self.temp_model.state_dict())

    def save(self, name):
        torch.save(self.main_model.state_dict(), name)


