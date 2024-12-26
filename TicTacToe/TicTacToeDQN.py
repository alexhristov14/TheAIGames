import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


def train(env, num_episodes=100, batch_size=64, gamma=0.99, lr=1e-3, buffer_capacity=10000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(5000)

    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01


    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()

            else:
                with torch.no_grad():
                    action = q_net(state_tensor).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                break


            if len(replay_buffer) > batch_size:
                
                transitions = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*transitions)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
            
                q_values = q_net(states).gather(1, actions)
                
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
                
                loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return q_net


env = gym.make("CartPole-v1")
model = train(env)


for _ in range(10000):
    state, _ = env.reset()
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    action = model(state_tensor).argmax().item()
    env.step(action)
    env.render(render_mode="human")
    