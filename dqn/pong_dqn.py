import random
from collections import deque
from itertools import count

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import QNet
from utils import ReplayBuffer
from wrappers import make_atari, wrap_pytorch

## Config
# general configs
batch_size = 32
gamma = 0.99
learning_starts = 10000
initial_epsilon = 1.0
final_epsilon = 0.01
final_exploration_frame = 100000
opt_algorithm = torch.optim.Adam
learning_rate = 0.0001
target_update_frequency = 1000
replay_memory_size = 100000
final_episode = 1000

# environment config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_epsilon(i):
    if i <= learning_starts:
        return initial_epsilon
    elif i >= final_exploration_frame:
        return final_epsilon
    else:
        return 1.0 + (final_epsilon-initial_epsilon) / (final_exploration_frame-learning_starts) * (i-learning_starts)


def select_action(state, current_net, eps, number_action):
    rand = random.random()
    assert 0.0 <= eps <= 1.0
    assert 0.0 <= rand <= 1.0

    if rand < eps:  # exploration
        action = random.randrange(number_action)
    else:  # exploitation
        with torch.no_grad():
            input = torch.from_numpy(state).to(device, torch.float32).unsqueeze(0)
            score = current_net(input)
            action = score.max(dim=1)[1].to(torch.int64).item()
    return action


def optimize_model(optimizer, policy_net, target_net, memory_batch):
    """
    Perform one step Q-learning on policy_net
    """
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory_batch
    state_batch =state_batch.to(device, torch.float32)
    action_batch = action_batch.to(device, torch.int64).view(-1,1)
    reward_batch = reward_batch.to(device, torch.float32)
    next_state_batch = next_state_batch.to(device, torch.float32)
    done_batch = done_batch.to(device, torch.float32)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_state_action_values = target_net(next_state_batch)
        next_state_values = next_state_action_values.max(1)[0]
        next_state_values = next_state_values * (1 - done_batch)  # no reward if this episode is done.
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

    # Compute Huber loss
    assert expected_state_action_values.requires_grad == False
    assert state_action_values.requires_grad == True
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Build environment
env = make_atari('PongNoFrameskip-v4', stack=2)
env = wrap_pytorch(env)

number_actions = env.action_space.n
replay_buffer = ReplayBuffer(replay_memory_size)

# Separate target net & policy net
input_shape = env.reset().shape
current_net = QNet(input_shape, number_actions).to(device)
target_net = QNet(input_shape, number_actions).to(device)  # with older weights
target_net.load_state_dict(current_net.state_dict())
target_net.eval()
optimizer = opt_algorithm(current_net.parameters(), lr=learning_rate)


n_episode = 1
episode_return = 0
best_return = 0
returns = []
state = env.reset()
for i in count():
    # env.render()
    eps = get_epsilon(i)
    action = select_action(state, current_net, eps, number_action=number_actions)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    episode_return += reward
    state = next_state

    # Perform one step optimization (on policy network)
    if i >learning_starts:
        memory_batch = replay_buffer.sample(batch_size)
        loss = optimize_model(optimizer, current_net, target_net, memory_batch)
    else:
        loss = 0

    # This episode is end
    if done:
        returns.append(episode_return)
        print(
            'episode {}, frame {}, return {}, loss {:.6f}, eps {:.6f}'.format(n_episode, i, episode_return, loss, eps))
        if episode_return>best_return or n_episode%100==0:
            # Save model
            torch.save(target_net.state_dict(), './checkpoints/pong_{}_{}.pt'.format(n_episode, episode_return))
            best_return = episode_return
        # New episode
        n_episode += 1
        episode_return = 0
        state = env.reset()

    # Update target network
    if i % target_update_frequency == 0:
        target_net.load_state_dict(current_net.state_dict())

    if n_episode == final_episode:
        break

env.close()