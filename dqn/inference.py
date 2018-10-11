import random

import torch
import gym

from model import QNet
from wrappers import make_atari, wrap_pytorch

model = 'pong_01_900.pt'
device = 'cuda'
epsilon = 0.01


def select_action(state, number_actions):
    eps = random.random()

    if eps < epsilon:
        action = random.randrange(number_actions)
    else:
        input = torch.from_numpy(state).to(device, torch.float32).unsqueeze(0)
        score = net(input)
        action = score.max(dim=1)[1].to(torch.int64).item()
    return action

# Build environment
env = make_atari('PongNoFrameskip-v4', stack=2)
env = wrap_pytorch(env)
env = gym.wrappers.Monitor(env, directory='./movie', force=True, video_callable=lambda x: True)
number_actions = env.action_space.n

# Separate target net & policy net
input_shape = env.reset().shape
net = QNet(input_shape, number_actions)
net.load_state_dict(torch.load(model))
net.eval().to(device)

for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        # env.render()
        action = select_action(state, number_actions=number_actions)
        next_state, reward, done, _ = env.step(action)
        state = next_state
env.close()