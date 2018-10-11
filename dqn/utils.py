import random
from collections import deque

import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(state,0), torch.tensor(action), torch.tensor(reward), torch.stack(next_state,0), torch.tensor(done)

    def __len__(self):
        return len(self.buffer)
