import random
from collections import namedtuple

from config import Config

Transition = namedtuple("Transition", ("state", "next_state", "action", "reward", "mask"))


class Memory:
    def __init__(self, config: Config):
        self.capacity = config.memory.memory_capacity
        self.memory = []
        self.index = 0

    def push(self, state, next_state, action, reward, mask):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, next_state, action, reward, mask)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        transition = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transition))   # 同じキー毎にまとめる
        return batch

    def __len__(self):
        return len(self.memory)
