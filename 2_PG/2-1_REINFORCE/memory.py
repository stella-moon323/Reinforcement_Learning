from collections import namedtuple

from config import Config

Transition = namedtuple("Transition", ("state", "action", "reward", "mask"))


class Memory:
    def __init__(self, config: Config):
        self.memory = []

    def push(self, state, action, reward, mask):
        self.memory.append(Transition(state, action, reward, mask))

    def sample(self):
        return Transition(*zip(*self.memory))   # 同じキー毎にまとめる

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
