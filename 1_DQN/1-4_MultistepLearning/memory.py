import random
from collections import namedtuple

from config import Config

Transition = namedtuple("Transition", ("state", "next_state", "action", "reward", "mask"))


class Memory:
    def __init__(self, config: Config):
        self.capacity = config.memory.memory_capacity
        self.memory = []
        self.index = 0
        self.gamma = config.agent.gamma
        self.n_step = config.memory.n_step
        self.reset_local()

    def reset_local(self):
        self.local_step = 0
        self.local_state = None
        self.local_action = None
        self.local_rewards = []

    def push(self, state, next_state, action, reward, mask):
        self.local_step += 1
        self.local_rewards.append(reward)
        if self.local_step == 1:
            self.local_state = state
            self.local_action = action
        if self.local_step == self.n_step:
            reward = 0
            for idx, local_reward in enumerate(self.local_rewards):
                reward += (self.gamma ** idx) * local_reward

            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.index] = Transition(self.local_state, next_state, self.local_action, reward, mask)
            self.index = (self.index + 1) % self.capacity
            self.reset_local()
        if mask == 0:
            self.reset_local()

    def sample(self, batch_size):
        transition = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transition))   # 同じキー毎にまとめる
        return batch

    def __len__(self):
        return len(self.memory)
