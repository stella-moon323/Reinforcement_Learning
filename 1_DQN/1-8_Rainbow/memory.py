import random
import numpy as np
from collections import namedtuple

from config import Config

Transition = namedtuple("Transition", ("state", "next_state", "action", "reward", "mask"))


class Memory:
    TdErrorEpsilon = 0.00001

    def __init__(self, config: Config):
        self.capacity = config.memory.memory_capacity
        self.memory = []
        self.td_error = []
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
                self.td_error.append(None)
            self.memory[self.index] = Transition(self.local_state, next_state, self.local_action, reward, mask)
            self.td_error[self.index] = 0
            self.index = (self.index + 1) % self.capacity
            self.reset_local()
        if mask == 0:
            self.reset_local()

    def sample(self, batch_size):
        if (batch_size * 2) <= len(self.memory):
            indice = self.get_prioritized_index(batch_size)
            transition = [self.memory[n] for n in indice]
        else:
            transition = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transition))   # 同じキー毎にまとめる
        return batch

    def get_all(self):
        return Transition(*zip(*self.memory))

    def update_td_error(self, td_error):
        self.td_error = td_error

    def get_prioritized_index(self, batch_size):
        # TD誤差の和
        sum_td_error = np.sum(np.absolute(self.td_error))
        sum_td_error += (self.TdErrorEpsilon * len(self.td_error))

        rand_list = np.random.uniform(0, sum_td_error, batch_size)
        rand_list = np.sort(rand_list)

        temp_td_error = 0
        index = 0
        indice = []
        for rand_num in rand_list:
            while temp_td_error < rand_num:
                temp_td_error += (abs(self.td_error[index]) + self.TdErrorEpsilon)
                index += 1
            if index >= len(self.td_error):
                index = len(self.td_error) - 1
            indice.append(index)
        return indice

    def __len__(self):
        return len(self.memory)
