import torch


class Memory:
    def __init__(self, agent_num, advantage_step, num_observations, gamma):
        self.advantage_step = advantage_step
        self.gamma = gamma
        self.index = 0
        self.states = torch.zeros(advantage_step + 1, agent_num, num_observations)
        self.masks = torch.ones(advantage_step + 1, agent_num, 1)
        self.rewards = torch.zeros(advantage_step, agent_num, 1)
        self.actions = torch.zeros(advantage_step, agent_num, 1).long()

        # 割引報酬和
        self.returns = torch.zeros(advantage_step + 1, agent_num, 1)

    def set_initial_state(self, state):
        self.states[0].copy_(state)

    def get_state(self, step):
        return self.states[step]

    def get_all_states_actions(self):
        num_observations = self.states.shape[2]
        states = self.states[:-1].view(-1, num_observations)
        actions = self.actions.view(-1, 1)
        return states, actions

    def insert(self, state, action, reward, mask):
        self.states[self.index + 1].copy_(state)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)
        self.index = (self.index + 1) % self.advantage_step

    def after_update(self):
        # 先頭に前回の状態を保持
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        # Advantageを計算
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            returns = self.returns[ad_step + 1]
            masks = self.masks[ad_step + 1]
            rewards = self.rewards[ad_step]
            self.returns[ad_step] = returns * self.gamma * masks + rewards
