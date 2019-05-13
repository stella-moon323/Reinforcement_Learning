import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Brain:
    def __init__(self, net, config):
        self.net = net
        self.value_loss_coef = config.brain.value_loss_coef
        self.entropy_coef = config.brain.entropy_coef
        self.max_grad_norm = config.brain.max_grad_norm

        # 最適化手法
        self.optimizer = optim.Adam(net.parameters(), lr=config.optim.lr)

    def update(self, memory, agent_num, advantage_step):
        states, actions = memory.get_all_states_actions()
        values, action_log_probs, entropy = self.evaluate_actions(states, actions)

        values = values.view(advantage_step, agent_num, 1)
        action_log_probs = action_log_probs.view(advantage_step, agent_num, 1)

        # Advantageの計算
        advantages = memory.returns[:-1] - values

        value_loss = advantages.pow(2).mean()  # Criticのlossを計算
        action_gain = -(action_log_probs * advantages.detach()).mean()   # Actorのgainを計算(Advantageはdetachして定数扱いにする)

        # 誤差関数の総和
        total_loss = (value_loss * self.value_loss_coef + action_gain - entropy * self.entropy_coef)

        # パラメータ更新
        self.net.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        # 勾配に制限をかける
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return total_loss.item()

    def evaluate_actions(self, states, actions):
        actor, critic = self.net(states)

        log_probs = F.log_softmax(actor, dim=1)  # 各状態での行動確率(log)を取得
        action_log_probs = log_probs.gather(1, actions)  # 選んでいた方の行動確率(log)を取得

        # エントロピー項の計算
        probs = F.softmax(actor, dim=1)  # 各状態での行動確率を取得
        entropy = -(log_probs * probs).sum(-1).mean()

        return critic, action_log_probs, entropy
