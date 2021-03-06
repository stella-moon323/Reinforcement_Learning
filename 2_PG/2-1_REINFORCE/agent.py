import gym
import numpy as np
import torch
import torchsummary
from tensorboardX import SummaryWriter
import os

from config import Config
from model import QNet
from memory import Memory


class Agent:
    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.env = gym.make(config.env.env_name)
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.logger.info("Observation nums: {}".format(self.num_observations))
        self.logger.info("Action nums: {}".format(self.num_actions))

        # モデル定義
        self.online_net = QNet(self.num_observations, self.num_actions, config.model.hidden_num)
        self.logger.debug(self.online_net)
        self.logger.debug('モデルパラメータ数: {}'.format(sum([p.data.nelement() for p in self.online_net.parameters()])))
        torchsummary.summary(self.online_net, tuple([self.num_observations]), device="cpu")

        # 最適化手法
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.optim.lr)

        if config.agent.use_visualize:
            print("Tensorboard : tensorboard --logdir [LOG_DIR]")
            tbx_dir = os.path.join(config.log.log_dir, config.com.name)
            self.tbx_writer = SummaryWriter(tbx_dir)
        else:
            self.tbx_writer = None

        # サポート対象のGPUがあれば使う
        if config.model.use_gpu:
            self.logger.info("Check GPU available")
            config.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            config.model.device = torch.device("cpu")
        self.logger.info("Use devide: {}".format(config.model.device))
        self.online_net.to(config.model.device)

        self.memory = Memory(config)

    def start(self):
        step_avg_last10 = np.zeros(10)    # 直近のStep数
        reward_avg_last10 = np.zeros(10)  # 直近の報酬
        complete_episode = 0              # 連続成功回数
        for episode in range(self.config.agent.max_episode):
            self.logger.debug("Episode start: {}".format(episode + 1))
            self.memory.clear()

            state = self.env.reset()
            episode_reward = 0
            loss_avg = AverageMeter()
            for step in range(self.config.agent.max_step):
                self.logger.debug("Step: {}, Observation: {}".format(step + 1, state))

                action = self.get_action(state)
                self.logger.debug("  - Action: {}".format(action))
                next_state, _, done, info = self.env.step(action)  # 報酬は別で計算

                # 報酬の計算
                if done:
                    mask = 0

                    # 途中で転んだら減点
                    if step < 195:
                        reward = -1
                        complete_episode = 0
                    else:
                        reward = +1
                        complete_episode += 1
                else:
                    mask = 1
                    reward = 0

                self.logger.debug("    - Reward: {}, Done: {}".format(reward, done))
                episode_reward += reward
                self.push_memory(state, action, reward, mask)
                state = next_state

                if done or (step == (self.config.agent.max_step - 1)):
                    step_avg_last10 = np.hstack((step_avg_last10[1:], step + 1))
                    reward_avg_last10 = np.hstack((reward_avg_last10[1:], episode_reward))
                    break

            # モデル更新
            loss = self.train_model()
            loss_avg.update(loss, 1)

            self.logger.info("Episode [{:3}]: Step:{}(Avg:{}), Reward:{:.3f}, loss(avg):{:.6f}".format(
                episode + 1, step + 1, step_avg_last10.mean(), episode_reward, loss_avg.avg))

            if self.tbx_writer is not None:
                self.tbx_writer.add_scalar("log/step", float(step_avg_last10.mean()), episode + 1)
                self.tbx_writer.add_scalar("log/reward", float(reward_avg_last10.mean()), episode + 1)
                self.tbx_writer.add_scalar("log/loss", float(loss_avg.avg), episode + 1)

            if complete_episode >= 10:
                # 10回連続成功で終了
                break

        self.logger.info("Training complete")

        if self.tbx_writer is not None:
            self.tbx_writer.close()

    def get_action(self, state):
        state_var = torch.Tensor(state).unsqueeze(0).to(self.config.model.device)
        with torch.no_grad():
            policy = self.online_net(state_var)[0].detach().numpy()
            action = np.random.choice(self.num_actions, 1, p=policy)[0]

        return action

    def push_memory(self, state, action, reward, mask):
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        self.memory.push(state, action_one_hot, reward, mask)

    def train_model(self):
        batch = self.memory.sample()
        device = self.config.model.device
        states = torch.Tensor(np.stack(batch.state)).to(device)
        actions = torch.Tensor(batch.action).float().to(device)
        rewards = torch.tensor(batch.reward).float().to(device)
        masks = torch.Tensor(batch.mask).to(device)

        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.agent.gamma * running_return * masks[t]
            returns[t] = running_return

        # パラメータ更新
        self.online_net.train()
        with torch.autograd.detect_anomaly():
            policies = self.online_net(states)
            policies = policies.view(-1, self.num_actions)
            log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)
            loss = (-log_policies * returns).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()


class AverageMeter:
    # 平均値測定器
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = (self.sum / self.count)
