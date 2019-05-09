import gym
import random
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
        self.online_net = QNet(self.num_observations, self.num_actions,
                               config.model.hidden_num, config.model.noisy_sigma, config.model.atom_nums)
        self.target_net = QNet(self.num_observations, self.num_actions,
                               config.model.hidden_num, config.model.noisy_sigma, config.model.atom_nums)
        self.update_target_net()
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

        self.epsilon = config.agent.epsilon_start
        self.memory = Memory(config)

        # 報酬の分布計算用
        self.dz = float(config.agent.v_max - config.agent.v_min) / (config.model.atom_nums - 1)
        self.z = torch.Tensor([config.agent.v_min + i * self.dz for i in range(config.model.atom_nums)])

    def start(self):
        step_avg_last10 = np.zeros(10)    # 直近のStep数
        reward_avg_last10 = np.zeros(10)  # 直近の報酬
        complete_episode = 0              # 連続成功回数
        for episode in range(self.config.agent.max_episode):
            self.logger.debug("Episode start: {}".format(episode + 1))

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
                self.push_memory(state, next_state, action, reward, mask)
                state = next_state

                if len(self.memory) > self.config.agent.batch_size:
                    # モデル更新
                    loss = self.train_model()
                    loss_avg.update(loss, self.config.agent.batch_size)

                if done or (step == (self.config.agent.max_step - 1)):
                    # ε-greedy更新
                    self.epsilon -= self.config.agent.epsilon_decay
                    self.epsilon = max(self.epsilon, self.config.agent.epsilon_min)

                    step_avg_last10 = np.hstack((step_avg_last10[1:], step + 1))
                    reward_avg_last10 = np.hstack((reward_avg_last10[1:], episode_reward))
                    break

            self.logger.info("Episode [{:3}]: Step:{}(Avg:{}), Reward:{:.3f}, loss(avg):{:.6f}".format(
                episode + 1, step + 1, step_avg_last10.mean(), episode_reward, loss_avg.avg))

            if self.tbx_writer is not None:
                self.tbx_writer.add_scalar("log/step", float(step_avg_last10.mean()), episode + 1)
                self.tbx_writer.add_scalar("log/reward", float(reward_avg_last10.mean()), episode + 1)
                self.tbx_writer.add_scalar("log/loss", float(loss_avg.avg), episode + 1)
                self.tbx_writer.add_scalar("log/epsilon", float(self.epsilon), episode + 1)

            if ((episode + 1) % self.config.agent.update_target_net_interval) == 0:
                self.update_target_net()

            # TD誤差の更新
            self.update_td_error()

            if complete_episode >= 10:
                # 10回連続成功で終了
                break

        self.logger.info("Training complete")

        if self.tbx_writer is not None:
            self.tbx_writer.close()

    def get_action(self, state):
        if self.epsilon <= np.random.uniform(0, 1):
            state_var = torch.Tensor(state).unsqueeze(0).to(self.config.model.device)
            with torch.no_grad():
                out = self.online_net(state_var).detach().squeeze(0)
                z_space = self.z.repeat(self.num_actions, 1)
                Q = torch.sum(out * z_space, dim=1)
                action = torch.argmax(Q).item()
        else:
            action = random.randrange(self.num_actions)
        return action

    def push_memory(self, state, next_state, action, reward, mask):
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        self.memory.push(state, next_state, action_one_hot, reward, mask)

    def get_m(self, rewards, masks, prob_next_states_action, batch_size):
        rewards = rewards.numpy()
        masks = masks.numpy()
        prob_next_states_action = prob_next_states_action.detach().numpy()
        m_prob = np.zeros([batch_size, self.config.model.atom_nums], dtype=np.float32)

        batch_id = range(batch_size)
        gamma = self.config.agent.gamma
        v_max = self.config.agent.v_max
        v_min = self.config.agent.v_min
        for j in range(self.config.model.atom_nums):
            tz = np.clip(rewards + masks * gamma * (v_min + j * self.dz), v_min, v_max)
            bj = (tz - v_min) / self.dz

            lj = np.floor(bj).astype(np.int64)
            uj = np.ceil(bj).astype(np.int64)
            blj = (bj - lj)
            buj = (uj - bj)

            m_prob[batch_id, lj[batch_id]] += ((1 - masks) + masks * (prob_next_states_action[batch_id, j])) * buj[batch_id]
            m_prob[batch_id, uj[batch_id]] += ((1 - masks) + masks * (prob_next_states_action[batch_id, j])) * blj[batch_id]

        return torch.tensor(m_prob)

    def calc_target_signal(self, next_states, actions, rewards, masks, batch_size):
        z_space = self.z.repeat(batch_size, self.num_actions, 1)
        prob_next_states = self.target_net(next_states)
        Q_next_state = torch.sum(prob_next_states * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states[i, action, :] for i, action in enumerate(next_actions)])

        m_prob = self.get_m(rewards, masks, prob_next_states_action, batch_size)
        m_prob = (m_prob / torch.sum(m_prob, dim=1, keepdim=True)).detach()
        return m_prob

    def train_model(self):
        batch = self.memory.sample(self.config.agent.batch_size)
        device = self.config.model.device
        states = torch.Tensor(np.stack(batch.state)).to(device)
        next_states = torch.Tensor(np.stack(batch.next_state)).to(device)
        actions = torch.Tensor(batch.action).float().to(device)
        rewards = torch.tensor(batch.reward).float().to(device)
        masks = torch.Tensor(batch.mask).to(device)

        # 教師信号の生成
        self.online_net.eval()
        self.target_net.eval()
        expand_dim_action = torch.unsqueeze(actions, -1)
        pred = torch.sum(self.online_net(states) * expand_dim_action.float(), dim=1)
        m_prob = self.calc_target_signal(next_states, actions, rewards, masks, self.config.agent.batch_size)

        # パラメータ更新
        self.online_net.train()
        with torch.autograd.detect_anomaly():
            loss = -torch.sum(m_prob * torch.log(pred + 1e-20), 1)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.online_net.reset_noise()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_td_error(self):
        batch = self.memory.get_all()
        device = self.config.model.device
        states = torch.Tensor(np.stack(batch.state)).to(device)
        next_states = torch.Tensor(np.stack(batch.next_state)).to(device)
        actions = torch.Tensor(batch.action).float().to(device)
        rewards = torch.tensor(batch.reward).float().to(device)
        masks = torch.Tensor(batch.mask).to(device)

        # TD誤差の計算
        self.online_net.eval()
        self.target_net.eval()
        expand_dim_action = torch.unsqueeze(actions, -1)
        pred = torch.sum(self.online_net(states) * expand_dim_action.float(), dim=1)
        m_prob = self.calc_target_signal(next_states, actions, rewards, masks, len(self.memory))

        td_error = -torch.sum(m_prob * torch.log(pred + 1e-20), 1)
        td_error = td_error.detach().numpy().tolist()
        self.memory.update_td_error(td_error)


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
