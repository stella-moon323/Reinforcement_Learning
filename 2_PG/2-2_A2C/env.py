import torch
import torch.nn.functional as F
import gym
import numpy as np

from config import Config
from model import Net
from brain import Brain
from memory import Memory


class Environment:
    def run(self, config: Config):
        env_name = config.env.env_name
        agent_num = config.agent.agent_num
        advantage_step = config.agent.advantage_step

        # 同時実行するエージェント数分の環境を作成
        envs = [gym.make(env_name) for i in range(agent_num)]
        num_observations = envs[0].observation_space.shape[0]
        num_actions = envs[0].action_space.n

        # モデル定義
        online_net = Net(num_observations, num_actions, config.model.hidden_num)
        global_brain = Brain(online_net, config)

        # メモリ
        memory = Memory(agent_num, advantage_step, num_observations, config.agent.gamma)

        # 各Agentに対するenvの初期化
        obs = np.array([envs[i].reset() for i in range(agent_num)])
        current_obs = torch.from_numpy(obs).float()
        memory.set_initial_state(current_obs)

        # step後の情報保持
        obs_np = np.zeros([agent_num, num_observations])
        reward_np = np.zeros([agent_num, 1])
        done_np = np.zeros([agent_num, 1])
        each_step = np.zeros(agent_num)  # 1episode中のstep数を計測

        # 終了判定用
        episode_rewards = torch.zeros([agent_num, 1])  # 現在の試行の報酬を保持
        final_rewards = torch.zeros([agent_num, 1])  # 最後の試行の報酬を保持

        episode = 0  # 先頭のAgentのEpisode数

        for episode in range(config.env.max_episode * agent_num):
            # Advantage学習のStep数毎に計算
            for adv_step in range(advantage_step):
                # 各Agentの行動を取得
                with torch.no_grad():
                    action = self.get_action(online_net, memory.get_state(adv_step))
                action_np = action.squeeze(1).numpy()

                for i in range(agent_num):
                    # 1step実行
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(action_np[i])

                    # Episode終了
                    if done_np[i]:
                        if i == 0:
                            print("Agent[{}] : Episode[{}] finished[{} Steps]".format(i, episode, each_step[i]))
                            episode += 1

                        # 報酬の設定
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 転んだ
                        else:
                            reward_np[i] = +1.0  # 立ち続けていた

                        each_step[i] = 0
                        obs_np[i] = envs[i].reset()  # 環境をリセット
                    else:
                        reward_np[i] = 0.0
                        each_step[i] += 1

                reward = torch.from_numpy(reward_np).float()
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])  # done==Trueなら0, Falseなら1

                # 最後の試行の報酬を保持
                episode_rewards += reward
                final_rewards *= masks  # done時は報酬を空にする
                final_rewards += (1 - masks) * episode_rewards  # done時の報酬を加算
                episode_rewards *= masks

                obs = np.array(obs_np)
                current_obs = torch.from_numpy(obs).float()
                current_obs *= masks  # done時は状態を空にする

                # メモリに追加
                memory.insert(current_obs, action.data, reward, masks)

            # 最後の状態の価値を取得
            with torch.no_grad():
                next_value = self.get_value(online_net, memory.get_state(-1).detach())

            # 割引報酬の計算
            memory.compute_returns(next_value)

            # ネットワーク更新
            loss = global_brain.update(memory, agent_num, advantage_step)
            memory.after_update()

            total_rewards = final_rewards.sum().numpy()
            print("Loss : {:+.6f}, Final rewards(sum) : {:+2.0f}".format(loss, total_rewards))
            # 全Agentが報酬をもらっていれば終了
            if total_rewards >= agent_num:
                print('Training complete')
                break

    def get_action(self, net, state):
        # 行動を確率的に選択
        actor, critic = net(state)
        action_probs = F.softmax(actor, dim=1)  # dim=1 : Actionの種類ごと
        action = action_probs.multinomial(num_samples=1)  # Actionを1つ選ぶ
        return action

    def get_value(self, net, state):
        # 状態価値を取得
        _, critic = net(state)
        return critic
