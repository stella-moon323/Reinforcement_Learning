import logging
import logzero
# import os


class Config:
    def __init__(self):
        self.com = CommonConfig()
        self.log = LogConfig(self.com)
        self.env = EnvConfig()
        self.agent = AgentConfig()
        self.model = ModelConfig()
        self.optim = AdamConfig()


class CommonConfig:
    def __init__(self):
        self.name = "REINFORCE"


class LogConfig:
    def __init__(self, com: CommonConfig):
        self.log_dir = "../../logs"
        self.name = "root"                  # loggerの名前、複数loggerを用意するときに区別できる
        self.level = logging.INFO           # 標準出力のログレベル
        self.formatter = logzero.LogFormatter(
            fmt='[%(levelname)s][%(asctime)s.%(msecs)d][%(module)s:%(lineno)d]: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')    # ログのフォーマット
        self.disableStderrLogger = False    # 標準出力をしないか
        self.logfile = None  # os.path.join(self.log_dir, com.name + ".log")    # ログファイルの格納先
        self.maxBytes = 10 * 1000 * 1000    # ログローテーションする際のファイルの最大バイト数
        self.backupCount = 10               # ログローテーションする際のバックアップ数
        self.fileLoglevel = logging.WARN    # ログファイルのログレベル


class EnvConfig:
    def __init__(self):
        self.env_name = "CartPole-v0"


class AgentConfig:
    def __init__(self):
        self.max_episode = 3000
        self.max_step = 200
        self.gamma = 0.99
        self.use_visualize = True


class ModelConfig:
    def __init__(self):
        self.use_gpu = False
        self.device = None
        self.hidden_num = 32


class AdamConfig:
    def __init__(self):
        self.lr = 0.001
