class Config:
    def __init__(self):
        self.com = CommonConfig()
        self.env = EnvConfig()
        self.agent = AgentConfig()
        self.model = ModelConfig()
        self.brain = BrainConfig()
        self.optim = AdamConfig()


class CommonConfig:
    def __init__(self):
        self.name = "A2C"


class EnvConfig:
    def __init__(self):
        self.env_name = "CartPole-v0"
        self.max_episode = 1000


class AgentConfig:
    def __init__(self):
        self.agent_num = 32
        self.advantage_step = 5
        self.gamma = 0.99


class ModelConfig:
    def __init__(self):
        self.use_gpu = False
        self.device = None
        self.hidden_num = 32


class BrainConfig:
    def __init__(self):
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5


class AdamConfig:
    def __init__(self):
        self.lr = 0.01
