import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    def __init__(self, input_nums, output_nums, sigma):
        super(NoisyLinear, self).__init__()

        self.input_nums = input_nums
        self.output_nums = output_nums
        self.sigma = sigma

        # 重みの平均 / 分散
        self.weight_mu = nn.Parameter(torch.empty(output_nums, input_nums))
        self.weight_sigma = nn.Parameter(torch.empty(output_nums, input_nums))

        # バイアスの平均 / 分散
        self.bias_mu = nn.Parameter(torch.empty(output_nums))
        self.bias_sigma = nn.Parameter(torch.empty(output_nums))

        # パラメータを登録
        self.register_buffer('weight_epsilon', torch.empty(output_nums, input_nums))
        self.register_buffer('bias_epsilon', torch.empty(output_nums))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_nums)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma / math.sqrt(self.input_nums))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma / math.sqrt(self.output_nums))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_nums)
        epsilon_out = self._scale_noise(self.output_nums)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        return F.linear(input,
                        self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon)


class QNet(nn.Module):
    def __init__(self, input_nums, output_nums, hidden_nums, sigma):
        super(QNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, hidden_nums),
            nn.ReLU(inplace=True),
        )
        self.noisy_linear = NoisyLinear(hidden_nums, output_nums, sigma)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def reset_noise(self):
        self.noisy_linear.reset_noise()

    def forward(self, x):
        return self.noisy_linear(self.layers(x))
