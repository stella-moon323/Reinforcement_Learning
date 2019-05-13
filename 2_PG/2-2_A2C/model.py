import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_nums, output_nums, hidden_nums):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, hidden_nums),
            nn.ReLU(inplace=True),
        )
        self.fc_actor = nn.Linear(hidden_nums, output_nums)
        self.fc_critic = nn.Linear(hidden_nums, 1)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        hid = self.layers(x)
        actor = self.fc_actor(hid)
        critic = self.fc_critic(hid)
        return actor, critic
