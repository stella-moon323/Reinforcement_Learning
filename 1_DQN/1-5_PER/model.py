import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_nums, output_nums, hidden_nums):
        super(QNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, output_nums),
        )

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)
