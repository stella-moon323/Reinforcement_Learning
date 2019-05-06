import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_nums, output_nums, hidden_nums):
        super(QNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, hidden_nums),
            nn.ReLU(inplace=True),
        )
        self.fc_adv = nn.Linear(hidden_nums, output_nums)
        self.fc_val = nn.Linear(hidden_nums, 1)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        hid = self.layers(x)
        adv = self.fc_adv(hid)
        val = self.fc_val(hid).expand(-1, adv.size(1))
        out = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return out
