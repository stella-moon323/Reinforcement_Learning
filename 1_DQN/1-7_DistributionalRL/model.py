import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_nums, output_nums, hidden_nums, atom_nums):
        super(QNet, self).__init__()

        self.output_nums = output_nums
        self.atom_nums = atom_nums
        self.layers = nn.Sequential(
            nn.Linear(input_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, hidden_nums),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nums, output_nums * atom_nums),
        )

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        z = out.view(-1, self.output_nums, self.atom_nums)
        p = nn.Softmax(dim=2)(z)
        return p
