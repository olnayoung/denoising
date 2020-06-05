import torch
import torch.nn as nn

import common

class Net(nn.Module):
    def __init__(self, conv=common.default_conv, res=common.ResBlock):
        super().__init__()

        self.conv1 = conv(3, 64, 3)
        self.conv2 = conv(64, 128, 3)
        self.conv3 = conv(128, 64, 3)
        self.conv4 = conv(64, 3, 3)

        self.res1 = res(conv, 128, 3)
        self.res2 = res(conv, 128, 3)

        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.relu(self.conv1(input))
        out = self.relu(self.conv2(out))

        out1 = self.res1(out)
        out1 += out
        out2 = self.res2(out1)
        out2 += out1

        out = self.relu(self.conv3(out2))
        out = self.relu(self.conv4(out))

        return out + input