from torch import nn as nn

from backbones.blocks import Conv2d_fw, Conv1d_fw, BatchNorm1d_fw,BatchNorm2d_fw, init_layer, Flatten, SimpleBlock, BottleneckBlock


class ResNet(nn.Module):
    maml = False  # Default

    def __init__(self, block, list_of_num_layers, list_of_out_dims, x_dim,flatten=True,fast_weight=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv1d_fw(x_dim, 64, kernel_size=1, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm1d_fw(64)
        else:
            conv1 = nn.Conv1d(x_dim, 64, kernel_size=1, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm1d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = x_dim
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool1d(1)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 1, 1]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,2,1)
        # x = x.permute(2,0,1)
        print("xshape=",x.shape)
        out = self.trunk(x)
        return out


def ResNet10(x_dim, fast_weight, flatten=True):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten=flatten,x_dim=x_dim,fast_weight=fast_weight)


def ResNet18(flatten=True):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)


def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)


def ResNet50(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], [256, 512, 1024, 2048], flatten)


def ResNet101(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], [256, 512, 1024, 2048], flatten)
