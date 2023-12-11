from torch import nn as nn

from backbones.blocks import ConvBlock, Flatten


def Conv4(x_dim, fast_weight):
    return ConvNet(4, x_dim=x_dim,fast_weight=fast_weight)


def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


def Conv4S():
    return ConvNetS(4)


def Conv4SNP():
    return ConvNetSNopool(4)

class ConvNet(nn.Module):
    def __init__(self, depth, x_dim, flatten=True, fast_weight=False):
        super(ConvNet, self).__init__()
        self.fast_weight = fast_weight
        trunk = []
        # for i in range(depth):
        #     indim = 3 if i == 0 else 64
        #     outdim = 64
        #     B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
        #     trunk.append(B)
        for i in range(depth):
            indim = x_dim if i == 0 else 16
            outdim = 16
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 16  #previous 1600

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,2,1)
        print("xinconv=",x.shape)

        out = self.trunk(x)
        print("convout",out.shape)
        return out

# class ConvNet(nn.Module):
#     def __init__(self, depth, flatten=True):
#         super(ConvNet, self).__init__()
#         trunk = []
#         for i in range(depth):
#             # indim = 16 if i == 0 else 64 #here 123123
#             indim = 16 if i == 0 else 2866 #here 123123
#             outdim = 64
#             B = ConvBlock(indim, outdim, 3,pool=(i < 4))  # only pooling for fist 4 layers
#             # print("B.shape",B.shape)
#             trunk.append(B)

#         if flatten:
#             trunk.append(Flatten())

#         self.trunk = nn.Sequential(*trunk)
#         self.final_feat_dim = 1600
#         # self.final_feat_dim = 11648

#     def forward(self, x):
#         out = self.trunk(x)
#         return out


class ConvNetNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 16 if i == 0 else 64#here 123123
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetS(nn.Module):  # For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten=True):
        super(ConvNetS, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())
        ####changed here123123
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


class ConvNetSNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64 #here 123123
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 5, 5]

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out
