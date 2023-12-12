# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from self_optimal_transport.self_optimal_transport import SOT
from backbones.blocks import Linear_fw

class ProtoNetSOT(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoNetSOT, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier1 = Linear_fw(75, n_way)
    def set_forward(self, x, is_feature=False):
        
        self.sot = SOT(distance_metric='cosine', 
                ot_reg=0.1, 
                sinkhorn_iterations=20, 
                sigmoid=False, 
                mask_diag=True, 
                max_scale=True)
        
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        scores = self.sot(scores)
        scores = self.classifier1(scores)
        return scores


    def set_forward_loss(self, x):
        # x = self.sot(x)
        print("x=",x,"xshape=",x.shape)
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())
        
        
        scores = self.set_forward(x)
        # scores = self.sot(scores)
        # print("scores.shape",scores.shape)
        # scores = self.classifier1(scores)
        # print("x=",y_query,"protosot",y_query.shape,"scores",scores,"scores.shape",scores.shape)
        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
