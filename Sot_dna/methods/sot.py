# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml
import sys
sys.path.append(r'../../fewshotbench_v2')
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate
from self_optimal_transport.self_optimal_transport import SOT

class SOTClASS(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, n_task, task_update_num, inner_lr, approx=False):
        super(SOTClASS, self).__init__(backbone, n_way, n_support, change_way=False)
        print("self.feat_dim=",self.feat_dim)
        # self.classifier = Linear_fw(self.feat_dim, n_way)
        self.classifier = Linear_fw(self.feat_dim, n_way)
        self.classifier1 = Linear_fw(75, n_way)
        self.classifier.bias.data.fill_(0)

        self.sot = SOT(distance_metric='cosine', 
                   ot_reg=0.1, 
                   sinkhorn_iterations=20, 
                   sigmoid=False, 
                   mask_diag=True, 
                   max_scale=True)
        
        if n_way == 1:
            self.type = "regression"
            self.loss_fn = nn.MSELoss()
        else:
            self.type = "classification"
            self.loss_fn = nn.CrossEntropyLoss()

        self.n_task = n_task
        self.task_update_num = task_update_num
        self.inner_lr = inner_lr
        self.approx = approx  # first order approx.
    # backbone
    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)

        # For regression tasks, these are not scores but predictions
        if scores.shape[1] == 1:
            scores = scores.squeeze(1)

        return scores

    def set_forward(self, x, y=None):

        if isinstance(x, list):  # If there are >1 inputs to model (e.g. GeneBac)
            if torch.cuda.is_available():
                x = [obj.cuda() for obj in x]
            x_var = [Variable(obj) for obj in x]
            x_a_i = [x_var[i][:, :self.n_support, :].contiguous().view(self.n_way * self.n_support,
                                                                *x[i].size()[2:]) for i in range(len(x))] #support set
            x_b_i = [x_var[i][:, self.n_support:, :].contiguous().view(self.n_way * self.n_query, *x[i].size()[2:]) for i in range(len(x))]  # query data

        else:
            if torch.cuda.is_available():
                x = x.cuda()
            x_var = Variable(x)
            x_a_i = x_var[:, :self.n_support, :].contiguous().view(self.n_way * self.n_support,
                                                                *x.size()[2:])  # support data
            x_b_i = x_var[:, self.n_support:, :].contiguous().view(self.n_way * self.n_query, *x.size()[2:])  # query data

        if y is None:  # Classification task, assign labels (class indices) based on n_way
            y_a_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support)))  # label for support data
        else:  # Regression task, keep labels as they are
            y_var = Variable(y)
            y_a_i = y_var[:, :self.n_support].contiguous().view(self.n_way * self.n_support,
                                                                *y.size()[2:])  # label for support data
        if torch.cuda.is_available():
            y_a_i = y_a_i.cuda()

        support_feature = self.sot(self.forward(x_a_i)) #(25,25)
        # query_feature = self.sot(self.forward(x_b_i)) #(75,75)
        query_feature = self.sot(self.forward(x_b_i)) #(75,75)
        
        
        print("query_feature=",query_feature)
        print("query_feature=",query_feature.shape)
        print("support_feature=",support_feature)
        print("support_feature=",support_feature.shape)
        
        # support_score = self.classifier(support_feature)
        query_score = self.classifier1(query_feature)
        
        scores = query_score
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('SOT performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x, y=None):
        scores = self.set_forward(x, y)

        if y is None:  # Classification task
            y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query)))
        else:  # Regression task
            y_var = Variable(y)
            y_b_i = y_var[:, self.n_support:].contiguous().view(self.n_way * self.n_query, *y.size()[2:])

        if torch.cuda.is_available():
            y_b_i = y_b_i.cuda()

        loss = self.loss_fn(scores, y_b_i)

        return loss

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        # train
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                assert self.n_way == x[0].size(
                    0), f"SOT do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"
            else:
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(
                    0), f"SOT do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"

            # Labels are assigned later if classification task
            if self.type == "classification":
                y = None

            loss = self.set_forward_loss(x, y)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({'loss/train': avg_loss / float(i + 1)})

    def test_loop(self, test_loader, return_std=False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                assert self.n_way == x[0].size(0), "SOT do not support way change"
            else:
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), "SOT do not support way change"

            if self.type == "classification":
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
            else:
                # Use pearson correlation
                acc_all.append(self.correlation(x, y))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        if self.type == "classification":
            print('%d Accuracy = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        else:
            # print correlation
            print('%d Correlation = %4.2f +- %4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
