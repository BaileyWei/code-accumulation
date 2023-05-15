import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import numpy as np


def multi_label_categorical_cross_entropy(y_pred: torch.FloatTensor, y_true: torch.LongTensor):
    """
    https://kexue.fm/archives/7359
    input: (batch_size, shaking_seq_len, type_size)
    target: (batch_size, shaking_seq_len, type_size)
    y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
         1 tags positive classes，0 tags negative classes(means tok-pair does not have this type of link).
    """

    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_neg = torch.concat([y_neg, zeros], axis=-1)
    y_pos = torch.concat([y_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)
    loss = neg_loss + pos_loss

    return loss.mean()


class MultiLabelCategoricalCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-100):
        super(MultiLabelCategoricalCrossEntropy, self).__init__()

        self.ignore_index = ignore_index

    def forward(self, input: torch.FloatTensor, target: torch.LongTensor):

        mask = target.ne(self.ignore_index)
        input = torch.masked_select(input, mask)
        target = torch.masked_select(target, mask)

        return multi_label_categorical_cross_entropy(input, target)


def batch_gather(data:torch.Tensor, index:torch.Tensor):

    length = index.shape[0]
    t_index = index.data.numpy()
    t_data = data.data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result))


def sparse_multilabel_categorical_cross_entropy(y_true, y_pred):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.concat([y_pred, zeros], axis=-1)
    y_pos_2 = batch_gather(y_pred, y_true)
    y_pos_1 = torch.concat([y_pos_2, zeros], axis=-1)
    pos_loss = torch.logsumexp(-y_pos_1, axis=-1)
    all_loss = torch.logsumexp(y_pred, axis=-1)
    aux_loss = torch.logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-12, 1)
    neg_loss = all_loss + torch.log(aux_loss)

    return pos_loss + neg_loss


class SparseMultiLabelCategoricalCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-100):
        super(SparseMultiLabelCategoricalCrossEntropy, self).__init__()

        self.ignore_index = ignore_index

    def forward(self, input: torch.FloatTensor, target: torch.LongTensor):

        mask = target.ne(self.ignore_index)
        input = torch.masked_select(input, mask)
        target = torch.masked_select(target, mask)

        return sparse_multilabel_categorical_cross_entropy(input, target)


class BCEWithLogitsLossWithMask(nn.Module):
    def __init__(self, ignore_index=-100, *kwargs):
        super(BCEWithLogitsLossWithMask, self).__init__()

        self.ignore_index = ignore_index
        self.loss_fct = BCEWithLogitsLoss(*kwargs)

    def forward(self, input: torch.FloatTensor, target: torch.LongTensor):

        mask = target.ne(self.ignore_index)
        input = torch.masked_select(input, mask)
        target = torch.masked_select(target, mask)

        return self.loss_fct(input, target)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: (batch_size, shaking_seq_len, type_size)
        target: (batch_size, shaking_seq_len,)

        """
        c = input.size()[-1]
        log_preds = torch.log_softmax(input, dim=-1)
        if self.reduction == 'sum':
            loss = - log_preds.sum()
        else:
            loss = - log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        loss = loss * self.eps / c
        loss = loss + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )

        return loss


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation
    """
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)

        return loss