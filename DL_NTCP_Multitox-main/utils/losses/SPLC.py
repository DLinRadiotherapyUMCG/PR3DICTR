import torch
from torch import nn
from torch.nn import functional as F



class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self,
                 tau: float = 0.6,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

        self.apply_SPLC = False

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        # SPLC missing label correction
        if self.apply_SPLC:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1), targets)
        
        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
