import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    """
    https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float or Tensor): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    def __init__(self, alpha = 0.25, gamma= 2.0, reduction = "none", eps=1e-8, margin=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.margin = margin

    def forward(self, inputs, targets):
        
        if self.margin is not None: # for Focal Margin Loss, we have to subtract the margin 
            # p2 = torch.clamp(torch.sigmoid(inputs - self.margin), min=self.eps, max=1-self.eps)
            # ce_loss = F.binary_cross_entropy(p2, targets, reduction="none")
            p = torch.clamp(torch.sigmoid(inputs - self.margin), min=self.eps, max=1-self.eps)
            ce_loss = F.binary_cross_entropy_with_logits(inputs - self.margin, targets, reduction="none")
        else:
            p = torch.clamp(torch.sigmoid(inputs), min=self.eps, max=1-self.eps)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        
        return loss
