"""
Loss functions and classes for training the pytorch models
"""
import math
import numpy as np
from typing import Optional, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss


class DiceLoss(torch.nn.Module):
    """
    https://neptune.ai/blog/pytorch-loss-functions
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)

    def forward(self, y_pred, y_true, smooth=1):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)
        y_pred = y_pred[:, 1]

        # Map [0, 1] to [-1, 1]
        # y_pred = (y_pred - 0.5) * 2
        # y_true = (y_true - 0.5) * 2

        # y_pred = y_pred.view(-1)
        # y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

        return 1 - dice



class F1_Loss(torch.nn.Module):
    '''
    Calculate F1 score. Can work with gpu tensors

    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = torch.nn.functional.one_hot(y_true, 2).to(torch.float32)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class L1Loss(torch.nn.Module):
    def __init__(self, num_ohe_classes, reduction):
        super().__init__()
        self.l1 = torch.nn.SmoothL1Loss(reduction=reduction, beta=0.05)
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.num_ohe_classes = num_ohe_classes

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(
            self.l1(self.softmax_act(y_pred), torch.nn.functional.one_hot(y_true, num_classes=self.num_ohe_classes).float()))

        return loss


class RankingLoss(torch.nn.Module):
    """
    In self.rank(): if y = 1 then it assumed the first input should be ranked higher (have a larger value) than the
    second input, and vice-versa for y = -1.

    y_pred.shape: torch.Size([7, 2])
    y_pred: tensor([[0.0916, 0.4190],
            [0.5505, 0.2531],
            [0.4484, 0.1766],
            [0.4597, 0.2745],
            [0.6642, 0.3163],
            [0.6221, 0.5239],
            [0.5956, 0.3042]], device='cuda:0', grad_fn=<AddmmBackward0>)
    y_true.shape: torch.Size([7])
    y_true: tensor([0, 0, 0, 1, 1, 0, 0], device='cuda:0')
    y: tensor([ 1.,  1.,  1., -1., -1.,  1.,  1.], device='cuda:0')

    """

    def __init__(self, reduction):
        super().__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.rank = torch.nn.MarginRankingLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)

        x1 = y_pred[:, 0]
        x2 = y_pred[:, 1]
        y = - (y_true - 0.5) * 2

        loss = self.rank(x1, x2, y)

        return loss


class SoftAUCLoss(torch.nn.Module):
    """
    Soft version of AUC that uses Wilcoxon-Mann-Whitney U statistic.

    Approximates the Area Under Curve score, using approximation based on the Wilcoxon-Mann-Whitney U statistic.

    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.

    Measures overall performance for a full range of threshold levels.

    Source: https://github.com/mysterefrank/pytorch_differentiable_auc/blob/master/auc_loss.py
    Source: https://github.com/tflearn/tflearn/blob/c0baee9d34f41b84dbc43ea28e37baa5dbd465e4/tflearn/objectives.py#L179
    Source: https://discuss.pytorch.org/t/loss-backward-found-long-expected-float/121743/4
    """

    def __init__(self):
        super().__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.sigmoid_act = torch.nn.Sigmoid()
        self.gamma = 0.2  # 0.7
        self.p = 3

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)[:, 1]
        # y_pred = self.sigmoid_act(y_pred)
        # Get the predictions of all the positive and negative examples
        pos = y_pred[y_true.bool()].view(1, -1)
        neg = y_pred[~y_true.bool()].view(-1, 1)
        # Compute Wilcoxon-Mann-Whitney U statistic
        difference = torch.zeros_like(pos * neg) + pos - neg - self.gamma
        masked = difference[difference < 0.0]
        loss = torch.sum(torch.pow(-masked, self.p))

        return loss


class CustomLoss(torch.nn.Module):
    def __init__(self, config, label_weights, reduction):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=label_weights, reduction=reduction, label_smoothing=config.label_smoothing)
        self.dice = DiceLoss()
        self.f1 = F1_Loss().to(config.device)
        self.l1 = L1Loss(num_ohe_classes=config.num_ohe_classes, reduction=reduction)
        self.ranking = RankingLoss(reduction=reduction)
        self.soft_auc = SoftAUCLoss()
        self.loss_weights = config.loss_weights

    def forward(self, y_pred, y_true):
        fcns = [self.ce, self.dice, self.f1, self.l1, self.ranking, self.soft_auc]

        loss = 0
        for i, (w, fcn) in enumerate(zip(self.loss_weights, fcns)):
            if w != 0:
                loss = loss + w * fcn(y_pred, y_true)

        return loss


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



class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.01, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss



class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "
    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 
   
     where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.
    .. note::
        Sigmoid will be done in loss. 
    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2
    """
    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'none'):
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_logits, targets):
        """
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = input_logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(input_logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




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
        





class SB_ECE_Loss(nn.Module):
    """
    Soft-Binning Expected Calibration Error (SB-ECE) 
    """
    def __init__(self,
                 n_bins: int = 10,
                 T: float = 0.01,
                 device: torch.device = torch.device('cpu'),
                 reduction: str = 'none') -> None:
        super(SB_ECE_Loss, self).__init__()

        self.n_bins = n_bins
        self.T = T
        self.device = device
        self.reduction = reduction
        
        # find the centers of each bin
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)[:-1]  #  (remove the last element as torch.linspace `does` include the end point (np.linspace does not))
        bin_width = ((bin_boundaries[1] - bin_boundaries[0])/2) 
        self.bin_centers = bin_boundaries + bin_width       

        self.columnwise_softmax = torch.nn.Softmax(dim=0)

        self.eps = 1e-8
    

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        N = len(logits)  # get the number of predictions in this batch
        m = self.n_bins
        confidences_all = torch.sigmoid(logits + self.eps).T
        targets_all = targets.T

        bin_center = self.bin_centers

        loss_template = torch.zeros_like(confidences_all, device=self.device, dtype=torch.float32)

        for idx, (confidences, targets) in enumerate(zip(confidences_all, targets_all)):
            #print(idx)
            #print(confidences)
            N = len(confidences)  # get the number of predictions in this batch
            r_hat = confidences.view(1, N)
            r_hat_matrix = r_hat.expand(m, N)
            epsilon_hat = bin_center.view(m, 1)
            epsilon_hat_matrix = epsilon_hat.expand(m, N).to(self.device)

            g = -(r_hat_matrix - epsilon_hat_matrix)**2 / self.T
            g_matrix = self.columnwise_softmax(g)

            correctness_vector = targets.float().view(1, N)
            correctness_matrix = correctness_vector.expand(m, N)
            A_matrix = correctness_matrix * g_matrix

            S = torch.sum(g_matrix, dim=1)
            S_check = torch.eq(S, torch.zeros(m).to(self.device)).float()
            A_hat = torch.div(torch.sum(A_matrix, dim=1), S + S_check + self.eps).view(m, 1)
            A_hat_matrix = A_hat.expand(m, N)
            temp_matrix = ((A_hat_matrix - r_hat_matrix)**2) * g_matrix

            loss =  (1/N * torch.sum(temp_matrix) + self.eps)**(1/2)

            loss_template[idx] = loss
        
        loss_template = torch.clamp(loss_template, min=0, max=1)
        return loss_template.T



        
        targets_all = targets
        #N = len(confidences)  # get the number of predictions in this batch
        #orig_shape = confidences_all.shape
        
        m = self.n_bins

        loss_template = torch.zeros_like(confidences_all)

        for idx, (confidences, targets) in enumerate(zip(confidences_all, targets_all)):
            N = torch.numel(confidences)

            bin_centers_matrix = self.bin_centers.view(m,1) .expand(m,N).to(self.device)  # bin centers as a matrix (each row contains a bin center, but the dimension is the same as the predictions matrix)
        
            pred_confidences_matrix = confidences.view(1,N).expand(m,N)  # predictions as a tensor of size m x N (each prediction is a column, and there is one row per bin)
            # compute the soft ECE bin probabilities
            
            g = -(pred_confidences_matrix-bin_centers_matrix)**2/ self.T  # soft bin function
            g_matrix = self.columnwise_softmax(g)        

            true_labels_matrix = targets.float().view(1,N).expand(m,N)
            Accuracy_matrix = true_labels_matrix * g_matrix
            bin_accuracies = torch.sum(Accuracy_matrix, dim = 1)  # accuracy of each bin

            S = torch.sum(g_matrix, dim = 1) # size of each bin
            S_check = torch.eq(S, torch.zeros(m).to(self.device)).float()  
            A_hat = torch.div(bin_accuracies, S+S_check).view(m,1)  # mean accuracy of all bins
            A_hat_matrix = A_hat.expand(m,N)
            temp_matrix = ((A_hat_matrix - pred_confidences_matrix)**2)*g_matrix

            temp2 = torch.sum(temp_matrix, dim = 0)

            loss_template[idx] = temp2
            
        return loss_template

        return (1/N*torch.sum(temp_matrix))**(1/2)
        #loss = torch.reshape(temp_matrix, orig_shape)
        #return loss





class SB_ECE_Loss_OLD(nn.Module):
    """
    Soft-Binning Expected Calibration Error (SB-ECE) 
    """
    def __init__(self,
                 n_bins: int = 10,
                 T: float = 0.01,
                 device: torch.device = torch.device('cpu'),
                 reduction: str = 'none') -> None:
        super(SB_ECE_Loss, self).__init__()

        self.n_bins = n_bins
        self.T = T
        self.device = device
        self.reduction = reduction
        
        # find the centers of each bin
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)[:-1]  #  (remove the last element as torch.linspace `does` include the end point (np.linspace does not))
        bin_width = ((bin_boundaries[1] - bin_boundaries[0])/2) 
        self.bin_centers = bin_boundaries + bin_width       

        self.columnwise_softmax = torch.nn.Softmax(dim=0)
    

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        confidences_all = torch.sigmoid(logits)
        targets_all = targets
        #N = len(confidences)  # get the number of predictions in this batch
        #orig_shape = confidences_all.shape
        
        m = self.n_bins

        loss_template = torch.zeros_like(confidences_all)

        for idx, (confidences, targets) in enumerate(zip(confidences_all, targets_all)):
            N = torch.numel(confidences)

            bin_centers_matrix = self.bin_centers.view(m,1) .expand(m,N).to(self.device)  # bin centers as a matrix (each row contains a bin center, but the dimension is the same as the predictions matrix)
        
            pred_confidences_matrix = confidences.view(1,N).expand(m,N)  # predictions as a tensor of size m x N (each prediction is a column, and there is one row per bin)
            # compute the soft ECE bin probabilities
            
            g = -(pred_confidences_matrix-bin_centers_matrix)**2/ self.T  # soft bin function
            g_matrix = self.columnwise_softmax(g)        

            true_labels_matrix = targets.float().view(1,N).expand(m,N)
            Accuracy_matrix = true_labels_matrix * g_matrix
            bin_accuracies = torch.sum(Accuracy_matrix, dim = 1)  # accuracy of each bin

            S = torch.sum(g_matrix, dim = 1) # size of each bin
            S_check = torch.eq(S, torch.zeros(m).to(self.device)).float()  
            A_hat = torch.div(bin_accuracies, S+S_check).view(m,1)  # mean accuracy of all bins
            A_hat_matrix = A_hat.expand(m,N)
            temp_matrix = ((A_hat_matrix - pred_confidences_matrix)**2)*g_matrix

            temp2 = torch.sum(temp_matrix, dim = 0)

            loss_template[idx] = temp2
            
        return loss_template

        return (1/N*torch.sum(temp_matrix))**(1/2)
        #loss = torch.reshape(temp_matrix, orig_shape)
        #return loss


class ESD_Loss(nn.Module):
    """
    Expected Squared Difference (ESD) loss as described in the paper ``
    """

    def __init__(self,
                 device: torch.device,
                 reduction: str = 'none') -> None:
        super(ESD_Loss, self).__init__()

        self.device = device

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        confidence1 = torch.sigmoid(logits)
        original_shape = confidence1.shape
        #N1 = len(confidence1)  # number of predictions in this batch
        N1 = torch.numel(confidence1)
        #confidence1 = torch.sigmoid(confidence1)
        confidence1= confidence1.view(N1)
        correct = targets.view(N1)

        val = correct.float() - confidence1   
        val = val.view(1,N1) 
        mask = torch.ones(N1,N1) - torch.eye(N1)
        mask = mask.to(self.device)
        confidence1_matrix = confidence1.expand(N1,N1)
        temp = (confidence1.view(1,N1).T).expand(N1,N1)
        tri = torch.le(confidence1_matrix,temp).float() 
        val_matrix = val.expand(N1,N1)
        x_matrix = torch.mul(val_matrix,tri)*mask
        g_bar = torch.sum(x_matrix, dim = 1)/(N1-1)
        x_matrix_squared = torch.mul(x_matrix, x_matrix)
        S_g = 1/(N1-2) * torch.sum(x_matrix_squared,dim=1) - (N1-1)/(N1-2) * torch.mul(g_bar,g_bar)

        # this is the ESD loss inside the square brackets of equation 10
        ESD_sum = torch.mul(g_bar, g_bar) - S_g/(N1-1)

        ESD_sum = torch.reshape(ESD_sum, original_shape)

        return ESD_sum
        # NOTE: below this is the original code; that only returns the mean ESD loss (not the per-item loss)
        #return ESD_sum
        reg_loss = torch.sum(ESD_sum)/N1
        
        return reg_loss





class WeightedLossWrapper(nn.Module):
    def __init__(self, loss_function, device, label_weights=None, label_pos_weights=None):
        super(WeightedLossWrapper, self).__init__()
        self.loss_function = loss_function
        self.device = device
        # weights = how much each classes' weight is weighted relative to the other classes
        if label_weights is None: 
            self.label_weights=None
        else: 
            self.label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device) # / sum(label_weights) # normalised to 1
        # pos_weight = how much each positive instance is weighted, relative to the negative class
        if label_pos_weights is None: 
            self.label_pos_weights=None
        else: 
            self.label_pos_weights = torch.tensor(label_pos_weights, dtype=torch.float32, device=device) # / sum(label_pos_weights) # normalised to 1
        
        

    def forward(self, input, target):
        # get the loss as normal (using the loss function)
        batch_loss = self.loss_function(input, target)

        # now apply the per-class weights to the loss
        if self.label_weights is not None:
            batch_loss *= self.label_weights

        # apply the pos_weight to the losses
        if self.label_pos_weights is not None:
            pos_mask = torch.isin(target, torch.tensor([1], device=self.device)) # only the positive examples
            pos_weight_multiplier = pos_mask * self.label_pos_weights         # per-class pos_weight (2 for first, 3 for second)
            pos_weight_multiplier[pos_weight_multiplier==0] = 1
            batch_loss *= pos_weight_multiplier

        # masking out the missing labels is done in the main code 
        return batch_loss





class MLLSC(nn.Module):
    """
    Implementation of MLLSC loss function from "Multi Label Loss Correction against Missing and Corrupted Labels" - Ghiassi et al. (2022)
    """

    def __init__(self,
                 variant = "BCE", 
                 tau_pos: float = 0.55,
                 tau_neg: float = 0.6,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(MLLSC, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

        self.apply_MLLSC = False
        self.variant = variant.lower()  # "BCE" | "Focal" ("FocalMargin" if margin>0 in config)

        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.Focal_loss = FocalLoss(margin=self.margin, gamma=self.gamma, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        if self.variant == "bce":
            loss = self.forward_BCE(logits, targets)
        elif self.variant == "focal":
            loss = self.forward_Focal(logits, targets)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward_BCE(self, logits, targets):
        
        if self.apply_MLLSC:
            preds = torch.sigmoid(logits)  # sigmoid the logits (result is probabilities)

            # loss for positive and negative labels
            L_pos = torch.where(
                preds >= self.tau_pos, # condition
                torch.log(logits),                  # value if condition is true
                torch.log(1 - logits))                          # value if condition is false
            
            L_neg = torch.where(
                preds < self.tau_neg, # condition
                torch.log(1- logits),                  # value if condition is true
                torch.log(logits))                          # value if condition is false
            
            loss = -((targets * L_pos) + ((1 - targets) * L_neg))
                    
        else: # standard BCE
            loss = self.BCE_loss(logits, targets)

        return loss

    def forward_Focal(self, logits, targets):
        if self.apply_MLLSC:
            preds = torch.sigmoid(logits)
            P_km = torch.sigmoid(logits - self.margin)

            # loss for positive and negative labels
            L_pos = torch.where(
                preds >= self.tau_pos, # condition
                ((1-P_km)**self.gamma)*torch.log(P_km),              # value if condition is true
                (P_km**self.gamma)*torch.log(1 - P_km)               # value if condition is false
                )                          
            
            L_neg = torch.where(
                preds < self.tau_neg, # condition
                (preds**self.gamma)*torch.log(1-preds),              # value if condition is true
                ((1 - preds)**self.gamma)*torch.log(preds)               # value if condition is false
                )                          
            
            loss = -((targets * L_pos) + ((1 - targets) * L_neg))

        else:
            loss = self.Focal_loss(logits, targets)

        return loss



class AuxillaryLossWrapper(nn.Module):
    def __init__(self,
                 config,
                 main_loss_function, 
                 ) -> None:
        super(AuxillaryLossWrapper, self).__init__()

        self.main_loss_function = main_loss_function
        
        if config.auxiliary_loss_func_name == "SB-ECE":
            self.aux_loss_function = SB_ECE_Loss(n_bins=config.SB_ECE_n_bins, T=config.SB_ECE_temperature, device=config.device, reduction='none')
        elif config.auxiliary_loss_func_name == "ESD":
            self.aux_loss_function = ESD_Loss(device=config.device, reduction='none')

    def forward(self, logits, targets):
        main_loss = self.main_loss_function(logits, targets)
        aux_loss = self.aux_loss_function(logits, targets)

        #print(main_loss, aux_loss)

        return main_loss + aux_loss




def class_ratio_counter(config, train_dict):
    num_endpoints = len(config.endpoint_list)
    num_patients =  np.zeros(num_endpoints) # counts the number of patients that have either 0 or 1 for each endpoint (i.e. that are not missing data)
    class_counts = np.zeros(num_endpoints)  # counts the number of patients that are equal to 1 (i.e. have the toxicity)
    for patient in train_dict:
        labels = np.array(patient['label_list'])
        data_present = labels>=0
        num_patients += data_present
        labels[labels<0]=0
        class_counts += labels

    pos_weights = num_patients/(2*class_counts)
    inv_class_counts = num_patients - class_counts

    neg_weights = num_patients/(2*inv_class_counts)

    # print("POS WEIGHTS", class_counts)
    # print("TOTAL NUM  ", num_patients)
  
    return inv_class_counts/class_counts  # weights the positive samples against the number of negative samples


def get_loss_function(config, train_dict, val_dict):
    """
    CrossEntropyLoss: softmax is computed as part of the loss. In other words, the model outputs should be logits,
        i.e. the model output should NOT be softmax'd.
    BCEWithLogitsLoss: sigmoid is computed as part of the loss. In other words, the model outputs should be logits,
        i.e. the model output should NOT be sigmoid'd.

    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
    Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

    Args:
        config: (class object) configuration parameters
        train_dict: (dict) dictionary containing the training data (for setting the label weights)
        val_dict: (dict) dictionary containing the validation data
    Returns:
        loss function (class object)

    """
    loss_function_name = config.loss_function_name
    label_weights = config.label_weights
    label_pos_weights = config.label_pos_weights
    reduction = config.loss_reduction

    if loss_function_name == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                  label_smoothing=config.label_smoothing)
    elif loss_function_name == 'bce':
        if label_pos_weights == True:
            label_pos_weights = class_ratio_counter(config=config, train_dict=train_dict)
            label_pos_weights = torch.tensor(label_pos_weights, dtype=torch.float32, device=config.device)
            # print(" BCE POS WEIGHTS")
            # print(label_pos_weights)
        else:
            label_pos_weights = None
            # print(" BCE POS WEIGHTS")
            # print(label_pos_weights)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=label_pos_weights)
        # TODO: replace weight with pos_weight to deal with imbalanced classes
    #elif loss_function_name == 'dice':
    #    loss_function = DiceLoss()
    elif loss_function_name == 'f1':
        loss_function = F1_Loss().to(config.device)
    #elif loss_function_name == 'ranking':
    #    loss_function = RankingLoss(reduction=reduction)
    elif loss_function_name == 'soft_auc':
        loss_function = SoftAUCLoss()
    elif loss_function_name == 'mse':
        loss_function = MSELoss(reduction="none")
    #elif loss_function_name == 'custom':
    #    loss_function = CustomLoss(config=config, reduction=reduction)
    elif loss_function_name == 'focal':
        loss_function = FocalLoss(reduction=reduction, alpha=config.alpha, gamma=config.gamma, margin=config.margin)  # TODO: add the parameters alpha and gamma
    elif loss_function_name == 'asymmetric':
        loss_function = AsymmetricLossOptimized(gamma_neg=config.gamma_neg, gamma_pos=config.gamma_pos)  # TODO: add the parameters
    elif loss_function_name == 'hill':
        loss_function = Hill(lamb=config.lamb, margin=config.margin, gamma=config.gamma, reduction=reduction)
    elif loss_function_name.lower() == 'splc':
        loss_function = SPLC(tau=config.tau, margin=config.margin, gamma=config.gamma, reduction=reduction)
    elif loss_function_name.lower() == 'mllsc':
        loss_function = MLLSC(variant=config.mllsc_variant, tau_pos=config.tau_pos, tau_neg=config.tau_neg, margin=config.margin, gamma=config.gamma, reduction=reduction)

    else:
        raise ValueError('Invalid loss_function_name: {}.'.format(loss_function_name))
    
    if config.auxiliary_loss_func_name != None:
        loss_function = AuxillaryLossWrapper(config=config, main_loss_function=loss_function)
        
        loss_function.pos_weight = label_pos_weights  # needed to check if BCE is using label_pos_weights

    # if (label_weights is not None) or (label_pos_weights is not None):
    #     # weight the losses per class
    #     loss_function = WeightedLossWrapper(loss_function, device=config.device, label_weights=label_weights, label_pos_weights=None)
    return loss_function



def calc_mixup_loss(predictions, targets, mixup_y_true_indexes, loss_function, mixup_lambda):
    #batch_loss = loss_function(predictions, targets)
    y_a = targets
    y_b = targets[mixup_y_true_indexes]

    #print(predictions.shape)
    #print(targets.shape)
    predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
    #print(predictions.shape)
    #print(targets.shape)
    # print(y_a)
    # print(y_b)
    # print(mixup_y_true_indexes)


    #predictions = torch.reshape(predictions, y_a.shape).to(predictions.dtype)
    #predictions = torch.reshape(predictions, y_b.shape).to(predictions.dtype)

    y_a_loss = loss_function(predictions, y_a)
    y_b_loss = loss_function(predictions, y_b)

    batch_loss = (mixup_lambda * y_a_loss) + ((1 - mixup_lambda) * y_b_loss)

    return batch_loss




def calculate_loss(cfg, outputs_dict, labels_dict, valid_endpoints_as_tensor, loss_function, mixup_y_true_indexes=None, mixup_lambda=None):

    predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32) # transposed! so that num columns = num toxicities
    targets = torch.stack(list(labels_dict.values()), dim=1).type(torch.float32)

    # compute loss
    if (cfg.perform_mixup == True) and (mixup_y_true_indexes != None) and (mixup_lambda != None):
        batch_loss = calc_mixup_loss(predictions, targets, mixup_y_true_indexes, loss_function, mixup_lambda)
    else:
        #print("PREDS", predictions.shape)
        #print("TARGS", targets.shape)
        predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
        #print(predictions.shape)
        batch_loss = loss_function(predictions, targets)
    #print(batch_loss)
    # mask out the losses of the missing labels
    
    #batch_loss *= mask
    
    per_toxicity_loss_dict = dict()
    axis = 0
    if not cfg.loss_function_name == 'mse':
        mask = (targets >= valid_endpoints_as_tensor[0]) & (targets <= valid_endpoints_as_tensor[1])
        #mask = torch.isin(targets, valid_endpoints_as_tensor)
        #print("B1", batch_loss)
        batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
        #print("B2", batch_loss)
        batch_loss *= mask
        #print("B3", batch_loss)

        # find the mean loss of the non-missing labels
        batch_loss_mean = batch_loss.sum() / mask.sum()
        # find the loss per endpoint (the column-wise mean of the batch_loss tensor)
        per_toxicity_loss = batch_loss.sum(axis=axis) / mask.sum(axis=axis)
    else:
        batch_loss_mean = batch_loss.mean()

        per_toxicity_loss = batch_loss.mean(axis=axis)

    #print(batch_loss_mean)
       

    for idx, endpoint in enumerate(cfg.endpoint_list):
        value = per_toxicity_loss[idx].item()
        per_toxicity_loss_dict[endpoint] = value if (not math.isnan(value)) else 0
    

    # also get the non-weighted loss
    # if we are not weighting the loss, then the unweighted_batch_loss_mean = batch_loss_mean, and unweighted_per_toxicity_loss_dict = per_toxicity_loss_dict
    unweighted_per_toxicity_loss_dict = dict()
    if cfg.label_pos_weights:
    #if loss_function.pos_weight:
        pos_weights = loss_function.pos_weight

        unweighting_factor = torch.where(targets != 1, torch.ones_like(targets), pos_weights.expand(batch_loss.shape))
        unweighted_batch_loss = batch_loss / unweighting_factor
        unweighted_batch_loss_mean = unweighted_batch_loss.sum() / mask.sum()

        unweighted_per_toxicity_loss = unweighted_batch_loss.sum(axis=axis) / mask.sum(axis=axis)
        for idx, endpoint in enumerate(cfg.endpoint_list):
            value = unweighted_per_toxicity_loss[idx].item()
            unweighted_per_toxicity_loss_dict[endpoint] = value if (not math.isnan(value)) else 0
        
    else:
        unweighted_batch_loss_mean = batch_loss_mean
        for idx, endpoint in enumerate(cfg.endpoint_list):
            unweighted_per_toxicity_loss_dict[endpoint] = per_toxicity_loss_dict[endpoint]

    #print(batch_loss_mean)
    batch_loss_mean = torch.clamp(batch_loss_mean, min=0, max=10000) 

    return batch_loss_mean, per_toxicity_loss_dict, unweighted_batch_loss_mean, unweighted_per_toxicity_loss_dict

