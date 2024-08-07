import torch
from utils.losses.F1 import F1_Loss
from utils.losses.dice import DiceLoss
from utils.losses.L1 import L1Loss
from utils.losses.ranking import RankingLoss
from utils.losses.softAUC import SoftAUCLoss



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