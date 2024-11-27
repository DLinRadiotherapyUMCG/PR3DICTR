import torch
from torch import nn



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
