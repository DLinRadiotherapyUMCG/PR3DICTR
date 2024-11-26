import torch

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