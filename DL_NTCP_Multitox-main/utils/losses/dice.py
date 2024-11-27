import torch

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
