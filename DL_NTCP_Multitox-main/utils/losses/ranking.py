import torch


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