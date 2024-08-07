import torch


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
