import torch

def calc_mixup_loss(output_dict, targets, loss_function, mixup_indexes, mixup_lambda):
    """
    Compute the loss for the mixup training strategy.
    Args:
        output_dict (dict): the model output dictionary
        targets (torch.Tensor): the target tensor
        loss_function (torch.nn.Module): the loss function
        mixup_indexes (torch.Tensor): the indexes for the mixup
        mixup_lambda (float): the mixup lambda value
    Returns:
        loss (torch.Tensor): the loss value
    """

    y_a_loss = loss_function(output_dict, targets)
    y_b_loss = loss_function(output_dict, targets[mixup_indexes])

    loss = (mixup_lambda * y_a_loss) + ((1 - mixup_lambda) * y_b_loss)

    return loss