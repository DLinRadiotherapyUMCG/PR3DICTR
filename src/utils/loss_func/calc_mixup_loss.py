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
        loss_dict (dict): a dictionary with the loss per endpoint
    """

    A_loss_val, A_loss_dict = loss_function(output_dict, targets)
    B_loss_val, B_loss_dict = loss_function(output_dict, targets[mixup_indexes])

    # loss value
    loss = (mixup_lambda * A_loss_val) + ((1 - mixup_lambda) * B_loss_val)

    # loss dictionary (loss per endpoint)
    loss_dict = {}
    for key in A_loss_dict.keys():
        loss_dict[key] = (mixup_lambda * A_loss_dict[key]) + ((1 - mixup_lambda) * B_loss_dict[key])

    return loss, loss_dict