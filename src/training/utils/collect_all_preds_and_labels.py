import torch

def collect_all_preds_and_labels(labels, label_types, preds_dict, labels_dict, targets, outputs):
    """
    Function that collects all of the predictions and labels for each batch.
    The preds_dict and labels_dict are updated with the 'targets' and 'outputs' from the current batch.
    Args:
        labels (list): a list of label names
        label_types (list): a list of label types (e.g. 'Binary', 'Event')
        preds_dict (dict): a dictionary of predictions for each label
        labels_dict (dict): a dictionary of labels for each label
        targets (torch.Tensor): the targets for the current batch
        outputs (dict): a dictionary of model outputs for the current batch (for each label)
    Returns:
        preds_dict (dict): updated dictionary of predictions for each label
        labels_dict (dict): updated dictionary of labels for each label
    """

    sigmoid_act = torch.nn.Sigmoid()
    identity_act = torch.nn.Identity()

    targets_idx = 0

    for lab_idx, label in enumerate(labels):
        if label_types[lab_idx] == 'Binary':
            endpoint_preds = sigmoid_act(outputs[label])
            endpoint_labels = targets[:, targets_idx].unsqueeze(-1)  # Get the binary label (single column)

            targets_idx += 1
                    
        elif label_types[lab_idx] == 'Event':
            # If the label is an event, we need to reshape the predictions and targets
            endpoint_preds = identity_act(outputs[label])
            endpoint_labels = targets[:, targets_idx:targets_idx+2] #  Get both event and days labels
            
            targets_idx += 2

        # collect all of the predictions and labels 
        if len(preds_dict[label]) == 0:
            preds_dict[label] = endpoint_preds.detach()
            labels_dict[label] = endpoint_labels.detach()
        else:
            preds_dict[label] = torch.cat([preds_dict[label], endpoint_preds.detach()], dim=0)
            labels_dict[label] = torch.cat([labels_dict[label], endpoint_labels.detach()], dim=0)

    return preds_dict, labels_dict