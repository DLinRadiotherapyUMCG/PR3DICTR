import torch


def collect_all_preds_and_labels(labels, label_types, preds_dict, labels_dict, targets, outputs):
    sigmoid_act = torch.nn.Sigmoid()
    identity_act = torch.nn.Identity()

    
    lab_idx = 0
    for label in labels:
        if label_types[lab_idx] == 'Binary':
            endpoint_preds = sigmoid_act(outputs[label])
            endpoint_labels = targets[:, lab_idx].unsqueeze(-1)  # Get the binary label (single column)

            lab_idx += 1
                    
        elif label_types[lab_idx] == 'Event':
                        # If the label is an event, we need to reshape the predictions and targets
            endpoint_preds = identity_act(outputs[label])
            endpoint_labels = targets[:, lab_idx:lab_idx+2] #  Get both event and days labels
            
            lab_idx += 2
        
        # collect all of the predictions and labels 
        if len(preds_dict[label]) == 0:
            preds_dict[label] = endpoint_preds
            labels_dict[label] = endpoint_labels
        else:
            preds_dict[label] = torch.cat([preds_dict[label], endpoint_preds], dim=0)
            labels_dict[label] = torch.cat([labels_dict[label], endpoint_labels], dim=0)
        
    return preds_dict, labels_dict