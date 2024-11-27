





def move_batch_to_device(batch, device):
    # batch is a dictionary
    inputs, clinical_features, targets = batch['input'], batch['features'], batch['label_list']
    inputs = inputs.to(device=device)
    clinical_features = clinical_features.to(device=device)
    targets = targets.to(device=device)
    return inputs, clinical_features, targets