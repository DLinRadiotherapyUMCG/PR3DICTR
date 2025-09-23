def collect_metadata(dataloader):
    """
    Collects metadata about the dataset from a dataloader, including image dimensions, number of clinical features, and number of labels.
    Args:
        dataloader (DataLoader): a MONAI DataLoader object
    Returns:
        metadata (dict): a dictionary containing the metadata
    """
    # collect some metadata about the image dimensions, number of features, etc.
    example_data = next(iter(dataloader))
    
    #batch_size1, channels, depth, height, width = example_data['input'].shape
    batch_size2, n_features = example_data['features'].shape
    batch_size3, n_labels = example_data['label_list'].shape

    assert batch_size2 == batch_size3 # all batch sizes should be the same

    metadata = {
        "n_features": n_features,
        "n_labels": n_labels,
    }

    # collect images metadata only if images are present

    if 'input' in example_data.keys():
        batch_size1, channels, depth, height, width = example_data['input'].shape
    
        assert batch_size1 == batch_size2 # all batch sizes should be the same

        metadata["channels"] = channels
        metadata["depth"] = depth
        metadata["height"] = height
        metadata["width"] = width

    else:
        metadata["channels"] = 0
        metadata["depth"] = 0
        metadata["height"] = 0
        metadata["width"] = 0
    
    return metadata