# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:47:47 2024

@author: HoekL02
"""

#%% 

class HNCDataset(Dataset):
    """
    Dataset class for the HNC dataset.
    Returns a tuple of image_stack, clinical_features, and label.
    """

    def __init__(self, csv_path, config, patient_ids, augment = False, split = False, train = True, splitVar = "Split"):
        self.images_path, self.label_column = config['paths']['images'], config['columns']['label']
        self.config = config
        self.augment = augment

        # Read the csv file
        delimiterFound = get_delimiter(csv_path)
        df = pd.read_csv(csv_path, delimiter=delimiterFound, dtype={'PatientID': str})


        # if(train):
        #     df = df[df[splitVar] == "train"]
        # else:
        #     df = df[df[splitVar] == "val"]     

        # Filter the data based on the patient ids, if provided.
        if patient_ids:
            df = df[df['PatientID'].isin(patient_ids)]

        self.df = df

        # Define the MONAI transformations
        prob = config['augmentation']['prob']
        strength = config['augmentation']['strength']

        self.transforms = augmentation(config)

        for transform in self.transforms.transforms:
            transform.set_random_state(seed=config['general']['seed'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        patient_id = self.df.iloc[idx]['PatientID']
        # Construct the file paths for the CT, dose, and segmentation map images
        ct_path = self.images_path + str(patient_id).rjust(7,'0') + '/ct.npy'
        dose_path = self.images_path + str(patient_id).rjust(7,'0') + '/rtdose.npy'
        segmentation_map_path = self.images_path + str(patient_id).rjust(7,'0') + '/segmentation_map.npy'

        # Load the images
        ct = np.load(ct_path)
        dose = np.load(dose_path)
        segmentation_map = np.load(segmentation_map_path)

        # Need to be 4 dimensions
        if(len(ct.shape) == 3):
            ct = np.expand_dims(ct,axis = 0)
            dose = np.expand_dims(dose,axis = 0)
            segmentation_map = np.expand_dims(segmentation_map,axis = 0)

        # Stack the images vertically
        image_stack = np.vstack((ct, dose, segmentation_map), dtype=np.float32)

        # Preprocess the image_stack
        image_stack = preprocess_image_stack(image_stack, self.config).astype(np.float32)

        # Apply the MONAI transformations
        if self.augment:
            image_stack = self.transforms(image_stack)

        # If cropping is enabled, crop the image
        if self.config['data']['preprocessing']['crop']:
            cropped_image_stack = np.empty((image_stack.shape[0], *self.config['data']['preprocessing']['crop_shape']))
            for i in range(image_stack.shape[0]):
                cropped_image_stack[i] = center_crop_3d(image_stack[i], self.config['data']['preprocessing']['crop_shape'])
            image_stack = cropped_image_stack.astype(np.float32)

        # Get the label
        label = np.array([self.df[self.df['PatientID'] == patient_id][self.label_column].values[0]]).astype(np.float32)

        # Get the clinical features
        clinical_features = [self.df[self.df['PatientID'] == patient_id][feature].values[0] for feature in
                             self.config['columns']['clinical_features']]
        clinical_features = np.array(clinical_features).astype(np.float32)

        return image_stack, clinical_features, label
    
def load_dataset(config, csv_path, patient_ids = None, augment= False, split = False, train = True, splitVar = "Split"):
    """
    Loads data for a single csv file.
    :param csv_path:
    :param config:
    :param patient_ids:
    :return: PyTorch Dataset and metadata
    """    
    # Create an instance of the HNCDataset
    dataset = HNCDataset(csv_path, config, patient_ids, augment=augment, 
                         split = split, train = train, splitVar = splitVar)

    # Get an example input to determine the metadata
    example_input, _, _ = dataset[0]
    channels, depth, height, width = example_input.shape

    n_features = len(config['columns']['clinical_features'])

    metadata = {
        "channels": channels,
        "depth": depth,
        "height": height,
        "width": width,
        "n_features": n_features,
    }

    # Return the dataset and the metadata
    return dataset, metadata