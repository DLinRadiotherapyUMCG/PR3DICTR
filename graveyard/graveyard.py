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






















def load_dataset_total(config, patient_ids = None):    # TODO: re-name to 'load_dataframes' ?
    """
    Load the all datasets with handling the options in
    the config file. This includes the csvFile
    """
    # Get paths of the files
    trainfile = os.path.join(config['paths']['csv'], config['data']['trainfile'])
    valfile = os.path.join(config['paths']['csv'], config['data']['valfile'])

    # There are 3 options to get the train, validation data
    #   1) There are 2 seperate files
    #   2) Single file that contains splitVar
    #   3) Single file with no splitVar will custom be split in --> train,var,test (warning: test need to be unique by seed --> need to check)

    # Check if 1 single file is given (testData is not always available)
    testDf = pd.DataFrame()
    if(trainfile == valfile):
        # Single file to split
        delimiterFound = get_delimiter(trainfile)
        totalDf = pd.read_csv(trainfile, delimiter=delimiterFound, dtype={'PatientID': str})
        totalDf = removePtnsExcluded(totalDf, config)
        totalDf = ValidateImageDataExists(config, totalDf)
        
        if patient_ids:
            totalDf = totalDf[totalDf['PatientID'].isin(patient_ids)]

        if(config['data']['splitvar'] != ""):
            splitVar = config['data']['splitvar']
            trainDf = totalDf[totalDf[splitVar] == "Train"]
            valDf = totalDf[totalDf[splitVar] == "Val"] 
            testDf = totalDf[totalDf[splitVar] == "Test"] 

            # BUG: This is how Daniel defined the splits
            if len(trainDf) == 0 and len(valDf) == 0:
                trainDf = totalDf[totalDf[splitVar] == "train_val"]
                testDf = totalDf[totalDf[splitVar] == "test"] 

                print("length test set", len(testDf))


            #if(config['data']['equalizer']['isEnabled']):
            #    trainDf = label_equalizer(trainDf, config)
        else:    
            # Need to split manual
            trainDf,valDf,testDf = data_split(totalDf, config, split=[0.7,0.15,0.15])

    else:
        # Two seperate files that are allready split
        delimiterFound = get_delimiter(trainfile)
        trainDf = pd.read_csv(trainfile, delimiter=delimiterFound, dtype={'PatientID': str})
        delimiterFound = get_delimiter(valfile)
        valDf = pd.read_csv(valfile, delimiter=delimiterFound, dtype={'PatientID': str})

    # Write information about data
    #print(f"Patient collection --> Train: {trainDf.shape[0]}, Validation: {valDf.shape[0]}")
    #if(testDf.shape[0] != 0):all you ne
    #    print(f"Patient collection --> Test: {testDf.shape[0]}")


    # if in test mode, and we want to use a subset of the data, then subsample the total dataset
    if config['general']['testMode'] and "n_patients_total" in config['data']:
        num_patients_sample = config['data']['n_patients_total']
        trainDf, valDf, testDf = subsample_datasets(num_patients_sample, trainDf, valDf, testDf)
        #print(len(trainDf), len(valDf), len(testDf))
    
    # Check and validate if KFolds settings are active
    trainDataset_Collection = []
    valDataset_Collection = []
    testDataset_Collection = []
    if(config["data"]["kFolds"]["isEnabled"] and config["data"]["kFolds"]["Iterations"] <= config["data"]["kFolds"]["Splits"]):
        # Multiple training and val datasets
        mergeDf = pd.concat([trainDf,valDf])
        labels = mergeDf[config['columns']['label']]
        # encode the labels (makes it possible to use StratifiedKFold for multi-label problems, as it only works on binary or multi-class)
        encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in labels.values])
         
        skf = StratifiedKFold(n_splits=config["data"]["kFolds"]["Splits"], shuffle=True, random_state=config["general"]["seed"])
        for i, (train_index, val_index) in enumerate(skf.split(mergeDf,encoded_labels)):
            trainDf_sel = mergeDf.iloc[train_index]
            if(config['data']['equalizer']['isEnabled']):
                trainDf_sel = label_equalizer(trainDf_sel, config)
            valDf_sel = mergeDf.iloc[val_index]

            # Check sanity is correct
            if(Complete_SanityCheck(config,[trainDf_sel,valDf_sel,testDf])):
                raise Exception("ABORT: Datasets contain identical patients! NOT ALLOWED!")

            trainDataset_Collection.append(trainDf_sel)
            valDataset_Collection.append(valDf_sel)
            testDataset_Collection.append(testDf)


            if(i == config["data"]["kFolds"]["Iterations"] - 1):
                break
    else:   
        # Single train and val dataset
        if(config['data']['equalizer']['isEnabled']):
                trainDf = label_equalizer(trainDf, config)
        if(Complete_SanityCheck(config,[trainDf,valDf,testDf])):
                raise Exception("ABORT: Datasets contain identical patients! NOT ALLOWED!")
        
        if(config['general']['testMode'] and trainDf.shape[0] > 100):
            # Only use 100 patients for training dataset
            trainDf = trainDf.iloc[:100]

        trainDataset_Collection.append(trainDf)
        valDataset_Collection.append(valDf)
        testDataset_Collection.append(testDf)
    

    logging.info(f"Patient amount in datasets: Train = {trainDataset_Collection[0].shape[0]}, Validation = {valDataset_Collection[0].shape[0]}, Test = {testDataset_Collection[0].shape[0]}")

    return [trainDataset_Collection, valDataset_Collection, testDataset_Collection]#, metadata



