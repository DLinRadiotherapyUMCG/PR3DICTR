# Toxicity Prediction Uncertainty

## Scripts

### train_dl.py

Perform a training run of the model on the full training set.

### train_dl_ensemble.py

Perform a training run of the multi-model ensemble on the bootstrapped training sets.

### fit_lr.py

Fit a logistic regression model to the full dataset.

### fit_lr_ensemble.py

Fit multiple logistic regression models to the bootstrapped training sets.

### test_dl_ensemble.py

Test the multi-model ensemble on the test set. Saves the predictions to a csv file.

### test_lr_ensemble.py

Test the logistic regression ensemble on the test set. Saves the predictions to a csv file.

### make_datasets.py

This script is used to generate the datasets for training a model.
For each toxicity, it generates one test split, one validation split, and 6 training splits: one full training split, and 5 bootstrapped training splits.

Data is read from the `private/dataset` folder.
This folder should contain a subfolder with a csv file with patients for each toxicity, containing all features.
Additionally, it should contain a subfolder with a folder for each patient, containing their CT, RTDOSE, and Segmentation map.

Example:
```
private
└── dataset
    ├── per_toxicity_csv
    │   ├── toxicity1.csv
    │   ├── toxicity2.csv
    │   └── ...
    └── patients
        ├── 0000001
        │   ├── ct.npy
        │   ├── rtdose.npy
        │   └── segmentation_map.npy
        ├── 0000002
        │   ├── ct.npy
        │   ├── rtdose.npy
        │   └── segmentation_map.npy
        └── ...
```

The processed datasets are stored in the `private/bagged_datasets` folder.

