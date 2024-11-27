import torch

ENSEMBLE_MEMBERS = 5
TOXICITIES = ['Xerostomia_M06', 'Xerostomia_M06_slim']

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.15
TEST_SPLIT = 0.15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

VERBOSE = True