import random

import numpy as np
import torch
from monai.utils import set_determinism


def set_random_seed(seed):
    """
    Set the random seed.
    :param seed:
    :return:
    """

    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
