# Common utils.
# Contributer(s): Neil Z. Shao
# All rights reserved 2022.
import os
from collections import namedtuple

########################################
def set_seed(seed=0):
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass

    try:
        import random
        random.seed(seed)
    except:
        pass

    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass
