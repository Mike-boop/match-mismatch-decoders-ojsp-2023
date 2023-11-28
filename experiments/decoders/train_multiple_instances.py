import os
import json
import glob
import torch
import random

import numpy as np

from torch.utils.data import ConcatDataset, DataLoader
from torchtnt.utils.loggers import CSVLogger

from experiments.decoders.decoders import DilatedConvNet, DilatedConvNetSymmetrisedOutputs
from experiments.decoders.train_instance import train_instance
from experiments.decoders.utils import (
    get_session_datasets_from_session_files, training_loop, validation_loop, init_weights
)



if __name__ == '__main__':

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    model = DilatedConvNetSymmetrisedOutputs()

    response = 'env'
    use_baseline = False
    eval_only = False

    for inst in range(100):

        # baseline model
        train_instance(config_path,
                       DilatedConvNet(),
                       inst=inst,
                       response='env')
        
        # envelope-based model
        train_instance(config_path,
                       DilatedConvNetSymmetrisedOutputs(),
                       inst=inst,
                       response='env')
        
        # ffr-based model
        train_instance(config_path,
                       DilatedConvNetSymmetrisedOutputs(),
                       inst=inst,
                       response='ffr')