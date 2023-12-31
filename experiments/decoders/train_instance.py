import os
import json
import glob
import torch
import random

import numpy as np

from torch.utils.data import ConcatDataset, DataLoader
from torchtnt.utils.loggers import CSVLogger

from experiments.decoders.decoders import DilatedConvNet, DilatedConvNetSymmetrisedOutputs
from experiments.decoders.utils import (
    get_session_datasets_from_session_files, training_loop, validation_loop, init_weights
)

torch.set_default_dtype(torch.float32)

    
def testing_loop(model, loss_fn, split_data_dir, test_subjects, feature_name, fs=64, sub_specific_stimuli=False, device='cuda'):

    results_dict = {}

    for subject in test_subjects:
        
        subject_session_files = glob.glob(os.path.join(split_data_dir, f'test_-_sub-{subject:03d}*eeg.npy'))
        
        if len(subject_session_files) == 0:
            continue

        test_datasets = get_session_datasets_from_session_files(subject_session_files,feature_name, fs=fs)
        test_datasets = ConcatDataset(test_datasets)
        test_loader = DataLoader(test_datasets, batch_size=64, num_workers=1)

        acc, loss = validation_loop(model, loss_fn, test_loader, device=device)

        results_dict[f'sub-{subject:03d}'] = {'acc':acc, 'loss':loss}

    return results_dict


def train_instance(
        config_path,
        model,
        inst=0,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        response='env',
        device='cuda',
        train_kwargs = {'max_epochs':50, 'patience':5},
        eval_only=False
        ):
    
    # SEED
    random.seed(inst)
    np.random.seed(inst)
    torch.random.manual_seed(inst)


    # LOAD CONFIG
    with open(config_path, 'r') as f:
        config = json.load(f)

    # RESULTS FILENAMES
    if isinstance(model, DilatedConvNet): # if using baseline model
        checkpointfile = f'baseline_ckpt_inst-{inst:02d}.pt'
        logfile = f'baseline_training_log_inst-{inst:02d}.csv'
        eval_file = f'baseline_eval_inst-{inst:02d}.csv'
    else:
        checkpointfile = f'{response}_ckpt_inst-{inst:02d}.pt'
        logfile = f'{response}_training_log_inst-{inst:02d}.csv'
        eval_file = f'{response}_eval_inst-{inst:02d}.csv'

    # DATA DIRECTORIES
    split_data_dir = os.path.join(config['root_results_dir'], "split_data_sparrkulee", response)
    model_savedir = os.path.join(config['root_results_dir'], response, 'trained_models')
    os.makedirs(model_savedir, exist_ok=True)
    logger = CSVLogger(os.path.join(model_savedir, logfile))

    # GET FEATURE AND SAMPLE RATE
    if response == 'env':
        feature = 'env'
        fs = 64
    elif response == 'ffr':
        feature = 'mods'
        fs = 512
    else:
        raise ValueError('Invalid response type!')
    

    # PREPARE DNN, OPTIMIZER, EARLY STOPPING
    model = model.to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)

    best_loss = np.inf
    patience_counter=0
    best_weights = None
    

    if not eval_only:
    
        # LOAD DATA
        train_datasets = get_session_datasets_from_session_files(
            glob.glob(os.path.join(split_data_dir, 'train*eeg.npy')),
            feature_name=feature,
            shuffle=True,
            fs=fs
        )

        val_datasets = get_session_datasets_from_session_files(
            glob.glob(os.path.join(split_data_dir, 'val*eeg.npy')),
            feature_name=feature,
            shuffle=True,
            fs=fs
        )

        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)

        val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=1, pin_memory=True, drop_last=True)


        # PERFORM TRAINING
        for epoch in range(train_kwargs['max_epochs']):

            # step epoch
            model_state_dict, train_acc, train_loss = training_loop(model,
                                                                    optimizer,
                                                                    loss_fn,
                                                                    train_loader=train_loader,
                                                                    tqdm_description_preamble=f"Epoch [{epoch}/{train_kwargs['max_epochs']}]",
                                                                    device=device)
            scheduler.step()
            val_acc, val_loss = validation_loop(model, loss_fn, val_loader, device=device)
            
            # logging
            logger.log('val_acc', val_acc, epoch)
            logger.log('val_loss', val_loss, epoch)
            logger.log('train_acc', val_acc, epoch)
            logger.log('train_loss', val_loss, epoch)

            # early stopping 
            if val_loss >= best_loss:
                patience_counter+=1

                if patience_counter >= train_kwargs['patience']:
                    break

            else:
                patience_counter = 0
                best_loss = val_loss
                best_weights = model_state_dict

                # save ckpt
                torch.save(best_weights, os.path.join(model_savedir, checkpointfile))


    # EVALUATE
    model.load_state_dict(torch.load(os.path.join(model_savedir, checkpointfile), map_location=torch.device(device)))
    
    test_subjects = np.arange(1, 72)
    results_dict = testing_loop(model, loss_fn, split_data_dir, test_subjects, feature, fs=fs, device=device)
    print('mean: ', np.mean([results_dict[k]['acc'] for k in results_dict]))

    with open(os.path.join(model_savedir, eval_file), 'w') as f:
        json.dump(results_dict, f, indent=4)



if __name__ == '__main__':

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    model = DilatedConvNetSymmetrisedOutputs()
    inst = 0

    response = 'env'
    eval_only = False

    # train model and evaluate on test split of development dataset
    train_instance(config_path,
                model,
                inst=inst,
                response=response,
                eval_only=eval_only)