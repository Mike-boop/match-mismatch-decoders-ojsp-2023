import os
import json
import glob
import torch

import numpy as np

from torch import sigmoid
from torch.utils.data import DataLoader

from experiments.decoders.decoders import DilatedConvNet, DilatedConvNetSymmetrisedOutputs
from experiments.decoders.utils import SessionDataset

torch.set_default_dtype(torch.float32)

    
def get_decoder_outputs_for_session(decoder, eeg_session_file, feature, fs=64, device='cuda'):

    session_dataset = SessionDataset(eeg_session_file, feature_name=feature, fs=fs, dtype=np.float32)
    # if outputs are to be averaged, important not to shuffle the data!
    loader = DataLoader(session_dataset, batch_size=512, num_workers=1, shuffle=False)

    predictions = []
    targets = []

    with torch.no_grad():

        for (eeg, m, mm) in loader:

            eeg = torch.transpose(eeg.to(device, dtype=torch.float), 1, 2)
            m = torch.transpose(m.to(device, dtype=torch.float), 1, 2)
            mm = torch.transpose(mm.to(device, dtype=torch.float), 1, 2)

            targ = torch.hstack(
                [torch.ones((m.shape[0],), device=device), torch.zeros((m.shape[0],), device=device)]
                )

            pred = decoder(torch.vstack([eeg, eeg]), torch.vstack([m, mm]), torch.vstack([mm, m])).flatten()
            pred = sigmoid(pred)
            predictions.append(pred.cpu())
            targets.append(targ.cpu())

    return np.hstack(predictions), np.hstack(targets)


def get_all_decoder_outputs(
        decoder_ckpt_path,
        split_data_dir,
        outputs_savedir,
        device='cuda',
        split='test'):

    # DECODER INFO FROM CKPT FNAME
    instance = os.path.basename(decoder_ckpt_path).split('inst-')[1]
    instance = int(instance.replace('.pt', ''))
    response = os.path.basename(os.path.dirname(os.path.dirname(decoder_ckpt_path)))


    # GET FEATURE AND SAMPLE RATE
    if response == 'env':
        feature = 'env'
        fs = 64
    elif response == 'ffr':
        feature = 'mods'
        fs = 512
    else:
        raise ValueError('Invalid response type!')

    # LOAD WEIGHTS
    if 'baseline' in decoder_ckpt_path:
        decoder = DilatedConvNet()
        output_files_prefix = f'baseline_-_{instance:02d}'
    else:
        decoder = DilatedConvNetSymmetrisedOutputs()
        output_files_prefix = f'{response}_-_{instance:02d}'
        
    decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location=torch.device(device)))
    decoder.to(device)

    # GET DECODER OUTPUTS
    session_files = glob.glob(os.path.join(split_data_dir, f'{split}_-_*_-_eeg.npy'))

    # GET OUTPUTS FOR ALL EEG SESSIONS
    for eeg_session_file in session_files:
        predictions, targets = get_decoder_outputs_for_session(decoder, eeg_session_file, feature, fs=fs, device='cuda')

        np.savez(
            os.path.join(outputs_savedir, output_files_prefix+'_-_'+os.path.basename(eeg_session_file).replace('eeg', 'outputs').replace('.npy', '.npz')),
            predictions=predictions,
            targets=targets
            )


if __name__ == '__main__':

    use_baseline = False
    response = 'env'
    seed = 0

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)


    # get the outputs for the test portion of the development dataset
    for model, response in [('baseline', 'env'), ('env', 'env'), ('ffr', 'ffr')]:

        split_data_dir = os.path.join(config['root_results_dir'], "split_data_sparrkulee", response)
        outputs_savedir = os.path.join(config['root_results_dir'], 'decoder_outputs_sparrkulee', response)
        os.makedirs(outputs_savedir, exist_ok=True)
        trained_models_dir = os.path.join(config['root_results_dir'], response, 'trained_models')

        for each_checkpoint in glob.glob(os.path.join(trained_models_dir, f'{model}*.pt')):

            get_all_decoder_outputs(
                each_checkpoint,
                split_data_dir,
                outputs_savedir,
                device='cuda',
                split='test')
            

    # get the outputs for the heldout dataset
    for model, response in [('baseline', 'env'), ('env', 'env'), ('ffr', 'ffr')]:

        split_data_dir = os.path.join(config['root_results_dir'], "heldout_data_sparrkulee", response)
        outputs_savedir = os.path.join(config['root_results_dir'], 'decoder_outputs_sparrkulee', response)
        os.makedirs(outputs_savedir, exist_ok=True)
        trained_models_dir = os.path.join(config['root_results_dir'], response, 'trained_models')

        for each_checkpoint in glob.glob(os.path.join(trained_models_dir, f'{model}*.pt')):

            get_all_decoder_outputs(
                each_checkpoint,
                split_data_dir,
                outputs_savedir,
                device='cuda',
                split='heldout')
            

    # get the outputs for the icl dataset
    for model, response in [('baseline', 'env'), ('env', 'env'), ('ffr', 'ffr')]:

        split_data_dir = os.path.join(config['root_results_dir'], "heldout_data_icl", response)
        outputs_savedir = os.path.join(config['root_results_dir'], 'decoder_outputs_icl', response)
        os.makedirs(outputs_savedir, exist_ok=True)
        trained_models_dir = os.path.join(config['root_results_dir'], response, 'trained_models')

        for each_checkpoint in glob.glob(os.path.join(trained_models_dir, f'{model}*.pt')):

            get_all_decoder_outputs(
                each_checkpoint,
                split_data_dir,
                outputs_savedir,
                device='cuda',
                split='heldout')