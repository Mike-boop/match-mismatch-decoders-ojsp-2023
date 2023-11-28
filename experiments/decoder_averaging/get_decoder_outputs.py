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

    
def get_decoder_outputs(decoder: torch.nn.Module, eeg_session_file, feature, fs=64, device='cuda'):

    session_dataset = SessionDataset(eeg_session_file, feature_name=feature, fs=fs, dtype=np.float32)
    loader = DataLoader(session_dataset, batch_size=512, num_workers=1)

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


if __name__ == '__main__':

    use_baseline = False
    response = 'env'
    seed = 0

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)


    if use_baseline:
        model_handle = DilatedConvNet
        checkpointfile = f'baseline_ckpt_{seed:02d}.pt'
        logfile = f'baseline_training_log_{seed:02d}.csv'
        eval_file = f'baseline_eval_{seed:02d}.csv'
    else:
        model_handle = DilatedConvNetSymmetrisedOutputs
        checkpointfile = f'{response}_ckpt_{seed:02d}.pt'
        logfile = f'{response}_training_log_{seed:02d}.csv'
        eval_file = f'{response}_eval_{seed:02d}.csv'


    if response == 'env':
        feature = 'envelope'
        fs = 64
    elif response == 'ffr':
        feature = 'modulations'
        fs = 512
    else:
        raise ValueError('Invalid response type!')


    # set up directories
    split_data_dir = os.path.join(config['root_results_dir'], "split_data_sparrkulee", response)
    model_savedir = config['results_dir'][response]['trained_models']
    outputs_savedir = os.path.join(os.path.join(config['root_results_dir'], "decoder_outputs_sparrkulee", response))

    eeg_session_files = glob.glob(os.path.join(split_data_dir, 'test*eeg.npy'))
    

    # experiment settings
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = model_handle()
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_savedir, checkpointfile), map_location=torch.device(device)))


    # get the sigmoid outputs

    for each_file in eeg_session_files:

        predictions, targets = get_decoder_outputs(model, each_file, feature, fs=fs, device=device)

        if use_baseline:
            np.savez(
                os.path.join(outputs_savedir, f'inst-{seed:02d}_-_baseline_-_'+os.path.basename(each_file).replace('eeg', 'outputs')),
                predictions=predictions,
                targets=targets
                )
        else:
            np.savez(
                os.path.join(outputs_savedir, f'inst-{seed:02d}_-_{response}_-_'+os.path.basename(each_file).replace('eeg', 'outputs')),
                predictions=predictions,
                targets=targets
                )