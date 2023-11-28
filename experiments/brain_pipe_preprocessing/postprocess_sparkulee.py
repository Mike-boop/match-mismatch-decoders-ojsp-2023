""" Code to split the EEG data in training, validation, testing, heldout subsets """
import os
import json
import glob

import numpy as np
from scipy.stats import zscore


if __name__ == '__main__':


    response = 'ffr' # change to 'env' if postprocessing env data

    if response == 'ffr':
        fs = 512
        feature = 'mods'
    elif response == 'env':
        fs = 64
        feature = 'env'
    else:
        raise ValueError('invalid response type')
    

    # configure directories
    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    sparrkulee_download_dir = config['sparrkulee_download_dir']
    preprocessed_data_dir = os.path.join(config['root_results_dir'], "preprocessed_data_sparrkulee", response)
    split_data_dir = os.path.join(config['root_results_dir'], "split_data_sparrkulee", response)
    heldout_data_dir = os.path.join(config['root_results_dir'], "heldout_data_sparrkulee", response)

    os.makedirs(split_data_dir, exist_ok=True)
    os.makedirs(heldout_data_dir, exist_ok=True)

    # 1. development dataset splitting
    dev_subjects = np.arange(1,72)
    dev_eeg_files = sum(
        [glob.glob(os.path.join(preprocessed_data_dir, f'sub-{s:03d}*eeg.npy')) for s in dev_subjects], []
    )
    dev_eeg_files = list(map(lambda x: os.path.basename(x), dev_eeg_files))
    dev_stimuli = map(lambda x: x.split('_-_')[3], list(dev_eeg_files))

    # dev data splitting & normalisation
    for each_stimulus in dev_stimuli:

        if 'artefact' in each_stimulus:
            continue

        # extract the splitting indices from the raw speech file. This ensures that the split data are 
        # aligned for different sample rates (required for composite decoder!).
        dev_speech_file = os.path.join(sparrkulee_download_dir, 'stimuli', 'eeg', f'{each_stimulus}.npz')
        speech_data = np.load(dev_speech_file)
        length, stimulus_fs = len(speech_data['audio']), speech_data['fs']

        train_onset, train_offset = int(fs*length*0.0/stimulus_fs), int(fs*length*0.8/stimulus_fs)
        val_onset, val_offset   = int(fs*length*0.8/stimulus_fs), int(fs*length*0.9/stimulus_fs)
        test_onset, test_offset = int(fs*length*0.9/stimulus_fs), int(fs*length*1.0/stimulus_fs)

        # normalise, then split data
        dev_feature_file = os.path.join(preprocessed_data_dir, f'{each_stimulus}_-_{feature}.npy')
        dev_data_files = [dev_feature_file] + list(filter(lambda x: f'{each_stimulus}_-_' in x, dev_eeg_files))
        for each_dev_data_file in dev_data_files:

            dev_data = zscore(np.load(os.path.join(preprocessed_data_dir, each_dev_data_file)), axis=0)

            np.save(os.path.join(split_data_dir, 'train_-_'+os.path.basename(each_dev_data_file)), dev_data[train_onset:train_offset])
            np.save(os.path.join(split_data_dir, 'val_-_'+os.path.basename(each_dev_data_file)), dev_data[val_onset:val_offset])
            np.save(os.path.join(split_data_dir, 'test_-_'+os.path.basename(each_dev_data_file)), dev_data[test_onset:test_offset])

    
    # 2. heldout data normalisation
    heldout_subjects = np.arange(72,86)
    heldout_eeg_files = sum(
        [glob.glob(os.path.join(preprocessed_data_dir, f'sub-{s:03d}*eeg.npy')) for s in heldout_subjects], []
    )
    heldout_eeg_files = list(map(lambda x: os.path.basename(x), heldout_eeg_files))
    heldout_stimuli = map(lambda x: x.split('_-_')[3], heldout_eeg_files)

    for each_heldout_file in list(heldout_eeg_files):

        heldout_data = zscore(np.load(os.path.join(preprocessed_data_dir, each_heldout_file)), axis=0)
        np.save(os.path.join(heldout_data_dir, 'heldout_-_'+os.path.basename(each_heldout_file)), heldout_data)

    for each_heldout_stimulus in list(heldout_stimuli):

        heldout_stim_file = each_heldout_stimulus+f'_-_{feature}.npy'

        heldout_data = zscore(np.load(os.path.join(preprocessed_data_dir, heldout_stim_file)), axis=0)
        np.save(os.path.join(heldout_data_dir, 'heldout_-_'+heldout_stim_file), heldout_data)