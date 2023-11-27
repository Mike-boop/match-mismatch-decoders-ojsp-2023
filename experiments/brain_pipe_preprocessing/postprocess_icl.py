""" Code for cropping out the 'English sentences' which were embedded in the Dutch stories, and normalising data """
import os
import json
import glob

import numpy as np
from scipy.stats import zscore


if __name__ == '__main__':

    response = 'env' # change to 'env' if postprocessing env data

    if response == 'ffr':
        fs = 512
        feature = 'modulations'
    elif response == 'env':
        fs = 64
        feature = 'envelope'
    else:
        raise ValueError('invalid response type')
    

    # configure directories
    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    icl_download_dir = config['icl_download_dir']
    preprocessed_data_dir = os.path.join(config['root_results_dir'], "preprocessed_data_icl", response)
    heldout_data_dir = os.path.join(config['root_results_dir'], "heldout_data_icl", response)


    # crop out the english segments
    dutch_conditions = ['cleanDutch', 'lbDutch', 'mbDutch', 'hbDutch']

    for condition in dutch_conditions:

        # load the onsets and offsets of the english sentences, in samples at 44100Hz sample rate.
        # the sentences are very short (maybe 3s in duration), so you could exclude this step.
        english_onsets_info = json.load(open(os.path.join(icl_download_dir, 'audiobooks', condition, 'english_onsets_info.json'), 'r'))

        for part in range(1,5):

            english_onsets = english_onsets_info[f'part_{part}']["onsets"]
            english_offsets = english_onsets_info[f'part_{part}']["offsets"]

            # get the corresponding onsets/offsets of the Dutch portions of the data
            dutch_onsets = [0] + english_offsets
            dutch_offsets = english_onsets + [None]

            stimulus_file = os.path.join(preprocessed_data_dir, f'{condition}_part_{part}-story_-_{feature}.npy')
            eeg_files = os.path.join(preprocessed_data_dir, f'*{condition}_part_{part}-story_-_eeg.npy')

            # crop the eeg
            for each_eeg_file in eeg_files:

                eeg_data = np.load(each_eeg_file)
                
                eeg_segments = []
                for i in range(len(dutch_onsets)):
                    if dutch_offsets[i] is None:
                        eeg_segments.append(eeg_data[(dutch_onsets[i]*fs)//44100:])
                    else:
                        eeg_segments.append(eeg_data[(dutch_onsets[i]*fs)//44100:(dutch_offsets[i]*fs)//44100])

                cropped_eeg_data = np.vstack(eeg_segments)

                np.save(os.path.join(heldout_data_dir, 'heldout_-_' + os.path.basename(each_eeg_file)), zscore(cropped_eeg_data, axis=0))

        # crop the stimulus feature
        stimulus_data = np.load(stimulus_file)
            
        stimulus_segments = []
        for i in range(len(dutch_onsets)):
            if dutch_offsets[i] is None:
                stimulus_segments.append(stimulus_data[(dutch_onsets[i]*fs)//44100:])
            else:
                stimulus_segments.append(stimulus_data[(dutch_onsets[i]*fs)//44100:(dutch_offsets[i]*fs)//44100])

        cropped_stimulus_data = np.vstack(stimulus_segments)

        np.save(os.path.join(heldout_data_dir, 'heldout_-_' + os.path.basename(stimulus_file)), zscore(cropped_stimulus_data, axis=0))


    # normalise the other conditions
    english_conditions = ['clean', 'lb', 'mb', 'hb', 'fM', 'fW']
    english_data_files = glob.glob(os.path.join(preprocessed_data_dir, '*_condition_*.npy'))

    for file in english_data_files:

        data = np.load(file)
        np.save(os.path.join(heldout_data_dir, 'icl_-_' + os.path.basename(file)), zscore(data, axis=0))
