import os
import glob
import json
import pickle
import numpy as np


icl_conditions = [
    'clean', 'lb', 'mb', 'hb',
    'cleanDutch', 'lbDutch', 'mbDutch', 'hbDutch',
    'fM-story', 'fM-distractor', 'fM-attention',
    'fW-story', 'fW-distractor', 'fW-attention'
]


if __name__ == '__main__':

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # load composite decoder LDA weights
    composite_decoder_path = os.path.join(os.path.join(config['root_results_dir'], "composite_decoder"))
    with open(composite_decoder_path, 'rb') as f:
        clf = pickle.load(f)

    averaged_env_outputs_dir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", "env"))
    averaged_ffr_outputs_dir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", "ffr", "composite_decoder.pkl"))
    results_savedir = os.path.join(os.path.join(config['root_results_dir'], 'decoder_evaluation_results'))


    # calculate accuracies for test split, heldout split
    n_avg = 100 # number of averaged decoders we are going to use
    results_dict = {}

    for condition in icl_conditions:

        results_dict[condition] = {}

        for sub in range(20):
            
            results_dict[condition][f'sub-{sub:03d}'] = {
                'envelope': {'predicted_labels':[], 'targets':[]},
                'ffr': {'predicted_labels':[], 'targets':[]},
                'composite': {'predicted_labels':[], 'targets':[]}
                }


            averaged_env_outputs_files = glob.glob(
                os.path.join(averaged_env_outputs_dir, f'env_-_avg-{n_avg:03d}_-_heldout_-_sub-{sub:03d}*_{condition}_outputs.npz')
                )
            averaged_ffr_outputs_files = list(map(lambda x: x.replace('env', 'ffr'), averaged_env_outputs_files))


            for i in range(len(averaged_env_outputs_files)):

                env_data = np.load(averaged_env_outputs_files[i])
                ffr_data = np.load(averaged_ffr_outputs_files[i])

                env_predictions = env_data['predictions']
                ffr_predictions = ffr_data['predictions']
                targets = env_data['targets']
                max_len = min(len(env_predictions, ffr_predictions))

                results_dict[condition][f'sub-{sub:03d}']['envelope']['predicted_labels'].append(env_predictions[:max_len]>0)
                results_dict[condition][f'sub-{sub:03d}']['envelope']['targets'].append(env_data['targets'])
                results_dict[condition][f'sub-{sub:03d}']['ffr']['predicted_labels'].append(ffr_predictions[:max_len]>0)
                results_dict[condition][f'sub-{sub:03d}']['ffr']['targets'].append(env_data['targets'])

                lda_features = np.vstack([env_predictions[:max_len], ffr_predictions[:max_len]]).T
                results_dict[condition][f'sub-{sub:03d}']['composite']['predicted_labels'].append(clf.predict(lda_features).squeeze())
                results_dict[condition][f'sub-{sub:03d}']['composite']['targets'].append(env_data['targets'])

            
            for decoder_key in results_dict[condition][f'sub-{sub:03d}']:
                
                hits = np.concatenate(results_dict[condition][f'sub-{sub:03d}'][decoder_key]['predicted_labels']) == \
                    np.concatenate(results_dict[condition][f'sub-{sub:03d}'][decoder_key]['targets'])
                
                n_observations = len(hits)
                accuracy = hits.sum()/len(hits)

                results_dict[condition][f'sub-{sub:03d}'][decoder_key] = {
                    'n_obs': n_observations,
                    'acc': accuracy
                }

            
        with open(os.path.join(results_savedir, f'icl_heldout_results.json')) as f:
            json.dump(results_dict, f, indent=4)