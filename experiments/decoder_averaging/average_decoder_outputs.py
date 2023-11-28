import os
import glob
import json
import numpy as np

np.random.seed(0)


def get_averaged_outputs(outputs_folder, prefix, draw):



    outputs_files = [
        os.path.join(outputs_folder, f'{prefix}_-_{i:02d}_outputs.npz') for i in draw
    ]

    output_data = [np.load(x) for x in outputs_files]
    predictions = [x['predictions'] for x in output_data]
    targets = output_data[0]['targets']


    return np.mean(predictions, axis=0), targets


if __name__ == '__main__':

    response = 'env'

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)


    outputs_savedir = os.path.join(os.path.join(config['root_results_dir'], "decoder_outputs_sparrkulee", response))
    averaged_outputs_savedir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", response))

    outputs_files = glob.glob(os.path.join(outputs_savedir, f'*.npz'))
    sessions = ['_-_'.join(os.path.basename(x).split('_-_')[1:]) for x in outputs_files]

    for each_session in sessions:

        instances_outputs_files = glob.glob(os.path.join(outputs_savedir, 'inst-*' + each_session))

        targets = np.load(instances_outputs_files[0])['targets']
        session_outputs = [
            np.load(x)['predictions'] for x in instances_outputs_files
        ]
        average_outputs = np.mean(session_outputs, axis=0)

        np.save(os.path.join(averaged_outputs_savedir), f'n_avg-{len(instances_outputs_files)}')