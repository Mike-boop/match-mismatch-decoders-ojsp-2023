import os
import glob
import json
import random
import numpy as np


def get_average_decoder_outputs(
        n_avg,
        decoder_outputs_dir,
        averaged_outputs_savedir,
        use_baseline=False,
        draw_random_instances=False
        ):

    # GET ALL AVAILABLE DECODER OUTPUTS FILES
    outputs_files = glob.glob(os.path.join(decoder_outputs_dir, f'*.npz'))

    if use_baseline:
        outputs_files = filter(lambda x: 'baseline' in x, outputs_files)
        prefix = 'baseline'
    else:
        outputs_files = list(filter(lambda x: 'baseline' not in x, outputs_files))
        prefix = os.path.basename(decoder_outputs_dir)

    # EXTRACT AVAILABLE EEG SESSIONS
    sessions = ['_-_'.join(os.path.basename(x).split('_-_')[2:]) for x in outputs_files]

    # AVERAGE THE OUTPUTS OF THE VARIOUS DECODER INSTANCES
    for each_session in sessions:

        output_files_all_instances = sorted(glob.glob(os.path.join(decoder_outputs_dir, f'{prefix}_-_*_-_{each_session}')))
        assert len(output_files_all_instances) >= n_avg

        if draw_random_instances:
            random.shuffle(output_files_all_instances)

        output_files = output_files_all_instances[:n_avg]

        outputs_n_instances = [np.load(x)['predictions'] for x in output_files]
        targets_n_instances = [np.load(x)['targets'] for x in output_files]
        assert all([np.all(x == targets_n_instances[0]) for x in targets_n_instances])
        targets = targets_n_instances[0]

        avg_outputs = np.mean(outputs_n_instances, axis=0)
        np.savez(
            os.path.join(averaged_outputs_savedir, f'{prefix}_-_avg-{n_avg:03d}' + '_-_' + each_session),
            predictions=avg_outputs, targets=targets
            )

if __name__ == '__main__':

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # change the following settings to obtain the averaged predictions on e.g. the heldout datasets
    n_avg = 1
    response = 'env'
    use_baseline = False
    decoder_outputs_dir = os.path.join(os.path.join(config['root_results_dir'], "decoder_outputs_sparrkulee", response))
    averaged_outputs_savedir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", response))

    get_average_decoder_outputs(
        n_avg,
        decoder_outputs_dir,
        averaged_outputs_savedir,
        use_baseline=use_baseline
        )
    

    n_avg = 1
    response = 'ffr'
    use_baseline = False
    decoder_outputs_dir = os.path.join(os.path.join(config['root_results_dir'], "decoder_outputs_sparrkulee", response))
    averaged_outputs_savedir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", response))

    get_average_decoder_outputs(
        n_avg,
        decoder_outputs_dir,
        averaged_outputs_savedir,
        use_baseline=use_baseline
        )