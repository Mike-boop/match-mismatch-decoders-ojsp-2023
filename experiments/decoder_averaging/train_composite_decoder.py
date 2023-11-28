import os
import glob
import json
import pickle
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
                                       
if __name__ == '__main__':

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    averaged_env_outputs_dir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", "env"))
    averaged_ffr_outputs_dir = os.path.join(os.path.join(config['root_results_dir'], "averaged_outputs_sparrkulee", "ffr"))
    composite_decoder_savedir = os.path.join(os.path.join(config['root_results_dir'], "composite_decoder"))
    os.makedirs(composite_decoder_savedir, exist_ok=True)


    # extract the averaged decoder outputs; fit a linear classifer

    n_avg = 1
    averaged_env_outputs_files = glob.glob(os.path.join(averaged_env_outputs_dir, f'env_-_avg-{n_avg:03d}*outputs.npz'))
    averaged_ffr_outputs_files = list(map(lambda x: x.replace('env', 'ffr'), averaged_env_outputs_files))

    all_env_predictions = []
    all_ffr_predictions = []
    all_targets = []

    for i in range(len(averaged_env_outputs_files)):

        env_data = np.load(averaged_env_outputs_files[i])
        ffr_data = np.load(averaged_ffr_outputs_files[i])

        env_predictions = env_data['predictions']
        ffr_predictions = ffr_data['predictions']
        targets = env_data['targets']
        max_len = min(len(env_predictions, ffr_predictions))

        all_env_predictions.append(env_predictions[:max_len])
        all_ffr_predictions.append(ffr_predictions[:max_len])
        all_targets.append(targets[:max_len])

    # 1. perform K-fold CV to get an idea of whether this is working

    clf = LinearDiscriminantAnalysis()
    lda_features = np.vstack([np.concatenate(all_env_predictions), np.concatenate(all_ffr_predictions)]).T
    lda_targets = np.concatenate(targets)

    scores = cross_val_score(clf, lda_features, lda_targets, cv=10)
    print('Mean score: ', np.mean(scores))

    # 2. train LDA classifier using entire test dataset (for later application to heldout/ICL datasets)

    clf.fit(lda_features, lda_targets)
    pickle.dump(clf, os.path.join(composite_decoder_savedir, 'composite_decoder.pkl'))