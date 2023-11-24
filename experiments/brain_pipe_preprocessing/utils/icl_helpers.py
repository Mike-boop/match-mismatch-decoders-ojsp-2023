import glob
import os
import h5py

from scipy.io import wavfile
from brain_pipe.dataloaders.base import DataLoader


icl_ch_names = [
    'AF3','AF4','AF7','AF8','C1','C2','C3','C4','C5',
    'C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz',
    'Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1',
    'FC2','FC3','FC4','FC5','FC6','Fp1','Fp2','FT10',
    'FT7','FT8','FT9','Fz','O1','O2','Oz','P1','P2',
    'P3','P4','P5','P6','P7','P8','PO3','PO7','PO8','POz',
    'Pz','FCz','T7','T8','TP10','TP7','TP8','TP9','AFz'
]


def icl_metadata_key_fn(data_dict):

    if "savename" in data_dict:
        return "savename"

    raise ValueError("No savename in the data_dict.")


def icl_stimulus_load_fn(path):

    condition = os.path.basename(os.path.dirname(path))
    _, part, suffix = os.path.splitext(os.path.basename(path))[0].split('_')
    savename = f'{condition}_part_{part}-{suffix}.npy'

    fs, audio = wavfile.read(path)
    return {'data':audio, 'sr':fs, 'savename':savename}


def icl_filename_fn(data_dict, feature_name="", set_name=None):

    if 'savename' in data_dict:
        return data_dict['savename']
    if 'stimulus_savename' in data_dict:
        return data_dict['stimulus_savename']


class ICLEEGLoader(DataLoader):

    def __init__(
        self,
        icl_download_dir,
        key="path",
        has_length=True,
    ):

        super().__init__(has_length=has_length)

        self.download_dir = icl_download_dir
        self.key = key

        self.h5_keys = self._get_trial_keys()


    def _get_trial_keys(self):

        h5_files = glob.glob(os.path.join(self.download_dir, 'eeg', '*.h5'))
        all_keys = []

        for each_file in h5_files:
            
            with h5py.File(each_file, 'r') as f:
                conditions = f.keys()

                for each_condition in conditions:

                    if each_condition in ['ch_names', 'srate']:
                        continue

                    parts = f[each_condition].keys()

                    for each_part in parts:

                        all_keys.append( (each_file, '/'.join([each_condition, each_part])) )

        return all_keys


    def h5_key_to_data_dict(self, h5_key):

        with h5py.File(h5_key[0], 'r') as f:

            condition, part = h5_key[1].split('/')
            part = int(part.replace('part_', ''))
            sub = int(os.path.splitext(os.path.basename(h5_key[0]))[0].replace('P',''))

            savename = f'sub-{sub:03d}_-_trial-{part:03d}_-_{condition}_-_{condition}-part_{part}-story'
            if condition in ['fM', 'fW']:
                savename+= f'_-_{condition}-part_{part}-distractor_-_eeg.npy'
            else:
                savename+= f'_-_eeg.npy'

            data_dict = {'data':f[h5_key[1]][:], 'ch_names':[x.decode("utf-8") for x in f['ch_names'][:]], 'mne_montage':'easycap-M1', 'data_fs': f['srate'][:], 'savename':savename, 'data_path':savename}

        return data_dict
    
    def __len__(self):
        
        return len(self.h5_keys)

    def __iter__(self):

        self.counter = 0
        return self


    def __next__(self):

        if self.counter<len(self):
            x = self.h5_key_to_data_dict(self.h5_keys[self.counter])
            self.counter+=1
            return x
        else:
            raise StopIteration