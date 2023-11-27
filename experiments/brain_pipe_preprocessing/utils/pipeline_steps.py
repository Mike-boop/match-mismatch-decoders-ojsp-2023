import logging
import mne

import librosa as lb
import numpy as np

from scipy.signal import filtfilt
from brain_pipe.pipeline.base import PipelineStep
from experiments.brain_pipe_preprocessing.utils.stimuli import get_envelope_modulations


class Transpose(PipelineStep):


    def __init__(
        self,
        data_keys=[]
    ):

        self.data_keys = data_keys

    def __call__(self, data_dict):
        for key in self.data_keys:
            data_dict[key] = data_dict[key].T

        return data_dict
    

class EnvelopeModulations(PipelineStep):


    def __init__(
        self,
        stimulus_data_key="stimulus_data",
        stimulus_sr_key="stimulus_sr",
        output_key="modulations_data",
        target_fs=500
    ):

        self.stimulus_data_key = stimulus_data_key
        self.stimulus_sr_key = stimulus_sr_key
        self.output_key = output_key
        self.target_fs = target_fs

    def __call__(self, data_dict):
        
        if self.stimulus_data_key not in data_dict:
            logging.warning(
                f"Found no stimulus data in {data_dict} for step "
                f"{self.__class__.__name__}, skipping..."
            )
            return data_dict
        data = data_dict[self.stimulus_data_key]
        sr = data_dict[self.stimulus_sr_key]

        # resample 16kHz
        ds = 16000
        resampled = lb.resample(data, orig_sr=sr, target_sr=ds)

        # get envelope modulations at 500 Hz
        modes = get_envelope_modulations(resampled, target_fs=self.target_fs)

        data_dict[self.output_key] = modes[:, None]
        data_dict[self.output_key+'_fs'] = self.target_fs

        return data_dict
    

class FirFiltFilt(PipelineStep):

    def __init__(
        self,
        filter_taps,
        data_key="data",
        axis=1,
        copy_data_dict=False
    ):
        super(FirFiltFilt, self).__init__(copy_data_dict=copy_data_dict)
        self.data_keys = self.parse_dict_keys(data_key)
        self.filter_taps = filter_taps
        self.axis=axis


    def __call__(self, data_dict):
        data_dict = super(FirFiltFilt, self).__call__(data_dict)

        for from_key, to_key in self.data_keys.items():
            data_dict[to_key] = filtfilt(
                self.filter_taps, 1.0, data_dict[from_key], axis=self.axis
            )
        return data_dict
    

class DropChannels(PipelineStep):
    """
    Interpolate some sensor locations
    """

    def __init__(
        self,
        channels_to_drop,
        data_key="data",
        copy_data_dict=False
    ):
        super(DropChannels, self).__init__(copy_data_dict=copy_data_dict)
        self.data_keys = self.parse_dict_keys(data_key)
        self.channels_to_drop = channels_to_drop


    def __call__(self, data_dict):
        
        data_dict = super(DropChannels, self).__call__(data_dict)
        drop_idxs = [data_dict['ch_names'].index(ch) for ch in self.channels_to_drop]
        keep_idxs = [i for i in range(len(data_dict['ch_names'])) if i not in drop_idxs]

        for from_key, to_key in self.data_keys.items():
            
            data_dict[to_key] = data_dict[to_key][keep_idxs, :]

        data_dict['ch_names'] = [data_dict['ch_names'][i] for i in keep_idxs]

        return data_dict
    

class ReorderChannels(PipelineStep):
    """
    Interpolate some sensor locations
    """

    def __init__(
        self,
        new_ch_order,
        data_key="data",
        copy_data_dict=False
    ):
        super(ReorderChannels, self).__init__(copy_data_dict=copy_data_dict)
        self.data_keys = self.parse_dict_keys(data_key)
        self.new_ch_names = new_ch_order


    def __call__(self, data_dict):
        
        data_dict = super(ReorderChannels, self).__call__(data_dict)
        idxs = [data_dict['ch_names'].index(ch) for ch in self.new_ch_names]

        for from_key, to_key in self.data_keys.items():
            
            data_dict[to_key] = data_dict[to_key][idxs, :]

        data_dict['ch_names'] = self.new_ch_names

        return data_dict
    

def get_interpolation_matrix(ch_from, ch_to, montage_from='easycap-M1', montage_to='easycap-M1', sfreq=64):

    info_from = mne.create_info(ch_names=ch_from, sfreq=64, ch_types='eeg')
    info_to = mne.create_info(ch_names=ch_to, sfreq=64, ch_types='eeg')

    info_from.set_montage(mne.channels.make_standard_montage(montage_from))
    info_to.set_montage(mne.channels.make_standard_montage(montage_to))

    pos_from = np.array(list(info_from.get_montage().get_positions()['ch_pos'].values()))
    pos_to = np.array(list(info_to.get_montage().get_positions()['ch_pos'].values()))

    M = mne.channels.interpolation._make_interpolation_matrix(pos_from, pos_to)

    return M


class InterpNewChannels(PipelineStep):
    """
    Interpolate some sensor locations
    """

    def __init__(
        self,
        channels_to,
        montage_to='easycap-M1',
        retain_original_channels=True,
        data_key="data",
        copy_data_dict=False
    ):
        super(InterpNewChannels, self).__init__(copy_data_dict=copy_data_dict)
        self.data_keys = self.parse_dict_keys(data_key)
        self.channels_to = channels_to
        self.montage_to = montage_to

        self.retain_original_channels = retain_original_channels


    def __call__(self, data_dict):
        data_dict = super(InterpNewChannels, self).__call__(data_dict)
        M = get_interpolation_matrix(data_dict['ch_names'], self.channels_to, montage_from=data_dict['mne_montage'], montage_to=self.montage_to)

        for from_key, to_key in self.data_keys.items():
            
            interpolated_channels = M @ data_dict[from_key]

            if self.retain_original_channels:
                data_dict[to_key] = np.vstack([data_dict[to_key], interpolated_channels])
            else:
                data_dict[to_key] = interpolated_channels

        if self.retain_original_channels:
            data_dict['ch_names'] += [*self.channels_to]
        else:
            data_dict['ch_names'] = self.channels_to

        return data_dict