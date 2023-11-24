import pyNSL
import numpy as np

from scipy.signal import firwin2, filtfilt, resample_poly


def get_envelope_modulations(track, filt_bank = 'CF_8kHz', f_thr=300,target_fs=None):
    
    '''
    CREDIT: Jonas Auernheimer

    envelope modulation feature:
    
    - track: array representing pressure waveform
    - sr: the original sampling frequency of track
    - filt_bank: filter bank audio signal from aud24.mat
    - frame_len: frame length of auditory spectrogram (e.g. 1ms -> 1kHz sampling rate)
    - f_thr: the lowest centre frequency to consider for the envelope_modulations feature
    - sos: the IIR system used to filter the envelope modulations in the range of F0
    '''

    CFs = pyNSL.pyNSL.get_filterbank()[filt_bank].squeeze()[:-1]
    pick_CFs = np.where(CFs > f_thr)[0]
    frame_len = 2 # 2 ms frame length = 500 Hz sample rate
    fs = 1/(frame_len*1e-3)
    
    # calculate auditory spectrogram and sum over higher frequency bins
    aud_spec = pyNSL.pyNSL.wav2aud(track, 16000, [frame_len, 8, -2, -1]) # track fs is 16000
    freq_mod = []
    
    for i in pick_CFs:
        freq_mod.append(aud_spec[:,i])

    env_mod = np.stack(freq_mod).mean(0).squeeze()

    # filter the envelope modulations
    taps = firwin2(249, [0, 60, 70, 220, 250], [0.0, 0.0, 1.0, 1.0, 0.0], fs=fs)

    env_mod = filtfilt(taps, 1.0, env_mod)

    # resample to target sfreq
    if target_fs is not None:
        env_mod = resample_poly(env_mod, target_fs, fs)
    
    return env_mod
