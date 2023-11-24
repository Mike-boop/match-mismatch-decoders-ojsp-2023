import os
import gzip
import logging
import librosa

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from typing import Any, Dict
from brain_pipe.preprocessing.brain.link import BIDSStimulusInfoExtractor


sparrkulee_ch_names = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
    'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
    'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
    'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
    ]


class BIDSAPRStimulusInfoExtractor(BIDSStimulusInfoExtractor):
    """Extract BIDS compliant stimulus information from an .apr file."""

    def __call__(self, brain_dict: Dict[str, Any]):
        """Extract BIDS compliant stimulus information from an events.tsv file.

        Parameters
        ----------
        brain_dict: Dict[str, Any]
            The data dict containing the brain data path.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        event_info = super().__call__(brain_dict)
        # Find the apr file
        path = brain_dict[self.brain_path_key]
        apr_path = "_".join(path.split("_")[:-1]) + "_eeg.apr"
        # Read apr file
        apr_data = self.get_apr_data(apr_path)
        # Add apr data to event info
        for e_i in event_info:
            e_i.update(apr_data)
        return event_info

    def get_apr_data(self, apr_path: str):
        """Get the SNR from an .apr file.

        Parameters
        ----------
        apr_path: str
            Path to the .apr file.

        Returns
        -------
        Dict[str, Any]
            The SNR.
        """

        apr_data = {}
        tree = ET.parse(apr_path)
        root = tree.getroot()

        if '.apx' in root.attrib['experiment_file']:
            apr_data['stimulus'] = root.attrib['experiment_file'].replace('.apx', '')
            apr_data['stimulus'] = '_'.join(apr_data['stimulus'].split('_')[1:])
        if '.xml' in root.attrib['experiment_file']:
            apr_data['stimulus'] = root.attrib['experiment_file'].replace('.xml', '')

        # Get SNR
        interactive_elements = root.findall(".//interactive/entry")
        for element in interactive_elements:
            description_element = element.find("description")
            if description_element.text == "SNR":
                apr_data["snr"] = element.find("new_value").text
        if "snr" not in apr_data:
            logging.warning(f"Could not find SNR in {apr_path}.")
            apr_data["snr"] = 100.0
        return apr_data


def default_librosa_load_fn(path):
    """Load a stimulus using librosa.

    Parameters
    ----------
    path: str
        Path to the audio file.

    Returns
    -------
    Dict[str, Any]
        The data and the sampling rate.
    """
    data, sr = librosa.load(path, sr=None)
    return {"data": data, "sr": sr}


def default_npz_load_fn(path):
    """Load a stimulus from a .npz file.

    Parameters
    ----------
    path: str
        Path to the .npz file.

    Returns
    -------
    Dict[str, Any]
        The data and the sampling rate.
    """
    np_data = np.load(path)
    return {
        "data": np_data["audio"],
        "sr": np_data["fs"],
    }


DEFAULT_LOAD_FNS = {
    ".wav": default_librosa_load_fn,
    ".mp3": default_librosa_load_fn,
    ".npz": default_npz_load_fn,
}


def temp_stimulus_load_fn(path):
    """Load stimuli from (Gzipped) files.

    Parameters
    ----------
    path: str
        Path to the stimulus file.

    Returns
    -------
    Dict[str, Any]
        Dict containing the data under the key "data" and the sampling rate
        under the key "sr".
    """
    if path.endswith(".gz") and os.path.exists(path):
        with gzip.open(path, "rb") as f_in:
            data = dict(np.load(f_in))
        return {
            "data": data["audio"],
            "sr": data["fs"],
        }
    
    # stimulus file may already have been decompressed
    path = path.replace('.gz', '')

    extension = "." + ".".join(path.split(".")[1:])
    if extension not in DEFAULT_LOAD_FNS:
        raise ValueError(
            f"Can't find a load function for extension {extension}. "
            f"Available extensions are {str(list(DEFAULT_LOAD_FNS.keys()))}."
        )
    load_fn = DEFAULT_LOAD_FNS[extension]
    return load_fn(path)


def bids_filename_fn(data_dict, feature_name, set_name=None):
    """Default function to generate a filename for the data.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: str
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.

    Returns
    -------
    str
        The filename.
    """

    bdf_fpath = data_dict["data_path"]
    filename = os.path.basename(bdf_fpath).split("_eeg")[0]

    subject = filename.split("_")[0]
    session = filename.split("_")[1]
    run     = filename.split("_")[3]

    sub_idx = int(subject.replace('sub-',''))
    run_idx = int(run.replace('run-',''))

    # get the listening condition
    events_info_fpath = bdf_fpath.replace('eeg.bdf', 'events.tsv')
    apr_fpath = bdf_fpath.replace('.bdf', '.apr')

    if os.path.exists(events_info_fpath):
        events_info_df = pd.read_csv(events_info_fpath, delimiter='\t', header=0)

        stimulus = events_info_df['stim_file'].item()
        stimulus = stimulus.split('/')[1].replace('.npz', '').replace('.gz', '')
        duration_s = events_info_df['duration'].item()+2
        snr = events_info_df['snr'].item()

    # at time of writing event file does not contain correct SNRs or video files
    if os.path.exists(apr_fpath):
        apr_data = BIDSAPRStimulusInfoExtractor().get_apr_data(apr_fpath)
        snr = apr_data['snr']
        if sub_idx != 33: stimulus = apr_data['stimulus'] # apr file messed up for sub 33

    if snr is None or int(snr) == 100.0:
        condition = 'natural'
    else:
        condition = 'SiN'

    if stimulus in ['podcast_10', 'podcast_10_video'] and sub_idx in range(27,37):
        condition = 'av'
        stimulus = stimulus.replace('_video', '')
        logging.info('%s: video condition', os.path.basename(bdf_fpath))


    filename = f"sub-{sub_idx:03d}_-_trial-{run_idx:03d}_-_{condition}_-_{stimulus}_-_{feature_name}"

    if set_name is not None:
        filename += f"_set-{set_name}"

    return filename + ".npy"