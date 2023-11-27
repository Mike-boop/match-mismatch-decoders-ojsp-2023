import os
import json
import logging

from scipy.signal import butter

from brain_pipe.pipeline.default import DefaultPipeline
from brain_pipe.preprocessing.brain.artifact import InterpolateArtifacts, ArtifactRemovalMWF
from brain_pipe.preprocessing.resample import ResamplePoly
from brain_pipe.preprocessing.filter import SosFiltFilt
from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference
from brain_pipe.preprocessing.stimulus.load import LoadStimuli
from brain_pipe.save.default import DefaultSave
from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from brain_pipe.dataloaders.path import GlobLoader
from brain_pipe.runner.default import DefaultRunner
from brain_pipe.utils.log import default_logging, DefaultFormatter

from experiments.brain_pipe_preprocessing.utils.pipeline_steps import (
    InterpNewChannels, DropChannels, ReorderChannels, Transpose
)
from experiments.brain_pipe_preprocessing.utils.icl_helpers import (
    ICLEEGLoader, icl_ch_names, icl_stimulus_load_fn, icl_metadata_key_fn, icl_filename_fn
)
from experiments.brain_pipe_preprocessing.utils.sparrkulee_helpers import sparrkulee_ch_names


def run_audio_preprocessing_pipeline(icl_download_dir, preprocessed_data_savedir, nb_processes=-1, overwrite=False, log_path='icl-env-preprocessing.log'):

    # LOGGING
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])

    stimulus_steps = DefaultPipeline(
        steps=[
            LoadStimuli(load_fn=icl_stimulus_load_fn, load_from={"stimulus_path": "stimulus"}),
            GammatoneEnvelope(),
            ResamplePoly(64, "envelope_data", "stimulus_sr"),
            DefaultSave(
                preprocessed_data_savedir,
                to_save={
                    "envelope": "envelope_data"
                },
                overwrite=overwrite,
                filename_fn=icl_filename_fn
            ),
        ],
        on_error=DefaultPipeline.RAISE,
    )

    data_loader = GlobLoader(
        [os.path.join(os.path.join(icl_download_dir, 'audiobooks', '*', '*.wav'))],
        key="stimulus_path",
        )

    DefaultRunner(
        nb_processes=nb_processes,
        logging_config=lambda: None,
    ).run(
        [(data_loader, stimulus_steps)],
    )


def run_eeg_preprocessing_pipeline(icl_download_dir, preprocessed_data_savedir, nb_processes=-1, overwrite=False, log_path='icl-env-preprocessing.log'):

    # LOGGING
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])

    missing_channels = [x for x in sparrkulee_ch_names if x not in icl_ch_names]
    extra_channels = [x for x in icl_ch_names if x not in sparrkulee_ch_names]

    eeg_steps = DefaultPipeline([
        SosFiltFilt(
            butter(1, 0.5, "highpass", fs=1000, output="sos"),
            emulate_matlab=True,
            axis=1,
        ),
        InterpolateArtifacts(),
        ArtifactRemovalMWF(reference_channels=(0,1,2,3,32,33,38,62)),#FP1 FP2 AF3 AF4 AF7 AF8 AFz Fz
        ResamplePoly(64, axis=1),
        InterpNewChannels(missing_channels),
        DropChannels(extra_channels),
        ReorderChannels(sparrkulee_ch_names),
        CommonAverageRereference(),
        Transpose(data_keys=['data']),
        DefaultSave(
            preprocessed_data_savedir,
            {"eeg": "data"},
            overwrite=overwrite,
            clear_output=True,
            filename_fn=icl_filename_fn,
            metadata_key_fn=icl_metadata_key_fn
        ),
        ])
    
    data_loader = ICLEEGLoader(icl_download_dir)

    DefaultRunner(
        nb_processes=nb_processes,
        logging_config=lambda: None,
    ).run(
        [(data_loader, eeg_steps)],
    )


if __name__ == '__main__':

    os.environ["HDF5_USE_FILE_LOCKING"] = "False"

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)


    icl_download_dir = config['icl_download_dir']
    preprocessed_data_savedir = os.path.join(config['root_results_dir'], "preprocessed_data_icl", "env")
    os.makedirs(preprocessed_data_savedir, exist_ok=True)

    run_audio_preprocessing_pipeline(icl_download_dir, preprocessed_data_savedir, nb_processes=8, overwrite=True)
    run_eeg_preprocessing_pipeline(icl_download_dir, preprocessed_data_savedir, nb_processes=4, overwrite=True)
