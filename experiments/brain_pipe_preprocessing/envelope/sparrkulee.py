import json
import logging
import os

import scipy.signal
import scipy.signal.windows

from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope

from brain_pipe.dataloaders.path import GlobLoader
from brain_pipe.pipeline.default import DefaultPipeline
from brain_pipe.preprocessing.brain.artifact import InterpolateArtifacts, ArtifactRemovalMWF
from brain_pipe.preprocessing.brain.eeg.biosemi import biosemi_trigger_processing_fn
from brain_pipe.preprocessing.brain.eeg.load import LoadEEGNumpy
from brain_pipe.preprocessing.brain.epochs import SplitEpochs
from brain_pipe.preprocessing.brain.link import LinkStimulusToBrainResponse

from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference
from brain_pipe.preprocessing.brain.trigger import AlignPeriodicBlockTriggers
from brain_pipe.preprocessing.filter import SosFiltFilt
from brain_pipe.preprocessing.resample import ResamplePoly
from brain_pipe.preprocessing.stimulus.load import LoadStimuli
from brain_pipe.runner.default import DefaultRunner
from brain_pipe.save.default import DefaultSave
from brain_pipe.utils.log import default_logging, DefaultFormatter
from brain_pipe.utils.path import BIDSStimulusGrouper

from experiments.brain_pipe_preprocessing.utils.pipeline_steps import Transpose
from experiments.brain_pipe_preprocessing.utils.sparrkulee_helpers import (
    BIDSAPRStimulusInfoExtractor, temp_stimulus_load_fn, bids_filename_fn
)


def run_preprocessing_pipeline(
    root_dir,
    preprocessed_stimuli_dir,
    preprocessed_eeg_dir,
    nb_processes=-1,
    overwrite=False,
    log_path="sparrKULee.log",
):
    """Construct and run the preprocessing on SparrKULee.

    Parameters
    ----------
    root_dir: str
        The root directory of the dataset.
    preprocessed_stimuli_dir:
        The directory where the preprocessed stimuli should be saved.
    preprocessed_eeg_dir:
        The directory where the preprocessed EEG should be saved.
    nb_processes: int
        The number of processes to use. If -1, the number of processes is
        automatically determined.
    overwrite: bool
        Whether to overwrite existing files.
    log_path: str
        The path to the log file.
    """
    #########
    # PATHS #
    #########
    os.makedirs(preprocessed_eeg_dir, exist_ok=True)
    os.makedirs(preprocessed_stimuli_dir, exist_ok=True)

    ###########
    # LOGGING #
    ###########
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])

    ################
    # DATA LOADING #
    ################
    logging.info("Retrieving BIDS layout...")
    data_loader = GlobLoader(
        [os.path.join(root_dir, "sub-*", "*", "eeg", "*.bdf*")],
        filter_fns=[lambda x: "restingState" not in x],
        key="data_path",
    )

    #########
    # STEPS #
    #########

    stimulus_steps = DefaultPipeline(
        steps=[
            LoadStimuli(load_fn=temp_stimulus_load_fn),
            GammatoneEnvelope(output_key='envelope_data'),
            ResamplePoly(64, "envelope_data", "stimulus_sr"),
            DefaultSave(
                preprocessed_stimuli_dir,
                to_save={
                    "env": "envelope_data",
                },
                overwrite=overwrite,
                clear_output=False
            ),
        ],
        on_error=DefaultPipeline.RAISE,
    )

    eeg_steps = [
        LinkStimulusToBrainResponse(
            stimulus_data=stimulus_steps,
            extract_stimuli_information_fn=BIDSAPRStimulusInfoExtractor(),
            grouper=BIDSStimulusGrouper(
                bids_root=root_dir,
                mapping={"stim_file": "stimulus_path", "trigger_file": "trigger_path"},
                subfolders=["stimuli", "eeg"],
            ),
        ),
        LoadEEGNumpy(unit_multiplier=1e6, channels_to_select=list(range(64))),
        SosFiltFilt(
            scipy.signal.butter(1, 0.5, "highpass", fs=1024, output="sos"),
            emulate_matlab=True,
            axis=1,
        ),
        InterpolateArtifacts(),
        AlignPeriodicBlockTriggers(biosemi_trigger_processing_fn),
        SplitEpochs(),
        ArtifactRemovalMWF(),
        CommonAverageRereference(),
        ResamplePoly(64, axis=1),
        Transpose(data_keys=['data']),
        DefaultSave(
            preprocessed_eeg_dir,
            {"eeg": "data"},
            overwrite=overwrite,
            clear_output=True,
            filename_fn=bids_filename_fn,
        ),
    ]

    #########################
    # RUNNING THE PIPELINE  #
    #########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")

    # Create data_dicts for the EEG files
    # Create the EEG pipeline
    eeg_pipeline = DefaultPipeline(steps=eeg_steps)

    DefaultRunner(
        nb_processes=nb_processes,
        logging_config=lambda: None,
    ).run(
        [(data_loader, eeg_pipeline)],
    )


if __name__ == "__main__":

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    sparrkulee_download_dir = config['sparrkulee_download_dir']
    preprocessed_stimuli_path = os.path.join(config['root_results_dir'], "preprocessed_data_sparrkulee", "env")
    preprocessed_eeg_path = preprocessed_stimuli_path
    
    n_processes = 8
    overwrite = False

    logpath = os.path.join(os.path.dirname(__file__), 'sparrkulee-env-preprocessing.log')

    # Run the preprocessing pipeline
    run_preprocessing_pipeline(
        sparrkulee_download_dir,
        preprocessed_stimuli_path,
        preprocessed_eeg_path,
        n_processes,
        overwrite,
        logpath
    )