"""
Helper functions related to splitting the data into trial and enrollment utterances.
"""

import os
import json
import logging

from spkanon_eval.datamodules import sort_datafile

LOGGER = logging.getLogger("progress")


# TODO: this expects the data to be sorted by speaker_id, but we sort it now by duration
def split_trials_enrolls(
    exp_folder: str,
    anonymized_enrolls: bool,
    root_folder: str = None,
    anon_folder: str = None,
    enrolls: list = None,
) -> tuple[str, str]:
    """
    Split the evaluation data into trial and enrollment datafiles. The first utt of
    each speaker is the trial utt, and the rest are enrollment utts. If the root folder
    is given, it is replaced in the trial with the folder where the anonymized
    evaluation data is stored (`exp_folder/results/anon_eval`). The root folder is None
    if we are evaluating the baseline, where speech is not anonymized.

    Args:
        exp_folder: path to the experiment folder.
        anonymized_enrolls: whether the anonymized or original versions of the enrollment
            utterances should be consider. Generally, this depends on whether they were
            anonymized with or without consistent targets in the inference run.
        root_folder (optional): root folder of the data. If we are computing a baseline
            with original data, this is null.
        anon_folder (optional): folder where the anonymized evaluation data is stored.
            It it is not given, we assume that it is the same as the experiment folder.
        enrolls: list of files defining the enrollment data. Each of these files
            contains one filename per line.

    Returns:
        paths to the created trial and enrollment datafiles

    Raises:
        ValueError: if one of the speakers only has one utterance. Each speaker should
            have at least two utterances, one for trial and one for enrollment.
    """

    LOGGER.info("Splitting evaluation data into trial and enrollment data")
    datafile = os.path.join(exp_folder, "data", "eval.txt")
    f_trials = os.path.join(exp_folder, "data", "eval_trials.txt")
    f_enrolls = os.path.join(exp_folder, "data", "eval_enrolls.txt")
    
    if os.path.exists(f_trials):
        LOGGER.warning("Datafile splits into trial and enrolls already exist, skipping")
        return f_trials, f_enrolls

    if root_folder is not None:
        anon_datafile = os.path.join(exp_folder, "data", "anon_eval.txt")
        anon_data = dict()
        for line in open(anon_datafile):
            obj = json.loads(line.strip())
            fname = os.path.splitext(os.path.basename(obj["path"]))[0]
            anon_data[fname] = line            
    else:
        anon_datafile = None

    if root_folder is None:
        LOGGER.info("No root folder given: original trial data will be used.")

    if anon_folder is None:
        anon_folder = exp_folder

    # if enrolls are given, use them to split the data
    if enrolls is not None:
        trial_writer = open(f_trials, "w")
        enroll_writer = open(f_enrolls, "w")
        enroll_fnames = list()

        # gather the filenames of the enrollment data
        for enroll_file in enrolls:
            with open(enroll_file) as f:
                for line in f:
                    enroll_fnames.append(line.strip())

        # split the data into trial and enrollment data
        for line in open(datafile):
            obj = json.loads(line.strip())
            fname = os.path.splitext(os.path.basename(obj["path"]))[0]
            if fname in enroll_fnames:
                enroll_writer.write(
                    anon_data[fname] if anon_datafile and anonymized_enrolls else line
                )
            else:
                trial_writer.write(anon_data[fname] if anon_datafile else line)

        trial_writer.close()
        enroll_writer.close()

    # if no enrolls, the trial is the first utt of each speaker
    else:
        current_spk = None
        objects = [json.loads(line) for line in open(datafile)]
        objects = sorted(objects, key=lambda x: x["speaker_id"])

        spk_objs = list()
        for obj in objects:
            spk = obj["speaker_id"]
            if current_spk is None:
                current_spk = spk
            elif spk != current_spk:
                split_speaker(spk_objs, f_trials, f_enrolls, anon_folder, root_folder)
                spk_objs = list()
                current_spk = spk
            spk_objs.append(obj)

        split_speaker(spk_objs, f_trials, f_enrolls, anon_folder, root_folder)
    
    # sort the files according to their duration
    sort_datafile(f_trials)
    sort_datafile(f_enrolls)

    return f_trials, f_enrolls


def split_speaker(
    spk_data: list[dict],
    trial_file: str,
    enroll_file: str,
    exp_folder: str,
    root_folder: str = None,
) -> None:
    """
    Split the speaker's data into trial and enrollment data. The first utt is the trial
    utt, and the rest are enrollment utts. If the root folder is given, it is replaced
    in the trial with the folder where the anonymized evaluation data is stored
    (`exp_folder/results/eval`).

    Args:
        spk_data: list of datafile objects from one speaker.
        trial_file: path to the trial datafile.
        enroll_file: path to the enrollment datafile.
        exp_folder: path to the experiment folder.
        root_folder (optional): root folder of the data.

    Raises:
        ValueError: if one of the speakers only has one utterance. Each speaker should
            have at least two utterances, one for trial and one for enrollment.
    """

    if len(spk_data) == 1:
        error_msg = f"Speaker {spk_data[0]['speaker_id']} has only one utterance"
        LOGGER.error(error_msg)
        raise ValueError(error_msg)

    trial_sample, enroll_data = spk_data[0], spk_data[1:]
    if root_folder is not None:
        trial_sample["path"] = trial_sample["path"].replace(
            root_folder, os.path.join(exp_folder, "results", "eval")
        )

    with open(trial_file, "a") as f:
        f.write(json.dumps(trial_sample) + "\n")
    with open(enroll_file, "a") as f:
        for enroll_utt in enroll_data:
            f.write(json.dumps(enroll_utt) + "\n")
