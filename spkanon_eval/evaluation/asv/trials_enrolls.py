"""
Helper functions related to splitting the data into trial and enrollment utterances.
"""

import os
import json
import logging
from typing import TextIO
import random

from spkanon_eval.datamodules import sort_datafile

LOGGER = logging.getLogger("progress")


def split_trials_enrolls(
    exp_folder: str,
    anonymized_enrolls: bool,
    root_folder: str = None,
    anon_folder: str = None,
    trials: list[str] = None,
    enrolls: list[str] = None,
) -> tuple[str, str]:
    """
    Split the evaluation data into trial and enrollment datafiles.

    ## Splitting strategy

    - If both trials and enrolls are passed, use them and discard the rest.
    - If enrolls are passed, but not trials, every utterance that is not part of enrolls
        is added to trials.
    - If trials are passed but not enrolls, same as before but vice versa.
    - If neither trials nor enrolls are passed, divide them 50/50 randomly.

    ## Using anonymized data

    If the root folder is passed, it is replaced in the trial with the folder where the
    anonymized evaluation data is stored (`exp_folder/results/anon_eval`).
    If `anonymized_enrolls` is True, the same is done for enrolls as well.
    The root folder is None if we are evaluating the baseline, where speech is not anonymized.

    Args:
        exp_folder: path to the experiment folder.
        anonymized_enrolls: whether the anonymized or original versions of the enrollment
            utterances should be consider. Generally, this depends on whether they were
            anonymized with or without consistent targets in the inference run.
        root_folder (optional): root folder of the original data. We use it to replace
            the original path with the anonymized one.
            If we are computing a baseline with original data, this is null.
        anon_folder (optional): folder where the anonymized evaluation data is stored.
            It it is not given, we assume that it is the same as the experiment folder.
        trials, enrolls (optional): list of files defining the enrollment data. Each of
            these files contains one filename per line.

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

    anonymized_trials = True
    if root_folder is None:
        anonymized_trials = False
        LOGGER.info("No root folder given: original trial data will be used.")
    elif anon_folder is None:
        anon_folder = exp_folder

    # create the file writers and define which data is anonymized
    is_anonymized = {"trials": anonymized_trials, "enrolls": anonymized_enrolls}
    splits = ["trials", "enrolls"]
    writers = dict()
    for split, split_dump_f in zip(splits, [f_trials, f_enrolls]):
        writers[split] = open(split_dump_f, "w")

    # gather the filenames of the trial and enrollment data, if any
    fnames = dict()
    for split, split_files in zip(splits, [trials, enrolls]):
        fnames[split] = list()
        if split_files is not None:
            for f in split_files:
                fnames[split].extend([line.strip() for line in open(f)])

    both_passed = trials is not None and enrolls is not None
    one_passed = trials is not None or enrolls is not None

    def write_line(split: str, line: str):
        """
        Write the line to the given split. If the split should be anonymized, replace the
        original path with the anonymized one. For this we need `root_folder` and
        `anon_folder`.

        Args:
            split: the split to which the line should be written (trials or enrolls).
            line: the original line from the datafile that should be dumped.
        """

        # check that all the necessary arguments are present
        if is_anonymized[split] and (not root_folder or not anon_folder):
            error_msg = (
                "`root_folder` and `anon_folder` are needed to find the anonymized path"
            )
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # replace the original path with the anonymized ones if needed
        if is_anonymized:
            obj = json.loads(line)
            obj["path"] = obj["path"].replace(
                root_folder, os.path.join(anon_folder, "results", "eval")
            )
            line = json.dumps(obj)

        writers[split].write(line + "\n")

    # select a splitting strategy depending on whether lists are passed
    if both_passed or one_passed:

        for line in open(datafile):
            obj = json.loads(line.strip())
            fname = os.path.splitext(os.path.basename(obj["path"]))[0]

            # check that the fname is only present in one of the lists, if any
            if fname in fnames["trials"] and fname in fnames["enrolls"]:
                error_msg = f"{fname} is part of both trials and enrolls"
                LOGGER.error(error_msg)
                raise RuntimeError(error_msg)

            # try adding it to a list, and continue if it's added
            is_written = False
            for split in splits:
                if fname in fnames[split]:
                    write_line(split, line)
                    is_written = True

            if is_written or both_passed:
                continue

            # if only one list was passed, add this line to the other
            for split in splits:
                if len(fnames[split]) == 0:
                    write_line(split, line)

    # trials and enrolls are both null: split data of each speaker randomly 50/50
    else:
        # group the objects according to the speaker ID
        speaker_lines = dict()
        for line in open(datafile):
            spk_id = json.loads(line)["speaker_id"]
            if spk_id not in speaker_lines:
                speaker_lines[spk_id] = list()

            speaker_lines[spk_id].append(line)

        # split the objects of each speaker
        for lines in speaker_lines.values():
            random.shuffle(lines)
            mid = len(lines) // 2
            for line_idx, line in enumerate(lines):
                split = "trials" if line_idx < mid else "enrolls"
                write_line(split, line)

        # sort the files according to their duration
        sort_datafile(f_trials)
        sort_datafile(f_enrolls)

    for writer in writers.values():
        writer.close()

    return f_trials, f_enrolls
