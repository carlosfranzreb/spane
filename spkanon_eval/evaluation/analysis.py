import os
import logging
import json
from collections.abc import Callable
from typing import Any

import numpy as np


LOGGER = logging.getLogger("progress")


def analyse_results(
    dump_folder: str,
    datafile: str,
    data: Any,
    analyse_func: Callable,
    headers_func: Callable,
):
    """
    Analyse the results of an evaluation for a given datafile and store them in the
    given dump folder, in a file that averages over the samples of the datafile. The
    analysis is run for the whole datafile, and also for its individual population
    segments (gender, age, or whatever is defined in the datafile).

    Args:
    - dump_folder: the folder where to store the results; each characteristic is stored
        in a separate file, for all the datafiles.
    - datafile: the name of the datafile whose results should be analysed.
    - data: the results of the evaluation that will be analysed here. Their format must
        conform with the expected input of the `analyse_func` function.
    - analyse_func: function that computes some aggregated statistics given
        evaluation results.
    - headers_func: function to create a dump file for dumping the analysis results with
        the appropriate headers.
    """

    datafile_name = os.path.splitext(os.path.basename(datafile))[0]
    LOGGER.info(f"Analysing the results for {datafile_name}")
    chars, speakers = get_characteristics(datafile)
    utts = np.arange(len(speakers))  # the indices of the utterances

    # dump the averages of the whole dataset
    results = analyse_func(np.ones(len(utts), dtype=bool), data)
    dump_file = os.path.join(dump_folder, "all.txt")
    if not os.path.exists(dump_file):
        headers_func(dump_file)
    with open(dump_file, "a", encoding="utf-8") as f:
        f.write(f"{datafile_name} ")
        f.write(" ".join(results) + "\n")

    # perform the analysis for each characteristic
    for key, values in chars.items():
        # define the dump file for this char and create it if needed
        dump_file = os.path.join(dump_folder, f"{key}.txt")
        if not os.path.exists(dump_file):
            headers_func(dump_file, key)

        # determine which indices to use (either speakers or utterances)
        indices = utts if key.startswith("utt_") else speakers

        # iterate over the values of this speaker char and write the results
        for value in values:
            val_utts = np.isin(indices, values[value])
            results = analyse_func(val_utts, data)
            with open(dump_file, "a", encoding="utf-8") as f:
                f.write(f"{datafile_name} {value} ")
                f.write(" ".join(results) + "\n")


def get_characteristics(datafile: str) -> tuple[dict, list, list]:
    """
    Given a datafile, return a dictionary that maps each characteristic to the
    speakers assigned to each of its values. The speaker IDs are replaced by their
    indices in the list of labels. These should be unique. If a characteristic starts
    with "utt_", it is considered to be a per-utterance characteristic, and the values
    are not aggregated over the speakers. These store audiofile paths instead of
    speaker IDs, as these are unique.The output's structure is:

        {
            "char1": {
                "val1": ["spk1", "spk2", ...],
                "val2": ["spk3", "spk4", ...],
                ...
            },
            "utt_char2": {
                "val1": ["utt1", "utt2", ...],
                "val2": ["utt3", "utt4", ...],
                ...
            },
            ...
        }

    Args:
        datafile (str): the path to the datafile

    Returns:
        tuple[dict, list, list]: the dictionary of characteristics, the list of
            speaker indices, and the list of utterance indices
    """

    chars = dict()  # maps each char value to the speaker or utterance indices
    speakers_idx = list()  # maps each utterance to its speaker index in `speakers`
    speakers = list()  # stores the speaker IDs

    with open(datafile) as f:
        for utt_idx, line in enumerate(f):
            # parse the line and get the speaker and utterance IDs
            obj = json.loads(line)
            if obj["label"] in speakers:
                spk_idx = speakers.index(obj["label"])
            else:
                spk_idx = len(speakers)
                speakers.append(obj["label"])
            speakers_idx.append(spk_idx)

            # iterate over the chars in this object
            for key, value in obj.items():
                # skip the keys that are not speaker characteristics
                if key in ["path", "text", "duration", "speaker_id", "label"]:
                    continue

                # pick an index depending on the characteristic (utt or spk)
                idx = utt_idx if key.startswith("utt_") else spk_idx
                new_arr = np.array([idx])

                # store the char value in the dictionary
                if key not in chars:
                    chars[key] = {value: new_arr}
                elif value not in chars[key]:
                    chars[key][value] = new_arr
                elif idx not in chars[key][value]:
                    chars[key][value] = np.concatenate((chars[key][value], new_arr))

    return chars, speakers_idx
