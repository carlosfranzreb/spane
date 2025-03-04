"""
About the dataset:

    - `subset` defines the dataset that is stored in the datafile: dev or test.
    - Information about the speakers is stored in `folder/linguistic_background.csv`.
    - `folder/subset/segments` contains the utterance ID, the conversation ID, and the
        start and end time of the utterance.
    - `folder/subset/texts` maps utterance IDs to the corresponding texts.
    - `folder/subset/utt2spk` maps utterance IDs to speaker IDs.
    - The audiofiles of the dataset are stored in `folder/data`, each named with the
    conversation ID.
"""

import os
import json
from argparse import ArgumentParser
import logging
import time

import pandas
import torchaudio


SAMPLE_RATE = 32000


def create_file(folder, subset, dump_file, root_folder, max_duration=None):
    """
    - We remove the samples that are longer than the max. duration, defined by the
        max_duration parameter.
    - We split the audiofiles into utterances, store them within the dataset under
        the "clips" directory, and create a datafile for these clips instead of the
        original audiofiles, which contain the whole conversations.
    """
    # store the speaker information in a dict
    speakers = pandas.read_csv(os.path.join(folder, "linguistic_background.csv"))
    convos = list()
    for line in open(os.path.join(folder, subset, "conv.list")):
        line = line.strip()
        if "_" in line:
            convos.append(line.split("_")[0])
        else:
            convos.append(line)

    # create the mapping from utterance ID to speaker ID
    utt2spk = {}
    for line in open(os.path.join(folder, subset, "utt2spk")):
        utt_id, spk_id = line.strip().split()
        utt2spk[utt_id] = spk_id

    # store the texts in a dict
    texts = {}
    for line in open(os.path.join(folder, subset, "text")):
        utt_id, text = line.strip().split(maxsplit=1)
        texts[utt_id] = text

    # create a writer object for the dump file and create the new directory
    writer = open(dump_file, "w")
    os.makedirs(os.path.join(folder, "clips"), exist_ok=True)

    # get the current year
    current_year = int(time.strftime("%Y"))

    # iterate over the speakers
    for spk in speakers["PARTICIPANT_ID"]:
        # skip speakers that are not in the conversations of this subset
        if spk.rsplit("-", 1)[0] not in convos:
            logging.debug(f"Skipping speaker {spk}")
            continue

        # iterate over the utterances of the subset
        for line in open(os.path.join(folder, subset, "segments")):
            utt_id, conv_id, start, end = line.strip().split()

            # skip utterances that are not from the current speaker
            if utt2spk[utt_id] != spk:
                continue

            conv_file = os.path.join(folder, "data", conv_id + ".wav")
            duration = float(end) - float(start)

            # skip samples that are too long if necessary
            if max_duration is not None and duration > max_duration:
                logging.warn(f"Skipping utterance {utt_id} because it is too long")
                continue

            # extract the utterance from the conversation and dump it
            utt_file = os.path.join(folder, "clips", utt_id + ".wav")
            if not os.path.exists(utt_file):
                audio, _ = torchaudio.load(
                    conv_file,
                    frame_offset=int(float(start) * SAMPLE_RATE),
                    num_frames=int(duration * SAMPLE_RATE),
                )
                torchaudio.save(utt_file, audio, SAMPLE_RATE)

            # get the speaker information
            speaker_info = speakers[speakers["PARTICIPANT_ID"] == utt2spk[utt_id]]
            l1 = speaker_info.iloc[0, 12]
            l2 = speaker_info.iloc[0, 20]
            if isinstance(l2, float) or l2 == "No":
                n_languages = len(set(l1.split(",")))
            else:
                n_languages = len(set(l1.split(",") + l2.split(",")))
            try:
                start_english_age = int(speaker_info.iloc[0, 21]) - int(
                    speaker_info.iloc[0, 6]
                )
            except ValueError:  # native speakers
                start_english_age = int(speaker_info.iloc[0, 6])

            try:
                obj = {
                    "path": utt_file.replace(root_folder, "{root}"),
                    "label": spk,
                    "duration": round(duration, 2),
                    "text": texts[utt_id],
                    "gender": speaker_info.iloc[0, 5],
                    "age": current_year - int(speaker_info.iloc[0, 6]),
                    "ethnicity": speaker_info.iloc[0, 7],
                    "education": speaker_info.iloc[0, 8],
                    "accent": speaker_info.iloc[0, 19],
                    "n_languages": n_languages,
                    "start_english_age": start_english_age,
                }
                writer.write(json.dumps(obj) + "\n")
            except RuntimeError as err:
                logging.error(f"Error while processing utterance {utt_id}:")
                logging.error(err)
    writer.close()


def get_speaker_data(spk_id, speakers_file):
    """
    Given the speakers file of VCTK and a speaker ID, return its age, gender, accent
    and region as a dict.
    """
    with open(speakers_file) as f:
        for line in f:
            if line.startswith(spk_id):
                data = line.strip().split()
                return {
                    "age": int(data[1]),
                    "gender": data[2],
                    "accent": data[3],
                    "region": data[4],
                }


if __name__ == "__main__":
    # configure logging with timestamp and level and dump to logs/create_dataset/{timestamp}.log
    logging.basicConfig(
        filename=os.path.join(
            "logs", "create_dataset", f"edacc_{int(time.time())}.log"
        ),
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # define and parse the arguments
    parser = ArgumentParser()
    parser.add_argument("--folder", help="Path to the EdAcc folder")
    parser.add_argument("--subset", help="Subset to process (dev or test)")
    parser.add_argument("--dump_file", help="Path to the dump file (TXT)")
    parser.add_argument("--root_folder", help="Path that will be replaced with {root}")
    parser.add_argument(
        "--max_duration",
        type=int,
        help="Min. no. of utterances per speaker",
        default=None,
    )
    args = parser.parse_args()

    # run the script
    create_file(
        args.folder, args.subset, args.dump_file, args.root_folder, args.max_duration
    )
