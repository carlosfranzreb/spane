"""
Create the data file as is expected by the dataset: as a txt file with one
dict per line, with the fields "path": str, "duration": float and "label":
int. Given our use case, we store the speaker ID in the label field. We also store
the transcript and gender, which are used in the evaluation.
"""

import os
import json
from argparse import ArgumentParser
import torchaudio
from tqdm import tqdm


def create_file(folder, dump_file, root_folder, max_duration):
    """
    - The Librispeech folder contains one folder per speaker, and each of those folders,
    contains a chapter, and that chapter folder contains FLAC files.
    - The SPEAKERS.TXT file is one folder above the given folder, and it contains the
    gender of each speaker.
    - We remove the samples that are longer than the max. duration, defined by the
        max_duration parameter.
    """

    if max_duration is None:
        max_duration = float("inf")

    # get the parent folder of the given folder
    speakers_file = os.path.join(os.path.dirname(folder), "SPEAKERS.TXT")
    # create a writer object for the dump file
    writer = open(dump_file, "w")
    # iterate over the files in the folder
    for dirpath, _, filenames in tqdm(os.walk(folder)):
        for fname in filenames:
            if fname.endswith(".trans.txt"):
                with open(os.path.join(dirpath, fname)) as f:
                    for line in f:
                        words = line.split(" ")
                        id, text = words[0], " ".join(words[1:])
                        audiofile = os.path.join(dirpath, id + ".flac")
                        audio, sample_rate = torchaudio.load(audiofile)
                        spk_id = id.split("-")[0]
                        duration = audio.shape[1] / sample_rate
                        if duration <= max_duration:
                            writer.write(
                                json.dumps(
                                    {
                                        "path": audiofile.replace(
                                            root_folder, "{root}"
                                        ),
                                        "text": text.strip(),
                                        "duration": duration,
                                        "label": spk_id,
                                        "gender": get_gender(spk_id, speakers_file),
                                        "dataset": "librispeech",
                                    }
                                )
                                + "\n"
                            )
    writer.close()


def get_gender(spk_id, speakers_file):
    """Given the speakers file of LibriSpeech and a speaker ID, return the gender."""
    with open(speakers_file) as f:
        for line in f:
            if line.startswith(spk_id):
                return line.split("|")[1].strip()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder", help="Path to the LibriSpeech directory")
    parser.add_argument("dump_file", help="Path to the dump file")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    parser.add_argument(
        "--max_duration", type=int, help="Min. no. of utterances per speaker"
    )
    args = parser.parse_args()
    create_file(args.folder, args.dump_file, args.root_folder, args.max_duration)
