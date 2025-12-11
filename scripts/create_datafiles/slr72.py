"""
Creates the data file for the Open SLR dataset 72: Crowdsourced high-quality Colombian
Spanish speech.
"""

import os
import json
from argparse import ArgumentParser

import torchaudio
from tqdm import tqdm


def create_file(
    dataset_dir: str, dump_file: str, root_folder: str, max_duration: int = None
):
    """
    - `dataset_dir` comprises all the required dataset information:
        - The waveforms as .wav files, directly under this directory.
        - `line_index_female.tsv` and its male version, with the transcripts.
    - We remove the samples that are longer than the max. duration, defined by the
        max_duration parameter.
    """

    if max_duration is None:
        max_duration = float("inf")

    # gather the transcripts
    transcripts = dict()
    for gender in ["female", "male"]:
        transcripts[gender] = dict()
        f = os.path.join(dataset_dir, f"line_index_{gender}.tsv")
        for line in open(f):
            fname, transcript = line.strip().split(maxsplit=1)
            transcripts[gender][fname] = transcript

    # create a writer object for the dump file
    writer = open(dump_file, "w")

    # iterate over the files in the folder
    for f in tqdm(os.listdir(dataset_dir)):
        if not f.endswith(".wav"):
            continue

        # find transcript and speaker
        fname = os.path.splitext(f)[0]
        spk = fname.split("_")[1]
        gender = "female" if fname.startswith("cof") else "male"
        text = transcripts[gender][fname]
        gender = gender[0].upper()

        # get the audio duration and check for max. duration
        audiofile = os.path.join(dataset_dir, f)
        audio, sample_rate = torchaudio.load(audiofile)
        duration = audio.shape[1] / sample_rate
        if duration > max_duration:
            print(f"Skipping {audiofile}, duration={duration}")
            continue

        # add audio to datafile
        writer.write(
            json.dumps(
                {
                    "path": audiofile.replace(root_folder, "{root}"),
                    "text": text,
                    "duration": round(duration, 2),
                    "label": spk,
                    "gender": gender,
                    "dataset": "slr72",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    parser.add_argument("dump_file", help="Path to the dump file")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    parser.add_argument(
        "--max_duration", type=int, help="Max. duration for filtering utterances"
    )
    args = parser.parse_args()
    create_file(
        args.dataset_dir,
        args.dump_file,
        args.root_folder,
        args.max_duration,
    )
