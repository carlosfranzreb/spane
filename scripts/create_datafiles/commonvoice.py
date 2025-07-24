"""
Create a datafile from a TSV file from Common Voice. Store all speaker data like
age, gender, accent, etc. in the datafile. This is useful to analyze the evaluation
results.
"""


import os
import json
from argparse import ArgumentParser
import torchaudio
import pandas as pd


def create_file(data_file, dump_file, root_folder, max_duration, min_utts=2):
    """
    We assume that the data file is a TSV file with the following columns:
    - speaker ID
    - path to the audio file
    - transcript

    - We remove the samples of speakers with less than the min. no. of samples, defined
        by the min_utts parameter. It should be at least 2 for ASV evaluation.
    - We remove the samples that are longer than the max. duration, defined by the
        max_duration parameter.
    - We store the speaker ID in the label field. We also store the transcript, which
        may be used in the evaluation. It will not be parsed by NeMo's dataloader.
    - We assume that the audio files are stored in the same folder as the data file,
        under the "clips" folder. This is true for all official Common Voice data files.
    """
    # open the files
    writer = open(dump_file, "w")
    df = pd.read_csv(data_file, sep="\t")
    clips_folder = os.path.join(os.path.dirname(data_file), "clips")
    # remove speakers with less than the min. no. of samples
    df = df.groupby("client_id").filter(lambda x: len(x) >= min_utts)
    # write the data
    for i, row in df.iterrows():
        audiofile = os.path.join(clips_folder, row["path"])
        audio, sample_rate = torchaudio.load(os.path.join(clips_folder, audiofile))
        duration = audio.shape[1] / sample_rate
        # don't add samples that are too long
        if duration <= max_duration:
            writer.write(
                json.dumps(
                    {
                        "path": audiofile.replace(root_folder, "{root}"),
                        "text": row["sentence"],
                        "duration": duration,
                        "label": row["client_id"],  # speaker ID
                        "gender": row["gender"],
                        "accent": row["accents"],
                        "age": row["age"],
                        "dataset": "common_voice"
                    }
                )
                + "\n"
            )
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_file", help="Path to the CV file (TSV)")
    parser.add_argument("dump_file", help="Path to the dump file (TXT)")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    parser.add_argument(
        "--max_duration", type=int, help="Max. duration of the samples (in seconds)"
    )
    parser.add_argument(
        "--min_utts", type=int, help="Min. no. of utterances per speaker"
    )
    args = parser.parse_args()
    create_file(
        args.data_file,
        args.dump_file,
        args.root_folder,
        args.max_duration,
        args.min_utts,
    )
