"""
Script to create a datafile for the MSP Podcast, given the labels_consensus.json file.

Available test sets:
- Test1 set: We use segments from 237 speakers - 30,647 segments
- Test2 set: We select 117 podcasts to create this test set. Instead of retrieving the samples using machine learning models, we randomly select 14,815 segments from 117 speakers. Segments from these 117 podcasts are not included i
n any other partition.
- A new test set (Test3) has been introduced, consisting of 2,347 unique segments with
balanced representation based on primary categorical emotions (Anger, Sadness, Happiness,
Surprise, Fear, Disgust, Contempt, Neutral).

Dimensions
- valence (1-very negative; 7-very positive)
- arousal (1-very calm; 7-very active)
- dominance (1-very weak; 7-very strong)
"""

import os
import json
from argparse import ArgumentParser

import torchaudio


EMOTIONS = {
    "N": "neutral",
    "X": "no agreement",
    "H": "happy",
    "O": "other",
    "A": "angry",
    "S": "sad",
    "F": "fear",
    "D": "disgust",
    "C": "contempt",
    "U": "surprise",
}


def create_file(folder: str, dataset: str, dump_file: str, root_folder: str):
    """ """

    writer = open(dump_file, "w")
    labels_f = os.path.join(folder, "Labels", "labels_consensus.json")
    labels = json.load(open(labels_f))
    for file, annotations in labels.items():

        if annotations["Split_Set"] != dataset:
            continue

        audiofile = os.path.join(folder, "Audios", file)

        # load the audio and get the duration
        path = os.path.join(folder, audiofile)
        audio, sample_rate = torchaudio.load(path)
        duration = audio.shape[1] / sample_rate

        # write the line to the dump file
        obj = {
            "path": path.replace(root_folder, "{root}"),
            "duration": round(duration, 2),
            "label": annotations["SpkrID"],
            "gender": annotations["Gender"],
            "utt_emotion": EMOTIONS[annotations["EmoClass"]],
            "utt_arousal": annotations["EmoAct"],
            "utt_dominance": annotations["EmoDom"],
            "utt_valence": annotations["EmoVal"],
            "dataset": "msp_podcast",
        }
        writer.write(json.dumps(obj) + "\n")

    writer.close()


if __name__ == "__main__":
    # define and parse the arguments
    parser = ArgumentParser()
    parser.add_argument("folder", help="Path to the MSP-Podcast folder")
    parser.add_argument(
        "dataset",
        help="Dataset that you want to store (called 'Split_Set' in the annotations)",
    )
    parser.add_argument("dump_file", help="Path to the dump file (TXT)")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    args = parser.parse_args()

    # run the script
    create_file(args.folder, args.dataset, args.dump_file, args.root_folder)
