"""
RAVDESS is an emotional dataset, where the information of each utterance is summarized
in the filename. The filename is of the form:
    03-01-06-01-02-01-12.wav

Filename identifiers

1. Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
2. Vocal channel (01 = speech, 02 = song).
3. Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
4. Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
5. Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
6. Repetition (01 = 1st repetition, 02 = 2nd repetition).
7. Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""

import os
import json
from argparse import ArgumentParser

import torchaudio


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]
INTENSITIES = ["normal", "strong"]
STATEMENTS = ["Kids are talking by the door", "Dogs are sitting by the door"]
REPETITIONS = [1, 2]


def create_file(folder, dump_file, root_folder):
    """
    - The Librispeech folder contains one folder per speaker, and each of those folders,
    contains a chapter, and that chapter folder contains FLAC files.
    - The SPEAKERS.TXT file is one folder above the given folder, and it contains the
    gender of each speaker.
    """

    # create a writer object for the dump file
    writer = open(dump_file, "w")

    # iterate over the files
    for actor_dir in os.listdir(folder):
        if not actor_dir.startswith("Actor_"):
            continue
        for audiofile in os.listdir(os.path.join(folder, actor_dir)):
            # skip non-audio files
            if not audiofile.endswith(".wav"):
                continue

            # load the audio and get the duration
            path = os.path.join(folder, actor_dir, audiofile)
            audio, sample_rate = torchaudio.load(path)
            duration = audio.shape[1] / sample_rate

            # write the line to the dump file
            obj = {
                "path": path.replace(root_folder, "{root}"),
                "duration": round(duration, 2),
                "label": actor_dir,
                **get_utt_info(audiofile),
            }
            writer.write(json.dumps(obj) + "\n")

    writer.close()


def get_utt_info(filename):
    """
    Given the filename of a RAVDESS audio file, return a dict with the following keys:
        - emotion, intensity, statement, repetition

    "utt_" is prepended to each key, to highlight that these characteristics are
    specific to the utterance, and not to the speaker.
    """

    data = [int(i) - 1 for i in os.path.splitext(filename)[0].split("-")]
    return {
        "gender": "M" if data[-1] % 2 == 0 else "F",
        "utt_emotion": EMOTIONS[data[2]],
        "utt_intensity": INTENSITIES[data[3]],
        "text": STATEMENTS[data[4]],
        "utt_text": STATEMENTS[data[4]].split()[0],
        "utt_repetition": REPETITIONS[data[5]],
        "dataset": "ravdess"
    }


if __name__ == "__main__":
    # define and parse the arguments
    parser = ArgumentParser()
    parser.add_argument(
        "folder", help="Path to the RAVDESS folder (Audio_Speech_Actors_01-24)"
    )
    parser.add_argument("dump_file", help="Path to the dump file (TXT)")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    args = parser.parse_args()

    # run the script
    create_file(args.folder, args.dump_file, args.root_folder)
