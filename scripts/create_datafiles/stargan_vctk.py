"""
val. data: src/featproc/models/StarGANv2VC/Data/val_list.txt
"""


import os
import json
from argparse import ArgumentParser
import torchaudio


def create_file(stargan_file, vctk_folder, microphone, dump_file, root_folder):
    """
    - The Librispeech folder contains one folder per speaker, and each of those folders,
    contains a chapter, and that chapter folder contains FLAC files.
    - The SPEAKERS.TXT file is one folder above the given folder, and it contains the
    gender of each speaker.
    """
    # get the parent folder of the given folder
    speakers_file = os.path.join(vctk_folder, "speaker-info.txt")
    # create a writer object for the dump file
    writer = open(dump_file, "w")
    # iterate over the datafile
    for line in open(stargan_file):
        path = line.strip().replace("./Data/", "")
        vctk_speaker, end = path.split("/")  # speaker label as defined in VCTK
        (
            utt_file,
            stargan_speaker,
        ) = end.split(
            "|"
        )  # spk label from StarGAN
        utt = utt_file.replace(".wav", "")  # utterance ID
        if len(utt) < 3:  # prepend zeros to utterance ID
            utt = (3 - len(utt)) * "0" + utt
        text_file = os.path.join(
            vctk_folder, "txt", vctk_speaker, f"{vctk_speaker}_{utt}.txt"
        )
        audiofile = os.path.join(
            vctk_folder,
            "wav48_silence_trimmed",
            vctk_speaker,
            f"{vctk_speaker}_{utt}_mic{microphone}.flac",
        )
        try:
            audio, sample_rate = torchaudio.load(audiofile)
            obj = {
                "path": audiofile.replace(root_folder, "{root}"),
                "text": open(text_file).read().strip(),
                "duration": audio.shape[1] / sample_rate,
                "label": stargan_speaker,
                "vctk_label": vctk_speaker,
            }
            obj.update(get_speaker_data(vctk_speaker, speakers_file))
            writer.write(json.dumps(obj) + "\n")
        except RuntimeError as err:
            print(err)
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
                    "dataset": "vctk",
                }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("stargan_file", help="Path to the StarGAN file")
    parser.add_argument("vctk_folder", help="Path to the VCTK folder")
    parser.add_argument("microphone", help="Microphone ID (1 or 2)")
    parser.add_argument("dump_file", help="Path to the dump file (TXT)")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    args = parser.parse_args()
    create_file(
        args.stargan_file,
        args.vctk_folder,
        args.microphone,
        args.dump_file,
        args.root_folder,
    )
