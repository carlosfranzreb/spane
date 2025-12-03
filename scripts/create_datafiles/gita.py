"""
Create the data file as is expected by the dataset: as a txt file with one dict per
line, with the fields "path": str, "duration": float and "label": int. The speaker ID
is stored in the label field. Whether the speaker has PD is stored in the "pd" field.

We also store speaker information such as gender and age, which are used in the
evaluation.

python scripts/create_datafiles/gita.py /cfs/collections-new/speech_parkinson_corpora/data/gita READ-TEXT data/gita/read_text.txt /cfs/collections-new/speech_parkinson_corpora/data
"""

import os
import json
from argparse import ArgumentParser
import csv

import torchaudio
from tqdm import tqdm


READ_TEXT = """Ayer fui al médico.
¿Qué le pasa? Me preguntó.
Yo le dije: Ay doctor! Donde pongo el dedo me duele.
Tiene la uña rota?
Sí.
Pues ya sabemos queé es. Deje su cheque a la salida."""

def create_file(
    dataset_dir: str, task: str, dump_file: str, root_folder: str, max_duration: int = None
):
    """
    - audio_dir: comprises all the required dataset information:
        - `audios` directory, comprising all the audio files from GITA.
        - `metadata.csv` file with the speaker information.
        - `transcripts` directory with the transcripts in txt files. TODO
    - We remove the samples that are longer than the max. duration, defined by the
        max_duration parameter.
    """

    if max_duration is None:
        max_duration = float("inf")

    # gather the speaker information
    metadata_f = os.path.join(dataset_dir, "metadata.csv")
    reader = csv.DictReader(open(metadata_f))
    spk_info = {row["subject_id"]: row for row in reader}

    # create a writer object for the dump file
    writer = open(dump_file, "w")

    # iterate over the files in the folder
    audios_dir = os.path.join(dataset_dir, "audios")
    for f in tqdm(os.listdir(audios_dir)):
        if not f.endswith(".wav"):
            continue
        
        # get info from fname and check task
        fname = os.path.splitext(f)[0]
        group, audio_task, spk, utt = fname.split("_")
        if audio_task != task:
            continue
        
        # get the transcript
        if task == "READ-TEXT":
            text = READ_TEXT.replace("\n", " ")

        # get the audio duration and check for max. duration
        audiofile = os.path.join(audios_dir, f)
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
                    "pd": len(spk_info[spk]["UPDRS"]) > 0,
                    "gender": spk_info[spk]["sex"],
                    "UPDRS": spk_info[spk]["UPDRS"],
                    "UPDRS-speech": spk_info[spk]["UPDRS-speech"],
                    "H/Y": spk_info[spk]["H/Y"],
                    "age": spk_info[spk]["age"],
                    "age_decade": str(spk_info[spk]["age"][0]),
                    "time_after_diagnosis": spk_info[spk]["time after diagnosis"],
                    "dataset": "gita",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        writer.flush()

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder", help="Path to the GITA directory")
    parser.add_argument("task", help="Task to get the data from (e.g. READ-TEXT)")
    parser.add_argument("dump_file", help="Path to the dump file")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    parser.add_argument(
        "--max_duration", type=int, help="Min. no. of utterances per speaker"
    )
    args = parser.parse_args()
    create_file(
        args.folder, args.task, args.dump_file, args.root_folder, args.max_duration
    )
