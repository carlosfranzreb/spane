"""
Create the data file as is expected by the dataset: as a txt file with one dict per
line, with the fields "path": str, "duration": float and "label": int. The speaker ID
is stored in the label field. Whether the speaker has PD is stored in the "pd" field.

We also store speaker information such as gender and age, which are used in the
evaluation. The structure expected for the dataset is the same as Gimeno's:
<https://github.com/david-gimeno/interpreting-ssl-parkinson-speech>

To use them as targets for PD detection, I normalize them with ffmpeg normalize:

> os.system(f"ffmpeg-normalize {path} -o {dump_path} -ar 16000 -f")
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

SENTENCE_MAPPING = {
    "JUAN": "JUAN SE ROMPIÓ UNA PIERNA CUANDO IBA EN LA MOTO",
    "LAURA": "LAURA SUBE AL TREN QUE PASA",
    "LOSLIBROS": "LOS LIBROS NUEVOS NO CABEN EN LA MESA DE LA OFICINA",
    "LUISA": "LUISA REY COMPRA EL COLCHÓN DURO QUE TANTO LE GUSTA",
    "MICASA": "MI CASA TIENE TRES CUARTOS",
    "OMAR": "OMAR QUE VIVE CERCA TRAJO MIEL",
    "PREOCUPADO": "ESTOY MUY PREOCUPADO CADA VEZ ME ES MÁS DIFÍCIL HABLAR",
    "PRECUPADO": "ESTOY MUY PREOCUPADO CADA VEZ ME ES MÁS DIFÍCIL HABLAR",
    "ROSITA": "ROSITA NIÑO QUE PINTA BIEN DONÓ SUS CUADROS AYER",
    "TRISTE": "ESTOY MUY TRISTE AYER VI MORIR A UN AMIGO",
    "VISTE": "VISTE LAS NOTICIAS YO VI GANAR LA MEDALLA DE PLATA EN PESAS. ESE MUCHACHO TIENE MUCHA FUERZA",
}


def create_file(
    dataset_dir: str,
    splits_dir: str,
    task: str,
    dump_file: str,
    root_folder: str,
    max_duration: int = None,
):
    """
    - `dataset_dir` comprises all the required dataset information:
        - `norm_audios` directory, comprising all the audio files from GITA.
            Audio normalization is done with ffmpeg-normalize.
        - `metadata.csv` file with the speaker information.
    - `splits_dir` defines the test set of each fold. Only these test files are
        retrieved, and marked with their corresponding fold. `splits_dir` should
        contain 5 subdirectories named `fold_0`, `fold_1`, etc.
    - We remove the samples that are longer than the max. duration, defined by the
        max_duration parameter.
    """

    if max_duration is None:
        max_duration = float("inf")

    # gather the speaker information
    metadata_f = os.path.join(dataset_dir, "metadata.csv")
    reader = csv.DictReader(open(metadata_f))
    spk_info = {row["subject_id"]: row for row in reader}

    # get the task's test files of each fold from the splits directory
    test_files = list()
    for fold_idx in range(5):
        fold = f"fold_{fold_idx}"
        test_f = os.path.join(splits_dir, fold, "test.csv")
        reader = csv.DictReader(open(test_f))
        test_files.append([])
        for row in reader:
            if task != row["task_id"]:
                continue

            label = "PD" if row["label"] == "1" else "HC"
            fname = f"{label}_{task}_{row['sample_id']}"
            test_files[-1].append(fname)

    # create a writer object for the dump file
    writer = open(dump_file, "w")

    # iterate over the files in the folder
    audios_dir = os.path.join(dataset_dir, "norm_audios")
    for f in tqdm(os.listdir(audios_dir)):
        if not f.endswith(".wav"):
            continue

        # find fold or continue if not found
        fname = os.path.splitext(f)[0]
        fold_idx = -1
        for idx, files in enumerate(test_files):
            if fname in files:
                fold_idx = idx
                break

        if fold_idx == -1:
            continue

        # get info from fname and check task
        fname = os.path.splitext(f)[0]
        group, audio_task, spk, utt = fname.split("_")
        if audio_task != task:
            continue

        # get the transcript
        match task:
            case "SUSTAINED-VOWELS":
                text = utt[0]
            case "WORDS" | "DDK":
                text = utt
            case "READ-TEXT":
                text = READ_TEXT.replace("\n", " ")
            case "SENTENCES":
                text = SENTENCE_MAPPING[utt]
            case _:
                text = ""

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
                    "is_male": spk_info[spk]["sex"] == "M",
                    "UPDRS": spk_info[spk]["UPDRS"],
                    "UPDRS-speech": spk_info[spk]["UPDRS-speech"],
                    "H-Y": spk_info[spk]["H/Y"],
                    "age": spk_info[spk]["age"],
                    "age_decade": str(spk_info[spk]["age"][0]),
                    "time_after_diagnosis": spk_info[spk]["time after diagnosis"],
                    "fold": fold_idx,
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
    parser.add_argument("dataset_dir", help="Path to the GITA directory")
    parser.add_argument("splits_dir", help="Path to the GITA splits")
    parser.add_argument("task", help="Task to get the data from (e.g. READ-TEXT)")
    parser.add_argument("dump_file", help="Path to the dump file")
    parser.add_argument("root_folder", help="Path that will be replaced with {root}")
    parser.add_argument(
        "--max_duration", type=int, help="Max. duration for filtering utterances"
    )
    args = parser.parse_args()
    create_file(
        args.dataset_dir,
        args.splits_dir,
        args.task,
        args.dump_file,
        args.root_folder,
        args.max_duration,
    )
