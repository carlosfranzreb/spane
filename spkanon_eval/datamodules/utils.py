import json
import os
import logging

from omegaconf import OmegaConf


LOGGER = logging.getLogger("progress")


def prepare_datafile(stage: str, config: OmegaConf, log_dir: str) -> str:
    """
    Add the root folder to the paths of the audiofiles and store the resulting
    datafiles within the experiment folder, merged into a single datafile named after
    the stage.

    Args:
        stage: stage of the datafiles to create (eval, train_eval, targets)
        config: config object, containing:
        - the datafiles under `data.datasets.{stage}`,
        - the root folder under `data.config.root_folder`,
        - the min. duration under `data.config.min_duration`,
        - the max. duration under `data.config.max_duration`.

    Raises:
        FileExistsError: if the new datafile already exists.

    Returns:
        The path to the new datafile.
    """

    LOGGER.info(f"Preparing datafile for stage {stage}")

    datafiles = config.data.datasets[stage]
    root_folder = config.data.config.root_folder
    min_dur = config.data.config.get("min_duration", 0)
    max_dur = config.data.config.get("max_duration", float("inf"))
    new_df = os.path.join(log_dir, "data", f"{stage}.txt")
    if os.path.exists(new_df):  # datafile already modified
        raise FileExistsError(f"New filepath `{new_df}` already exists")
    else:
        os.makedirs(os.path.dirname(new_df), exist_ok=True)

    prepared_objs = list()
    n_seen_speakers = 0
    for datafile in datafiles:
        too_short, too_long, included = list(), list(), list()
        speaker_ids = list()
        with open(datafile) as reader:
            for line in reader:
                obj = json.loads(line)
                dur = obj["duration"]
                if dur < min_dur:
                    too_short.append(dur)
                    continue
                if dur > max_dur:
                    too_long.append(dur)
                    continue
                included.append(dur)
                obj["path"] = obj["path"].replace("{root}", root_folder)
                spk = datafile + "_" + obj["label"]
                if spk not in speaker_ids:
                    speaker_ids.append(spk)
                obj["speaker_id"] = n_seen_speakers + speaker_ids.index(spk)
                prepared_objs.append(obj)

        n_seen_speakers += len(speaker_ids)

        LOGGER.info(
            f"{len(included)} samples included ({round(sum(included) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_short)} samples too short ({round(sum(too_short) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_long)} samples too long ({round(sum(too_long) / 3600, 3 )} h)"
        )

    prepared_objs.sort(key=lambda x: x["duration"], reverse=True)
    with open(new_df, "w") as new_df_writer:
        for obj in prepared_objs:
            new_df_writer.write(json.dumps(obj) + "\n")

    LOGGER.info(f"Done with datafile prep for stage {stage}")
    return new_df


def sort_datafile(path: str):
    """
    Sort the samples of the datafile according to its duration in reverse order (the
    longest sample goes first), and dump them again into the same file.
    """
    objs = [json.loads(line) for line in open(path)]
    objs.sort(key=lambda x: x["duration"], reverse=True)
    with open(path, "w") as f:
        for obj in objs:
            f.write(json.dumps(obj) + "\n")
