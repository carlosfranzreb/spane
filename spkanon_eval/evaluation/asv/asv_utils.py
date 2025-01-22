import os
import logging
from typing import Callable

import numpy as np
from sklearn.metrics import roc_curve, RocCurveDisplay
from matplotlib import pyplot as plt
from tqdm import tqdm
import plda

from spkanon_eval.evaluation.analysis import get_characteristics


LOGGER = logging.getLogger("progress")


def compute_eer(
    trials: np.array, enrolls: np.array, llrs: np.array
) -> tuple[np.array, np.array, np.array, int]:
    """
    Compute the equal error rate (EER) for the given LLRs. The EER is the threshold
    that minimizes the absolute difference between the false positive rate (FPR)
    and the false negative rate (FNR). We compute it with sklearn's roc_curve.

    Args:
        trials: shape (n_pairs) - the trial speakers
        enrolls: shape (n_pairs) - the enroll speakers
        llrs: shape (n_pairs) - the log-likelihood ratio of each trial-enroll pair

    Returns:
        fpr: shape (n_pairs) - the false positive rates (FPR) for each threshold
        tpr: shape (n_pairs) - the true positive rates (TPR) for each threshold
        thresholds: shape (n_pairs) - the thresholds
        key: index of the threshold that is closest to the EER
    """
    # check that there are trials and enrolls
    if len(trials) == 0 or len(enrolls) == 0:
        LOGGER.warning("There are no trials or enrolls; cannot compute EER")
        return None, None, None, -1
    # compute the ROC curve
    same_speaker = trials == enrolls
    fpr, tpr, thresholds = roc_curve(same_speaker, llrs)
    # check that there are no NaNs
    if np.any(np.isnan(fpr)):
        LOGGER.warning("There are no different-speaker pairs; cannot compute EER")
        key = -1
    elif np.any(np.isnan(tpr)):
        LOGGER.warning("There are no same-speaker pairs; cannot compute EER")
        key = -1
    # compute the EER threshold
    else:
        key = np.nanargmin(np.absolute(((1 - tpr) - fpr)))
    return fpr, tpr, thresholds, key


def analyse_results(datafile: str, llr_file: str) -> None:
    """
    Compute and dump the EER and ROC curve for the whole dataset and each of its
    subsets w.r.t speaker characteristics present in the file (age, gender, etc.).
    """
    LOGGER.info(f"Analysing ASV results for {datafile}")
    # get the filename of the datafile
    fname = os.path.splitext(os.path.basename(datafile))[0]
    # get the dump folder for this datafile (the LLR file's folder)
    dump_folder = os.path.dirname(llr_file)
    # read the llr file and extract the source & target speakers and the llrs
    trials, enrolls, llrs = np.hsplit(np.load(llr_file), 3)

    # compute the EER of the whole dataset and dump the ROC curve
    LOGGER.info("Computing EER of the whole dataset and dumping ROC curve")
    fpr, tpr, thresholds, key = compute_eer(trials, enrolls, llrs)
    # create the eer file with the headers if it doesn't exist yet
    eer_file = os.path.join(os.path.dirname(dump_folder), "eer.txt")
    if not os.path.exists(eer_file):
        with open(eer_file, "w") as f:
            f.write("dataset n_pairs threshold eer\n")
    # dump the EER and the threshold to the eer file
    eer = (fpr[key] + (1 - tpr[key])) / 2
    with open(eer_file, "a") as f:
        f.write(f"{fname} {llrs.size} {thresholds[key]} {eer}\n")
    # dump the ROC curve
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.plot([1, 0], [0, 1], "k--")
    plt.savefig(os.path.join(dump_folder, "roc_curve.png"))

    # store the spk label for each value of each speaker char.
    speaker_chars, _ = get_characteristics(datafile)

    # for each speaker char., compute the EER for all its values
    for key, values in speaker_chars.items():
        LOGGER.info(f"Computing EER for all values of {key}")
        for value in values:
            # get the indices of the rows that contain the values
            indices = np.where(
                np.logical_and(
                    np.isin(trials, values[value]), np.isin(enrolls, values[value])
                )
            )[0]
            # compute the EER and dump it to the dump file if it was computed
            fpr, tpr, thresholds, eer_key = compute_eer(
                trials[indices], enrolls[indices], llrs[indices]
            )
            if eer_key >= 0:
                dump_file = os.path.join(os.path.dirname(dump_folder), f"eer_{key}.txt")
                eer = (fpr[eer_key] + (1 - tpr[eer_key])) / 2
                if not os.path.exists(dump_file):
                    with open(dump_file, "w") as f:
                        f.write(f"dataset {key} n_pairs threshold eer\n")
                with open(dump_file, "a") as f:
                    f.write(f"{fname} {value} {indices.size} ")
                    f.write(f"{thresholds[eer_key]} {eer}\n")


def compute_llrs(
    plda: plda.Classifier, vecs: np.array, chunk_size: int
) -> tuple[np.array, np.array]:
    """
    Compute the log-likelihood ratios (LLRs) of all pairs of trial and enrollment
    utterances. For each speaker, the first utterance is considered the trial
    utterance and the rest are considered the enrollment utterances.
    The LLRs are calculated as in the plda package, but the code is adapted to our
    use case, where every vector is used multiple times. We therefore compute the
    marginal LLs beforehand once, and use them to compute the LLRs for all pairs.

    Args:
    - vecs: dict with two keys, trials and enrolls, each containing a numpy array
        containing speaker embeddings.

    Returns:
    - llrs: numpy array with the LLRs
    - indices: numpy array containing the indices of trial and enroll utterances
        that were used to compute each LLR. The number of pairs equals all possible
        combinations of trial and enroll utts.
    """
    LOGGER.info("Computing LLRs for all pairs of trial and enrollment utterances")
    return iterate_over_chunks(vecs, compute_llrs_chunk, chunk_size, plda=plda)


def iterate_over_chunks(
    vecs: np.array, chunk_func: Callable, chunk_size: int, **kwargs
) -> tuple[np.array, np.array]:
    """
    Iterate over all possible pairs of trial and enrollment utterances, and compute
    the `chunk_func` output for each chunk of pairs.
    """
    # compute all pairs of trial and enrollment utterances
    indices = np.dstack(
        np.meshgrid(
            np.arange(vecs["trials"].shape[0]), np.arange(vecs["enrolls"].shape[0])
        )
    ).reshape(-1, 2)

    # iterate over chunks of `chunk_size` pairs to avoid memory issues
    scores = None
    for i in tqdm(range(0, indices.shape[0], chunk_size)):
        idx = indices[i : i + chunk_size]
        chunk_scores = chunk_func(vecs, idx, **kwargs)
        scores = (
            np.concatenate((scores, chunk_scores))
            if scores is not None
            else chunk_scores
        )
    return scores, indices


def compute_llrs_chunk(
    vecs: np.array, indices: np.array, plda: plda.Classifier
) -> np.array:
    """Compute the LLRs for the given chunk of trial and enroll utterances."""
    data = {"trials": dict(), "enrolls": dict()}
    for i, key in enumerate(data):
        data[key]["idx"] = indices[:, i]
        data[key]["idx_unique"], data[key]["idx_inverse"] = np.unique(
            data[key]["idx"], return_inverse=True
        )
        data[key]["vecs"] = vecs[key][data[key]["idx_unique"]]
        # add a new dimension to the vectors to conform to PLDA's input format
        data[key]["vecs"] = data[key]["vecs"][:, np.newaxis, :]
        # compute the marginal log-likelihoods of the vectors
        data[key]["lls"] = plda.model.calc_logp_marginal_likelihood(data[key]["vecs"])
        # map vecs and lls back to the original indices
        data[key]["vecs"] = data[key]["vecs"][data[key]["idx_inverse"]]
        data[key]["lls"] = data[key]["lls"][data[key]["idx_inverse"]]
    # compute the LLRs for the current chunk
    pairs = np.concatenate(
        [data["trials"]["vecs"], data["enrolls"]["vecs"]],
        axis=1,
    )
    pair_lls = plda.model.calc_logp_marginal_likelihood(pairs)
    chunk_llrs = pair_lls - (data["trials"]["lls"] + data["enrolls"]["lls"])
    return chunk_llrs


def compute_dists(vecs: np.array, chunk_size: int) -> tuple[np.array, np.array]:
    """
    This is the analogous function to `compute_llrs`, but for the cosine similarity.
    See `compute_llrs` for more details.
    """
    LOGGER.info(
        "Computing spkemb dists. for all pairs of trial and enrollment utterances"
    )
    return iterate_over_chunks(vecs, compute_dists_chunk, chunk_size)


def compute_dists_chunk(vecs: np.array, indices: np.array) -> np.array:
    """
    Compute the cosine similarities for the given chunk of trial and enroll
    utterances.
    """
    trials = vecs["trials"][indices[:, 0]]
    enrolls = vecs["enrolls"][indices[:, 1]]
    dot_product = np.sum(trials * enrolls, axis=-1)
    trials_norm = np.linalg.norm(trials, axis=-1)
    enrolls_norm = np.linalg.norm(enrolls, axis=-1)
    return dot_product / (trials_norm * enrolls_norm)
