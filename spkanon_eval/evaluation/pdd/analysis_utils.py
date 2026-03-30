"""Helper functions to analyze the results of the PDD evaluation."""

import numpy as np
from sklearn.metrics import classification_report


def analyse_func(indices: np.ndarray, data: np.ndarray) -> list[str]:
    """
    - Data is Nx2 array with columns (x, y)
    - Filter the samples with the given indices.
    - Return the class-specific and weighted precisions, recalls and F1-scores.
        See `headers_func` below for their order.
    """
    this_data = data[indices]
    report = classification_report(this_data[:, 1], this_data[:, 0], output_dict=True)
    results = list()
    for key in ["0", "1", "weighted avg"]:
        if key not in report:
            results.extend([-1., -1., -1., 0.])
            continue

        results.extend(list(report[key].values()))

    results = [str(round(v, 2)) for v in results]
    return results


def headers_func(dump_file: str, key: str = None):
    """
    Create the header of the dump file. If a key is given, write it to the right of the
    datafile. Here are the meanings of the abbreviations:
    - hc: healthy control
    - pd: Parkinson's disease
    - w: weighted average
    - p, r, f1, s: precision, recall, f1-score, support
    """
    with open(dump_file, "w") as f:
        f.write("dataset ")
        if key is not None:
            f.write(f"{key} ")

        f.write("hc_p hc_r hc_f1 hc_s pd_p pd_r pd_f1 pd_s w_p w_r w_f1 w_s\n")
