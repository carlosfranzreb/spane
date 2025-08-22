# Speaker Anonymization Evaluation (SpAnE)

![Build and test](https://github.com/carlosfranzreb/spkanon_eval/actions/workflows/build.yml/badge.svg)
![coverage badge](./coverage.svg)

Evaluation framework for speaker anonymizers.

## Installation

The framework can be installed with `pip`, but requires system packages and a second repository (NISQA) for the naturalness evaluation.
The script `build/framework.sh` creates a conda environment, installs the framework there and runs the tests.
It is also the script used in the GitHub CD pipeline.
Please use it to install the framework.

Note that we expect this repository to be installed inside a directory where you implement your anonymizers and run the experiments, as shown below.
The build script and the tests also assume this.

### Expected structure

```linux
my_repo/
  venv/
  logs/
  spkanon_eval/
    spkanon_eval/
    tests/
    spkanon_models
    NISQA/
    ...
```

## Paper results

Here are links to the results of the papers I've written using the framework.

### SPSC 2023: A Comprehensive Evaluation Framework for Speaker Anonymization Systems

```tex
@inproceedings{franzreb2023comprehensive,
  title={A Comprehensive Evaluation Framework for Speaker Anonymization Systems},
  author={Franzreb, Carlos and Polzehl, Tim and Moeller, Sebastian},
  booktitle={Proc. 3rd Symposium on Security and Privacy in Speech Communication},
  year={2023},
}
```

Introduces the framework, including the utility evaluation with pre-trained models and the use of the [EdAcc](https://groups.inf.ed.ac.uk/edacc/) for the privacy evaluation. 
I have removed them from the current version to simplify the repository.
Here is the [commit under which the results can be found](https://github.com/carlosfranzreb/spkanon_eval/tree/28f27eb).
The notebooks summarizing the results are under `scripts`.

### SPSC 2025: Optimizing the Dataset for the Privacy Evaluation of Speaker Anonymizers

Introduces the Librispeech training and evaluation datasets, after experimenting with different configurations.
The results are published as a [release of this repository](https://github.com/carlosfranzreb/spane/releases/tag/paper_results).

### Interspeech 2025: Private kNN-VC: Interpretable Anonymization of Converted Speech

```tex
@inproceedings{franzreb25_interspeech,
  title     = {{Private kNN-VC: Interpretable Anonymization of Converted Speech}},
  author    = {{Carlos Franzreb and Arnab Das and Tim Polzehl and Sebastian Möller}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{3224--3228}},
  doi       = {{10.21437/Interspeech.2025-820}},
}
```

This anonymizer was run with SpAnE and evaluated with the [VPC 2024 framework](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/tree/main).
The results, a self-contained demo and some samples can be found as [a separate repository](https://github.com/carlosfranzreb/private_knnvc).
Private kNN-VC is also part of the [spkanon_models repository](https://github.com/carlosfranzreb/spkanon_models).

### Pre-print: Improving the Speaker Anonymization Evaluation's Robustness to Target Speakers with Adversarial Learning

[Link to pre-print](https://www.arxiv.org/abs/2508.09803).

Note that this work has not been reviewed yet.
The results are published as a [release of this repository](https://github.com/carlosfranzreb/spane/releases/tag/paper_results_2).

## Existing anonymizers

We have moved the anonymizers to a separate repository, as well as the build scripts required for them.
You can find the components, build instructions and evaluation results in the [spkanon_models repository](https://github.com/carlosfranzreb/spkanon_models).

## Evaluate your anonymizer

To evaluate your own model, you have to implement the required wrappers. We also have implemented several components which you might find useful.
Read about them in the [component documentation](docs/components.md).

Alternatively, you can define an `infer` method on your model and replace the current model in the `spkanon_eval/main.py` file.
The `infer` method should anonymize and unpad batches.
See `spane.anonymizer.Anonymizer.infer` to learn how we do it.

