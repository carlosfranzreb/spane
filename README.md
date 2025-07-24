# Speaker Anonymization Evaluation (SpAnE)

![Build and test](https://github.com/carlosfranzreb/spkanon_eval/actions/workflows/build.yml/badge.svg)

Evaluation framework for speaker anonymization models.

## Installation

The framework can be installed with `pip install .`, but requires system packages and a second repository (NISQA) for the naturalness evaluation.
The script `build/framework.sh` creates a conda environment, installs the framework there and runs the tests.
Please run

Not that we expect this repository to be installed inside another one where you implement your models and run the experiments, as shown below.
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

## Full results of the SPSC 2023 paper

The results that were used on the aforementioned paper can be found on a previous commit of this repository. We have removed them from the current version to simplify the repository. Here is a link under which the results can be found: <https://github.com/carlosfranzreb/spkanon_eval/tree/28f27eb>. The notebooks summarizing the results are under `scripts`.

## Existing anonymizers

We have moved the anonymization models to a separate repository, as well as the build scripts required for them.
You can find the components, build instructions and evaluation results in the `spkanon_models` repository: <https://github.com/carlosfranzreb/spkanon_models>.

## Evaluate your anonymizer

To evaluate your own model, you have to implement the required wrappers. We also have implemented several components which you might find useful.
Read about them in the [component documentation](docs/components.md).

Alternatively, you can define an `infer` method on your model and replace the current model in the `spkanon_eval/main.py` file.
The `infer` method should anonymize and unpad batches.
See `featex_eval.anonymizer.Anonymizer.infer` to learn how we do it.

## Citation

```tex
@inproceedings{franzreb2023comprehensive,
  title={A Comprehensive Evaluation Framework for Speaker Anonymization Systems},
  author={Franzreb, Carlos and Polzehl, Tim and Moeller, Sebastian},
  booktitle={Proc. 3rd Symposium on Security and Privacy in Speech Communication},
  year={2023},
}
```
