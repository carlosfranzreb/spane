# Speaker Anonymization Evaluation (SpAnE)

![Workflow badge](https://github.com/carlosfranzreb/spkanon/actions/workflows/build.yml/badge.svg)

Evaluation framework for speaker anonymization models.

## Installation

The evaluation framework can be installed with `pip install .`. Alternatively, the script `build/framework.sh` creates a conda environment and installs the framework there, as well as `ffmpeg`. Audiofiles are loaded with `torchaudio`. To load MP3 files, `ffmpeg` is required. You can install it with `conda install 'ffmpeg<5'`, as is done in the build script.

If you want to evaluate the naturalness of your synthesized speech with NISQA, clone the repository: `git clone https://github.com/gabrielmittag/NISQA.git`. This is also done in the build script.

Once the framework is installed, you can run the tests with `python -m unittest discover -s spkanon_eval/tests`. We expect you to clone this repository inside another one where you implement your models and run the experiments, as shown below. The tests also assume this. The test audio files are already part of the framework.

### Expected structure

```linux
my_repo/
  my_models/
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

## Existing anonymization models

We have moved the anonymization models to a separate repository, as well as the build scripts required for them. They are:

- **STT-TTS with Whisper & FastPitch**: extracts the text from the input speech and synthesizes it with one of the 20 FastPitch target speakers.
- **StarGANv2-VC**: voice conversion model trained with 20 target speakers of VCTK.
- **SoftVC**: install it with build/softvc.sh does not require any further installation. Read more about this pipeline here.

You can find the components, build instructions and evaluation results in the `spkanon_models` repository: <https://github.com/carlosfranzreb/spkanon_models>.

## Evaluate your anonymization model

To evaluate your own model, you have to implement the required wrappers. We also have implemented several components which you might find useful. Read about them [here](docs/components.md). You can also look at the existing anonymization models to learn more about this framework. They are stored in [this repository](https://github.com/carlosfranzreb/spkanon_models).

Alternatively, you can define an `infer` method on your model and replace the current model in the `spkanon_eval/main.py` file. The `infer` method should anonymize and unpad batches. See `featex_eval.anonymizer.Anonymizer.infer` to learn how we do it.

## Citation

```tex
@inproceedings{franzreb2023comprehensive,
  title={A Comprehensive Evaluation Framework for Speaker Anonymization Systems},
  author={Franzreb, Carlos and Polzehl, Tim and Moeller, Sebastian},
  booktitle={Proc. 3rd Symposium on Security and Privacy in Speech Communication},
  year={2023},
}
```
