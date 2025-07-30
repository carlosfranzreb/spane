#! /bin/bash
# Create a conda environment with the framework
# This script should be called from the parent folder of the repository
conda update -n base -c defaults conda
conda create -p ./venv python=3.10 -y
eval "$(conda shell.bash hook)" && conda activate ./venv

# ffmpeg is required to load MP3 files
conda install -y 'ffmpeg<5'

# install pip dependencies
pip install --no-input --upgrade pip
pip install --no-input -e ./spane

# clone NISQA
git clone https://github.com/gabrielmittag/NISQA.git ./spane/NISQA
export PYTHONPATH=$(pwd)/spane:$PYTHONPATH

# run tests with coverage check
bash ./spane/build/run_tests.sh
