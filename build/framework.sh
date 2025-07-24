# ! This script should be called from the parent folder of the repository

# create and activate conda environment
conda update -n base -c defaults conda
conda create -p ./venv python=3.10 -y
eval "$(conda shell.bash hook)" && conda activate ./venv

# ffmpeg is required to load MP3 files
conda install -y 'ffmpeg<5'

# install pip dependencies
pip install --no-input --upgrade pip
pip install --no-input -e ./spkanon_eval

# clone NISQA
git clone https://github.com/gabrielmittag/NISQA.git ./spkanon_eval/NISQA
export PYTHONPATH=$(pwd)/spkanon_eval:$PYTHONPATH

# run tests
python -m unittest discover -s spkanon_eval/tests -p "test_*.py"