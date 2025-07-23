# create and activate conda environment
conda update -n base -c defaults conda
conda create -p ./venv python=3.11 -y
eval "$(conda shell.bash hook)" && conda activate ./venv

# ffmpeg is required to load MP3 files
conda install -y 'ffmpeg<5'

# install pip dependencies
pip install --no-input --upgrade pip
pip install --no-input .

# clone NISQA
git clone https://github.com/gabrielmittag/NISQA.git