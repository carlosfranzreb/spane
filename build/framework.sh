# move repository to a different directory
mkdir spkanon
mv * spkanon/
mv .[^.]* spkanon/ 2>/dev/null

# create and activate conda environment
conda update -n base -c defaults conda
conda create -p ./venv python=3.11 -y
eval "$(conda shell.bash hook)" && conda activate ./venv

# ffmpeg is required to load MP3 files
conda install -y 'ffmpeg<5'

# install pip dependencies
pip install --no-input --upgrade pip
pip install --no-input spkanon_eval

# clone NISQA
cd spkanon_eval
git clone https://github.com/gabrielmittag/NISQA.git
cd ..

# run tests
python -m unittest discover -s spkanon_eval/tests -p "test_*.py"