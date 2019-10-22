# #!/bin/bash
# 
set -x
# 
# # Don't download stuff to the git repo, that's messy.
pushd ${HOME}
# 
# Update packages
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install bzip2
sudo apt-get tmux
sudo apt-get htop

ANACONDA_INSTALLER="Anaconda3-5.2.0-Linux-x86_64.sh"
wget "https://repo.continuum.io/archive/$ANACONDA_INSTALLER"
bash "$ANACONDA_INSTALLER"

source ${HOME}/.bashrc

${HOME}/anaconda3/bin/pip install --upgrade pip
${HOME}/anaconda3/bin/jupyter notebook --generate-config

# Copy Jupyter config
popd
mkdir ${HOME}/.jupyter
cp -v $(dirname $0)/support/jupyter_notebook_config.py ${HOME}/.jupyter/jupyter_notebook_config.py

cd ${HOME}/quora/misc/
conda env create -f enviro_debian.yml

cd ${HOME}/quora/eda
gdown https://drive.google.com/uc?id=1DRNWMId4T-0qP6EGBmMqbPWErUpuSS-n
cd ${HOME}/quora/embeddings
gdown https://drive.google.com/uc?id=1yCmhJJq0uJhDgjlW4PzfXlXhbs4DpIIK
cd ${HOME}
