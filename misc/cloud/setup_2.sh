#!/bin/bash

cd ${HOME}/quora/eda
gdown https://drive.google.com/uc?id=1DRNWMId4T-0qP6EGBmMqbPWErUpuSS-n
cd ${HOME}/quora/embeddings
gdown https://drive.google.com/uc?id=1yCmhJJq0uJhDgjlW4PzfXlXhbs4DpIIK
cd ${HOME}

git clone https://github.com/Existantialdredd/vimrc.git ~/vimrc
cp ${HOME}/vimrc/.vimrc ~

cd ${HOME}/quora/misc/
conda env create -f enviro_debian.yml
