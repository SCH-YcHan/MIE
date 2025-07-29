#!/bin/bash

# Activate the HuBMAP virtual environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate HuBMAP

# Set Python encoding to UTF-8
export PYTHONIOENCODING=UTF-8

# Change to the target directory
cd ~/Desktop/glomer

# Define arrays for the models, folds and their respective paths
models=("unet-x2-glomer" "unet-x3-glomer" "unet-x4-glomer")
folds=("fold0" "fold1" "fold2" "fold3" "fold4")

# Loop through the models and folds and run the tests
for model in "${models[@]}"; do
  for fold in "${folds[@]}"; do
    for iter_model in "${models[@]}"; do
      work_dir="work_dirs_test/${model}-${iter_model}-${fold}"
      config_file="work_dirs/${model}-${fold}/${model}-${fold}.py"
      checkpoint_file="work_dirs/${iter_model}-${fold}/iter_80000.pth"
      python mmsegmentation/tools/test.py ${config_file} ${checkpoint_file} --work-dir ${work_dir}
    done
  done
done