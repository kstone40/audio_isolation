#!/bin/bash

module load anaconda3

#conda activate audio

conda activate /SFS/user/ry/stonekev/miniconda3/envs/audio/

cd /SFS/user/ry/stonekev/audio/audio_isolation/

python train_script.py #--config config/test_auto.yml