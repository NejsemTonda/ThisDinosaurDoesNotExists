#!/bin/bash

#SBATCH --clusters parlab
#SBATCH --partition ffa
#SBATCH --job-name dcgan_exec
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 2g
#SBATCH --time 10:00:00


source ./.venv/bin/activate
python ./dcgan.py --threads=32 --epochs=1 --save_model=True --resume_training=True
deactivate
