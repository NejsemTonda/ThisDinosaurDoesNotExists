#!/bin/bash

#SBATCH --clusters gpulab
#SBATCH --partition gpu-ffa
#SBATCH --job-name dcgan_exec
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 2g
#SBATCH --time 10:00:00
#SBATCH --gpus 1


source ./.venv/bin/activate
python ./dcgan.py --threads=32 --epochs=80 --save_to_dir="models/dcgan/" --resume_from="models/dcgan/checkpoint.pt"
deactivate
