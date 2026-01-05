#!/bin/bash

#SBATCH --clusters gpulab
#SBATCH --partition gpu-ffa
#SBATCH --job-name vae_exec
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 2g
#SBATCH --time 10:00:00
#SBATCH --gpus 1


source ./.venv/bin/activate
python ./vae.py --epochs=180 --threads=32 --save_to_dir="models/vae/" --resume_from="models/vae/vae_model.pt"
deactivate
