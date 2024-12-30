#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=universe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=./slurmlogs/%j_download.log

source ~/envs/dtt/bin/activate

srun dtt download -c configs/config.yaml 