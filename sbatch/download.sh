#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=1
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_download.log

source ~/envs/dtt/bin/activate

# Train a tokenizer on a dataset AND use it to 
# encode that same dataset  
srun dtt download -c configs/config.yaml 