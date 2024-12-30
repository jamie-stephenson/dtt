#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=universe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=
#SBATCH --time=1:00:00
#SBATCH --output=./slurmlogs/%j_tokenize.log

source ~/envs/dtt/bin/activate

# Train a tokenizer on a dataset AND use it to 
# encode that same dataset  
mpirun --bind-to none --mca btl_tcp_if_include eno1 dtt tokenize -c configs/config.yaml
srun --ntasks=2 --ntasks-per-node=1 cp -r data/* ~/data/