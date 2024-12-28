#!/bin/bash
# Script to set up local enivronment on each node 

mount_dir="$1"

mkdir -p $mount_dir/slurmlogs

#-PYTHON ENVIRONMENT--
# If you want a specific python version you can use deadsnakes:
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 -y install python3.12-venv
python3 -m venv ~/envs/dtt
source ~/envs/dtt/bin/activate
pip install https://github.com/jamie-stephenson/bpekit/releases/download/v0.1.0-test/bpekit-0.1.0-cp310-abi3-linux_x86_64.whl
deactivate
#---------------------     

#------OPEN MPI-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y openmpi-bin openmpi-common libopenmpi-dev 
#---------------------