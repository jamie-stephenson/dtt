#!/bin/bash
# Script to set up local enivronment on each node 

mount_dir="$1"

#-PYTHON ENVIRONMENT--
# If you want a specific python version you can use deadsnakes:
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 -y install python3.12-venv
python3 -m venv ~/envs/dtt
source ~/envs/dtt/bin/activate
pip install git+https://github.com/jamie-stephenson/dtt.git
deactivate
#---------------------     

#------OPEN MPI-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y openmpi-bin openmpi-common libopenmpi-dev 
#---------------------