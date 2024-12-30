#!/bin/bash
# Script to set up local enivronment on each node 

mount_dir="$1"

current_host=$(hostname)

if [ "$current_host" = "node00" ]; then
    git clone https://github.com/jamie-stephenson/dtt.git $mount_dir/dtt
    mkdir -p $mount_dir/dtt/slurmlogs
fi

#-PYTHON ENVIRONMENT--
# If you want a specific python version you can use deadsnakes:
sudo NEEDRESTART_MODE=l add-apt-repository -y ppa:deadsnakes/ppa
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 -y install python3.11 python3.11-venv python3.11-dev
python3.11 -m venv ~/envs/dtt
source ~/envs/dtt/bin/activate
pip install $mount_dir/dtt
deactivate
#---------------------     

#------OPEN MPI-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y openmpi-bin openmpi-common libopenmpi-dev 
#---------------------