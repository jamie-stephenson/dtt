#!/bin/bash
# Script to set up local enivronment on each node 

mount_dir="$1"

current_host=$(hostname)

if [ "$current_host" = "node00" ]; then
    git clone https://github.com/jamie-stephenson/dtt.git $mount_dir/dtt
    mkdir -p $mount_dir/dtt/slurmlogs
fi

#-PYTHON ENVIRONMENT--
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_PROJECT_ENVIRONMENT="~/envs/dtt"
uv --project $mount_dir/dtt venv ~/envs/dtt --python 3.11
source ~/envs/dtt/bin/activate
uv pip install --project $mount_dir/dtt/ $mount_dir/dtt/
deactivate
#---------------------     

#------OPEN MPI-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y openmpi-bin openmpi-common libopenmpi-dev 
#---------------------