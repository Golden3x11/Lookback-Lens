#!/bin/bash
#SBATCH --partition plgrid-gpu-a100
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1G
#SBATCH --time 2:00:00
#SBATCH --job-name jupyter-notebook-tunnel
#SBATCH --output jupyter-log-%J.txt
#SBATCH --gpus=1

 
## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
user=$USER
 
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -o ServerAliveInterval=300 -N -L $ipnport:$ipnip:$ipnport ${user}@ares.cyfronet.pl
    -----------------------------------------------------------------
 
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
 
# module load jupyterlab/3.1.6-gcccore-11.2.0 scipy-bundle/2021.10-intel-2021b
 
## start an ipcluster instance and launch jupyter server
jupyter-lab --no-browser --port=$ipnport --ip=$ipnip
