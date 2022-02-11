#!/bin/bash

# Job name:
#SBATCH --job-name=slip
#SBATCH --account=co_songlab
#SBATCH --qos=songlab_htc3_normal
#SBATCH --partition=savio3_htc
#
# Number of nodes:
#SBATCH --nodes=1
#
#SBATCH --ntasks-per-node=40
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=06:00:00
#
# Memory required:
#SBATCH --mem=10G
#
# Output and email
#SBATCH --output=/global/scratch/projects/fc_songlab/nthomas/slip/log/regression_%j.out
#SBATCH --error=/global/scratch/projects/fc_songlab/nthomas/slip/log/regression_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nthomas@berkeley.edu
## Command(s) to run:
module load ml/tensorflow/2.5.0-py37
source env activate slip
python run_regression_main.py $1
