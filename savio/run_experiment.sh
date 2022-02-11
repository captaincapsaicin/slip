#!/bin/bash

# Job name:
#SBATCH --job-name=slip
#
# Account:
#SBATCH --account=co_songlab
#
# QoS:
#SBATCH --qos=songlab_htc3_normal
#
# Partition:
#SBATCH --partition=savio3_htc
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=1
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
## Command(s) to run:
module load ml/tensorflow/2.5.0-py37
source env activate slip
python run_regression_main.py $1
