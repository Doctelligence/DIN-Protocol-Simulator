#!/bin/bash

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=48
#SBATCH --partition=short
#SBATCH --job-name=fl-100
#SBATCH --time=00:40:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=abraham.nash@cs.ox.ac.uk

module purge
module load Anaconda3/2020.11
module load foss/2020a

source activate $DATA/mpienv
export FL_CONFIG_FILE=~/myspace/fl-main/Mobilenetv1-Fl/config_file/test_config.yaml
mpirun -n 100 python DistributedTrainer.py

