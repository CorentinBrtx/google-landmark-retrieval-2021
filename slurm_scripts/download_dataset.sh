#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=%x.o%j
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=cpu_long

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate landmark-retrieval

# Run python script
python ~/google-landmark-retrieval-2021/src/data/download.py -d $WORKDIR/landmark-retrieval -b 0 -e 499 -n 10000
