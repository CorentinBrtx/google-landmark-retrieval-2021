#!/bin/bash
#SBATCH --job-name=landmark_retrieval_training
#SBATCH --output=%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=corentin.berteaux@student-cs.fr
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate landmark-retrieval

# Run training
python ~/google-landmark-retrieval-2021/src/scripts/train.py -d $WORKDIR/landmark-retrieval -f 512 -n 1000 -b 32 -lr 1e-3 --image_size 224 --log_interval 10 --num_workers 4