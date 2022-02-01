#!/bin/bash
#SBATCH --job-name=landmark_retrieval_extraction
#SBATCH --output=%x.o%j
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate landmark-retrieval

# Run inference
python ~/google-landmark-retrieval-2021/src/scripts/test.py $WORKDIR/landmark-retrieval/models/b4_m0_380/latest_checkpoint.pth -d $WORKDIR/landmark-retrieval/index --o $WORKDIR/landmark-retrieval --efficientnet efficientnet-b4 --feature-size 512 --batch-size 32 --image-size 380 --num-workers 8 --incomplete-model
