#!/bin/bash

# slurm options
#SBATCH -c 20
#SBATCH -o train.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Loading the required module
echo "Setting up environment..."
source /home/gridsan/ndoshi/venvs/pytorch-basic/bin/activate

# Run the script
echo "Starting training..."
python train.py
echo "Should I be here?"
