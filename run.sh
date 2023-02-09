#!/bin/bash
#SBATCH -p a100_normal_q
#SBATCH --account=vos
#SBATCH --time=12:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=log.out
#SBATCH --error=log.err

module load Anaconda3/2020.11
source activate tf

python3.9 /home/niche/find_ants/main.py