#!/bin/bash
#SBATCH -p a100_normal_q
#SBATCH --account=vos
#SBATCH --time=12:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=out.log
#SBATCH --error=out.err

module load site/tinkercliffs-rome_a100/easybuild/setup
module load Anaconda3/2020.11
source activate tf

python3.9 /home/niche/find_ants/optimization.py
