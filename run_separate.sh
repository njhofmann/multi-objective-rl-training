#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --job-name=run
#SBATCH --mem=16G
#SBATCH --gres=gpu:p100:1
#SBATCH --output=sbatch/exec.%j.out

export CUBLAS_WORKSPACE_CONFIG=:16:8
module load cuda/11.0
echo nvcc --version
conda activate mulit-objective-rl
python main.py --seed $1 --env $2 --agent-method $3 --train-method $4