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
conda activate multi-objective-rl
python main.py --save-dirc exp_results/cartpole_linear_sum_hp --seed $1 --env $2 --agent-method shared --lr $3 --arch $4 --train-method linear-sum --entropy-weight $5 --batch-size $6