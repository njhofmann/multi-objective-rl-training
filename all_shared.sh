#!/bin/bash
# TODO entropy
for seed in 3 7 15 # 32 43 56 90
do
  for env in cartpole #acrobot mountaincar
  do
    for batch_size in 16 32
    do
      for lr in .01 .001
      do
        for arch in 64 128 256
        do
          for entropy in 0.0 0.01 0.001
          do
            sbatch run_shared.sh $seed $env $lr $arch $entropy $batch_size
          done
        done
      done
    done
  done
done
