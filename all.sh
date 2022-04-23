#!/bin/bash
for seed in 3 7 15 # 32 43 56 90
do
  for network in separate shared 
  do
    for train in linear-sum # mmdm
    do
      for env in acrobot mountaincar #cartpole
      do
        sbatch run.sh $seed $env $network $train 
      done
    done
  done
done
