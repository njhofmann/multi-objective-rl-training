- login into cluster
- switch to compute node: `srun --partition=short --nodes=1 --cpus-per-task=1 --pty /bin/bash`
- install latest version of Miniconda: `wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- start install: `bash Miniconda-3-latest-Linux-x86_64.sh`
- activate Miniconda env: `source/miniconda3/bin/activate`
- create new Conda env: `conda creat --name <env-name> python=<version>`
- activate: `conda activate <env-name>`
- load CUDA: `module load cuda/11.0`
  - check version with `nvcc --version`
- install TensorFlow & dependencies: 
  - `conda install -c anaconda tensorflow-gpu=2.4`
  - `conda install -c conda-forge gym=0.19.0`  
- check desired TensorFlow version matches CUDA version
- test install with: `python -c 'import tensorflow as tf;  print(tf.test.is_built_with_cuda())'`
  - should be on a GPU node