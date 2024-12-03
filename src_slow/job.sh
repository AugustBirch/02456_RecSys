#!/bin/sh
#BSUB -J Train
#BSUB -o logs/Train%J.out
#BSUB -e logs/Train%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 5
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8G]"
#BSUB -W 12:00
#BSUB -N

#BSUB 
# end of BSUB options

module load cuda/12.3.2
module load cudnn/v8.9.7.29-prod-cuda-12.X
module load tensorrt/8.6.1.6-cuda-12.X
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.3.2

python3 train.py