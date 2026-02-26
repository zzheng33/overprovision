#!/bin/bash

module use /soft/modulefiles
module load cuda/12.3.0
module load cmake
module load gcc/12.2.0  
module load openmpi/4.1.1-gcc
module load public_mkl/2019
export CUDA_DIR=/soft/compilers/cuda/cuda-12.3.0
export PCM_NO_MSR=1
export PCM_KEEP_NMI_WATCHDOG=1

dcgmi dmon -i 0 -e 1008,1007,1006,1002,100,155,1005 -d 300 -c 1



# # suite 0: ECP 
# # suite 1: ALTIS
# # suite 2: ML
# # suite 3: Hec
# # suite 4: spec


/home/ac.zzheng/env/ml/bin/python3 exp_power_cap.py --suite 2










