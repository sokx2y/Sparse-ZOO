#!/bin/bash
source /capsule/home/hzheng/cuda.sh
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/capsule/home/xiangyuxing/hf_home

source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/torch210cu118


python single_linear_sim.py \
    --batch_size 16 \
    --cxt_len 64 \
    --in_features 1024 \
    --out_features 1024