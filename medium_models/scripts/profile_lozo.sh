#!/bin/bash

# CustomLinear功能配置文件
# 通过profile_lozo.sh调用lozo.sh来实现CustomLinear功能

source /capsule/home/hzheng/cuda.sh
# 可选：为了安全，先取消任何全局 LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
# 再显式注入本次作业需要的路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 加载 Conda 环境
source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/torch210cu118

echo "=== CustomLinear功能配置 ==="

LOG_DIR_PREFIX="testprofile3" # 通过这个设置log保存目录

# ----------------------------
# 每次debug前要修改logprefix （2:base； 3:delta）
# 目前的输出： models中的 logits&loss      lozotrainer中的 inputs loss1&2
# ----------------------------


# CustomLinear功能配置
ENABLE_CUSTOM_LINEAR=false
CUSTOM_LINEAR_PLOT_DIR="./custom_linear_plots"
PLOT_INTERVAL=50 

# diffLinear功能配置
ENABLE_DIFFENRENTIAL_LINEAR=false
ENABLE_DIFFENRENTIAL_VALIDATION=false
ENABLE_ACCURATE_DIFF=true
DIFFERENTIAL_PLOT_DIR="./diff_linear_plots"
DIFFERENTIAL_VALIDATION_FILE="./diff_linear_plots/validation_file"

# quantize diffLinear 功能配置
ENABLE_QDIFFLINEAR=false
ACT_QUANT_PATTERN="per_token"
ACT_BIT=4
WEIGHT_BIT=8
ENABLE_X=true
ENABLE_DIFFX=true
ENABLE_W=true
ENABLE_DIFFW=true
MX_A_ELEM_FORMAT="fp8_e4m3"
MX_DIFFA_ELEM_FORMAT="fp8_e4m3"
MX_W_ELEM_FORMAT="fp4_e2m1"
MX_DIFFW_ELEM_FORMAT="fp4_e2m1"
MX_QUAN=true

APPLY_FORWARD_DELTA=true

rm data/k-shot-1k-test/SST-2/16-42/cached_*

# 基本任务配置
TASK=SST-2
K=16
SEED=42
BS=64
LR=1e-7
EPS=1e-3
MODEL=/lamport/shared/hzheng/workspace/model/roberta-large
RANK=4
STEP_INTERVAL=10
STEP=50 # keep the step small for only profiling

EVALUATE_DURING_TRAINING=false



# 导出所有环境变量
export ENABLE_CUSTOM_LINEAR
export CUSTOM_LINEAR_PLOT_DIR
export PLOT_INTERVAL
export TASK
export K
export SEED
export BS
export LR
export EPS
export MODEL
export RANK
export STEP_INTERVAL
export STEP
export LOG_DIR_PREFIX
export ENABLE_DIFFENRENTIAL_LINEAR
export ENABLE_DIFFENRENTIAL_VALIDATION
export DIFFERENTIAL_PLOT_DIR
export DIFFERENTIAL_VALIDATION_FILE
export ENABLE_ACCURATE_DIFF
export EVALUATE_DURING_TRAINING

export ENABLE_QDIFFLINEAR
export ACT_QUANT_PATTERN
export ACT_BIT
export WEIGHT_BIT
export ENABLE_X
export ENABLE_DIFFX
export ENABLE_W
export ENABLE_DIFFW
export MX_A_ELEM_FORMAT
export MX_DIFFA_ELEM_FORMAT
export MX_W_ELEM_FORMAT
export MX_DIFFW_ELEM_FORMAT
export MX_QUAN
export APPLY_FORWARD_DELTA



echo "配置完成，开始调用lozo.sh..."
echo "QdiffLinear: $ENABLE_QDIFFLINEAR"
echo "任务: $TASK, K=$K, SEED=$SEED"
echo "模型: $MODEL, RANK=$RANK"

# 调用lozo.sh
bash scripts/lozo.sh

