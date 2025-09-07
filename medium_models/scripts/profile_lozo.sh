#!/bin/bash

# CustomLinear功能配置文件
# 通过profile_lozo.sh调用lozo.sh来实现CustomLinear功能

echo "=== CustomLinear功能配置 ==="

# CustomLinear功能配置
ENABLE_CUSTOM_LINEAR=true
CUSTOM_LINEAR_PLOT_DIR="./custom_linear_plots"
PLOT_INTERVAL=50 
LOG_DIR_PREFIX=test # 通过这个设置log保存目录

rm data/k-shot-1k-test/SST-2/16-42/cached_*

# 基本任务配置
TASK=SST-2
K=16
SEED=42
BS=64
LR=1e-7
EPS=1e-3
MODEL=roberta-large
RANK=4
STEP_INTERVAL=100
STEP=50 # keep the step small for only profiling



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

echo "配置完成，开始调用lozo.sh..."
echo "CustomLinear: $ENABLE_CUSTOM_LINEAR"
echo "图表目录: $CUSTOM_LINEAR_PLOT_DIR"
echo "任务: $TASK, K=$K, SEED=$SEED"
echo "模型: $MODEL, RANK=$RANK"

# 调用lozo.sh
bash scripts/lozo.sh

