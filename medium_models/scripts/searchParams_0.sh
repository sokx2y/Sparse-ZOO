#!/bin/bash

# 固定参数
TASK="SST-2"
K="512"
STEP="150000"
EVAL_STEP="5000"
MODEL="/lamport/shared/hzheng/workspace/model/opt-350m"
MODELNAME="opt-350m"
FEW_SHOT_TYPE="prompt"

# 参数集合
STEP_INTERVAL_SETS=("50")
WD_SETS=("0")
BS_SETS=("64")
LR_SETS=("2.5e-6")
EPS_SETS=("1e-3")
SEED_SETS=("42")
LOZO_OPTIMIZER_SETS=("sgd")
BETA1_SETS=("0.9")
RANK_SETS=("8")

LORA_PROFILES=(
  "8 16 0.0"
)

QUANTIZE=true
APPLY_PRUNE=true

QUAN_PROFILES=(
    "int4 int8 fp8_e4m3 fp8_e4m3 fp8_e4m3 fp8_e4m3"
)

PRUNE_PROFILES=(
  '[
      {
          "pattern": "^model\\.decoder\\.layers\\.0\\.self_attn\\.(q_proj|k_proj)$",
          "m": 24,
          "n": 32
      },
      {
          "pattern": "^model\\.decoder\\.layers\\.(?:[1-9]|[1-9]\\d+)\\.self_attn\\.(q_proj|k_proj)$",
          "m": 24,
          "n": 32
      },
      {
          "pattern": "^model\\.decoder\\.layers\\.(?:[0-9]|1[0-2])\\.self_attn\\.(v_proj|out_proj)$",
          "m": 24,
          "n": 32
      },
      {
          "pattern": "^model\\.decoder\\.layers\\.(?:1[3-9]|[2-9]\\d+)\\.self_attn\\.(v_proj|out_proj)$",
          "m": 24,
          "n": 32
      },
      {
          "pattern": "^model\\.decoder\\.layers\\.(?:[0-8])\\.fc[12]$",
          "m": 24,
          "n": 32
      },
      {
          "pattern": "^model\\.decoder\\.layers\\.(?:9|[1-9]\\d+)\\.fc[12]$",
          "m": 16,
          "n": 32
      }
  ]'
)

#

# PRUNE_PROFILES=(
# '[
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:[0-9]|1[0-2])\\.attention\\.self\\.[qkv]\\w*$",
#         "m": 12,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:1[3-9]|[2-9]\\d+)\\.attention\\.self\\.[qkv]\\w*$",
#         "m": 8,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:[0-9]|1[0-2])\\.attention\\.output\\.dense$",
#         "m": 12,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:1[3-9]|[2-9]\\d+)\\.attention\\.output\\.dense$",
#         "m": 8,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:[0-9]|1[0-2])\\.intermediate\\.dense$",
#         "m": 12,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:1[3-9]|[2-9]\\d+)\\.intermediate\\.dense$",
#         "m": 8,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:[0-9]|1[0-2])\\.output\\.dense$",
#         "m": 12,
#         "n": 16
#     },
#     {
#         "pattern": "^roberta\\.encoder\\.layer\\.(?:1[3-9]|[2-9]\\d+)\\.output\\.dense$",
#         "m": 8,
#         "n": 16
#     }
# ]'
# )

# 实验编号设置
NUM_START=5  # 设置起始编号
NUM=$NUM_START

# 日志目录
LOG_DIR="/lamport/makkapakka/xiangyuxing/LOZO/LOZO/medium_models/logs_dir"
mkdir -p "$LOG_DIR"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export TQDM_DISABLE=1

main_loop() {
  # 根据QUANTIZE和APPLY_PRUNE设置循环数组
  local quan_profiles_to_use=()
  if $QUANTIZE; then
      quan_profiles_to_use=("${QUAN_PROFILES[@]}")
  else
      quan_profiles_to_use=("")  # 空元素确保至少一次循环
  fi

  local prune_profiles_to_use=()
  if $APPLY_PRUNE; then
      prune_profiles_to_use=("${PRUNE_PROFILES[@]}")
  else
      prune_profiles_to_use=("")  # 空元素确保至少一次循环
  fi

  for BS in "${BS_SETS[@]}"; do
    for LR in "${LR_SETS[@]}"; do
      for EPS in "${EPS_SETS[@]}"; do
        for SEED in "${SEED_SETS[@]}"; do
          for LOZO_OPTIMIZER in "${LOZO_OPTIMIZER_SETS[@]}"; do
            for BETA1 in "${BETA1_SETS[@]}"; do
              for RANK in "${RANK_SETS[@]}"; do
                for STEP_INTERVAL in "${STEP_INTERVAL_SETS[@]}"; do
                  for WD in "${WD_SETS[@]}"; do
                    for LORA_PROFILE in "${LORA_PROFILES[@]}"; do
                      for QUAN_PROFILE in "${quan_profiles_to_use[@]}"; do
                        for PRUNE_RULES in "${prune_profiles_to_use[@]}"; do
                          # 解析LORA_PROFILE
                          read LORA_R LORA_ALPHA LORA_DROPOUT <<< "$LORA_PROFILE"
                          LORA_TAG="r${LORA_R}a${LORA_ALPHA//[^0-9]/}d${LORA_DROPOUT//[^0-9]/}"
                          

                          # 根据QUANTIZE状态处理量化参数
                          if $QUANTIZE; then
                              read MLP_W MLP_A ATTN_W ATTN_A LORA_W LORA_A <<< "$QUAN_PROFILE"
                              # 生成精简量化标签
                              QUAN_TAG="a${MLP_A//[^0-9]/}w${MLP_W//[^0-9]/}l${LORA_A//[^0-9]/}"
                              [[ $MLP_A == fp* ]] && QUAN_TAG="a${MLP_A:2:2}w${MLP_W//[^0-9]/}l${LORA_A:2:2}"
                          else
                              QUAN_TAG="noquant"
                          fi

                          # 根据APPLY_PRUNE状态处理剪枝标签
                          if $APPLY_PRUNE; then
                              # 使用tr删除所有空白符
                              COMPACT_PRUNE_RULES=$(echo "$PRUNE_RULES" | tr -d '[:space:]')
                              
                              # 生成剪枝标签（根据紧凑后的字符串调整）
                              PRUNE_TAG=$(echo "$COMPACT_PRUNE_RULES" | 
                                         sed -E 's/.*"m":([0-9]+),"n":([0-9]+).*/m\1n\2/g' |
                                         tr -d '[]{}"' | tr ',' '_')
                              
                              echo "PRUNE_RULES: $PRUNE_RULES"
                              echo "COMPACT_PRUNE_RULES: $COMPACT_PRUNE_RULES"
                              echo "PRUNE_TAG: $PRUNE_TAG"
                              #PRUNE_TAG=$(echo "$PRUNE_RULES" | sed -E 's/.*"m":([0-9]+),"n":([0-9]+).*/m\1n\2/g' | tr -d '[]{}"' | tr ',' '_')
                          else
                              PRUNE_TAG="noprune"
                          fi

                          # 构建日志文件名
                          LOG_NAME="Exp${NUM}_${TASK}_lozo_bs${BS}_lr${LR}_eps${EPS}_seed${SEED}"
                          LOG_NAME+="_opt${LOZO_OPTIMIZER}_beta${BETA1}_si${STEP_INTERVAL}_wd${WD}"
                          LOG_NAME+="_lora${LORA_TAG}"
                          LOG_NAME+="_rank${RANK}_${QUAN_TAG}_${PRUNE_TAG}"
                          LOG_FILE="${LOG_DIR}/${LOG_NAME}.log"

                          # 构建TAG参数
                          EXP_TAG="k${K}-${MODELNAME}"
                          EXP_TAG+="_${LORA_TAG}"
                          EXP_TAG+="_quan${QUAN_TAG}_prune${PRUNE_TAG}"

                          # 准备参数列表
                          CMD_ARGS=(
                            "FEW_SHOT_TYPE=$FEW_SHOT_TYPE"
                            "TASK=$TASK"
                            "K=$K"
                            "BS=$BS"
                            "LR=$LR"
                            "EPS=$EPS"
                            "SEED=$SEED"
                            "STEP=$STEP"
                            "EVAL_STEP=$EVAL_STEP"
                            "STEP_INTERVAL=$STEP_INTERVAL"
                            "WD=$WD"
                            "RANK=$RANK"
                            "LOZO_OPTIMIZER=$LOZO_OPTIMIZER"
                            "BETA1=$BETA1"
                            "MODEL=$MODEL"
                            "MODELNAME=$MODELNAME"
                            "QUANTIZE=$QUANTIZE"
                            "APPLY_PRUNE=$APPLY_PRUNE"
                            "LORA_R=$LORA_R"
                            "LORA_ALPHA=$LORA_ALPHA"
                            "LORA_DROPOUT=$LORA_DROPOUT"
                            "TAG=$EXP_TAG"
                          )

                          # 添加量化参数（仅当启用时）
                          if $QUANTIZE; then
                              CMD_ARGS+=(
                                "MLP_W_ELEM_FORMAT=$MLP_W"  # int4 
                                "MLP_A_ELEM_FORMAT=$MLP_A"  # int8
                                "ATTN_W_ELEM_FORMAT=$ATTN_W"  # int4
                                "ATTN_A_ELEM_FORMAT=$ATTN_A"  # int8
                                "LORA_W_ELEM_FORMAT=$LORA_W"  # fp8
                                "LORA_A_ELEM_FORMAT=$LORA_A"  # fp8
                              )
                          fi

                          # 添加剪枝参数（仅当启用时）
                          if $APPLY_PRUNE; then
                              CMD_ARGS+=("PRUNE_RULES=$COMPACT_PRUNE_RULES")
                          fi

                          # 启动并行任务
                          current_num=$NUM
                          device_id=$((current_num % 10))
                          (
                            export CUDA_VISIBLE_DEVICES=$device_id
                            echo "Experiment #${current_num} on GPU ${device_id}: ${CMD_ARGS[*]}"
                            env "${CMD_ARGS[@]}" bash lozo.sh >> "$LOG_FILE" 2>&1
                          ) &

                          sleep 1
                          ((NUM++))
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

main_loop
echo "Total experiments: $((NUM - NUM_START))"
