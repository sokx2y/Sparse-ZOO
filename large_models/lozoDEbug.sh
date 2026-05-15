#!/bin/bash
source /capsule/home/hzheng/cuda.sh
# 可选：为了安全，先取消任何全局 LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
# 再显式注入本次作业需要的路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/capsule/home/xiangyuxing/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export LOCAL_DATASETS_DIR=/capsule/home/xiangyuxing/local_datasets

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# 加载 Conda 环境
source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/torch210cu118

export CUDA_VISIBLE_DEVICES=7
export WANDB_DISABLED=true
export TQDM_DISABLE=1

# MODEL=${MODEL:-/capsule/home/xiangyuxing/hf_offline/opt-6.7b}
MODEL=${MODEL:-"/lamport/shared/hzheng/workspace/model/opt-350m"}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

LOG_DIR_PREFIX=${LOG_DIR_PREFIX:-"LogPaper"}

BS=${BS:-16}
EPS=${EPS:-1e-3}
# TRAIN=${TRAIN:-1000}
# DEV=${DEV:-500}
# EVAL=${EVAL:-1000}
# STEPS=${STEPS:-20000}
# EVAL_STEPS=${EVAL_STEPS:-4000}

TRAIN=${TRAIN:-64}
DEV=${DEV:-32}
EVAL=${EVAL:-32}
STEPS=${STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-10}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

LR=2e-7
TASK=RTE
SEED=0
RANK=2
# STEP_INTERVAL=100
STEP_INTERVAL=10
Tainer=LOZO

# forward_delta 
APPLY_FORWARD_DELTA=${APPLY_FORWARD_DELTA:-true}
ENABLE_X=${ENABLE_X:-true}
ENABLE_DIFFX=${ENABLE_DIFFX:-true}
ENABLE_W=${ENABLE_W:-true}
ENABLE_DIFFW=${ENABLE_DIFFW:-true}
MX_A_ELEM_FORMAT=${MX_A_ELEM_FORMAT:-"fp8_e4m3"}
MX_DIFFA_ELEM_FORMAT=${MX_DIFFA_ELEM_FORMAT:-"fp4_e2m1"}
MX_W_ELEM_FORMAT=${MX_W_ELEM_FORMAT:-"fp8_e4m3"}
MX_DIFFW_ELEM_FORMAT=${MX_DIFFW_ELEM_FORMAT:-"fp4_e2m1"}

TRAINABLE_MODE=${TRAINABLE_MODE:-"all"}
DEBUG_PHASE1_COMPARE=${DEBUG_PHASE1_COMPARE:-true}
DEBUG_FD_EXACT_EFFECTIVE_DELTA=${DEBUG_FD_EXACT_EFFECTIVE_DELTA:-false}

LOAD_FLOAT16=${LOAD_FLOAT16:-true}

LOAD_ARGS=""
DTYPE_TAG="32"
if [ "$LOAD_FLOAT16" = "true" ]; then
    LOAD_ARGS="--load_float16"
    DTYPE_TAG="16"
fi

case $TASK in
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

TAG=$Tainer-$MODEL_NAME-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL-$RANK-$TASK-$DTYPE_TAG

if [ "$APPLY_FORWARD_DELTA" = "true" ]; then
    ENABLE="Quantdiff-${TRAINABLE_MODE}"

    if [ "$ENABLE_X" = "true" ]; then
        ENABLE="${ENABLE}-x${MX_A_ELEM_FORMAT}"
    fi
    if [ "$ENABLE_DIFFX" = "true" ]; then
        ENABLE="${ENABLE}-dx${MX_DIFFA_ELEM_FORMAT}"
    fi
    if [ "$ENABLE_W" = "true" ]; then
        ENABLE="${ENABLE}-w${MX_W_ELEM_FORMAT}"
    fi
    if [ "$ENABLE_DIFFW" = "true" ]; then
        ENABLE="${ENABLE}-dw${MX_DIFFW_ELEM_FORMAT}"
    fi
else
    ENABLE="normal-${TRAINABLE_MODE}"
fi

ENABLE="${ENABLE}"

echo "Quantize pattern: $ENABLE"

# 创建日志目录
mkdir -p ${LOG_DIR_PREFIX}/log_dir
# Redirect all output to a log file based on the TAG
# exec &> >(tee "${LOG_DIR_PREFIX}/log_dir/${TASK}-${GR_TAG}-${TAG}.log")
exec > "${LOG_DIR_PREFIX}/log_dir/${TAG}-${ENABLE}.log" 2>&1   
PHASE1_JSONL="${LOG_DIR_PREFIX}/log_dir/${TAG}-${ENABLE}.jsonl"


echo $TAG
echo "Task: $TASK"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "RANK: $RANK"
echo "STEP INTERVAL: $STEP_INTERVAL"
echo "apply_forward_delta": "$APPLY_FORWARD_DELTA"

python run_lozoDEbug.py \
    --model_name $MODEL_NAME --model_path $MODEL\
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG-${ENABLE} --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $Tainer \
    $LOAD_ARGS \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --step_interval $STEP_INTERVAL \
    --rank_r $RANK \
    --apply_forward_delta $APPLY_FORWARD_DELTA --trainable_mode $TRAINABLE_MODE\
    --mx_w_elem_format $MX_W_ELEM_FORMAT --mx_a_elem_format $MX_A_ELEM_FORMAT --mx_diffw_elem_format $MX_DIFFW_ELEM_FORMAT --mx_diffa_elem_format $MX_DIFFA_ELEM_FORMAT --enable_x $ENABLE_X --enable_diffx $ENABLE_DIFFX --enable_w $ENABLE_W --enable_diffw $ENABLE_DIFFW \
    --debug_phase1_compare $DEBUG_PHASE1_COMPARE \
    --debug_phase1_jsonl $PHASE1_JSONL \
    --debug_fd_exact_effective_delta $DEBUG_FD_EXACT_EFFECTIVE_DELTA \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 





