#!/bin/bash
source /capsule/home/hzheng/cuda.sh

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/capsule/home/xiangyuxing/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export LOCAL_DATASETS_DIR=/capsule/home/xiangyuxing/local_datasets

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/lozo_llama3

export CUDA_VISIBLE_DEVICES=7
export WANDB_DISABLED=true
export TQDM_DISABLE=1

MODEL=${MODEL:-/capsule/home/xiangyuxing/hf_offline/Llama-3.2-3B}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

LOG_DIR_PREFIX=${LOG_DIR_PREFIX:-"LogPaper"}

# Phase1 debug: 小步数、小数据、小 batch
BS=${BS:-1} #
EPS=${EPS:-1e-3}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
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
TASK=CB
SEED=0
RANK=2
STEP_INTERVAL=10
Tainer=LOZO

# Phase1 检查 forward_delta，但关闭 MX 量化，跑 diffnormal
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
DEBUG_PHASE1C_DELTA=${DEBUG_PHASE1C_DELTA:-true}
DEBUG_PHASE1C_PARAM_NAMES=${DEBUG_PHASE1C_PARAM_NAMES:-""}


case $TASK in
    CB)
        DEV=100
        ;;
    Copa)
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

TAG=PHASE1C-$Tainer-$MODEL_NAME-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL-$RANK-$TASK
PHASE1C_JSONL="${LOG_DIR_PREFIX}/log_dir/${TAG}-${ENABLE}.jsonl"
LOG_FILE="${LOG_DIR_PREFIX}/log_dir/${TAG}-${ENABLE}.log"


if [ "$APPLY_FORWARD_DELTA" = "true" ]; then
    ENABLE="diffnormal-${TRAINABLE_MODE}"
else
    ENABLE="normal-${TRAINABLE_MODE}"
fi

echo "Debug mode: Phase1"
echo "Pattern: $ENABLE"

mkdir -p ${LOG_DIR_PREFIX}/log_dir

PHASE1_JSONL="${LOG_DIR_PREFIX}/log_dir/${TAG}-${ENABLE}.jsonl"
LOG_FILE="${LOG_DIR_PREFIX}/log_dir/${TAG}-${ENABLE}.log"

exec > "$LOG_FILE" 2>&1

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
echo "apply_forward_delta: $APPLY_FORWARD_DELTA"
echo "debug_phase1_jsonl: $PHASE1_JSONL"
echo "log_file: $LOG_FILE"

python run_lozo_DEbug.py \
    --model_name $MODEL_NAME --model_path $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG-${ENABLE} \
    --tag $TAG \
    --train_set_seed $SEED \
    --num_train $TRAIN \
    --num_dev $DEV \
    --num_eval $EVAL \
    --logging_steps 1 \
    --max_steps $STEPS \
    --trainer $Tainer \
    --load_float16 \
    --learning_rate $LR \
    --zo_eps $EPS \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type "constant" \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_total_limit 1 \
    --eval_steps $EVAL_STEPS \
    --save_steps $EVAL_STEPS \
    --train_as_classification \
    --step_interval $STEP_INTERVAL \
    --rank_r $RANK \
    --apply_forward_delta $APPLY_FORWARD_DELTA \
    --trainable_mode $TRAINABLE_MODE \
    --mx_w_elem_format $MX_W_ELEM_FORMAT \
    --mx_a_elem_format $MX_A_ELEM_FORMAT \
    --mx_diffw_elem_format $MX_DIFFW_ELEM_FORMAT \
    --mx_diffa_elem_format $MX_DIFFA_ELEM_FORMAT \
    --enable_x $ENABLE_X \
    --enable_diffx $ENABLE_DIFFX \
    --enable_w $ENABLE_W \
    --enable_diffw $ENABLE_DIFFW \
    --debug_phase1_compare false \
    --debug_phase1c_delta $DEBUG_PHASE1C_DELTA \
    --debug_phase1c_jsonl $PHASE1C_JSONL \
    --debug_phase1c_param_names "$DEBUG_PHASE1C_PARAM_NAMES" \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"




