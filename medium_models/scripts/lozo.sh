#!/bin/bash

source /capsule/home/hzheng/cuda.sh
# 可选：为了安全，先取消任何全局 LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
# 再显式注入本次作业需要的路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 加载 Conda 环境
source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/torch210cu118

export CUDA_VISIBLE_DEVICES=4
export WANDB_DISABLED=true
export TQDM_DISABLE=1

FEW_SHOT_TYPE=${FEW_SHOT_TYPE:-"prompt"}

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-5e-7}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-20000}
EVAL_STEP=${EVAL_STEP:-1000}
STEP_INTERVAL=${STEP_INTERVAL:-50}
RANK=${RANK:-8}
LOZO_OPTIMIZER=${LOZO_OPTIMIZER:-'sgd'}
BETA1=${BETA1:-0.9}
MODEL=${MODEL:-"/lamport/shared/hzheng/workspace/model/roberta-large"}
# MODEL=${MODEL:-"/lamport/shared/hzheng/workspace/model/opt-350m"}
MODELNAME=${MODELNAME:-"roberta-large"}
# MODELNAME=${MODELNAME:-"opt"}
LOG_DIR_PREFIX=${LOG_DIR_PREFIX:-"LogPaperSVD"}


# lora参数
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}

# CustomLinear & diffLinear 都是 test 用的
# CustomLinear相关参数
ENABLE_CUSTOM_LINEAR=${ENABLE_CUSTOM_LINEAR:-false}
CUSTOM_LINEAR_PLOT_DIR=${CUSTOM_LINEAR_PLOT_DIR:-"./custom_linear_plots"}
PLOT_INTERVAL=${PLOT_INTERVAL:-100}
# diffLinear相关参数
ENABLE_DIFFENRENTIAL_LINEAR=${ENABLE_DIFFENRENTIAL_LINEAR:-false}
ENABLE_DIFFENRENTIAL_VALIDATION=${ENABLE_DIFFENRENTIAL_VALIDATION:-false}
ENABLE_ACCURATE_DIFF=${ENABLE_ACCURATE_DIFF:-true}
DIFFERENTIAL_PLOT_DIR=${DIFFERENTIAL_PLOT_DIR:-"./diff_linear_plots"}
DIFFERENTIAL_VALIDATION_FILE=${DIFFERENTIAL_VALIDATION_FILE:-"./diff_linear_plots/validation_file"}
# QdiffLinear 相关参数
ENABLE_QDIFFLINEAR=${ENABLE_QDIFFLINEAR:-false}
MX_QUAN=${MX_QUAN:-true}
ACT_QUANT_PATTERN=${ACT_QUANT_PATTERN:-"per_channel"}
ACT_BIT=${ACT_BIT:-8}
WEIGHT_BIT=${WEIGHT_BIT:-4}


# ----------------- Here! ------------------
# svd
svd_lora_compensation=true
svd_lora_act_scales_path=/capsule/home/xiangyuxing/Sparse-ZOO/SVD-ZOO-Quant/outputs/roberta-large/fp16_roberta_actscales/act_scales.pt
svd_lora_smooth_alpha=0.5
svd_lora_checkpoint_path=/capsule/home/xiangyuxing/Sparse-ZOO/SVD-ZOO-Quant/outputs/roberta-large/roberta_residual_nvfp4_nvfp8_float16_svd_rank32_1.0
svd_lora_rank=32
svd_lora_quant_format="nvfp4"  # quantize base_weight
quant_format_wdx=${quant_format_wdx:-"nvfp4"}
appro_SVD=${appro_SVD:-false}
COMPENSATION_MODE="${COMPENSATION_MODE:-residual}"
# forward_delta 
APPLY_FORWARD_DELTA=${APPLY_FORWARD_DELTA:-true}
ENABLE_X=${ENABLE_X:-true}
ENABLE_DIFFX=${ENABLE_DIFFX:-true}
ENABLE_W=${ENABLE_W:-true}
ENABLE_DIFFW=${ENABLE_DIFFW:-true}
MX_A_ELEM_FORMAT=${MX_A_ELEM_FORMAT:-"fp8_e4m3"}
MX_DIFFA_ELEM_FORMAT=${MX_DIFFA_ELEM_FORMAT:-"fp4_e2m1"}
MX_W_ELEM_FORMAT=${MX_W_ELEM_FORMAT:-"fp8_e4m3"}
MX_DIFFW_ELEM_FORMAT=${MX_DIFFW_ELEM_FORMAT:-"fp8_e4m3"}

TRAINABLE_MODE=${TRAINABLE_MODE:-"all"}



LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "Step: $STEP; Eval step: $EVAL_STEP"
echo "MODEL: $MODEL"
echo "Custom Plot Interval: $PLOT_INTERVAL"
echo "QdiffLinear: $ENABLE_QDIFFLINEAR"
echo "mx_quan: $MX_QUAN"
echo "apply_forward_delta": "$APPLY_FORWARD_DELTA"



GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP-step-interval$STEP_INTERVAL-rank$RANK
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=$MODELNAME-$COMPENSATION_MODE-$svd_lora_rank-fp8

if [ "$APPLY_FORWARD_DELTA" = "true" ]; then
    ENABLE="SVDQuantdiff-${TRAINABLE_MODE}-${svd_lora_quant_format}-${quant_format_wdx}"
else
    ENABLE="normal-${TRAINABLE_MODE}"
fi

ENABLE="${ENABLE}"

echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"
echo "Quantize pattern: $ENABLE"

EVALUATE_DURING_TRAINING=${EVALUATE_DURING_TRAINING:-true}


# 创建日志目录
mkdir -p ${LOG_DIR_PREFIX}/log_dir
# Redirect all output to a log file based on the TAG
# exec &> >(tee "${LOG_DIR_PREFIX}/log_dir/${TASK}-${GR_TAG}-${TAG}.log")
exec > "${LOG_DIR_PREFIX}/log_dir/${TASK}-${GR_TAG}-${TAG}-${ENABLE}.log" 2>&1   


TYPE=$FEW_SHOT_TYPE GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K ENABLE=$ENABLE\
    bash scripts/run_fewshot_lozo.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP --weight_decay $WD --zo_eps $EPS \
    --zero_order_optim --lr_scheduler_type "constant" --optimizer "sgd" --efficient_zero_order \
    --lozo_optimizer $LOZO_OPTIMIZER --beta1 $BETA1 --step_interval $STEP_INTERVAL --rank $RANK \
    --enable_custom_linear $ENABLE_CUSTOM_LINEAR --custom_linear_plot_dir $CUSTOM_LINEAR_PLOT_DIR --plot_interval $PLOT_INTERVAL \
    --enable_differential_linear $ENABLE_DIFFENRENTIAL_LINEAR --enable_differential_validation $ENABLE_DIFFENRENTIAL_VALIDATION --enable_accurate_diff $ENABLE_ACCURATE_DIFF --differential_plot_dir $DIFFERENTIAL_PLOT_DIR --differential_validation_file $DIFFERENTIAL_VALIDATION_FILE \
    --enable_QdiffLinear $ENABLE_QDIFFLINEAR --mx_quan $MX_QUAN --mx_w_elem_format $MX_W_ELEM_FORMAT --mx_a_elem_format $MX_A_ELEM_FORMAT --mx_diffw_elem_format $MX_DIFFW_ELEM_FORMAT --mx_diffa_elem_format $MX_DIFFA_ELEM_FORMAT --act_quant_pattern $ACT_QUANT_PATTERN --act_bit $ACT_BIT --weight_bit $WEIGHT_BIT --enable_x $ENABLE_X --enable_diffx $ENABLE_DIFFX --enable_w $ENABLE_W --enable_diffw $ENABLE_DIFFW \
    --use_uv_diffw True --trainable_mode $TRAINABLE_MODE \
    --apply_forward_delta $APPLY_FORWARD_DELTA --use_forward_delta_loss $APPLY_FORWARD_DELTA\
    --load_best_model_at_end True --evaluation_strategy steps --save_strategy steps --save_total_limit 1 --evaluate_during_training $EVALUATE_DURING_TRAINING\
    --save_steps $EVAL_STEP \
    --svd_lora_compensation $svd_lora_compensation --svd_lora_act_scales_path $svd_lora_act_scales_path --svd_lora_smooth_alpha $svd_lora_smooth_alpha \
    --svd_lora_checkpoint_path $svd_lora_checkpoint_path --svd_lora_rank $svd_lora_rank --svd_lora_quant_format $svd_lora_quant_format \
    --quant_format_wdx $quant_format_wdx \
    --appro_SVD $appro_SVD \
    --debug_forward_delta --debug_forward_delta_steps 10 --debug_forward_delta_tol 1e-6 --debug_forward_delta_abort False \
    --compare_seed --compare_seed_steps 30 \
    $@
