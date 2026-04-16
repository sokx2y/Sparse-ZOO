#!/bin/bash

source /capsule/home/hzheng/cuda.sh
unset LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/capsule/home/xiangyuxing/hf_home

source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/torch210cu118
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

mkdir -p ./result

BATCH_SIZE=16
CXT_LEN=64
IN_FEATURES=1024
OUT_FEATURES=1024

I_PREC=4
KV_PREC=8
W_PREC=4
OUTPUT_PREC=16

# PE_array size and area/enery use bitmod settings
PE_DP_SIZE=1   # this is not bitmod, but bitfusion. so we set it as 1, meaning each PE finish 1 element dot product in 1 cycle
PE_ENERGY=0.56
PE_AREA=1507.7
PE_ARRAY_H=32
PE_ARRAY_W=32

# 我们这里设置铺满整个Fusion PE 的点乘为 16bits * 16bits, 也即一个FU内部是 8*8 的 BitBricks
BASE_ACTIVATION_PREC=16     
BASE_WEIGHT_PREC=16

IS_GENERATION=0
IS_LOSSLESS=0
IS_BIT_SERIAL=0

echo "single_linear_sim_bitFusion parameters:"
echo "  batch_size=${BATCH_SIZE}"
echo "  cxt_len=${CXT_LEN}"
echo "  in_features=${IN_FEATURES}"
echo "  out_features=${OUT_FEATURES}"
echo "  i_prec=${I_PREC}"
echo "  kv_prec=${KV_PREC}"
echo "  w_prec=${W_PREC}"
echo "  output_prec=${OUTPUT_PREC}"
echo "  pe_dp_size=${PE_DP_SIZE}"
echo "  pe_energy=${PE_ENERGY}"
echo "  pe_area=${PE_AREA}"
echo "  pe_array_h=${PE_ARRAY_H}"
echo "  pe_array_w=${PE_ARRAY_W}"
echo "  base_activation_prec=${BASE_ACTIVATION_PREC}"
echo "  base_weight_prec=${BASE_WEIGHT_PREC}"
echo "  is_generation=${IS_GENERATION}"
echo "  is_lossless=${IS_LOSSLESS}"
echo "  is_bit_serial=${IS_BIT_SERIAL}"

GENERATION_FLAG=""
LOSSLESS_FLAG=""
BIT_SERIAL_FLAG=""

if [[ ${IS_GENERATION} -eq 1 ]]; then
    GENERATION_FLAG="--is_generation"
fi

if [[ ${IS_LOSSLESS} -eq 1 ]]; then
    LOSSLESS_FLAG="--is_lossless"
fi

if [[ ${IS_BIT_SERIAL} -eq 1 ]]; then
    BIT_SERIAL_FLAG="--is_bit_serial"
fi

python single_linear_simBB.py \
    --batch_size ${BATCH_SIZE} \
    --cxt_len ${CXT_LEN} \
    --in_features ${IN_FEATURES} \
    --out_features ${OUT_FEATURES} \
    --i_prec ${I_PREC} \
    --kv_prec ${KV_PREC} \
    --w_prec ${W_PREC} \
    --output_prec ${OUTPUT_PREC} \
    --pe_dp_size ${PE_DP_SIZE} \
    --pe_energy ${PE_ENERGY} \
    --pe_area ${PE_AREA} \
    --pe_array_h ${PE_ARRAY_H} \
    --pe_array_w ${PE_ARRAY_W} \
    --base_activation_prec ${BASE_ACTIVATION_PREC} \
    --base_weight_prec ${BASE_WEIGHT_PREC} \
    ${GENERATION_FLAG} \
    ${LOSSLESS_FLAG} \
    ${BIT_SERIAL_FLAG} \
    2>&1 | tee ./result/single_linear_sim_bitFusion.log

