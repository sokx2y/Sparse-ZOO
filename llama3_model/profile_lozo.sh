#!/bin/bash
# profile_lozo.sh
# 小样本/小步数调试：通过设置环境变量覆盖 lozo.sh 的默认值，然后调用 lozo.sh


source /capsule/home/hzheng/cuda.sh
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 加载 Conda 环境
source ~/.bashrc
conda activate /capsule/home/xiangyuxing/oldmkpk/conda_envs/lozo_llama3


# 只覆盖 lozo.sh 里支持 ${VAR:-default} 的变量
export LOG_DIR_PREFIX=${LOG_DIR_PREFIX:-"testprofileSVD"}

# export MODEL=${MODEL:-/capsule/home/xiangyuxing/hf_offline/opt-13b}
MODEL=${MODEL:-/capsule/home/xiangyuxing/hf_offline/Llama-3.2-3B}


export MODE=${MODE:-ft}     

export BS=${BS:-16}
export EPS=${EPS:-1e-3}

export TRAIN=${TRAIN:-32}
export DEV=${DEV:-16}
export EVAL=${EVAL:-64}

export STEPS=${STEPS:-100}
export EVAL_STEPS=${EVAL_STEPS:-10}



echo "LOG_DIR_PREFIX=$LOG_DIR_PREFIX"
echo "MODEL=$MODEL"
echo "MODE=$MODE"
echo "BS=$BS EPS=$EPS"
echo "TRAIN=$TRAIN DEV=$DEV EVAL=$EVAL"
echo "STEPS=$STEPS EVAL_STEPS=$EVAL_STEPS"

bash lozo.sh "$@"


