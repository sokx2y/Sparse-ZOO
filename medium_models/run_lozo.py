"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List
import torch
import torch.nn.functional as F

import numpy as np
import traceback

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, PreTrainedTokenizerBase
from transformers.pipelines import model_info
from src.modeling_roberta import RobertaConfig
from src.modeling_opt import OPTConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.linearhead_trainer import LinearHeadTrainer
from src.dataset import FewShotDataset, OurInputFeatures
from src.models import MODEL_TYPES, resize_token_type_embeddings, convert_opt_model
from src.LOZOtrainer import LowRankTrainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from src.custom_linear import get_all_custom_layers, replace_linear_layers_with_custom
from src.diff_linear import replace_linear_layers_with_differential, get_all_differential_layers, DifferentialLinear
from src.diff_fake_quant import QdiffLinear, replace_linear_with_Qdifflinear, get_all_Qdifflayers
from src.diff_fake_quant_mx import diffLinear, diffLayerNorm, diffEmbedding, QdiffLinear
from src.diff_fake_quant_mx import QuantizeRobertaForLOZO

from filelock import FileLock
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def apply_trainable_mode(model, mode: str, logger=None):
    if mode is None or mode.lower() == "all":
        if logger: logger.info("[trainable_mode] all params (default).")
        return

    mode = mode.lower()
    if mode != "linear_only":
        raise ValueError(f"Unknown trainable_mode: {mode}")

    # 0) 先全冻结
    for p in model.parameters():
        p.requires_grad_(False)

    # 1) 处理 OPT 常见的 tied embeddings（lm_head.weight 与 embed_tokens.weight 共享）
    in_emb = getattr(model, "get_input_embeddings", lambda: None)()
    out_emb = getattr(model, "get_output_embeddings", lambda: None)()
    tied_weight = None
    if in_emb is not None and hasattr(in_emb, "weight"):
        tied_weight = in_emb.weight  # 你要求 embedding 不动 -> 这个 weight 必须保持冻结

    tied = (
        tied_weight is not None
        and out_emb is not None
        and hasattr(out_emb, "weight")
        and (out_emb.weight is tied_weight)
    )
    if tied and logger:
        logger.warning(
            "[trainable_mode] Detected tied input/output embeddings. "
            "To keep embeddings frozen, lm_head.weight will also be frozen unless you set --untie_emb."
        )

    # 2) 只解冻“线性层类”的 weight/bias
    for m in model.modules():
        cls = m.__class__.__name__.lower()
        if ("embedding" in cls) or ("layernorm" in cls) or ("layer_norm" in cls):
            continue

        w = getattr(m, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.ndim == 2:
            # 如果是 tied embedding 的那块 weight，跳过（否则会把 embedding 也解冻）
            if tied_weight is not None and w is tied_weight:
                continue
            w.requires_grad_(True)

            b = getattr(m, "bias", None)
            if isinstance(b, torch.nn.Parameter):
                b.requires_grad_(True)

    # 3) 打印确认
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_total = sum(x[1] for x in trainable)
    if logger:
        logger.info(f"[trainable_mode] linear_only trainable params: {trainable_total}/{total} ({trainable_total/total:.2%})")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    l2_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use L2 loss (only makes a difference in standard FT)."}
    )
    use_task_word: bool = field(
        default=False,
        metadata={'help': 'uses the task words MLM logit for kernel computation'}
    )

    # LoRA arguments: only for BERT-type model
    apply_lora: bool = field(
        default=False,
        metadata={'help': 'use LoRA for finetuning'}
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={'help': 'initialization scale for one of the low rank matrices in lora'}
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={'help': 'inner rank for lora matrices'}
    )

    # Calibration
    sfc: bool = field(
        default=False,
        metadata={"help": "Whether to use surface form calibration."}
    )

    icl_sfc: bool = field(
        default=False,
        metadata={"help": "Use in-context learning demos in sfc."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    sfc_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "SFC prompt"}
    )

    template: Optional[str] = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: Optional[str] = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: Optional[int] = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: Optional[int] = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: Optional[int] = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: Optional[int] = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: Optional[str] = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )
    
    # CustomLinear相关参数
    enable_custom_linear: bool = field(
        default=False,
        metadata={"help": "是否启用CustomLinear层来记录奇偶数次推理数据"}
    )
    
    custom_linear_plot_dir: str = field(
        default=None,
        metadata={"help": "CustomLinear数据分布图保存目录"}
    )
    
    plot_interval: int = field(
        default=100,
        metadata={"help": "每隔多少次推理绘制一次图表"}
    )
    
    # DiffLinear 相关参数
    enable_differential_linear: bool = field(
        default=False,
        metadata={"help": "启用DifferentialLinear层来优化计算偶数次的output"}
    )
    
    enable_differential_validation: bool = field(
        default=False,
        metadata={"help": "是否计算diff_linear与正常linear的误差"}
    )
    
    enable_accurate_diff: bool = field(
        default=False,
        metadata={"help": "是否启用acc_diff_linear"}
    )

    differential_plot_dir: str = field(
        default=None,
        metadata={"help": "DifferentialLinear验证误差图保存目录"}
    )

    differential_reset_interval: int = field(
        default=1000,
        metadata={"help": "每隔多少次推理重置一次缓存状态（防止长时间运行的数值累积误差）"}
    )

    differential_validation_file: str = field(
        default=None,
        metadata={"help": "保存差分计算验证结果的JSON文件路径"}
    )
    
    
    # Quantize QdiffLinear 相关参数
    enable_QdiffLinear: bool = field(
        default=False,
        metadata={"help":"Use QdiffLinear"}    
    )
    
    mx_quan: bool = field(
        default=False,
        metadata={"help":"Use microxcaling to quantize model"}    
    )

    apply_forward_delta: bool = field(
        default=False,
        metadata={"help":"Use all difflayer in diff_fake_quant_mx.py"}
    )
    
    mx_w_elem_format : str = field(
        default=None,
        metadata={"help": "choose your mx quantize format for weight"}
    )
    
    mx_a_elem_format : str = field(
        default=None,
        metadata={"help": "choose your mx quantize format for activation"}
    )
    
    mx_diffw_elem_format : str = field(
        default=None,
        metadata={"help": "choose your mx quantize format for diff_weight in QdiffLinear"}
    )
    
    mx_diffa_elem_format : str = field(
        default=None,
        metadata={"help": "choose your mx quantize format for diff_input in QdiffLinear"}
    )
    
    act_quant_pattern: str = field(
        default="per_token",
        metadata={"help": "Choose the pattern to quantize x and diffx"}
    )
    
    act_bit: int = field(
        default=8,
        metadata={"help": "choose the quantization precision for x and diffx"}
    )
    
    weight_bit: int = field(
        default=8,
        metadata={"help": "choose the quantization precision for w and diffw"}
    )
    
    enable_x: bool = field(
        default=False,
        metadata={"help": "quantize activation_odd"}
    )
    
    enable_diffx: bool = field(
        default=False,
        metadata={"help": "quantize diff_activation"}
    )
    
    enable_w: bool = field(
        default=False,
        metadata={"help": "quantize weight_even"}
    )
    
    enable_diffw: bool = field(
        default=False,
        metadata={"help": "quantize diff_weight"}
    )
    
    use_uv_diffw: bool = field(
        default=False,
        metadata={"help": "use uv from lozotrainer to replace diff_weight during qunatization"}
    )
    
    
    


    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=False,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    gpt3_demo_separator: str = field(
        default="\n\n\n",
        metadata={"help": "Separator between demonstrations"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: Optional[List[str]] = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."},

    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    
    # =================== LOZO Arguments =====================
    lowrank_zo: bool = field(
        default=True,
        metadata={"help": "where use low rank zeroth order"}
    )
    rank: int = field(
        default=2,
        metadata={"help": "rank of u and v"}
    )
    step_interval: int = field(
        default=50,
        metadata={"help": "frequency of updating v"}
    )
    zo_eps: float = field(
        default=1e-3,
        metadata={"help": "epsilon of lowrank"}
    )
    beta1: float = field(
        default=0.7,
        metadata={"help": "coefficient of momentum term"}
    )
    lozo_optimizer: str = field(
        default='sgd',
        metadata={"help": "optimizer selection"}
    )
    trainable_mode: str = field(
        default="all",
        metadata={"help": "Trainable params: all | linear_only (freeze embedding & layernorm)"}
    )
    
    # forward_delta Debugging 
    use_forward_delta_loss: bool = field(
        default=True,
        metadata={"help": "If True, compute (loss+, loss-) via forward_delta (one inference). If False, use two normal forwards."}
    )
    # Debug4Seed
    compare_seed: bool = field(default=False, metadata={"help": "Log zo_random_seed and u/v/z fingerprints for comparison."})
    compare_seed_steps: int = field(default=30, metadata={"help": "Log first N steps."})
    compare_seed_param: str = field(
        default="roberta.encoder.layer.20.attention.self.query.weight",
        metadata={"help": "Which parameter name to fingerprint u/v for (must be a 2D weight)."}
    )
    compare_seed_param_1d: str = field(
        default="roberta.encoder.layer.20.attention.self.query.bias",
        metadata={"help": "Which 1D parameter name to fingerprint z for."}
    )

    debug_forward_delta: bool = field(
        default=False,
        metadata={
            "help": "Sanity-check forward_delta by comparing: (a) true loss at theta+eps*delta via normal forward, "
                    "(b) loss_base returned by forward_delta, (c) true loss at theta-eps*delta via normal forward, "
                    "and (d) loss_perturbed returned by forward_delta."
        },
    )
    debug_forward_delta_steps: int = field(
        default=1,
        metadata={"help": "Run forward_delta sanity-check for the first N LOZO steps."},
    )
    debug_forward_delta_tol: float = field(
        default=1e-6,
        metadata={"help": "Absolute tolerance for forward_delta sanity-check (loss mismatch)."},
    )
    debug_forward_delta_abort: bool = field(
        default=True,
        metadata={"help": "If True, raise an error when sanity-check mismatch exceeds tol."},
    ) 
    # =======================================================
        
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training or at the."}
    )
    log_file: str = field(
        default='log'
    )

    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-parameter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    optimizer: str = field(
        default='adam',
        metadata={'help': 'choose sgd or adam. default is adam'}
    )
    optimizer_variant: str = field(
        default='',
        metadata={'help': 'define variants on optimizer: signgd'}
    )

    trainer: str = field(
        default="standard",
        metadata={"help": "Pick from {standard, kernel, linearhead}"}
    )
    from_linearhead: bool = field(
        default=False,
        metadata={"help": "Whether to initialize head with the linearhead solution. Works for both normal and kernel trainer."}
    )
    lp_early_stopping: bool = field(
        default=False,
        metadata={"help": "When on, increases the tolerance and lowers max_iter in scikit LogisticRegression solver to encourage early stopping."}
    )
    random_model_init: bool = field(
        default=False,
        metadata={'help': 'reinit the model randomly'}
    )
    sweep: bool = field(
        default=False,
        metadata={'help': 'configures the output directories to be informative when running W&B sweep'}
    )
    kernel_formula: str = field(
        default='sgd',
        metadata={"help": "choose kernel formula from {sgd, signgd, asymmetric_signgd}"}
    )
    kernel_solver: str = field(
        default="logistic",
        metadata={"help": "choose kernel solver from {lstsq, logistic, svr, svc, asym (only for asymmetric_signgd)}"}
    )
    load_kernels: Optional[str] = field(
        default=None,
        metadata={'help': 'when specified, loads the kernels from the folder given here'}
    )
    overwrite_kernels: bool = field(
        default=False,
        metadata={'help': 'when specified, overwrites the kernels in the output_dir and computes them from scratch'}
    )

    exclude_embeddings: bool = field(
        default=False,
        metadata={"help": "Don't use embeddings for kernel computation "}
    )
    exclude_head: bool = field(
        default=False,
        metadata={"help": "Don't use head for kernel computation "}
    )
    only_biases: bool = field(
        default=False,
        metadata={"help": "Only use bias parameters for kernel computation for BitFit-style kernel"}
    )
    exclude_first_layers: int = field(
        default=-1,
        metadata={'help': 'excludes the first N layers from kernel computation'}
    )
    sync_embedding_layers: bool = field(
        default=False,
        metadata={'help': 'sync the input embedding to match output embedding (use with --exclude_first_layers)'}
    )

    kernel_regularization: float = field(
        default=0.0,
        metadata={"help": "Regularization constant for kernel"}
    )
    kernel_gamma: float = field(
        default=1.0,
        metadata={"help": "Gamma for asymmetric kernel solver"}
    )
    binary_classification: bool = field(
        default=False,
        metadata={"help": "If num_classes=2, convert two softmax logits to single sigmoid logit"}
    )
    adjust_for_init: bool = field(
        default=False,
        metadata={'help': 'when on, trains kernel on y-f0 and adds f0 at test time'}
    )
    f0_scaling: float = field(
        default=1.0,
        metadata={'help': 'adjust label scaling, might help with --adjust_for_init perf'}
    )
    zero_order_optim: bool = field(
        default=False,
        metadata={'help': 'when on, trains the model by zero-order optimization'}
    )
    zero_order_eps: float = field(
        default=1e-3,
        metadata={'help': 'eps for zero order optim'}
    )
    prob_as_feature: bool = field(
        default=False,
        metadata={'help': 'in linear head, use log prob as feature'}
    )
    zero_order_use_trainer_optim: bool = field(
        default=False,
        metadata={"help": "Use trainer optimizer for zero order optimization"}
    )
    efficient_zero_order: bool = field(
        default=False,
        metadata={"help": "Efficient zero-order: resample noise vectors instead of saving them. enable different model loading using --hf_inference_model"}
    )
    hf_inference_model: bool = field(
        default=False,
        metadata={"help": "loads the HF model in inference mode across many GPUs. incompatible with --zero_order_use_trainer_optim."}
    )
    efficient_zero_order_fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 for efficient zero order"}
    )
    zero_order_sample_scheduler: Optional[str] = field(
        default=None,
        metadata={"help": "Have a sample scheduler. None, 'linear', 'power', or 'constant."}
    )
    scale_lr_with_samples: bool = field(
        default=False,
        metadata={"help": "Scales the LR proportionally to the number of z samples. --learning_rate will be the LR for one z sample."}
    )
    zero_order_sample: int = field(
        default=1,
        metadata={"help": "Sample times for zero-order estimate. If scheduler is 'linear', this number is the max sample number."}
    )
    zero_order_clip_grad: bool = field(
        default=False,
        metadata={"help": "Clip the norm of the gradient for zero order (only when using trainer optimizer)"}
    )
     
    # MeZO variants
    zo_by_layer: bool = field(
        default=False,
        metadata={"help": "For ZO: estimate the gradients on each layer individually, scales number of forward passes per grad step by a factor of L"}
    )
    zo_variant: Optional[str] = field(
        default=None,
        metadata={"help": "Choose the MeZO variant: grad_norm or param_norm (see documentation)"}
    )
    use_zo_grad_est: bool = field(
        default=False,
        metadata={"help": "Use zero-order estimate of the gradient for zo variants"}
    )
    recompute_norms: bool = field(
        default=False,
        metadata={'help': 'Recompute the grad or parameter norm (whichever was specified as --zo_variant) at the start of each epoch.'}
    )
    scale_norm_by_num_params: bool = field(
        default=False,
        metadata={'help': 'Scale grad or param norm by 1 / sqrt(num params)'}
    )
    norm_running_update: bool = field(
        default=False,
        metadata={"help": "When performing --zo_by_layer and using --zo_variant 'grad_norm', update the layer grad norms as they are recomputed at each step"}
    )
    change_grad_estimate: bool = field(
        default=False,
        metadata={"help": "Changes the expectation of the ZO gradient estimate according to zo_variant, instead of just modifying the variance"}
    )

    # prefix tuning hyperparameters
    prefix_tuning: bool = field(
        default=False,
        metadata={"help": "Prefix tuning"}
    )
    num_prefix: int = field(
        default=10,
        metadata={"help": "How many prefix tokens to use"}
    )
    no_reparam: bool = field(
        default=False,
        metadata={"help": "No reparameterization trick"}
    )
    prefix_init_by_real_act: bool = field(
        default=False,
        metadata={"help": "For no_reparam case, randomly sample words and take their actual key/value pairs as initialization"}
    )
    layer_wise_optim: bool = field(
        default=False,
        metadata={'help': 'Optimize layer-by-layer (only for prefix + ZO)'}
    )

    max_zo_forward_steps: int = field(
        default=0,
        metadata={'help': 'Stop at this number of ZO forward steps. The trainer will take whichever is reached first, max_steps or max_zo_forward_steps.'}
    )
    
    untie_emb: bool = field(
        default=False,
        metadata={"help": "Untie embeddings from lm head. Only work for OPT!!"}
    )
    tie_emb: bool = field(
        default=False,
        metadata={"help": "Tie embeddings from lm head. Only work for RoBERTa!!"}
    )
    
    optimize_acc: bool = field(
        default=False,
        metadata={"help": "Maximize accuracy instead of minimizing loss"}
    )


    ## hessian trainer args
    num_hvp_vecs: int = field(
        default=128,
        metadata={"help": "Number of vectors to use to estimate HVPs"}
    )
    mc_tol: float = field(
        default=0.1,
        metadata={"help": "Tolerance (on std dev) after which MC estimate is deemed converged"}
    )

    head_tuning: bool = field(
        default=False,
        metadata={"help": "Tune the head only"}
    )


@dataclass
class MyDataCollatorWithPadding:
    """
    Implements padding for LM-BFF inputs.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        mask_pos = []
        standard_features = []
        if features[0].sfc_input_ids is not None:
            sfc_batch = self.__call__([OurInputFeatures(input_ids=x.sfc_input_ids, attention_mask=x.sfc_attention_mask, mask_pos=x.sfc_mask_pos) for x in features])

        for item in features:
            standard_item = {}
            for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)
            mask_pos.append(item.mask_pos)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(mask_pos):
            batch["mask_pos"] = torch.tensor(mask_pos)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        if features[0].sfc_input_ids is not None:
            batch["sfc_input_ids"] = sfc_batch["input_ids"]
            batch["sfc_attention_mask"] = sfc_batch["attention_mask"]
            batch["sfc_mask_pos"] = sfc_batch["mask_pos"]
        return batch


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.sweep:
        now = datetime.now()
        dt_str = now.strftime('%m_%d_%H_%M_%S')
        training_args.output_dir = os.path.join(training_args.output_dir, dt_str)

    #if model_args.apply_lora:
    #    assert 'roberta' in model_args.model_name_or_path, 'LoRA only implemented for RoBERTa models'

    if training_args.kernel_formula == 'asymmetric_signgd':
        assert training_args.binary_classification, 'asymmetric solver not implemented for multi-class setting, use --binary_classification'

    if training_args.optimizer_variant != '':
        assert training_args.optimizer == 'sgd', 'variants on optimizer are only implemented for SGD'

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True


    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            new_sfc_template = data_args.sfc_prompt + ''
            old_template = old_template.replace('*cls*', '')
            old_template = old_template.replace('*bos*', '')
            if data_args.gpt3_in_context_head:
                new_template = new_template.replace('*cls*', '')
                new_template = new_template.replace('*bos*', '')

            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                if "opt" in model_args.model_name_or_path or "gpt" in model_args.model_name_or_path:
                    sub_template = sub_template + "*labelx_{}*".format(instance_id)
                else:
                    sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + data_args.gpt3_demo_separator + sub_template # Put context at the end
                    new_sfc_template = new_sfc_template + data_args.gpt3_demo_separator + sub_template
                else:
                    new_template = sub_template + data_args.gpt3_demo_separator + new_template # Put context at the beginning
                    new_sfc_template = sub_template + data_args.gpt3_demo_separator + new_sfc_template
            if data_args.gpt3_in_context_head:
                new_template = "*bos*" + new_template
                new_sfc_template = "*bos*" + new_sfc_template
            logger.info("| {} => {}".format(data_args.template, new_template))
            logger.info("New SFC template (in-context learning): {}".format(new_sfc_template))
            data_args.template = new_template
            if model_args.icl_sfc:
                data_args.icl_sfc_prompt = new_sfc_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config_kwargs = {'apply_lora': model_args.apply_lora,
                    'lora_alpha': model_args.lora_alpha,
                    'lora_r': model_args.lora_r}
    if model_args.apply_lora:
        if 'roberta' in model_args.model_name_or_path:
            config = RobertaConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                # apply_forward_delta=data_args.apply_forward_delta,
                # config其实无需修改， 真正的功能是后面的 replace 函数， provider 先在这里传 none , 目前是因为懒得修改了， 一些冗余的代码后面在修改，
                # uv_provider=None,   
                # z_provider=None,
                # enable_x=data_args.enable_x,
                # enable_diffx=data_args.enable_diffx,
                # enable_w=data_args.enable_w,
                # enable_diffw=data_args.enable_diffw,
                # mx_w_elem_format=data_args.mx_w_elem_format,
                # mx_a_elem_format=data_args.mx_a_elem_format,
                # mx_diffw_elem_format=data_args.mx_diffw_elem_format,
                # mx_diffa_elem_format=data_args.mx_diffa_elem_format,
                **config_kwargs)
        else:
            config = OPTConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                **config_kwargs
            )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir
        )

    if training_args.untie_emb:
        logger.warn("Untie embeddings and lm head")
        logger.warn("NOTE that this only works for OPT. By default RoBERTa model embeddings are already untied.")
        config.tie_word_embeddings = False

    if 'prompt' in model_args.few_shot_type:
        model_fn = MODEL_TYPES[config.model_type]
    elif model_args.few_shot_type == 'finetune':
        if training_args.from_linearhead:
            model_fn = MODEL_TYPES[config.model_type]
        else:
            model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )
    if "opt" in model_args.model_name_or_path:
        # Set SEP token
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = 0
    if "gpt2" in model_args.model_name_or_path:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id


    if training_args.hf_inference_model:
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-5}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}

        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            device_map='auto',
            torch_dtype=torch.float16 if training_args.efficient_zero_order_fp16 else torch.float32,
            max_memory=max_memory,
        )
    else:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    if training_args.tie_emb:
        logger.warn("Tie embeddings. Only work for RoBERTa (in our code by default they are not tied)")
        model.tie_emb()
    
    if training_args.head_tuning:
        if model.config.model_type == "roberta":
            head_name = "lm_head"

        for n, p in model.named_parameters():
            if head_name not in n:
                p.requires_grad = False 
            else:
                logger.info(f"Only tuning {n}")        

    tokenizer.model_type = model.config.model_type

    if training_args.exclude_first_layers != -1:
        model = convert_opt_model(model, config, training_args.exclude_first_layers)

    if training_args.prefix_tuning:
        from src.prefix import PrefixTuning
        PrefixTuning(model, num_prefix=training_args.num_prefix, reparam=not training_args.no_reparam, float16=training_args.efficient_zero_order_fp16, init_by_real_act=training_args.prefix_init_by_real_act)

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_train
        else None
    )
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_predict
        else None
    )

    set_seed(training_args.seed)

    if training_args.random_model_init:
        model.init_weights() # reinit weights to random

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    if eval_dataset.label_word_list is not None:
        model.label_word_list = torch.tensor(eval_dataset.label_word_list).long().to(training_args.device)
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    if model_args.apply_lora:
        for name, param in model.named_parameters():
            if (name.startswith('roberta') and "lora" not in name) or (name.startswith('opt') and "lora" not in name):
                param.requires_grad_(False)
    
    # 启用CustomLinear功能
    custom_layers = []
    if data_args.enable_custom_linear:
        logger.info("启用CustomLinear层来记录奇偶数次推理数据")
        
        # 创建保存目录
        os.makedirs(data_args.custom_linear_plot_dir, exist_ok=True)
        
        # 替换模型中的Linear层为CustomLinear层
        replace_linear_layers_with_custom(model)
        logger.info(f"Replace All Linear Layers with CustomLinear Layers")
        
        print(f"model:{model}")
        
    # 启用DiffLinear功能
    if data_args.enable_differential_linear:
        logger.info("启用DifferentialLinear层来优化偶数次推理")
        
        # 创建保存目录
        os.makedirs(data_args.differential_plot_dir, exist_ok=True)
        
        # 替换模型中的Linear层为DifferentialLinear层
        logger.info(model)
    
        replace_linear_layers_with_differential(model, enable_validation=data_args.enable_differential_validation, enable_accurate_diff=data_args.enable_accurate_diff, max_steps=training_args.max_steps)
        logger.info("Replace All Linear Layers with DifferentialLinear Layers")
        
        print(f"model:{model}")
        
        
    apply_trainable_mode(model, training_args.trainable_mode, logger=logger)
    
    
    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]

            num_sample = test_dataset.num_sample if eval_dataset is None else eval_dataset.num_sample
            logits = predictions.reshape([num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer_classes = {
        "standard": LowRankTrainer,
        "linearhead": LinearHeadTrainer,
        "Lowrank": LowRankTrainer
    }
    trainer_class = trainer_classes[training_args.trainer]
    trainer_kwargs = {}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=MyDataCollatorWithPadding(tokenizer),
        **trainer_kwargs
    )
    
    
    # apply_forward_delta 
    if data_args.apply_forward_delta:
        logger.info("replacing nnlayers in model with difflayers")
        uv_provider = trainer.make_uv_provider()
        z_provider = trainer.make_z_provider()
        
        if model.config.model_type == "roberta":
            QuantizeRobertaForLOZO(model=model,
                                   mx_w_elem_format=data_args.mx_w_elem_format, mx_a_elem_format=data_args.mx_a_elem_format, mx_diffw_elem_format=data_args.mx_diffw_elem_format, mx_diffa_elem_format=data_args.mx_diffa_elem_format, 
                                   enable_w=data_args.enable_w, enable_x=data_args.enable_x, enable_diffx=data_args.enable_diffx, enable_diffw=data_args.enable_diffw, 
                                   uv_provider=uv_provider, z_provider=z_provider)
            apply_trainable_mode(model, training_args.trainable_mode, logger=logger)
        logger.info("Replace All nnLayers with diffLayers to support forward_delta")
        print(f"model:{model}") 
        
    print("[ARGS CHECK]", getattr(training_args, "debug_forward_delta", "MISSING"),
      getattr(training_args, "debug_forward_delta_steps", "MISSING"),
      getattr(training_args, "debug_forward_delta_tol", "MISSING"),
      getattr(training_args, "debug_forward_delta_abort", "MISSING"),
      flush=True)
        
    
    # Use QdiffLinear
    if data_args.enable_QdiffLinear:
        logger.info("Use QdiffLinear to reduce the computing stress and accelerate LOZO")
        logger.info(model)
        
        uv_provider = trainer.make_uv_provider()
        
        replace_linear_with_Qdifflinear(model=model, mx_quan=data_args.mx_quan, mx_w_elem_format=data_args.mx_w_elem_format, mx_a_elem_format=data_args.mx_a_elem_format, mx_diffw_elem_format=data_args.mx_diffw_elem_format, mx_diffa_elem_format=data_args.mx_diffa_elem_format, act_quant=data_args.act_quant_pattern, weight_quant="per_outchannel", act_b=data_args.act_bit, weight_b=data_args.weight_bit, enable_w=data_args.enable_w, enable_x=data_args.enable_x, enable_diffx=data_args.enable_diffx, enable_diffw=data_args.enable_diffw, max_steps=training_args.max_steps, use_uv_diffw=data_args.use_uv_diffw, uv_provider = uv_provider)
        logger.info("Replace All Linear Layers with DifferentialLinear Layers")
        # calculate and validate if QdiffLinear works successfully
        Qdiff_layers = get_all_Qdifflayers(model)
        logger.info(f"we find {len(Qdiff_layers)} QdiffLinears which are successfully replaced")
        print(f"model:{model}")    

    # Calibration
    if model_args.sfc:
        inputs = tokenizer([data_args.sfc_prompt.replace("_", " ")], return_tensors="pt")
        logger.info(f"Calibrating SFC with prompt: {data_args.sfc_prompt}")
        logger.info("Inputs: {}".format(inputs.input_ids))
        inputs = inputs.to(model.device)
        with torch.no_grad():
            model.eval()
            logits = model(**inputs)[0]
        model.sfc_bias = F.log_softmax(logits.squeeze(0).detach())
        logger.info("SFC bias: {}".format(model.sfc_bias))


    # Training
    if training_args.do_train:
        train_result = trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)

        if training_args.trainer == "hessian":
            # Write the result to log
            with FileLock('log_hessian.lock'):
                with open('log_hessian', 'a') as f:
                    train_result.update(vars(model_args))
                    train_result.update(vars(training_args))
                    train_result.update(vars(data_args))
                    if 'evaluation_strategy' in train_result:
                        train_result.pop('evaluation_strategy')
                    f.write(str(train_result) + '\n')
            exit()

        # Use the early stop, so do not save the model in the end (unless specify save_at_last)

        if training_args.trainer == "standard" or training_args.trainer == "linearhead":
            if training_args.save_at_last:
                trainer.save_model(training_args.output_dir)

            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)
                torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
                torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
            
            if training_args.evaluate_during_training:
                # Reload the best checkpoint (for eval)
                # model.load_state_dict(trainer.best_model_ckpt)
                # if training_args.prefix_tuning:
                #     # We can load prefix by directly using load_state_dict
                #     model.load_state_dict(torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin")))
                # else:
                #     model = model_fn.from_pretrained(training_args.output_dir)
                # if training_args.exclude_first_layers != -1:
                #     model = convert_opt_model(model, config, training_args.exclude_first_layers)
                
                # model = model.to(training_args.device)
                
                # Now we just reload this from memory instead of disk <-- much faster
                trainer.model.load_state_dict(trainer.best_model_ckpt)

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir
    }

    eval_results = {}
    if training_args.do_eval :
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        ### Don't evaluate on mnli-mm for our purposes
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     test_datasets.append(
        #         FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", use_demo=('demo' in model_args.few_shot_type))
        #     )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)


    if trainer.is_world_process_zero():
        with FileLock('log.lock'):
            with open(training_args.log_file, 'a') as f:
                final_result.update(vars(model_args))
                final_result.update(vars(training_args))
                final_result.update(vars(data_args))
                if 'evaluation_strategy' in final_result:
                    final_result.pop('evaluation_strategy')
                f.write(str(final_result) + '\n')

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    # 如果启用了CustomLinear，在训练/评估完成后绘制数据分布图
    if data_args.enable_custom_linear:
        logger.info("开始绘制CustomLinear层的数据分布图...")
        plot_custom_linear_data(model, data_args)
    
    
    # 获取所有DifferentialLinear层
    diff_layers = get_all_differential_layers(model)
    logger.info(f"共找到 {len(diff_layers)} 个DifferentialLinear层")
                
    # 如果启用验证，绘制误差图
    if data_args.enable_differential_validation:
        logger.info("\n=== 验证结果分析 ===")
        for i, layer in enumerate(diff_layers):
            if layer.validation_errors and layer.sparsity_diff:
                logger.info(f"\ndiff层 {layer.layer_name} ({layer.in_features}→{layer.out_features}):")
                
                # 绘制误差图
                plot_path = os.path.join(data_args.differential_plot_dir, f'{layer.layer_name}.png')
                # layer.plot_validation_errors(save_path=plot_path, show_plot=False)
                layer.plot_sparsity_diff_distribution(save_path=plot_path, show_plot=False)
                layer.plot_activation_weight_3d(save_path=plot_path)
                logger.info(f"sparsity analyse 图已保存: {plot_path}")
    
    logger.info("\n测试完成！")
    
    
    return eval_results
    
    



def plot_custom_linear_data(model, data_args):
    from src.custom_linear import CustomLinear
    """
    绘制CustomLinear层的数据分布图并保存numpy文件
    
    Args:
        model: 模型
        data_args: 数据参数
        training_args: 训练参数
    """
    
    # 创建numpy数据保存目录
    numpy_data_dir = os.path.join(data_args.custom_linear_plot_dir, "numpy_data")
    os.makedirs(numpy_data_dir, exist_ok=True)
    logger.info(f"numpy数据将保存到: {numpy_data_dir}")
    
    # 为每个CustomLinear层绘制图表和保存数据
    # for i, layer in enumerate(custom_layers):
    for name, child in model.named_modules():
        if isinstance(child, CustomLinear):
            print(f"name:{name}")
            print(f"child:{child}")

            name = name.split(".")[3:]
            name = ".".join(name)
            
            logger.info(f"为 {name} 绘制图表并保存数据 (推理次数: {child.inference_count})")
            
            # 绘制数据分布图
            dist_plot_path = os.path.join(
                data_args.custom_linear_plot_dir, 
                f"layer_{name}_data_dist.png"
            )
            child.plot_data_distribution(save_path=dist_plot_path, show_plot=False)
            
            
            # 保存numpy数据文件
            numpy_save_path = os.path.join(numpy_data_dir, f"layer_{name}_data.npz")
            save_layer_data_to_numpy(child, numpy_save_path, name)
            
            # 保存数据摘要
            summary = child.get_data_summary()
            summary_path = os.path.join(
                data_args.custom_linear_plot_dir, 
                f"layer_{name}_summary.txt"
            )
            with open(summary_path, 'w') as f:
                f.write(f"CustomLinear Layer {name} 数据摘要\n")
                f.write(f"推理次数: {summary['inference_count']}\n")
                f.write(f"奇数次记录数: {summary['odd_records']}\n")
                f.write(f"偶数次记录数: {summary['even_records']}\n")
                f.write("\n统计信息:\n")
                for key, values in summary['stats'].items():
                    if values:
                        f.write(f"{key}: {len(values)} 个值\n")
                        f.write(f"  最新值: {values[-1]:.6f}\n")
                        if len(values) > 1:
                            f.write(f"  平均值: {np.mean(values):.6f}\n")
                            f.write(f"  标准差: {np.std(values):.6f}\n")
    
    logger.info(f"Saved data distribution plots to: {data_args.custom_linear_plot_dir}")
    logger.info(f"Saved numpy data to: {numpy_data_dir}")
    


def save_layer_data_to_numpy(layer, save_path, layer_idx):
    """
    将CustomLinear层的数据保存为numpy文件
    
    Args:
        layer: CustomLinear层
        save_path: 保存路径
        layer_idx: 层索引
    """
    # try:
    # 准备要保存的数据
    data_dict = {
        'layer_index': layer_idx,
        'inference_count': layer.inference_count,
        'odd_records_count': len(layer.odd_inputs),
        'even_records_count': len(layer.even_inputs),
        'max_records': layer.max_records,
        'record_interval': layer.record_interval
    }
    
        # 保存奇数次输入输出数据
    if layer.odd_inputs:
        odd_inputs_np = [(x.detach().cpu().numpy() if hasattr(x, 'detach') else (x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x))) for x in layer.odd_inputs]
        odd_outputs_np = [(x.detach().cpu().numpy() if hasattr(x, 'detach') else (x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x))) for x in layer.odd_outputs]
        data_dict['odd_inputs'] = np.array(odd_inputs_np, dtype=object)
        data_dict['odd_outputs'] = np.array(odd_outputs_np, dtype=object)
        data_dict['odd_inputs_shape'] = np.array([x.shape for x in odd_inputs_np], dtype=object)
        data_dict['odd_outputs_shape'] = np.array([x.shape for x in odd_outputs_np], dtype=object)

        # 保存偶数次输入输出数据
    if layer.even_inputs:
        even_inputs_np = [(x.detach().cpu().numpy() if hasattr(x, 'detach') else (x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x))) for x in layer.even_inputs]
        even_outputs_np = [(x.detach().cpu().numpy() if hasattr(x, 'detach') else (x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x))) for x in layer.even_outputs]
        data_dict['even_inputs'] = np.array(even_inputs_np, dtype=object)
        data_dict['even_outputs'] = np.array(even_outputs_np, dtype=object)
        data_dict['even_inputs_shape'] = np.array([x.shape for x in even_inputs_np], dtype=object)
        data_dict['even_outputs_shape'] = np.array([x.shape for x in even_outputs_np], dtype=object)
    
    # 保存统计信息
    data_dict['stats'] = layer.stats
    
    # 保存为npz文件
    np.savez_compressed(save_path, **data_dict)
    logger.info(f"第 {layer_idx} 层数据已保存到: {save_path}")
        
    # except Exception as e:
    #     logger.error(f"保存第 {layer_idx} 层数据时出错: {e}")


if __name__ == "__main__":
    main()
