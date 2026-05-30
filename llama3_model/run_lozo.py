import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import sys
from pathlib import Path
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict, field
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from LOZOtrainer import LowRankTrainer
from LOZOtrainer0 import LowRankTrainer as OrdinaryLowRankTrainer
import random
from diff_fake_quant_mx import QuantizeOPTForLOZO, QuantizeLlamaForLOZO
from modeling_opt import OPTForCausalLM
from modeling_llama import LlamaForCausalLM


@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_path: str = None
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    ## - LOZO: low rank zeroth-order (LOZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 
    
    # test LOZO muti-GPU
    debug_device_preflight: bool = field(
        default=False,
        metadata={"help": "Run one train forward_delta batch and one eval normal-forward batch before training."}
    )
    
    debug_device_preflight_only: bool = field(
        default=False,
        metadata={"help": "Run device preflight and exit without training."}
    )
    
    # LOZO
    zo_eps: float = 1e-3 # eps in LOZO
    step_interval: int = 50 # $\nu$ in LOZO
    rank_r: int = 2 # rank r in LOZO

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)
    
    # forward_delta
    apply_forward_delta: bool = field(
        default=False,
        metadata={"help":"Use all difflayer in diff_fake_quant_mx.py"}
    )
    
    trainable_mode: str = field(
        default="all",
        metadata={"help": "Trainable params: all | linear_only (freeze embedding & layernorm)"}
    )
    
    # Quantization 
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

    # Linear outlier profiling
    profile_linear: bool = field(
        default=False,
        metadata={"help": "Enable low-intrusion linear outlier profiling"}
    )
    profile_linear_dir: str = field(
        default="linear_profile",
        metadata={"help": "Directory for linear outlier profile plots and reports"}
    )
    profile_linear_max_calls_per_layer: int = field(
        default=2,
        metadata={"help": "Maximum profiled calls per linear layer"}
    )
    profile_linear_block_size: int = field(
        default=16,
        metadata={"help": "Block size for linear outlier block statistics"}
    )
    profile_linear_token_stride: int = field(
        default=1,
        metadata={"help": "Token stride for activation surface plots"}
    )
    profile_linear_channel_stride: int = field(
        default=4,
        metadata={"help": "Channel stride for activation surface plots"}
    )
    profile_linear_weight_stride: int = field(
        default=8,
        metadata={"help": "Weight stride for weight and delta-weight surface plots"}
    )
    profile_linear_layer_regex: str = field(
        default="",
        metadata={"help": "Optional regex selecting layer names for linear profiling"}
    )
    # LOZO perturbation distribution profiling
    profile_perturb_distribution: bool = field(
        default=False,
        metadata={"help": "Profile raw/direct/delta LOZO perturbation distributions"}
    )
    profile_perturb_distribution_dir: str = field(
        default="linear_outlier_plots/perturb_distribution",
        metadata={"help": "Directory for perturbation distribution plots and reports"}
    )
    profile_perturb_distribution_layer_regex: str = field(
        default=(
            r"^model\.layers\.(0|7|14|21|27)\..*"
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        ),
        metadata={"help": "Regex selecting Linear layers for perturbation profiling"}
    )
    profile_perturb_distribution_log_y: bool = field(
        default=False,
        metadata={"help": "Use log-density y axes for perturbation histograms"}
    )

    # SmoothQuant + SVD-LoRA compensation. Disabled by default.
    svd_lora_compensation: bool = field(
        default=False,
        metadata={"help": "Enable always-on SmoothQuant/SVD low-rank compensation factors."}
    )
    svd_lora_act_scales_path: str = field(
        default="",
        metadata={"help": "Path to precomputed SmoothQuant activation scales from SVD-ZOO-Quant/experiments/generate_act_scales_hf.py."}
    )
    # need to be same as the param in SVD-ZOO-Quant
    svd_lora_smooth_alpha: float = field(
        default=0.85,
        metadata={"help": "SmoothQuant alpha for SVD-LoRA compensation."}
    )
    
    svd_lora_checkpoint_path: str = field(
        default="",
        metadata={"help": "Path to a PEFT-style SVD-LoRA checkpoint generated by SVD-ZOO-Quant/experiments/init_lora_from_svd.py."}
    )
    svd_lora_rank: int = field(
        default=32,
        metadata={"help": "Rank of the offline SVD-LoRA compensation factors. Independent from LOZO perturbation rank_r."}
    )
    svd_lora_quant_format: str = field(
        default="nvfp4",
        metadata={"help": "Quantization format matching the offline SVD checkpoint: none, nvint4, nvfp4, mxint4, mxfp4."}
    )
    svd_lora_group_size: int = field(
        default=0,
        metadata={"help": "Group size for SVD-LoRA fake quantization. 0 uses the format default: 16 for NV, 32 for MX."}
    )
    svd_lora_adapter_name: str = field(
        default="default",
        metadata={"help": "Adapter name used in PEFT-style SVD-LoRA checkpoint keys."}
    )
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()
        self.linear_outlier_profiler = None
        self.perturb_distribution_profiler = None
        if self.args.profile_linear and self.args.local_rank <= 0:
            self._attach_linear_outlier_profiler()

    def _attach_linear_outlier_profiler(self):
        from linear_outlier_profile import LinearOutlierProfiler, attach_linear_outlier_profiler

        if self.linear_outlier_profiler is None:
            self.linear_outlier_profiler = LinearOutlierProfiler(
                output_dir=self.args.profile_linear_dir,
                max_calls_per_layer=self.args.profile_linear_max_calls_per_layer,
                block_size=self.args.profile_linear_block_size,
                token_stride=self.args.profile_linear_token_stride,
                channel_stride=self.args.profile_linear_channel_stride,
                weight_stride=self.args.profile_linear_weight_stride,
                layer_regex=self.args.profile_linear_layer_regex,
            )
        else:
            self.linear_outlier_profiler.close()

        attach_linear_outlier_profiler(self.model, self.linear_outlier_profiler)
        logger.info(f"Attached linear outlier profiler: {self.args.profile_linear_dir}")

    def _flush_linear_outlier_profiler(self):
        if self.linear_outlier_profiler is not None:
            self.linear_outlier_profiler.flush()
    
    def _attach_perturb_distribution_profiler(self):
        from perturb_distribution_profile import (
            PerturbDistributionProfiler,
            attach_perturb_distribution_profiler,
            make_linear_weight_mx_specs,
        )

        direct_format = self.args.mx_w_elem_format if self.args.enable_w else None
        delta_format = self.args.mx_diffw_elem_format if self.args.enable_diffw else None
        self.perturb_distribution_profiler = PerturbDistributionProfiler(
            output_dir=self.args.profile_perturb_distribution_dir,
            layer_regex=self.args.profile_perturb_distribution_layer_regex,
            direct_weight_mx_specs=make_linear_weight_mx_specs(direct_format),
            delta_weight_mx_specs=make_linear_weight_mx_specs(delta_format),
            log_y=self.args.profile_perturb_distribution_log_y,
        )
        attach_perturb_distribution_profiler(self.model, self.perturb_distribution_profiler)
        logger.info(
            "Attached perturb distribution profiler: %s",
            self.args.profile_perturb_distribution_dir,
        )

    def _flush_perturb_distribution_profiler(self):
        if self.perturb_distribution_profiler is not None:
            self.perturb_distribution_profiler.flush()

    def _add_svd_zoo_to_path(self):
        svd_root = Path(__file__).resolve().parents[1] / "SVD-ZOO-Quant"
        if not svd_root.exists():
            raise FileNotFoundError(f"SVD-ZOO-Quant directory not found: {svd_root}")
        if str(svd_root) not in sys.path:
            sys.path.insert(0, str(svd_root))

    def _svd_lora_quant_settings(self):
        quant_format = (self.args.svd_lora_quant_format or "none").lower()
        if quant_format in {"none", "fp", "fp16", "fp32"}:
            return "none", "none"
        if quant_format == "nvfp4":
            return "nv", "nvfp4"
        if quant_format == "nvint4":
            return "nv", "nvint4"
        if quant_format == "mxint4":
            return "mx", "mxint4"
        if quant_format == "mxfp4":
            return "mx", "mxfp4"
        raise ValueError(
            f"Unsupported --svd_lora_quant_format={self.args.svd_lora_quant_format}. "
            "Expected none, nvint4, nvfp4, mxint4, or mxfp4."
        )

    def _prepare_svd_lora_compensation(self, model, tokenizer):
        if not self.args.svd_lora_compensation:
            return None
        if not self.args.apply_forward_delta:
            raise ValueError("--svd_lora_compensation requires --apply_forward_delta")
        if not self.args.svd_lora_checkpoint_path:
            raise ValueError(
                "SVD-LoRA compensation requires a checkpoint generated by "
                "SVD-ZOO-Quant/experiments/init_lora_from_svd.py"
            )

        self._add_svd_zoo_to_path()
        from smoothquant.smooth import smooth_lm

        if not self.args.svd_lora_act_scales_path:
            raise ValueError(
                "SVD-LoRA compensation requires precomputed activation scales generated by "
                "SVD-ZOO-Quant/experiments/generate_act_scales_hf.py"
            )
        act_scales_path = Path(self.args.svd_lora_act_scales_path)
        if not act_scales_path.exists():
            raise FileNotFoundError(
                "SVD-LoRA compensation requires precomputed activation scales generated by "
                f"SVD-ZOO-Quant/experiments/generate_act_scales_hf.py, but not found: {act_scales_path}"
            )
        act_scales = torch.load(act_scales_path, map_location="cpu")

        logger.info("[SVD-LoRA] applying SmoothQuant smoothing")
        w0 = model.model.layers[0].self_attn.q_proj.weight.detach().float().clone()  # debug
        smooth_lm(model, act_scales, self.args.svd_lora_smooth_alpha)
        w1 = model.model.layers[0].self_attn.q_proj.weight.detach().float()  # debug
        print("[SMOOTH-CHECK] q_proj delta norm =", (w1 - w0).norm().item())

        checkpoint_path = Path(self.args.svd_lora_checkpoint_path)
        checkpoint_file = checkpoint_path / "adapter_model.bin" if checkpoint_path.is_dir() else checkpoint_path
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                "SVD-LoRA compensation requires a checkpoint generated by "
                f"SVD-ZOO-Quant/experiments/init_lora_from_svd.py, but not found: {checkpoint_file}"
            )
        logger.info("[SVD-LoRA] loading checkpoint from %s", checkpoint_file)
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        config_file = checkpoint_file.parent / "svd_lora_config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                "SVD-LoRA compensation requires checkpoint metadata svd_lora_config.json "
                f"next to {checkpoint_file}. Regenerate the checkpoint with "
                "SVD-ZOO-Quant/experiments/init_lora_from_svd.py."
            )
        with config_file.open("r", encoding="utf-8") as handle:
            self.svd_lora_metadata = json.load(handle)
        
        ckpt_quant_format = ""
        ckpt_rank = self.svd_lora_metadata.get("rank")
        if ckpt_rank is not None and int(ckpt_rank) != int(self.args.svd_lora_rank):
            raise ValueError(
                f"SVD-LoRA rank mismatch: checkpoint rank={ckpt_rank}, "
                f"--svd_lora_rank={self.args.svd_lora_rank}"
            )
        # ckpt_quant_format = (self.svd_lora_metadata.get("quant_format") or "").lower()
        # arg_quant_format = (self.args.svd_lora_quant_format or "none").lower()
        # if ckpt_quant_format and ckpt_quant_format != arg_quant_format:
        #     raise ValueError(
        #         f"SVD-LoRA quant_format mismatch: checkpoint quant_format={ckpt_quant_format}, "
        #         f"--svd_lora_quant_format={arg_quant_format}. The SVD target and LOZO base quantization "
        #         "must match."
        #     )
        mode = (self.svd_lora_metadata.get("compensation_mode") or "quant_error").lower()
        low_rank_input = (self.svd_lora_metadata.get("low_rank_input") or "").lower()
        logger.info("[SVD-LoRA] checkpoint compensation_mode=%s quant_format=%s", mode, ckpt_quant_format)
        if mode == "calib_residual":
            if arg_quant_format == "none" and low_rank_input == "quantized":
                logger.warning(
                    "[SVD-LoRA] calib_residual checkpoint used low_rank_input=quantized, "
                    "but runtime quant_format=none uses fp SVD branch input."
                )
            if arg_quant_format != "none" and low_rank_input == "fp":
                logger.warning(
                    "[SVD-LoRA] calib_residual checkpoint used low_rank_input=fp, "
                    "but runtime quantized SVD path uses quantized activation input."
                )
        return state_dict
        
    def get_max_memory(self, reserve_gb=6, low0_extra_reserve_gb=8):
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()
            free_gib = int(free / 1024**3)
    
            allowed = free_gib - reserve_gb
            if i == 0:
                # Keep more room on cuda:0 for input batches, logits, lm_head boundary, CUDA context, etc.
                allowed -= low0_extra_reserve_gb
    
            max_memory[i] = f"{max(1, allowed)}GiB"
        return max_memory


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 or self.args.load_bfloat16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name if self.args.model_path is None else self.args.model_path)
            is_llama = "llama" in self.args.model_name.lower() or getattr(config, "model_type", None) == "llama"
            if is_llama and hasattr(config, "use_cache"):
                config.use_cache = False
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM as HT_OPTForCausalLM
                model = HT_OPTForCausalLM.from_pretrained(
                    self.args.model_name if self.args.model_path is None else self.args.model_path,
                    config=config,
                )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                if is_llama:
                    model = LlamaForCausalLM.from_pretrained(
                        self.args.model_name if self.args.model_path is None else self.args.model_path,
                        config=config,
                    )
                else:
                    model = OPTForCausalLM.from_pretrained(
                        self.args.model_name if self.args.model_path is None else self.args.model_path,
                        config=config,
                    )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                max_memory = self.get_max_memory(reserve_gb=5, low0_extra_reserve_gb=0)
                logger.info(f"[DEVICE_MAP] max_memory = {max_memory}")
                logger.info(f"[DEVICE_MAP] visible cuda devices = {torch.cuda.device_count()}")
                if is_llama:
                    model = LlamaForCausalLM.from_pretrained(
                        self.args.model_name if self.args.model_path is None else self.args.model_path,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        max_memory=max_memory,
                        load_in_8bit=self.args.load_int8,
                    )
                else:
                    model = OPTForCausalLM.from_pretrained(
                        self.args.model_name if self.args.model_path is None else self.args.model_path,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        max_memory=max_memory,
                        load_in_8bit=self.args.load_int8,
                    )
            if is_llama and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
                
            logger.info(f"[DEVICE_MAP] hf_device_map = {getattr(model, 'hf_device_map', None)}")
            logger.info(f"[DEVICE_MAP] embed_tokens device = {model.model.embed_tokens.weight.device}")
            logger.info(f"[DEVICE_MAP] lm_head device before diff replace = {model.lm_head.weight.device}")
            
            model.eval()
            if getattr(config, "model_type", None) == "opt":
                print("do_layer_norm_before =", model.model.decoder.layers[0].do_layer_norm_before)
            print(model)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name if self.args.model_path is None else self.args.model_path, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        
        if is_llama:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = tokenizer.pad_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model, "generation_config"):
                model.generation_config.pad_token_id = tokenizer.pad_token_id

        self.svd_lora_state_dict = self._prepare_svd_lora_compensation(model, tokenizer)

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        self._flush_linear_outlier_profiler()
        return metrics
    
    def _run_device_preflight(self, trainer,raw_eval_samples=None):
        """
        Fast multi-device smoke test before real training.
    
        It checks:
        1. one LOZO forward_delta train step if apply_forward_delta=True
        2. one normal forward / option-loss eval batch
        3. Check final Framework.evaluate / one_step_pred path.
        """
    
        logger.info("========== DEVICE PREFLIGHT START ==========")
        logger.info(f"hf_device_map = {getattr(self.model, 'hf_device_map', None)}")
        logger.info(f"param dtypes = {sorted({str(p.dtype) for p in self.model.parameters()})}")
        logger.info(f"param devices = {sorted({str(p.device) for p in self.model.parameters()})}")
    
        # Preserve RNG state so the smoke test does not affect the actual run too much.
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
        def _reset_diff_counters_and_caches():
            # forward_delta modules keep inference_count; reset after preflight.
            for module in self.model.modules():
                if hasattr(module, "inference_count"):
                    module.inference_count = 0
        
            # LOZOtrainer caches u/v/z during lowrank_zo_step; clear them after preflight.
            for name in ["u", "v", "z"]:
                if hasattr(trainer, name):
                    obj = getattr(trainer, name)
                    if isinstance(obj, dict):
                        obj.clear()
        
            # Very important:
            # lowrank_zo_step uses `hasattr(self, "step")` to decide whether to start from step=0.
            # If preflight leaves step=0 but clears self.v, the first real training step becomes step=1
            # and tries to reuse missing self.v[name], causing KeyError.
            if hasattr(trainer, "step"):
                delattr(trainer, "step")
        
            if hasattr(trainer, "projected_grad"):
                trainer.projected_grad = None
        
            if hasattr(trainer, "zo_random_seed"):
                delattr(trainer, "zo_random_seed")
    
        try:
            # 1. Check train forward_delta path.
            if self.args.apply_forward_delta:
                logger.info("[PREFLIGHT] checking one train batch with lowrank_zo_step / forward_delta ...")
                train_batch = next(iter(trainer.get_train_dataloader()))
                loss = trainer.lowrank_zo_step(self.model, train_batch)
                logger.info(f"[PREFLIGHT] forward_delta train batch OK, loss={float(loss.detach().cpu())}")
    
                # lowrank_zo_step should restore params, but it increments counters and fills u/v/z caches.
                _reset_diff_counters_and_caches()
            else:
                logger.info("[PREFLIGHT] skip forward_delta check because apply_forward_delta=False")
    
            # 2. Check normal eval forward path.
            # This covers model(**inputs), wrapped forward_wrap_with_option_len, lm_head, logits/input_ids/labels devices.
            logger.info("[PREFLIGHT] checking one eval batch with normal forward / compute_loss ...")
            eval_batch = next(iter(trainer.get_eval_dataloader()))
            eval_batch = trainer._prepare_inputs(eval_batch)
    
            self.model.eval()
            with torch.inference_mode():
                with trainer.compute_loss_context_manager():
                    eval_loss = trainer.compute_loss(self.model, eval_batch)
    
            logger.info(f"[PREFLIGHT] normal eval batch OK, loss={float(eval_loss.detach().cpu())}")
            
            # 3. Check final Framework.evaluate / one_step_pred path.
            # This is different from Trainer.evaluate: after training, run_lozo resets
            # the option-loss wrapper and calls Framework.evaluate([], eval_samples).
            if raw_eval_samples is not None and len(raw_eval_samples) > 0 and not self.args.no_eval:
                logger.info("[PREFLIGHT] checking one final-eval style sample via Framework.one_step_pred ...")

                if self.args.only_train_option and not self.args.non_diff and hasattr(self.model, "original_forward"):
                    saved_forward = self.model.forward
                    self.model.forward = self.model.original_forward
                    try:
                        pred = self.one_step_pred([], raw_eval_samples[0], verbose=False)
                    finally:
                        self.model.forward = saved_forward
                else:
                    pred = self.one_step_pred([], raw_eval_samples[0], verbose=False)

                logger.info(f"[PREFLIGHT] final-eval style sample OK, pred={pred.predicted_candidate}")
            else:
                logger.info("[PREFLIGHT] skip final-eval style sample check")
                
        except Exception as e:
            logger.exception("[PREFLIGHT] FAILED. This usually means a device mismatch or unsupported forward path.")
            raise
    
        finally:
            _reset_diff_counters_and_caches()
    
            # Restore RNG states.
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    
        logger.info("========== DEVICE PREFLIGHT PASSED ==========")


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.trainer == "LOZO":
            print("tariner is LOZO")
            trainer_cls = LowRankTrainer if self.args.apply_forward_delta else OrdinaryLowRankTrainer
            trainer = trainer_cls(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())
            
        # Check: if weight tying?
        # w1 = self.model.model.decoder.embed_tokens.weight
        # w2 = self.model.lm_head.weight
        # print("HIHIHI")
        # print(w1 is w2)                               # 是否同一个 Parameter 对象
        # print(w1.data_ptr() == w2.data_ptr())         # 是否同一块显存

        # Add: apply Forward_delta
        if self.args.apply_forward_delta:
            logger.info("replacing nnlayers in model with difflayers")
            uv_provider = trainer.make_uv_provider()
            z_provider = trainer.make_z_provider()
            if self.args.svd_lora_compensation:
                from diff_fake_quant_SVD import (
                    QuantizeLlamaForLOZOSVD,
                    QuantizeOPTForLOZOSVD,
                    attach_svd_lora_to_diff_layers,
                )
                _, svd_base_quant_format = self._svd_lora_quant_settings()
        
            if self.model.config.model_type == "opt":
                quantize_opt = QuantizeOPTForLOZOSVD if self.args.svd_lora_compensation else QuantizeOPTForLOZO
                quantize_opt(
                    model=self.model,
                    mx_w_elem_format=self.args.mx_w_elem_format,
                    mx_a_elem_format=self.args.mx_a_elem_format,
                    mx_diffw_elem_format=self.args.mx_diffw_elem_format,
                    mx_diffa_elem_format=self.args.mx_diffa_elem_format,
                    enable_w=self.args.enable_w,
                    enable_x=self.args.enable_x,
                    enable_diffx=self.args.enable_diffx,
                    enable_diffw=self.args.enable_diffw,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                )
        
            elif self.model.config.model_type == "llama":
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
        
                if self.args.svd_lora_compensation:
                    QuantizeLlamaForLOZOSVD(
                        model=self.model,
                        uv_provider=uv_provider,
                        z_provider=z_provider,
                        quant_format=self.args.svd_lora_quant_format,
                        group_size=self.args.svd_lora_group_size,
                    )
                else:
                    QuantizeLlamaForLOZO(
                    model=self.model,
                    mx_w_elem_format=self.args.mx_w_elem_format,
                    mx_a_elem_format=self.args.mx_a_elem_format,
                    mx_diffw_elem_format=self.args.mx_diffw_elem_format,
                    mx_diffa_elem_format=self.args.mx_diffa_elem_format,
                    enable_w=self.args.enable_w,
                    enable_x=self.args.enable_x,
                    enable_diffx=self.args.enable_diffx,
                    enable_diffw=self.args.enable_diffw,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                    )
        
            else:
                raise NotImplementedError(
                    f"apply_forward_delta is not implemented for model_type={self.model.config.model_type}"
                )

            if self.args.svd_lora_compensation:
                if self.svd_lora_state_dict is None:
                    raise ValueError("SVD-LoRA compensation was enabled but no factors were prepared")
                svd_lora_metadata = getattr(self, "svd_lora_metadata", {})
                svd_mode = (svd_lora_metadata.get("compensation_mode") or "quant_error").lower()
                ckpt_lora_alpha = svd_lora_metadata.get("lora_alpha")
                lora_scaling = 1.0 if ckpt_lora_alpha is None else float(ckpt_lora_alpha) / max(int(self.args.svd_lora_rank), 1)
                attach_svd_lora_to_diff_layers(
                    self.model,
                    self.svd_lora_state_dict,
                    adapter_name=self.args.svd_lora_adapter_name,
                    svd_lora_rank=self.args.svd_lora_rank,
                    trainable=True,
                    freeze_base=True,
                    base_quant_format=svd_base_quant_format,
                    group_size=self.args.svd_lora_group_size,
                    compensation_mode=svd_mode,
                    lora_scaling=lora_scaling,
                )
        
            logger.info("Replace All nnLayers with diffLayers to support forward_delta")
            if self.args.profile_linear and self.args.local_rank <= 0:
                self._attach_linear_outlier_profiler()
            if self.args.profile_perturb_distribution and self.args.local_rank <= 0:
                self._attach_perturb_distribution_profiler()
            print(f"model:{self.model}")
        elif self.args.profile_perturb_distribution:
            raise ValueError(
                "--profile_perturb_distribution requires --apply_forward_delta True "
                "so selected layers receive provider u/v/scale."
            )
        
        # CheckAgain: if weight tying?
        # w1 = self.model.model.decoder.embed_tokens.weight
        # w2 = self.model.lm_head.weight
        # print(w1 is w2)                               # 是否同一个 Parameter 对象
        # print(w1.data_ptr() == w2.data_ptr())         # 是否同一块显存

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint
            
        if self.args.debug_device_preflight or self.args.debug_device_preflight_only:
            self._run_device_preflight(trainer, raw_eval_samples=eval_samples)
        if self.args.debug_device_preflight_only:
            logger.info("debug_device_preflight_only=True, exiting before trainer.train().")
            self._flush_linear_outlier_profiler()
            return

        # trainer.train(resume_from_checkpoint=last_checkpoint) 
        trainer.train(resume_from_checkpoint=None) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward
        self._flush_linear_outlier_profiler()
        self._flush_perturb_distribution_profiler()


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                print(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

if __name__ == "__main__": 
    
    main()
