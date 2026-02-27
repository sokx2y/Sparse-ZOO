########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class LowRankTrainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.named_parameters_to_optim = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        self.inference_step = 0
        self.u = {} 
        self.z = {}
        # self.u_seeds = {} 

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:   # 要是推理模型不加载训练所需的优化器和调度器
            return

        if self.optimizer is None:   # 初始化优化器
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:  # fix_layers是冻结层，该层参数不加入params中
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            # 选择Optimizer
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:   # 创建LR调度器
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def zo_forward(self, model, inputs, with_delta = False):   # 用于用ZO方法计算loss
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            self.inference_step = self.inference_step + 1
            # print(f"inference:{self.inference_step} shape: {inputs['input_ids'].shape}")
            with self.compute_loss_context_manager():
                if with_delta:
                    if not hasattr(model, "forward_delta"):
                        raise AttributeError(
                            "model 没有 forward_delta 方法，请为 model 添加此路径"
                        )
                    outputs = model.forward_delta(**inputs)

                    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
                        raise ValueError(
                            "期望 model.forward_delta 返回 (loss_base, loss_perturbed, ...)，"
                            f"但实际得到类型/长度为: {type(outputs)}, len={len(outputs) if isinstance(outputs, (tuple, list)) else 'N/A'}"
                        )
                    loss_base, loss_perturbed = outputs[0], outputs[1]
                else:
                    loss = self.compute_loss(model, inputs)
                    
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                if with_delta:
                    loss_base = loss_base.mean()
                    loss_perturbed = loss_perturbed.mean()
                else:
                    loss = loss.mean()
                
        if with_delta:
            return loss_base.detach(), loss_perturbed.detach()
        else:
            return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)
    
    def random_gaussian_matrix(self, m, n, device, dtype, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        random_matrix = torch.randn(m, n, device=device, dtype=dtype)
        return random_matrix
    


    # =========================================== LOZO Functions =======================================================================
    
    def lowrank_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector uv^t.
        """
        args = self.args
        step = self.step

        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                if step % args.step_interval == 0:
                    v = torch.randn(param.data.size(1), args.rank, device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v
                else:
                    v = self.v[name]
                    
                # new1: but failed: cached seed_u for regenration
                # base_seed = random_seed if random_seed is not None else self.zo_random_seed
                # u_seed = base_seed + hash(name) + step  
                # self.u_seeds[name] = u_seed
                # u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank, 
                                          # device=param.data.device, dtype=param.data.dtype,
                                          # random_seed=u_seed)

                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank, device=param.data.device, dtype=param.data.dtype)
                # new2: cached u (low-rank matrix memory cost)
                self.u[name] = u
                # print(f"step:{step}: name:{name}, real_u[0]:{u[0]}, real_v[0]:{v[0]}")
                param.data = param.data + scaling_factor * (u@v.t()) * self.args.zo_eps   # u*vT计算扰动矩阵
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                self.z[name] = z
                # print(f"step:{step}: name:{name}, real_v:{z[0]}")
                param.data = param.data + scaling_factor * z * self.args.zo_eps
    
    def make_z_provider(self):
        """
        provide z to replace diff_bias and diff_weight in layernorm
        """
        def provider(param_name, shape, device, dtype, inference_count):
            z = None
            if hasattr(self, "z"):
                z = self.z.get(param_name, None)
            if z is None:
                z = torch.normal(mean=0.0, std=1.0, size=shape, device=device, dtype=dtype)
            scale = -2 * self.args.zo_eps
            return z, scale
        
        return provider
    
    def make_uv_provider(self):
        """
        provide u,v^t to replace diff_weight in our quantize method(QdiffLinear).
        """
        def provider(param_name, shape, device, dtype, inference_count):
            out_f, in_f = shape
            # provide v from cache
            v = None
            if hasattr(self, "v"):
                v = self.v.get(param_name, None)
            if v is None: 
                v = torch.randn(in_f, self.args.rank, device=device, dtype=dtype)
                
            # But failed: provide u from regeneration(cache seed_u)
            # if param_name in self.u_seeds:
                # u_seed = self.u_seeds[param_name]
                # u = self.random_gaussian_matrix(m=out_f, n=self.args.rank, device=device, dtype=dtype, random_seed=u_seed)
            # else:
                # print(f"Error: no u_seeds cache to regeneration!")
            
            # use cache_u directly
            u = None
            if hasattr(self, "u"):
                u = self.u.get(param_name, None)
            if u is None: 
                u = torch.randn(out_f, self.args.rank, device=device, dtype=dtype)
            
            # The difference between the even iteration and the odd cached weights is −2×zo_eps×(UVT).
            scale = -2 * self.args.zo_eps
            return u, v, scale

        return provider
    
    def _should_probe(self, gs: int) -> bool:
        args = self.args
        if not getattr(args, "compare_seed", False):
            return False
        first_k = int(getattr(args, "compare_seed_steps", 0))
        if gs < first_k:
            return True
        # Probe around evaluation boundary
        if getattr(args, "evaluate_during_training", False):
            E = int(getattr(args, "eval_steps", 0))
            if E > 0:
                r = gs % E
                if r == 0 or r == 1 or r == (E - 1):
                    return True
        return False

    @staticmethod
    def _fp_tensor(t):
        import hashlib
        b = t.detach().cpu().numpy().tobytes()
        return hashlib.md5(b).hexdigest()[:8]
    
    def _debug_forward_delta_sanity_check(
        self,
        *,
        loss_plus_true: torch.Tensor,
        loss_minus_true: torch.Tensor,
        loss_plus_fd: torch.Tensor,
        loss_minus_fd: torch.Tensor,
    ) -> None:
        """Compare true +/- losses (two normal forwards) vs forward_delta returned losses."""
        tol = float(getattr(self.args, "debug_forward_delta_tol", 1e-6))
        abort = bool(getattr(self.args, "debug_forward_delta_abort", True))

        lp_t = float(loss_plus_true.detach().cpu())
        lm_t = float(loss_minus_true.detach().cpu())
        lp_f = float(loss_plus_fd.detach().cpu())
        lm_f = float(loss_minus_fd.detach().cpu())

        d_plus = abs(lp_t - lp_f)
        d_minus = abs(lm_t - lm_f)

        logger.info(
            "[debug_forward_delta] step=%s | plus_true=%.10f plus_fd=%.10f (abs diff=%.3e) | "
            "minus_true=%.10f minus_fd=%.10f (abs diff=%.3e)",
            getattr(self, "step", "?"),
            lp_t, lp_f, d_plus,
            lm_t, lm_f, d_minus,
        )

        if (d_plus > tol) or (d_minus > tol):
            msg = (
                f"[debug_forward_delta] FAILED step={getattr(self,'step','?')} tol={tol} | "
                f"plus_true={lp_t} plus_fd={lp_f} (abs diff={d_plus}) | "
                f"minus_true={lm_t} minus_fd={lm_f} (abs diff={d_minus})"
            )
            if abort:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)

    def lowrank_zo_step(self, model, inputs):   # 通过loss来用ZO方法估计梯度
        """
        Estimate gradient by Lowrank-zo. Return the loss from f(theta + uv^t)
        """
        args = self.args
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.v = {}
            self.v_old = {}
            self.exp_avg_m = {}
            self.exp_avg_sq = {}

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)
        
        # Debugging: log seed for comparison
        gs = int(getattr(self.state, "global_step", self.step))
        do_probe = self._should_probe(gs)
        mode_str = "fd" if getattr(args, "use_forward_delta_loss", True) else "base"
        if do_probe:
            logger.info("[seed_cmp] mode=%s step=%d zo_random_seed=%d",
                        "fd" if getattr(args, "use_forward_delta_loss", True) else "base",
                        self.step, self.zo_random_seed)


        # First function evaluation
        # and only one inference is enough, one inference, two loss
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        
        # Debugging: compare seed and uvz
        if do_probe:
            p2d = getattr(args, "compare_seed_param", "")
            p1d = getattr(args, "compare_seed_param_1d", "")
        
            # 2D u/v
            if hasattr(self, "u") and hasattr(self, "v") and (p2d in self.u) and (p2d in self.v):
                u = self.u[p2d]; v = self.v[p2d]
                logger.info("[uv_fp] mode=%s step=%d %s u_md5=%s v_md5=%s u_norm=%.6f v_norm=%.6f u00=%.6f v00=%.6f",
                            "fd" if getattr(args, "use_forward_delta_loss", True) else "base",
                            self.step, p2d,
                            self._fp_tensor(u), self._fp_tensor(v),
                            u.norm().item(), v.norm().item(),
                            u.flatten()[0].item(), v.flatten()[0].item())
            else:
                # 如果 NOT_FOUND，大概率是 compare_seed_param 的名字和你 model.named_parameters() 的真实名字不一致
                keys = list(self.u.keys())[:5] if hasattr(self, "u") else []
                logger.info("[uv_fp] mode=%s step=%d %s NOT_FOUND; example_keys=%s",
                            "fd" if getattr(args, "use_forward_delta_loss", True) else "base",
                            self.step, p2d, keys)
        
            # 1D z
            if hasattr(self, "z") and (p1d in self.z):
                z = self.z[p1d]
                logger.info("[z_fp] mode=%s step=%d %s z_md5=%s z_norm=%.6f z0=%.6f",
                            "fd" if getattr(args, "use_forward_delta_loss", True) else "base",
                            self.step, p1d,
                            self._fp_tensor(z), z.norm().item(), z.flatten()[0].item())
            else:
                keys = list(self.z.keys())[:5] if hasattr(self, "z") else []
                logger.info("[z_fp] mode=%s step=%d %s NOT_FOUND; example_keys=%s",
                            "fd" if getattr(args, "use_forward_delta_loss", True) else "base",
                            self.step, p1d, keys)
                
        # Debugging: forward_delta sanity-check
        gs = int(getattr(self.state, "global_step", self.step))
        do_probe = self._should_probe(gs)
        
        do_debug = bool(getattr(args, "debug_forward_delta", False)) and (
            gs < int(getattr(args, "debug_forward_delta_steps", 0)) or do_probe
        )

        if do_debug and (not getattr(args, "use_forward_delta_loss", True)):
            logger.warning("[debug_forward_delta] enabled but use_forward_delta_loss=False (base mode). "
                           "Sanity-check is intended for forward_delta mode, skipping.")
        loss_plus_true = None
        if do_debug and getattr(args, "use_forward_delta_loss", True):
            # true loss at theta + eps*delta using normal forward
            loss_plus_true = self.zo_forward(model, inputs, with_delta=False)


        # loss1 = self.zo_forward(model, inputs)
        if getattr(args, "use_forward_delta_loss", True):
            # one inference -> two losses
            loss1, loss2 = self.zo_forward(model, inputs, with_delta=True)
        else:
            # two normal forwards
            loss1 = self.zo_forward(model, inputs, with_delta=False)
            self.lowrank_zo_perturb_parameters(scaling_factor=-2)  # now at -eps
            loss2 = self.zo_forward(model, inputs, with_delta=False)
            self.lowrank_zo_perturb_parameters(scaling_factor=2)   # restore to +eps
        
        # print(f"step:{self.step}")
        # print(f"inputs:{inputs}")
        # print(f"loss1:{loss1}; loss2:{loss2}")

        # Second function evaluation
        # self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        # loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()   # ZO估计梯度

        #Debugging: sanity-check forward_delta vs true +/- (and gradient diff)
        if do_debug and getattr(args, "use_forward_delta_loss", True):
            eps = float(self.args.zo_eps)

            # true loss at theta - eps*delta (by moving parameters to -eps)
            self.lowrank_zo_perturb_parameters(scaling_factor=-2)  # now at -eps
            loss_minus_true = self.zo_forward(model, inputs, with_delta=False)
            self.lowrank_zo_perturb_parameters(scaling_factor=2)   # restore to +eps

            # compare losses
            self._debug_forward_delta_sanity_check(
                loss_plus_true=loss_plus_true,
                loss_minus_true=loss_minus_true,
                loss_plus_fd=loss1,
                loss_minus_fd=loss2,
            )

            # compare gradient estimate (what actually drives update)
            g_true = float(((loss_plus_true - loss_minus_true) / (2.0 * eps)).detach().cpu())
            g_fd = float(((loss1 - loss2) / (2.0 * eps)).detach().cpu())
            abs_g = abs(g_true - g_fd)
            rel_g = abs_g / (abs(g_true) + 1e-12)
            logger.info(
                "[debug_forward_delta][grad] step=%s | g_true=%.10e g_fd=%.10e | abs=%.3e rel=%.3e | eps=%.3e",
                getattr(self, "step", "?"),
                g_true, g_fd, abs_g, rel_g, eps
            )
        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        # self.lowrank_zo_perturb_parameters(scaling_factor=1)
        self.lowrank_zo_perturb_parameters(scaling_factor=-1)
        return loss1

    def lowrank_zo_update(self):  # SGD更新
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank, device=param.data.device, dtype=param.data.dtype)

                
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * (u@v.t()) + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * (u@v.t()))
            else:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        # self.lr_scheduler.step()
        if getattr(self, "optimizer", None) is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.step()
        
    def lowrank_zo_update_momentum(self):
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     
        
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank, device=param.data.device, dtype=param.data.dtype)
                
                if self.step % args.step_interval == 0:
                    if name in self.v_old:   
                        v_old = self.v_old[name]
                        n = v_old.shape[0]
                        self.exp_avg_m[name] = args.beta1 * self.exp_avg_m[name] @ v_old.t() @ v / n + (1 - args.beta1) * self.projected_grad * u 
                    else:
                        self.exp_avg_m[name] = self.projected_grad * u
                elif self.step % args.step_interval == args.step_interval - 1:
                    self.v_old[name] = v
                    self.exp_avg_m[name] = args.beta1 * self.exp_avg_m[name] + (1 - args.beta1) * self.projected_grad * u 
                else:
                    self.exp_avg_m[name] = args.beta1 * self.exp_avg_m[name] + (1 - args.beta1) * self.projected_grad * u 
                    
                
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    
                    param.data = param.data - self._get_learning_rate() * (self.exp_avg_m[name] @ v.t() + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.exp_avg_m[name] @ v.t())
            else:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if self.step == 0:
                    self.exp_avg_m[name] = self.projected_grad * z
                else:
                    self.exp_avg_m[name] = args.beta1 * self.exp_avg_m[name] + (1 - args.beta1) * self.projected_grad * z
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.exp_avg_m[name] + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.exp_avg_m[name])

        # self.lr_scheduler.step()
        if getattr(self, "optimizer", None) is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.step()
        
    # =====================================================================================================================================
        

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError
        # print("Sample %d zs" % (noise_sample_time))

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
        
        # batch = next(iter(train_dataloader))
        # self.debug_check_forward_delta_alignment_one_batch(model, batch)

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight

                # estimate c's (param or grad norm) on epoch 0
                if epoch == 0 and step == 0 and self.args.zo_variant is not None:
                    self.initialize_c(model, inputs)
                elif step == 0 and self.args.zo_variant is not None and self.args.recompute_norms:
                    self.initialize_c(model, inputs)
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                # -------------- this is the key in lozotrainer !!!! ---------------
                if self.args.lowrank_zo:
                    tr_loss_step = self.lowrank_zo_step(model, inputs)
                    # tr_loss += tr_loss_step
                    if self.args.lozo_optimizer == 'sgd':
                        self.lowrank_zo_update()
                    elif self.args.lozo_optimizer == 'sgdm':
                        self.lowrank_zo_update_momentum()
                    else:
                        raise ValueError(f"Unsupported optimizer: {self.args.lozo_optimizer}")
                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        logs["loss"] = tr_loss_step.item()
                        logs["learning_rate"] = self.args.learning_rate
                        logs["global_step"] = self.state.global_step
                        logs["max_steps"] = self.args.max_steps
                        logs["time"] = int(time.time() - start_time)
                        self.log(logs)
                        logger.info(str(logs))
                # ---------------------------------------------------------------
                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)

                    scheduler.step()

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        elif self.args.fp16:
                            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n,p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()

                        # scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            self.log(logs)
                            logger.info(str(logs))

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    
                    mode_str = "fd" if getattr(self.args, "use_forward_delta_loss", True) else "base"
                    logger.info("[eval_record] mode=%s gs=%d metrics=%s", mode_str, self.state.global_step, metrics)

                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
    
    
    



    # def _apply_cached_perturb(self, scaling_factor: float):
    #     """
    #     Apply perturbation using cached self.u/self.v/self.z WITHOUT resampling.
    #     This matches lowrank_zo_perturb_parameters but guarantees the same directions.
    #     """
    #     eps = self.args.zo_eps
    # 
    #     if not hasattr(self, "named_parameters_to_optim"):
    #         raise RuntimeError("named_parameters_to_optim not set. Call debug func after building it.")
    # 
    #     for name, param in self.named_parameters_to_optim:
    #         if param.data.ndim >= 2:
    #             if not hasattr(self, "u") or not hasattr(self, "v"):
    #                 raise RuntimeError("Missing self.u/self.v caches.")
    #             u = self.u.get(name, None)
    #             v = self.v.get(name, None)
    #             if u is None or v is None:
    #                 raise KeyError(f"Cached u/v not found for {name}. Name mismatch in provider?")
    #             # param += scaling_factor * (u @ v^T) * eps
    #             param.data.addmm_(u, v.t(), beta=1.0, alpha=scaling_factor * eps)
    #         else:
    #             if not hasattr(self, "z"):
    #                 raise RuntimeError("Missing self.z cache.")
    #             z = self.z.get(name, None)
    #             if z is None:
    #                 raise KeyError(f"Cached z not found for {name}. Name mismatch in provider?")
    #             # param += scaling_factor * z * eps
    #             param.data.add_(z, alpha=scaling_factor * eps)
    # 
    # 
    # def debug_check_forward_delta_alignment_one_batch(
    #     self,
    #     model,
    #     inputs,
    #     n_param_check: int = 8,
    # ):
    #     """
    #     One-batch alignment check:
    #       1) perturb +1 -> theta_plus
    #       2) compare:
    #            loss_plus_real (normal forward at theta_plus)
    #            vs loss_base_virtual (forward_delta first loss)
    #       3) go to theta_minus using cached directions (apply -2)
    #          compare:
    #            loss_minus_real (normal forward at theta_minus)
    #            vs loss_minus_virtual (forward_delta second loss)
    #       4) restore theta (apply +1 from theta_minus)
    #          check sampled parameter max error
    #     """
    #     model.eval()
    #     with torch.inference_mode():
    #         inputs = self._prepare_inputs(inputs)
    # 
    #         # Hard guards (otherwise your zo_forward protocol breaks)
    #         assert inputs.get("labels", None) is not None, "with_delta alignment check requires labels in inputs."
    #         assert not getattr(model, "return_full_softmax", False), "Disable return_full_softmax for ZO/delta training."
    # 
    #         # Build parameter list (same as your lowrank_zo_step)
    #         self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # 
    #         # Snapshot a few params to verify exact restore
    #         snap = []
    #         for n, p in self.named_parameters_to_optim:
    #             if len(snap) >= n_param_check:
    #                 break
    #             snap.append((n, p.detach().clone()))
    # 
    #         # Make sure caches exist (do not wipe existing training caches)
    #         if not hasattr(self, "u"): self.u = {}
    #         if not hasattr(self, "v"): self.v = {}
    #         if not hasattr(self, "z"): self.z = {}
    # 
    #         # 1) theta -> theta_plus (this also fills self.u/self.v/self.z used by providers)
    #         self.lowrank_zo_perturb_parameters(scaling_factor=1)
    # 
    #         # Compute loss at theta_plus via normal forward
    #         with self.compute_loss_context_manager():
    #             loss_plus_real = self.compute_loss(model, inputs).detach()
    # 
    #         # Compute (loss_base_virtual, loss_minus_virtual) via forward_delta at theta_plus
    #         with self.compute_loss_context_manager():
    #             out = model.forward_delta(**inputs)
    #         if not isinstance(out, (tuple, list)) or len(out) < 2:
    #             raise RuntimeError(f"forward_delta must return (loss_base, loss_pert, ...), got {type(out)} len={len(out)}")
    #         loss_base_virtual = out[0].detach()
    #         loss_minus_virtual = out[1].detach()
    # 
    #         # 2) theta_plus -> theta_minus using the SAME cached directions (apply -2)
    #         self._apply_cached_perturb(scaling_factor=-2)
    # 
    #         with self.compute_loss_context_manager():
    #             loss_minus_real = self.compute_loss(model, inputs).detach()
    # 
    #         # 3) restore theta_minus -> theta (apply +1)
    #         self._apply_cached_perturb(scaling_factor=+1)
    # 
    #         # Param restore check
    #         name2p = dict(model.named_parameters())
    #         max_param_err = 0.0
    #         for n, ref in snap:
    #             cur = name2p[n].detach()
    #             err = (cur - ref).abs().max().item()
    #             max_param_err = max(max_param_err, err)
    # 
    #         # Print report
    #         def _fmt(x):
    #             return float(x.item()) if torch.is_tensor(x) else float(x)
    # 
    #         print("====== LOZO forward_delta alignment check (one batch) ======")
    #         print(f"loss_plus_real     (normal @ theta+): { _fmt(loss_plus_real) }")
    #         print(f"loss_base_virtual  (delta  @ theta+): { _fmt(loss_base_virtual) }")
    #         print(f"abs diff (plus): { abs(_fmt(loss_plus_real) - _fmt(loss_base_virtual)) }")
    #         print("------------------------------------------------------------")
    #         print(f"loss_minus_real    (normal @ theta-): { _fmt(loss_minus_real) }")
    #         print(f"loss_minus_virtual (delta  @ theta-): { _fmt(loss_minus_virtual) }")
    #         print(f"abs diff (minus): { abs(_fmt(loss_minus_real) - _fmt(loss_minus_virtual)) }")
    #         print("------------------------------------------------------------")
    #         print(f"max_param_restore_err (sampled {len(snap)} params): {max_param_err}")
    #         print("============================================================")
    # 
    #         return {
    #             "loss_plus_real": loss_plus_real,
    #             "loss_base_virtual": loss_base_virtual,
    #             "loss_minus_real": loss_minus_real,
    #             "loss_minus_virtual": loss_minus_virtual,
    #             "max_param_restore_err": max_param_err,
    #         }
    # 