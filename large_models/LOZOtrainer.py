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

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers


_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class LowRankTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = {} 
        self.z = {}
        self.debug_forward_delta = False
        # debug controls
        self.debug_provider_hits = True
        self.debug_provider_max_print = 40
        self.debug_provider_seen = set()
        self.debug_roundtrip_param_names = [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.fc1.weight",
            "model.decoder.embed_tokens.weight",
        ]

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                        
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4 # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1: # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            # pass
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        
        
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                if args.trainer == "LOZO":
                    tr_loss_step = self.lowrank_zo_step(model, inputs)
                else:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "LOZO":
                        self.lowrank_zo_update()
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    # =========================================== LOZO Functions ==============================================================
    def _debug_resolve_param_name(self, param_dict, target_name):
        if target_name in param_dict:
            return target_name
        for name in param_dict.keys():
            if name.endswith(target_name):
                return name
        return None

    def _debug_capture_param_snapshot(self, model):
        param_dict = dict(model.named_parameters())
        snapshot = {}
        for target_name in self.debug_roundtrip_param_names:
            resolved_name = self._debug_resolve_param_name(param_dict, target_name)
            if resolved_name is None:
                print(f"[DEBUG][roundtrip] missing param in model.named_parameters(): {target_name}", flush=True)
                continue
            snapshot[resolved_name] = param_dict[resolved_name].detach().clone()
        return snapshot
    
    def _debug_report_roundtrip(self, model, snapshot, tag=""):
        if not snapshot:
            print(f"[DEBUG][roundtrip] no snapshot captured {tag}", flush=True)
            return
    
        param_dict = dict(model.named_parameters())
        max_diff_all = 0.0
        for name, ref in snapshot.items():
            if name not in param_dict:
                print(f"[DEBUG][roundtrip] param disappeared: {name}", flush=True)
                continue
            cur = param_dict[name].detach()
            diff = (cur - ref).abs().max().item()
            max_diff_all = max(max_diff_all, diff)
            print(f"[DEBUG][roundtrip] {name} max_abs_diff={diff:.10e}", flush=True)
    
        print(f"[DEBUG][roundtrip] overall max_abs_diff={max_diff_all:.10e} {tag}", flush=True)
    
    def _debug_log_provider(self, kind, param_name, *, z_hit=None, u_hit=None, v_hit=None, shape=None, inference_count=None):
        if not self.debug_provider_hits:
            return
    
        key = (kind, param_name)
        if key in self.debug_provider_seen:
            return
        if len(self.debug_provider_seen) >= self.debug_provider_max_print:
            return
    
        self.debug_provider_seen.add(key)
        parts = [f"[DEBUG][{kind}]", f"name={param_name}"]
        if inference_count is not None:
            parts.append(f"inference_count={inference_count}")
        if shape is not None:
            parts.append(f"shape={tuple(shape)}")
        if z_hit is not None:
            parts.append(f"z_hit={z_hit}")
        if u_hit is not None:
            parts.append(f"u_hit={u_hit}")
        if v_hit is not None:
            parts.append(f"v_hit={v_hit}")
        print(" ".join(parts), flush=True)
        
    def _debug_capture_old_forward_outputs(self, model, inputs, module_names):
        captured = {}
        hooks = []
    
        name_to_module = dict(model.named_modules())
    
        for name in module_names:
            if name not in name_to_module:
                print(f"[DEBUG][hook] missing module: {name}", flush=True)
                continue
    
            module = name_to_module[name]
    
            def make_hook(module_name):
                def hook(mod, inp, out):
                    out0 = out[0] if isinstance(out, tuple) else out
                    if torch.is_tensor(out0):
                        captured[module_name] = out0.detach().float().clone()
                return hook
    
            hooks.append(module.register_forward_hook(make_hook(name)))
    
        with torch.no_grad():
            loss = self.zo_forward(model, inputs, with_delta=False)   
    
        for h in hooks:
            h.remove()
    
        return captured, loss    
    
    def _debug_capture_forward_delta_outputs(self, model, inputs, module_names):
        captured = {}
        name_to_module = dict(model.named_modules())
        patched = []  # (module, original_forward_delta)
    
        def make_wrapper(module_name, orig_fn):
            def wrapped(*args, **kwargs):
                out = orig_fn(*args, **kwargs)
                # diff* 模块 forward_delta 期望返回 (base, diff)
                if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                    base, diff = out[0], out[1]
                    captured[module_name] = {
                        "base": base.detach().float().clone(),
                        "pert": (base + diff).detach().float().clone(),
                    }
                return out
            return wrapped
    
        # patch
        for name in module_names:
            if name not in name_to_module:
                print(f"[DEBUG][fd_hook] missing module: {name}", flush=True)
                continue
            m = name_to_module[name]
            if not hasattr(m, "forward_delta"):
                print(f"[DEBUG][fd_hook] module has no forward_delta: {name}", flush=True)
                continue
            orig = m.forward_delta
            m.forward_delta = make_wrapper(name, orig)
            patched.append((m, orig))
    
        try:
            with torch.no_grad():
                loss_base, loss_pert = self.zo_forward(model, inputs, with_delta=True)  
        finally:
            # restore
            for m, orig in patched:
                m.forward_delta = orig
    
        return captured, loss_base, loss_pert 
        
    # ------ DEBUGGING UPUPUP -----    

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
                    v = torch.randn(param.data.size(1), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v
                else:
                    v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank_r, device=param.data.device, dtype=param.data.dtype)
                # new2: cached u (low-rank matrix memory cost)
                self.u[name] = u
                # print(f"step:{step}: name:{name}, real_u[0]:{u[0]}, real_v[0]:{v[0]}")
                param.data = param.data + scaling_factor * (u@v.t()) * self.args.zo_eps
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
            # return z, 0
        
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
                v = torch.randn(in_f, self.args.rank_r, device=device, dtype=dtype)
            # use cache_u directly
            u = None
            if hasattr(self, "u"):
                u = self.u.get(param_name, None)
            if u is None: 
                u = torch.randn(out_f, self.args.rank_r, device=device, dtype=dtype)
            
            # The difference between the even iteration and the odd cached weights is −2×zo_eps×(UVT).
            scale = -2 * self.args.zo_eps
            return u, v, scale
            # return u, v, 0
             
        return provider
    
    
    
    def zo_forward(self, model, inputs, with_delta = False):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            inputs["return_dict"] = False
            # 临时debug用的，先关掉use_cache
            # inputs["use_cache"] = False
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
                    # print(f"[CHECK] base={loss_base.item():.6f} pert={loss_perturbed.item():.6f} ")

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


    def lowrank_zo_step(self, model, inputs):
        """
        Estimate gradient by Lowrank-zo. Return the loss from f(theta + uv^t)
        """
        args = self.args
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.v = {}

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling 
        self.zo_random_seed = np.random.randint(1000000000)
        
        
        # debug!!!
        if self.debug_forward_delta and self.step < 10:
            roundtrip_snapshot = self._debug_capture_param_snapshot(model)
            targets = [
                "model.decoder.embed_tokens",
                "model.decoder.embed_positions",
                "model.decoder.layers.0.self_attn.q_proj",
                "model.decoder.layers.0.self_attn.k_proj",
                "model.decoder.layers.0.self_attn.v_proj",
                "model.decoder.layers.0.self_attn.out_proj",
                "model.decoder.layers.0.fc1",
                "model.decoder.layers.0.fc2",
                "model.decoder.layers.0.self_attn_layer_norm",
                "model.decoder.layers.0.final_layer_norm",
            ]
        
            # old +eps
            self.lowrank_zo_perturb_parameters(scaling_factor=1)
            
            # for k in [
            #     "model.decoder.layers.0.self_attn_layer_norm.weight",
            #     "model.decoder.layers.0.self_attn_layer_norm.bias",
            #     "model.decoder.layers.0.final_layer_norm.weight",
            #     "model.decoder.layers.0.final_layer_norm.bias",
            # ]:
            #     print("[DEBUG][z-check]", k, "in self.z =", k in self.z)
                
            old_plus_out, loss_plus = self._debug_capture_old_forward_outputs(model, inputs, targets)
            # now run forward_delta at +eps
            new_out, loss_base, loss_pert = self._debug_capture_forward_delta_outputs(model, inputs, targets)
            
            # old -eps
            self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        
            ln_attn = model.model.decoder.layers[0].self_attn_layer_norm
            ln_ffn  = model.model.decoder.layers[0].final_layer_norm
            attn_ln_w_minus_old = ln_attn.weight.detach().clone()
            attn_ln_b_minus_old = ln_attn.bias.detach().clone()
            ffn_ln_w_minus_old  = ln_ffn.weight.detach().clone()
            ffn_ln_b_minus_old  = ln_ffn.bias.detach().clone()
            
            old_minus_out, loss_minus = self._debug_capture_old_forward_outputs(model, inputs, targets)
            # back to +eps then new forward_delta
            self.lowrank_zo_perturb_parameters(scaling_factor=2)
            
            scale = -2 * self.args.zo_eps
            attn_ln_w_minus_new = ln_attn.weight.detach() + self.z["model.decoder.layers.0.self_attn_layer_norm.weight"] * scale
            attn_ln_b_minus_new = ln_attn.bias.detach()   + self.z["model.decoder.layers.0.self_attn_layer_norm.bias"] * scale
            ffn_ln_w_minus_new  = ln_ffn.weight.detach()  + self.z["model.decoder.layers.0.final_layer_norm.weight"] * scale
            ffn_ln_b_minus_new  = ln_ffn.bias.detach()    + self.z["model.decoder.layers.0.final_layer_norm.bias"] * scale
        
            print("[DEBUG][ln-param] attn_ln.weight old_minus vs rebuild max_abs =",
                  float((attn_ln_w_minus_old - attn_ln_w_minus_new).abs().max()))
            print("[DEBUG][ln-param] attn_ln.bias   old_minus vs rebuild max_abs =",
                  float((attn_ln_b_minus_old - attn_ln_b_minus_new).abs().max()))
            print("[DEBUG][ln-param] ffn_ln.weight  old_minus vs rebuild max_abs =",
                  float((ffn_ln_w_minus_old - ffn_ln_w_minus_new).abs().max()))
            print("[DEBUG][ln-param] ffn_ln.bias    old_minus vs rebuild max_abs =",
                  float((ffn_ln_b_minus_old - ffn_ln_b_minus_new).abs().max()))
        
            
            
            
            print("[DEBUG] loss_plus(old)  =", float(loss_plus))
            print("[DEBUG] loss_base(new)  =", float(loss_base))
            print("[DEBUG] abs diff (+)    =", float((loss_plus - loss_base).abs()))
            print("[DEBUG] loss_minus(old) =", float(loss_minus))
            print("[DEBUG] loss_pert(new)  =", float(loss_pert))
            print("[DEBUG] abs diff (-)    =", float((loss_minus - loss_pert).abs()))
        
            g_old = ((loss_plus - loss_minus) / (2 * self.args.zo_eps)).item()
            g_new = ((loss_base - loss_pert) / (2 * self.args.zo_eps)).item()
            print("[DEBUG] proj_grad old =", g_old)
            print("[DEBUG] proj_grad new =", g_new)
            print("[DEBUG] grad abs diff =", abs(g_old - g_new))
        
            # layer-wise check（这次对拍是严谨的：old_plus 对 new_base，old_minus 对 new_pert）
            for name in targets:
                if name in old_plus_out and name in old_minus_out and name in new_out:
                    plus_err = (old_plus_out[name] - new_out[name]["base"]).abs().max().item()
                    minus_err = (old_minus_out[name] - new_out[name]["pert"]).abs().max().item()
                    print(f"[LAYER-CHECK] {name} plus_err={plus_err:.6e} minus_err={minus_err:.6e}", flush=True)
                else:
                    print(f"[LAYER-CHECK] missing {name} (old_plus={name in old_plus_out}, old_minus={name in old_minus_out}, new={name in new_out})", flush=True)
        
            # restore to original params
            self.lowrank_zo_perturb_parameters(scaling_factor=-1)
            self._debug_report_roundtrip(model, roundtrip_snapshot, tag=f"after +1/-2/+2/-1 at step={self.step}")


        # First function evaluation
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        loss1, loss2 = self.zo_forward(model, inputs, with_delta = True)
        

        # Second function evaluation
        # self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        # loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.lowrank_zo_perturb_parameters(scaling_factor=-1)
        return loss1


    def lowrank_zo_update(self):
        args = self.args

        # Reset the random seed for sampling 
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank_r, device=param.data.device, dtype=param.data.dtype)

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * (u@v.t()) + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * (u@v.t()))
            else:
                # Resample z for bias
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        self.lr_scheduler.step()
        
    def random_gaussian_matrix(self, m, n, device, dtype, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        random_matrix = torch.randn(m, n, device=device, dtype=dtype)
        return random_matrix

    ############## Misc overload functions ##############


    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
