import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from typing import Optional, List, Dict
import numpy as np
import os
import random
from typing import Optional
from typing import Union, Tuple

try:
    from mx.specs import MxSpecs,finalize_mx_specs
    from mx.linear import linear as mx_linear  
except Exception as e:
    raise ImportError(
        "microxcaling 未安装或不可用。请先 `pip install microxcaling` 并确保有 CUDA 环境。"
    ) from e
            
   
class diffLinear(nn.Linear):
    def __init__(
        self,
        layer_name,
        in_features: int, out_features: int, bias: bool=True, device=None, dtype=None,
        uv_provider=None, z_provider=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.inference_count = 0
        self.layer_name = layer_name
        self.uv_provider = uv_provider
        self.z_provider = z_provider
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)
    
    def forward_delta(self, input: torch.Tensor, diff_input: torch.Tensor):
        self.inference_count += 1
        output = F.linear(input, self.weight, self.bias)
        diff_output = F.linear(diff_input, self.weight, None)
        if self.uv_provider is not None:
            u, v, scale = self.uv_provider(
                self.layer_name + ".weight",
                self.weight.shape,
                self.weight.device,
                self.weight.dtype,
                self.inference_count
            )
            # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_u[0]:{u[0]}, fake_v[0]:{v[0]}")
            out_dim = diff_output.size(-1)
            r = v.size(-1)

            diff_out_2d = diff_output.reshape(-1, out_dim)    
            
            tmp = diff_input.reshape(-1, v.size(0)).matmul(v)  
            diff_out_2d.addmm_(tmp, u.t(), beta=1.0, alpha=scale)  
    
            tmp = input.reshape(-1, v.size(0)).matmul(v)       
            diff_out_2d.addmm_(tmp, u.t(), beta=1.0, alpha=scale)
            diff_output = diff_out_2d.view_as(diff_output)

        if self.z_provider is not None and (self.bias is not None):
            z, bias_scale = self.z_provider(
                self.layer_name + ".bias",
                self.bias.shape,
                self.bias.device,
                self.bias.dtype,
                self.inference_count
            )
            # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_z[0]:{z[0]}")
            diff_bias = z * bias_scale  # [out]
            view_shape = [1] * (diff_output.dim() - 1) + [-1]
            diff_output.add_(diff_bias.view(*view_shape))
            
        return output, diff_output
    
    
    def __repr__(self):
        return (f"diffLinear({self.in_features}, {self.out_features}, bias={self.bias is not None} ")

        
# Use mx to quantize diffLinear
class QdiffLinear(nn.Linear):
    def __init__(
        self, 
        enable_x, enable_diffx, enable_w, enable_diffw,
        layer_name: str, 
        in_features: int, out_features: int, bias: bool=True, device=None, dtype=None,  
        mx_w_elem_format=None, mx_a_elem_format=None, mx_diffw_elem_format=None, mx_diffa_elem_format=None,
        uv_provider=None, z_provider=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.inference_count = 0
        self.layer_name = layer_name
        self.uv_provider = uv_provider
        self.z_provider = z_provider
        
        self.enable_x = enable_x
        self.enable_diffx = enable_diffx
        self.enable_w = enable_w
        self.enable_diffw = enable_diffw

        # for mx quan
        self.mx_w_elem_format = mx_w_elem_format if enable_w else None
        self.mx_a_elem_format = mx_a_elem_format if enable_x else None
        self.mx_diffw_elem_format = mx_diffw_elem_format if enable_diffw else None
        self.mx_diffa_elem_format = mx_diffa_elem_format if enable_diffx else None
        # specs0: output_odd
        mx_specs0 = {
            'w_elem_format': self.mx_w_elem_format,
            'a_elem_format': self.mx_a_elem_format,
            'scale_bits': 8,
            'block_size': 16,
            'bfloat': 16,
            'custom_cuda': True,
            'quantize_backprop': False,
        }
        self.mx_specs0 = finalize_mx_specs(mx_specs0)
        # specs1: linear(diff_x, weight_even)
        mx_specs1 = {
            'w_elem_format': self.mx_w_elem_format,
            'a_elem_format': self.mx_diffa_elem_format,
            'scale_bits': 8,
            'block_size': 16,
            'bfloat': 16,
            'custom_cuda': True,
            'quantize_backprop': False,
        }
        self.mx_specs1 = finalize_mx_specs(mx_specs1)
        # specs2: linear(x_odd, z)
        mx_specs2 = {
            'w_elem_format': self.mx_diffw_elem_format,
            'a_elem_format': self.mx_a_elem_format,
            'scale_bits': 8,
            'block_size': 16,
            'bfloat': 16,
            'custom_cuda': True,
            'quantize_backprop': False,
        }
        self.mx_specs2 = finalize_mx_specs(mx_specs2)
        # spec3
        mx_specs3 = {
            'w_elem_format': self.mx_diffw_elem_format,
            'a_elem_format': self.mx_diffa_elem_format,
            'scale_bits': 8,
            'block_size': 16,
            'bfloat': 16,
            'custom_cuda': True,
            'quantize_backprop': False,
        }
        self.mx_specs3 = finalize_mx_specs(mx_specs3)
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)
    
    def forward_delta(self, input: torch.Tensor, diff_input: torch.Tensor) -> torch.Tensor:
        self.inference_count += 1
        # ground state output
        output = mx_linear(input, self.weight, self.bias, mx_specs=self.mx_specs0)
        
        # calculate diff_weight_output (with Quan)
        diff_weight_output = 0
        if self.uv_provider is not None:
            u, v, scale = self.uv_provider(self.layer_name + ".weight",
                               self.weight.shape,
                               self.weight.device,
                               self.weight.dtype,
                               self.inference_count)
            # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_u[0]:{u[0]}, fake_v[0]:{v[0]}")
            tmp = mx_linear(input, v.t(), None, mx_specs=self.mx_specs2)
            diff_weight_output = mx_linear(tmp, u, None, mx_specs=self.mx_specs2)
            diff_weight_output = diff_weight_output * scale
        
        diff_bias = 0
        if self.z_provider is not None and (self.bias is not None):
            z, bias_scale = self.z_provider(
                self.layer_name + ".bias",
                self.bias.shape,
                self.bias.device,
                self.bias.dtype,
                self.inference_count,
            )
            # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_z[0]:{z[0]}")
            diff_bias = bias_scale * z
            view_shape = [1] * output.dim()
            view_shape[-1] = diff_bias.size(0)
            diff_bias = diff_bias.view(*view_shape)
            diff_bias = diff_bias.expand_as(output)
            
        # calculate diff_act_output (with Quan)
        diff_act_output = mx_linear(diff_input, self.weight, None, mx_specs=self.mx_specs1)
        tmp = mx_linear(diff_input, v.t(), None, mx_specs=self.mx_specs3)   # [*, r]
        cross = mx_linear(tmp, u, None, mx_specs=self.mx_specs3) * scale    # [*, out]
        diff_act_output = diff_act_output + cross
        
        diff_output = diff_act_output + diff_weight_output + diff_bias
        # diff_output = diff_act_output + diff_weight_output + diff_bias
        
        return output, diff_output
    
    
    def __repr__(self):
        return (f"QdiffLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, "
                f"mx_w_elem_format={self.mx_w_elem_format}, mx_a_elem_format={self.mx_a_elem_format}, mx_diffw_elem_format={self.mx_diffw_elem_format}, mx_diffa_elem_format={self.mx_diffa_elem_format}, "
                f"enable_x={self.enable_x}, enable_diffx={self.enable_diffx}, enable_w={self.enable_w}, enable_diffw={self.enable_diffw})")


class diffEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False,
                 layer_name: str = "", uv_provider = None,
                 device=None, dtype=None,
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            **factory_kwargs,
        )
        self.layer_name = layer_name           
        self.uv_provider = uv_provider         
        self.inference_count = 0    
        
    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        return super().forward(input)           
    
    def forward_delta(self, input):  
        self.inference_count += 1
        # ground state output
        output = super().forward(input)  # [B, T, C]
        
        # calculate uv-diff part
        if self.uv_provider is None:
            diff_output = torch.zeros_like(output)
            return output, diff_output
        weight = self.weight                     # [num_embeddings, embedding_dim]
        num_embeddings, dim = weight.shape
        if self.layer_name:
            param_name = self.layer_name + ".weight"
        else:
            param_name = "weight"  # fallback，
        u, v, scale = self.uv_provider(
            param_name,
            weight.shape,
            weight.device,
            weight.dtype,
            self.inference_count,
        )
        # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_u[0]:{u[0]}, fake_v[0]:{v[0]}")
        u_rows = u[input]                      # [B, T, r]
        diff_output = torch.matmul(u_rows, v.t()) * scale
        
        return output, diff_output
        
    def __repr__(self) -> str:
        return (
            f"diffEmbedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}, "
            f"layer_name='{self.layer_name}')"
        )
        

class diffLayerNorm(nn.LayerNorm):
    def __init__(self, 
                 normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5, elementwise_affine: bool = True,
                 layer_name: str = "", z_provider=None, 
                 device=None, dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **factory_kwargs,)
        self.layer_name = layer_name
        self.z_provider = z_provider
        self.inference_count = 0
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)   
    
    def forward_delta(self, input: torch.Tensor, diff_input: torch.Tensor):
        self.inference_count += 1
        # ground state output
        output = super().forward(input)
        
        input_1 = input + diff_input
        if self.elementwise_affine:
            weight = self.weight
            bias = self.bias
        else:
            weight = None
            bias = None
        weight_1 = weight
        bias_1 = bias
        if self.z_provider is not None and self.elementwise_affine:
            if weight is not None:
                w_name = self.layer_name + ".weight" if self.layer_name else "weight"
                z_w, scale_w = self.z_provider(
                    w_name,
                    weight.shape,
                    weight.device,
                    weight.dtype,
                    self.inference_count,
                )
                weight_1 = weight + z_w * scale_w  # γ_new = γ + Δγ
                # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_z_w[0]:{z_w[0]}")
            if bias is not None:
                b_name = self.layer_name + ".bias" if self.layer_name else "bias"
                z_b, scale_b = self.z_provider(
                    b_name,
                    bias.shape,
                    bias.device,
                    bias.dtype,
                    self.inference_count,
                )
                bias_1 = bias + z_b * scale_b      # β_new = β + Δβ
                # print(f"step:{self.inference_count}: name:{self.layer_name}, fake_z_b[0]:{z_b[0]}")
        output_1 = F.layer_norm(input_1, self.normalized_shape, weight_1, bias_1, self.eps)
        diff_output = output_1 - output
        
        return output, diff_output
            

def get_all_QdiffLinear(model: nn.Module):
    QdiffLinears = []    
    def find_linears(module):
        for child in module.children():
            
            if isinstance(child, QdiffLinear):
                QdiffLinears.append(child)
            find_linears(child)
            
    find_linears(model)
    return QdiffLinears

def get_all_diffEmbedding(model: nn.Module):
    diff_embs = []
    def find_Embedding(module: nn.Module):
        for child in module.children():
            if isinstance(child, diffEmbedding):
                diff_embs.append(child)
            find_Embedding(child)

    find_Embedding(model)
    return diff_embs

def get_all_diffLayerNorm(model: nn.Module):
    diff_norm = []
    def find_LayerNorm(module: nn.Module):
        for child in module.children():
            if isinstance(child, diffLayerNorm):
                diff_norm.append(child)
            find_LayerNorm(child)

    find_LayerNorm(model)
    return diff_norm

# ---------------------------------------------------------------------


# This is the method about how to Quantize diff in current model 

def QuantizeRobertaForLOZO(
    enable_w, enable_x, enable_diffw, enable_diffx,
    model: nn.Module,
    mx_w_elem_format=None, mx_a_elem_format=None, mx_diffw_elem_format=None, mx_diffa_elem_format=None,
    uv_provider=None, z_provider=None,
):
   
   # 将roberta模型中的可训练权重转换成对应的diff模式，注意encoder中的linear转换为mx量化格式的Qdifflinear
   
    def replace_roberta_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # print(f"Checking layer: {full_name}")

            in_encoder = full_name.startswith("roberta.encoder")
            in_embeddings = full_name.startswith("roberta.embeddings")
            

            if in_embeddings and isinstance(child, nn.Embedding) and not isinstance(child, diffEmbedding):
                new_emb = diffEmbedding(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx,
                    max_norm=child.max_norm,
                    norm_type=child.norm_type,
                    scale_grad_by_freq=child.scale_grad_by_freq,
                    sparse=child.sparse,
                    layer_name=full_name,
                    uv_provider=uv_provider,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                new_emb.weight.data = child.weight.data.clone()
                setattr(module, name, new_emb)
                # print(f"Replace {full_name} with diffEmbedding")
                continue
            
            # 别忘记这里应该是QdiffLinear
            # if in_encoder and isinstance(child, nn.Linear) and not isinstance(child, QdiffLinear):
            #     new_qlinear = QdiffLinear(
            #         enable_x=enable_x,
            #         enable_diffx=enable_diffx,
            #         enable_w=enable_w,
            #         enable_diffw=enable_diffw,
            #         layer_name=full_name,
            #         in_features=child.in_features,
            #         out_features=child.out_features,
            #         bias=(child.bias is not None),
            #         device=child.weight.device,
            #         dtype=child.weight.dtype,
            #         mx_w_elem_format=mx_w_elem_format,
            #         mx_a_elem_format=mx_a_elem_format,
            #         mx_diffw_elem_format=mx_diffw_elem_format,
            #         mx_diffa_elem_format=mx_diffa_elem_format,
            #         uv_provider=uv_provider,
            #         z_provider=z_provider,
            #     )
            #     new_qlinear.weight.data = child.weight.data.clone()
            #     if child.bias is not None:
            #         new_qlinear.bias.data = child.bias.data.clone()
            #     setattr(module, name, new_qlinear)
            #     print(f"Replace {full_name} with QdiffLinear")
            #     continue
            if in_encoder and isinstance(child, nn.Linear) and not isinstance(child, diffLinear):
                new_dlinear = diffLinear(
                    layer_name=full_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                )
                new_dlinear.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_dlinear.bias.data = child.bias.data.clone()
                setattr(module, name, new_dlinear)
                print(f"Replace {full_name} with diffLinear")
                continue
            
            if (not in_encoder) and isinstance(child, nn.Linear) and not isinstance(child, diffLinear):
                new_dlinear = diffLinear(
                    layer_name=full_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                )
                new_dlinear.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_dlinear.bias.data = child.bias.data.clone()
                setattr(module, name, new_dlinear)
                print(f"Replace {full_name} with diffLinear")
                continue

            if isinstance(child, nn.LayerNorm) and not isinstance(child, diffLayerNorm):
                new_ln = diffLayerNorm(
                    normalized_shape=child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                    layer_name=full_name,
                    z_provider=z_provider,
                    device=child.weight.device if child.weight is not None else None,
                    dtype=child.weight.dtype if child.weight is not None else None,
                )
                if child.weight is not None:
                    new_ln.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_ln.bias.data = child.bias.data.clone()
                setattr(module, name, new_ln)
                print(f"Replace {full_name} with diffLayerNorm")
                continue

            replace_roberta_module(child, full_name)

    if getattr(model, "config", None) is not None and model.config.model_type == "roberta":
        replace_roberta_module(model)
        print(model)
    else:
        print("Model is not a RoBERTa model, skip QuantizeRobertaForLOZO.")


def QuantizeOPTForLOZO(
    enable_w, enable_x, enable_diffw, enable_diffx,
    model: nn.Module,
    mx_w_elem_format=None, mx_a_elem_format=None, mx_diffw_elem_format=None, mx_diffa_elem_format=None,
    uv_provider=None, z_provider=None,
):

    def replace_opt_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # print(f"Checking layer: {full_name}")

            in_decoder_layers = full_name.startswith("model.decoder.layers")
            is_lm_head = (full_name == "lm_head")
            
            if child.__class__.__name__ == "OPTLearnedPositionalEmbedding":
                child.layer_name = full_name
                child.uv_provider = uv_provider
                print(f"Patch {full_name} with layer_name/uv_provider for forward_delta")
                continue

            if isinstance(child, nn.Embedding) and not isinstance(child, diffEmbedding):
                new_emb = diffEmbedding(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx,
                    max_norm=child.max_norm,
                    norm_type=child.norm_type,
                    scale_grad_by_freq=child.scale_grad_by_freq,
                    sparse=child.sparse,
                    layer_name=full_name,
                    uv_provider=uv_provider,
                    device="meta",                 
                    dtype=child.weight.dtype,
                )
                new_emb.weight = child.weight
                setattr(module, name, new_emb)
                # print(f"Replace {full_name} with diffEmbedding (share weight)")
                continue

            # if in_decoder_layers and isinstance(child, nn.Linear) and not isinstance(child, QdiffLinear):
            #     new_qlinear = QdiffLinear(
            #         enable_x=enable_x,
            #         enable_diffx=enable_diffx,
            #         enable_w=enable_w,
            #         enable_diffw=enable_diffw,
            #         layer_name=full_name,
            #         in_features=child.in_features,
            #         out_features=child.out_features,
            #         bias=(child.bias is not None),
            #         device="meta",                
            #         dtype=child.weight.dtype,
            #         mx_w_elem_format=mx_w_elem_format,
            #         mx_a_elem_format=mx_a_elem_format,
            #         mx_diffw_elem_format=mx_diffw_elem_format,
            #         mx_diffa_elem_format=mx_diffa_elem_format,
            #         uv_provider=uv_provider,
            #         z_provider=z_provider,
            #     )
            #     new_qlinear.weight = child.weight
            #     if child.bias is not None:
            #         new_qlinear.bias = child.bias
            #     setattr(module, name, new_qlinear)
            #     print(f"Replace {full_name} with QdiffLinear (share params)")
            #     continue
            
            if in_decoder_layers and isinstance(child, nn.Linear) and not isinstance(child, diffLinear):
                new_head = diffLinear(
                    layer_name=full_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    device="meta",              
                    dtype=child.weight.dtype,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                )
                new_head.weight = child.weight
                if child.bias is not None:
                    new_head.bias = child.bias
                setattr(module, name, new_head)
                # print(f"Replace {full_name} with diffLinear (share params)")
                continue

            if is_lm_head and isinstance(child, nn.Linear) and not isinstance(child, diffLinear):
                new_head = diffLinear(
                    layer_name=full_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    device="meta",              
                    dtype=child.weight.dtype,
                    uv_provider=uv_provider,
                    z_provider=z_provider,
                )
                new_head.weight = child.weight
                if child.bias is not None:
                    new_head.bias = child.bias
                setattr(module, name, new_head)
                # print(f"Replace {full_name} with diffLinear (share params)")
                continue

            if isinstance(child, nn.LayerNorm) and not isinstance(child, diffLayerNorm):
                new_ln = diffLayerNorm(
                    normalized_shape=child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                    layer_name=full_name,
                    z_provider=z_provider,
                    device="meta",              
                    dtype=child.weight.dtype if child.weight is not None else None,
                )
                if child.weight is not None:
                    new_ln.weight = child.weight
                if child.bias is not None:
                    new_ln.bias = child.bias
                setattr(module, name, new_ln)
                # print(f"Replace {full_name} with diffLayerNorm (share params)")
                continue

            # 递归
            replace_opt_module(child, full_name)

    if getattr(model, "config", None) is not None and model.config.model_type == "opt":
        replace_opt_module(model)
        
        if getattr(model.config, "tie_word_embeddings", False):
            try:
                model.tie_weights()
                # tied 情况下，让 lm_head 的 delta key 复用 embed_tokens
                if hasattr(model, "lm_head") and isinstance(model.lm_head, diffLinear):
                    model.lm_head.layer_name = "model.decoder.embed_tokens"
                    model.lm_head.uv_provider = uv_provider
                    model.lm_head.z_provider = z_provider
                    print("[PATCH] tied lm_head delta key -> model.decoder.embed_tokens")
            except Exception as e:
                print(f"[WARN] model.tie_weights() failed: {e}")

        print(model)
    else:
        print("Model is not an OPT model, skip QuantizeOPTForLOZO.")