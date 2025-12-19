import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from typing import Optional, List, Dict
import numpy as np
import os
import random
from typing import Optional

try:
    from mx.specs import MxSpecs,finalize_mx_specs
    from mx.linear import linear as mx_linear  
except Exception as e:
    raise ImportError(
        "microxcaling 未安装或不可用。请先 `pip install microxcaling` 并确保有 CUDA 环境。"
    ) from e

# only for int quantize
@torch.no_grad()
def quantize_weight_per_outchannel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    q_max = 2 ** (n_bits - 1) - 1
    w_fp32 = w.detach().to(torch.float32)           
    scales = w_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / q_max
    w_q = (w_fp32 / scales).round() * scales
    return w_q.to(w.dtype)                          

@torch.no_grad()
def quantize_activation_per_token_absmax(x, n_bits=8):
    # x: [B, T, C]
    q_max = 2 ** (n_bits - 1) - 1
    x_fp32 = x.detach().to(torch.float32)
    scales = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / q_max  
    x_q = (x_fp32 / scales).round() * scales
    return x_q.to(x.dtype)

@torch.no_grad()
def quantize_activation_per_channel_absmax(x, n_bits=8):
    # x: [B, T, C] / [B, C]
    q_max = 2 ** (n_bits - 1) - 1
    x_fp32 = x.detach().to(torch.float32)
    if x_fp32.ndim == 1:
        scales = x_fp32.abs().amax().clamp_min(1e-5) / q_max   #
    else:
        reduce_dims = tuple(range(x_fp32.ndim - 1))            
        scales = x_fp32.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-5) / q_max  # [1,1,...,C]
    x_q = (x_fp32 / scales).round() * scales
    return x_q.to(x.dtype)                


class QdiffLinear(nn.Linear):
    def __init__(
        self,
        mx_quan, 
        act_quant, weight_quant, act_b, weight_b, 
        enable_x, enable_diffx, enable_w, enable_diffw,
        max_steps: int, layer_name: str, 
        in_features: int, out_features: int, bias: bool=True, device=None, dtype=None,  
        mx_w_elem_format=None, mx_a_elem_format=None, mx_diffw_elem_format=None, mx_diffa_elem_format=None,
        use_uv_diffw: bool=False, uv_provider=None,
        quantize_output=False,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.inference_count = 0
        self.layer_name = layer_name
        self.max_steps = max_steps
        
        self.mx_quan = mx_quan
        self.use_uv_diffw = use_uv_diffw
        self.uv_provider = uv_provider
        
        self.register_buffer('cached_input_odd', None, persistent=False)
        self.register_buffer('cached_output_odd', None, persistent=False)
        self.register_buffer('cached_weight_odd', None, persistent=False)   # 可以不用cache了 只需cache uv 的 low-rank matrix
        
        # for low-bit-int quan 
        self.act_b = act_b
        self.enable_x = enable_x
        self.enable_diffx = enable_diffx
        self.weight_quant_name = "per_outchannel"
        self.weight_quant = "per_outchannel"
        self.weight_b = weight_b
        self.enable_w = enable_w
        self.enable_diffw = enable_diffw
        
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.act_b)
        elif act_quant == "per_channel":
            self.act_quant_name = "per_channel"
            self.act_quant = partial(quantize_activation_per_channel_absmax, n_bits=self.act_b)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x
            
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
        
            
    
    def reset_inference_state(self):
        self.inference_count = 0
        self.cached_input_odd = None
        self.cached_output_odd = None
        self.cached_weight_odd = None
    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.inference_count += 1
        
        if self.inference_count % 2 == 1 or self.inference_count > 2 * self.max_steps:
            if not self.mx_quan:
                output = super().forward(input)
            else:
                output = mx_linear(input, self.weight, self.bias, mx_specs=self.mx_specs0)
            
            self.cached_input_odd = input.detach().clone()
            self.cached_output_odd = output.detach().clone()
            self.cached_weight_odd = self.weight.detach().clone()
            
            return output
        
        else:
            if self.cached_input_odd is None or self.cached_output_odd is None:
                raise RuntimeError(f"[{self.layer_name}] An even inference step requires cached data from the preceding odd step.")
            if input.shape != self.cached_input_odd.shape:  # during evaluate during training: On shape mismatch, use the regular Linear computation
                if not self.mx_quan:
                    return super().forward(input)
                else:
                    return mx_linear(input, self.weight, self.bias, mx_specs=self.mx_specs0)
            
            # calculate diff_act_output (with Quan)
            diff_input = input - self.cached_input_odd
            if not self.mx_quan:
                if self.enable_diffx:
                    q_diff_input = self.act_quant(diff_input)
                    # diffx_mse = F.mse_loss(q_diff_input, diff_input)   # For test&debug
                else:
                    q_diff_input = diff_input
                    # diffx_mse = 0
                if self.enable_w:  
                    q_weight = quantize_weight_per_outchannel_absmax(self.weight, n_bits=self.weight_b)
                    # w_mse = F.mse_loss(q_weight, self.weight)   # For test&debug
                else:
                    q_weight = self.weight
                    # w_mse = 0
                diff_act_output = F.linear(q_diff_input, q_weight, None)
            else:
                diff_act_output = mx_linear(diff_input, self.weight, None, mx_specs=self.mx_specs1)
                
            
            # calculate diff_weight_output (with Quan)
            if self.use_uv_diffw and (self.uv_provider is not None):
                u, v, scale = self.uv_provider(self.layer_name + ".weight",
                                   self.weight.shape,
                                   self.weight.device,
                                   self.weight.dtype,
                                   self.inference_count)
                tmp = mx_linear(self.cached_input_odd, v.t(), None, mx_specs=self.mx_specs2)
                diff_weight_output = mx_linear(tmp, u, None, mx_specs=self.mx_specs2)
                diff_weight_output = diff_weight_output * scale
                
                # For test&debug
                # print(f"inference:{self.inference_count}: name:{self.layer_name}, fake_u[0]:{u[0]}, fake_v[0]: {v[0]}")
                
                # weight_diff_uv = (u@v.t())*scale
                # weight_diff = self.weight - self.cached_weight_odd
                # num = (weight_diff - weight_diff_uv).norm(p=2)
                # den = weight_diff.norm(p=2).clamp_min(1e-12)
                # diffw_err = (num / den).item()
                # print(f"diffw_err:{diffw_err}")
                
                # diff_weight_output = mx_linear(self.cached_input_odd, weight_diff_uv, None, mx_specs=self.mx_specs2)   
                
            else:
                weight_diff = self.weight - self.cached_weight_odd
                if not self.mx_quan:
                    if self.enable_x:
                        q_cached_input_odd = self.act_quant(self.cached_input_odd)
                        # x_mse = F.mse_loss(q_cached_input_odd, self.cached_input_odd)   # For test&debug
                    else:
                        q_cached_input_odd = self.cached_input_odd
                        # x_mse = 0
                    if self.enable_diffw:
                        q_weight_diff = quantize_weight_per_outchannel_absmax(weight_diff, n_bits=self.weight_b)
                        # diffw_mse = F.mse_loss(q_weight_diff, weight_diff)    # For test&debug
                    else:
                        q_weight_diff = weight_diff
                        # diffw_mse = 0   
                    diff_weight_output = F.linear(q_cached_input_odd, q_weight_diff)
                else:
                    diff_weight_output = mx_linear(self.cached_input_odd, weight_diff, None, mx_specs=self.mx_specs2)
                
            output = self.cached_output_odd + diff_act_output + diff_weight_output
            output = self.output_quant(output)
            
            # For test&debug
            # diff_act_output_noquan = F.linear(diff_input,self.weight,None)
            # diff_weight_output_noquan = F.linear(self.cached_input_odd,weight_diff)
            # output_noquan = self.cached_output_odd + diff_act_output_noquan + diff_weight_output_noquan
            # output_mse = F.mse_loss(output, output_noquan)
            # print(f'inference_count: {self.inference_count}')
            # print(f'output_mse: {output_mse}')
            
            return output
    
    
    def __repr__(self):
        return (f"QdiffLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, mx_quan={self.mx_quan}, "
                f"mx_w_elem_format={self.mx_w_elem_format}, mx_a_elem_format={self.mx_a_elem_format}, mx_diffw_elem_format={self.mx_diffw_elem_format}, mx_diffa_elem_format={self.mx_diffa_elem_format}, "
                f"weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, act_b={self.act_b}, weight_b={self.weight_b}, "
                f"enable_x={self.enable_x}, enable_diffx={self.enable_diffx}, enable_w={self.enable_w}, enable_diffw={self.enable_diffw})")
    

# This is the method about how to Quantize diff in current model 
def replace_linear_with_Qdifflinear(
    mx_quan,
    act_quant, weight_quant, act_b, weight_b,
    enable_w, enable_x, enable_diffw, enable_diffx,
    model: nn.Module, max_steps: int = 50,
    mx_w_elem_format=None, mx_a_elem_format=None, mx_diffw_elem_format=None, mx_diffa_elem_format=None,
    quantize_output=False,
    use_uv_diffw: bool=False, uv_provider=None,
    ):
    
    def replace_roberta_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print(f'Checking layer: {full_name}')
            
            in_encoder = full_name.startswith("roberta.encoder")
            on_path_to_encoder = (full_name == "roberta") or (full_name == "roberta.encoder")
            
            if in_encoder and isinstance(child, nn.Linear) and not isinstance(child, QdiffLinear):
                new_QdiffLinear = QdiffLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                    layer_name=full_name,  
                    max_steps=max_steps,
                    act_quant = act_quant,
                    weight_quant = weight_quant,
                    quantize_output = quantize_output,
                    act_b = act_b,
                    weight_b = weight_b,
                    enable_w = enable_w,
                    enable_x = enable_x,
                    enable_diffw = enable_diffw,
                    enable_diffx = enable_diffx,
                    mx_quan = mx_quan,
                    mx_w_elem_format = mx_w_elem_format, 
                    mx_a_elem_format = mx_a_elem_format, 
                    mx_diffw_elem_format = mx_diffw_elem_format, 
                    mx_diffa_elem_format = mx_diffa_elem_format,
                    use_uv_diffw = use_uv_diffw,
                    uv_provider = uv_provider,
                )
                
                new_QdiffLinear.weight.data = child.weight.data.clone() 
                if child.bias is not None:
                    new_QdiffLinear.bias.data = child.bias.data.clone()
                setattr(module, name, new_QdiffLinear)
                print(f"Replace {full_name} with QdiffLinear")
            else:
                if in_encoder or on_path_to_encoder:
                    replace_roberta_module(child, full_name) 
                
    if model.config.model_type == "roberta":
        replace_roberta_module(model)
    
def get_all_Qdifflayers(model: nn.Module):
    FQuan_layers = []
    
    def find_layers(module):
        for child in module.children():
            
            if isinstance(child, QdiffLinear):
                FQuan_layers.append(child)
            find_layers(child)
            
    find_layers(model)
    return FQuan_layers


def reset_fquant_(model: nn.Module):
    layers = get_all_Qdifflayers(model)
    for layer in layers:
        layer.reset_inference_state()
    print(f"have resetted {len(layers)} QdiffLinear inference state")
    


        