import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import os


class CustomLinear(nn.Linear):
    """
    自定义Linear层，能够记录奇数次和偶数次推理的前向数据
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, 
            record_interval: int = 100):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # 记录推理次数
        self.inference_count = 0
        
        
        # 存储奇数次和偶数次的输入输出数据
        self.odd_inputs = []
        self.odd_outputs = []
        self.even_inputs = []
        self.even_outputs = []
        
        # 存储统计信息
        self.stats = {
            'odd_input_mean': [],
            'odd_input_std': [],
            'odd_output_mean': [],
            'odd_output_std': [],
            'even_input_mean': [],
            'even_input_std': [],
            'even_output_mean': [],
            'even_output_std': []
        }
        
        # 是否启用记录
        self.enable_recording = True
        
        # 最大记录数量（防止内存溢出）
        self.max_records = 1000
        self.record_interval = record_interval
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行正常的线性变换
        self.inference_count += 1
        # print(f"inference_count:{self.inference_count}")
        output = super().forward(input)
        
        # 如果启用记录，则记录数据
        if self.enable_recording and (self.inference_count % self.record_interval == 0 or self.inference_count % self.record_interval == 1):
            print(f'inference count:{self.inference_count}, running record')
            self._record_data(input.detach(), output.detach())
            
        return output
    
    def _record_data(self, input: torch.Tensor, output: torch.Tensor):
        """记录输入输出数据"""
        
        
        # 将tensor转换为numpy数组并存储
        input_np = input.cpu().numpy()
        output_np = output.cpu().numpy()
        
        if self.inference_count % 2 == 1:  # 奇数次
            if len(self.odd_inputs) < self.max_records:
                self.odd_inputs.append(input_np)
                self.odd_outputs.append(output_np)
                
                # 计算统计信息
                self._update_stats('odd', input_np, output_np)
        else:  # 偶数次
            if len(self.even_inputs) < self.max_records:
                self.even_inputs.append(input_np)
                self.even_outputs.append(output_np)
                
                # 计算统计信息
                self._update_stats('even', input_np, output_np)
    
    def _update_stats(self, parity: str, input_np: np.ndarray, output_np: np.ndarray):
        """更新统计信息"""
        input_mean = np.mean(input_np)
        input_std = np.std(input_np)
        output_mean = np.mean(output_np)
        output_std = np.std(output_np)
        
        self.stats[f'{parity}_input_mean'].append(input_mean)
        self.stats[f'{parity}_input_std'].append(input_std)
        self.stats[f'{parity}_output_mean'].append(output_mean)
        self.stats[f'{parity}_output_std'].append(output_std)
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        summary = {
            'inference_count': self.inference_count,
            'odd_records': len(self.odd_inputs),
            'even_records': len(self.even_inputs),
            'stats': self.stats.copy()
        }
        return summary
    
    def clear_records(self):
        """清除记录的数据"""
        self.odd_inputs.clear()
        self.odd_outputs.clear()
        self.even_inputs.clear()
        self.even_outputs.clear()
        
        for key in self.stats:
            self.stats[key].clear()
    
    def plot_data_distribution(self, save_path: Optional[str] = None, show_plot: bool = True):
        """绘制数据分布图"""
        if not self.odd_inputs and not self.even_inputs:
            print("没有数据可以绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'CustomLinear Layer Data Distribution (Inference Count: {self.inference_count})', fontsize=16)
        
        # 输入数据分布
        if self.odd_inputs:
            odd_input_flat = np.concatenate([x.flatten() for x in self.odd_inputs])
            axes[0, 0].hist(odd_input_flat, bins=50, alpha=0.7, label=f'Odd (n={len(self.odd_inputs)})', density=True)
            axes[0, 0].set_title('Odd Inference Input Distribution')
            axes[0, 0].set_xlabel('Input Values')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if self.even_inputs:
            even_input_flat = np.concatenate([x.flatten() for x in self.even_inputs])
            axes[0, 1].hist(even_input_flat, bins=50, alpha=0.7, label=f'Even (n={len(self.even_inputs)})', density=True)
            axes[0, 1].set_title('Even Inference Input Distribution')
            axes[0, 1].set_xlabel('Input Values')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 输出数据分布
        if self.odd_outputs:
            odd_output_flat = np.concatenate([x.flatten() for x in self.odd_outputs])
            axes[1, 0].hist(odd_output_flat, bins=50, alpha=0.7, label=f'Odd (n={len(self.odd_outputs)})', density=True)
            axes[1, 0].set_title('Odd Inference Output Distribution')
            axes[1, 0].set_xlabel('Output Values')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if self.even_outputs:
            even_output_flat = np.concatenate([x.flatten() for x in self.even_outputs])
            axes[1, 1].hist(even_output_flat, bins=50, alpha=0.7, label=f'Even (n={len(self.even_outputs)})', density=True)
            axes[1, 1].set_title('Even Inference Output Distribution')
            axes[1, 1].set_xlabel('Output Values')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()



def replace_linear_layers_with_custom(model: nn.Module):
    """
    将模型中的Linear层替换为CustomLinear层
    
    Args:
        model: 要替换的模型
        
    Returns:
        替换后的CustomLinear层列表
    """
    custom_layers = []
    
    def replace_module(module):
        for name, child in module.named_children():
            print(f'name:{name}')
            if isinstance(child, nn.Linear):
                # 创建新的CustomLinear层
                custom_layer = CustomLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                
                # 复制权重和偏置
                custom_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    custom_layer.bias.data = child.bias.data.clone()
                
                # 替换模块
                setattr(module, name, custom_layer)
                # custom_layers.append(custom_layer)
                
                print(f"Replace {name} with CustomLinear")
            else:
                replace_module(child)
    
    replace_module(model)
    # return custom_layers


def get_all_custom_layers(model: nn.Module) -> List[CustomLinear]:
    """
    获取模型中的所有CustomLinear层
    
    Args:
        model: 模型
        
    Returns:
        CustomLinear层列表
    """
    custom_layers = []
    
    def find_custom_layers(module):
        for child in module.children():
            if isinstance(child, CustomLinear):
                custom_layers.append(child)
            find_custom_layers(child)
    
    find_custom_layers(model)
    return custom_layers 