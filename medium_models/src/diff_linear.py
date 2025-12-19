import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.stats import norm, laplace
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


class DifferentialLinear(nn.Linear):
    """
    自定义Linear层，用以验证 one inference ZO 的可行性(noquan nosparse版本)：
    - 奇数次推理：正常计算Linear，并缓存输入和输出(input_odd, output_odd)
    - 偶数次推理：output = output_odd + linear(input - input_odd, weight) + linear(input_odd, z)   z为lozo中weight的扰动矩阵 
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None, enable_validation: bool = False, enable_accurate_diff: bool = False, max_steps: int = 50, 
                 layer_name: str = "unnamed"):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.inference_count = 0
        self.layer_name = layer_name  # 记录该层在模型中的名称
        
        # 缓存奇数次的输入激活和输出
        self.register_buffer('cached_input_odd', None, persistent=False)
        self.register_buffer('cached_output_odd', None, persistent=False)
        self.enable_accurate_diff = enable_accurate_diff
        if self.enable_accurate_diff:
            self.register_buffer('cached_weight_odd', None, persistent=False)  # Cache for odd weights
        
        # 用于检查tictoc计算的正确性
        self.enable_validation = enable_validation
        self.validation_errors = [] 
        self.max_validation_records = 1000  
        self.max_steps = max_steps
        
        # check sparsity
        self.sparsity_diff = []
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.inference_count += 1
        
        if self.inference_count % 2 == 1 or self.inference_count > 2*self.max_steps:         
            # 查看input.shape
            # print(f'[{self.layer_name}] 第{self.inference_count}次difflinear的输入shape是{input.shape}')
            # 正常的Linear计算
            output = super().forward(input)
            
            # 缓存输入和输出
            self.cached_input_odd = input.detach().clone()
            self.cached_output_odd = output.detach().clone()
            if self.enable_accurate_diff:
                self.cached_weight_odd = self.weight.detach().clone()
            
            return output
            
        else: 
            if self.cached_input_odd is None or self.cached_output_odd is None:
                raise RuntimeError(f"[{self.layer_name}] 偶数次推理需要先有奇数次推理的缓存数据")
            
            # 检查shape是否匹配
            # 查看input.shape
            # print(f'[{self.layer_name}] 第{self.inference_count}次difflinear的输入shape是{input.shape}')
            if input.shape != self.cached_input_odd.shape:
                # Shape不匹配时，使用正常的Linear计算
                # print(f"警告：[{self.layer_name}] 推理#{self.inference_count} Shape不匹配 "
                      # f"(当前: {input.shape} vs 缓存: {self.cached_input_odd.shape})，"
                      # f"使用正常Linear计算")
                return super().forward(input)
        
            # 计算diff部分
            diff_input = input - self.cached_input_odd
            diff_output = F.linear(diff_input, self.weight, None)
            if self.enable_accurate_diff:
                weight_diff = self.weight - self.cached_weight_odd
                weight_output = F.linear(self.cached_input_odd, weight_diff)
            # 最终输出
            if self.enable_accurate_diff:
                output = self.cached_output_odd + diff_output + weight_output
            else:
                output = self.cached_output_odd + diff_output
            
            # 是否validation
            if self.enable_validation:
                with torch.no_grad():
                    expected_output = super().forward(input)
                    
                    # 计算误差
                    error = torch.abs(output - expected_output)
                    max_error = torch.max(error).item()
                    mean_error = torch.mean(error).item()
                    
                    act_part = torch.mean(diff_output).item()
                    if self.enable_accurate_diff:
                        weight_part = torch.mean(weight_output).item()
                    else:
                        weight_part = 0
                        
                    # act_contribution = torch.abs(diff_output) / torch.abs(output)
                    # mean_act_contribution = torch.mean(act_contribution).item()
                    # if self.enable_accurate_diff:
                        # weight_contribution = torch.abs(weight_output) / torch.abs(output)
                        # mean_weight_contribution = torch.mean(weight_contribution).item()
                    # else:
                        # mean_weight_contribution = 0
                        
                    
                    # 记录误差（包括层名称）
                    if len(self.validation_errors) < self.max_validation_records:
                        self.validation_errors.append({
                            'layer_name': self.layer_name,
                            'inference_count': self.inference_count,
                            'max_error': max_error,
                            'mean_error': mean_error,
                            "act_part": act_part,
                            "weight_part": weight_part,
                            'expected_output': expected_output.cpu().numpy(),
                            'diff_output': output.cpu().numpy()
                        })
                        
                    # 记录sparsity数据
                    if len(self.sparsity_diff) < self.max_validation_records:
                        self.sparsity_diff.append({
                            # 'weight_odd': self.cached_weight_odd
                            # 'weight_even': self.weight
                            'diff_activation': diff_input.cpu().numpy(),
                            'diff_weight': weight_diff.cpu().numpy(),   
                            'activation_odd': self.cached_input_odd.cpu().numpy(),
                            'weight_even': self.weight.cpu().numpy()
                        })
                               
                    # Attention : 'diff_weight': weight_diff.cpu().numpy() canbe lowrank 分解 (based on lozo) 可以从lozotrainer直接传入分解的两个个lowrank
                    
                    # 如果误差过大，打印警告
                    if max_error > 1e-2:
                        print(f"警告：[{self.layer_name}] 推理#{self.inference_count} 差分计算误差较大(1e-2)！")
                        print(f"  最大误差: {max_error:.2e}")
                        print(f"  平均误差: {mean_error:.2e}")
            
            return output
    
    def reset_inference_state(self):
        """重置推理状态"""
        self.inference_count = 0
        self.cached_input_odd = None
        self.cached_output_odd = None
        self.validation_errors.clear()
        self.sparsity_diff.clear()
        
        
    def plot_sparsity_diff_distribution(self, save_path: Optional[str] = None, show_plot: bool = True, 
                                        num_random_samples: int = 1):
        """
        可视化稀疏差异的概率分布图，每次推理保存一个单独的图像文件，并标注推理次数。
        
        参数:
            save_path: 保存图片的路径
            show_plot: 是否显示图片
            num_random_samples: 随机选取的推理次数
        """
        if not self.sparsity_diff:
            print(f"[{self.layer_name}] 没有可用于可视化的稀疏差异数据")
            return
        
        # 随机选取推理次数
        num_samples = min(num_random_samples, len(self.sparsity_diff))
        if num_samples > 0:
            random.seed(42)  # 设计seed
            random_indices = random.sample(range(len(self.sparsity_diff)), num_samples)
            
            # 遍历每次推理，生成图像
            for idx, i in enumerate(random_indices):
                sparsity_data = self.sparsity_diff[i]
                diff_activation = sparsity_data['diff_activation']
                diff_weight = sparsity_data['diff_weight']
                activation_odd = sparsity_data['activation_odd']
                weight_even = sparsity_data['weight_even']
                
    
                # 定义绘制直方图和拟合分布的函数
                def plot_hist_with_fit(data, ax, title):
                    data = np.asarray(data).ravel()
                    ax.hist(data, bins=100, density=True, alpha=0.6, color='b', edgecolor='black')
                    xmin, xmax = ax.get_xlim()
                    x = np.linspace(xmin, xmax, 100)
                    
                    # 高斯分布拟合
                    gauss_pdf = norm.pdf(x, np.mean(data), np.std(data))
                    ax.plot(x, gauss_pdf, color='gold', linestyle='-', label='Gaussian Fit') 
                    
                    # 拉普拉斯分布拟合
                    laplace_pdf = laplace.pdf(x, np.mean(data), np.std(data))
                    ax.plot(x, laplace_pdf, 'g-', label='laplace Fit')
                    
                    ax.set_title(title)
                    ax.legend()
    
                # 创建每次推理的图像
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                fig.suptitle(f'inference times: {i} - Sparsity Difference Distributions - Layer: {self.layer_name}', fontsize=16)
                
                # 绘制 diff_activation 分布图
                plot_hist_with_fit(diff_activation, axs[0], 'diff_activation')
                
                # 绘制 diff_weight 分布图
                plot_hist_with_fit(diff_weight, axs[1], 'diff_weight')
                
                # 绘制 activation_odd 分布图
                plot_hist_with_fit(activation_odd, axs[2], 'activation_odd')
                
                # 绘制weight_even 分布图
                plot_hist_with_fit(weight_even, axs[3], 'weight_even')
                
                # 调整布局并保存/显示图像
                plt.tight_layout()
                save_directory = os.path.dirname(save_path)
                if save_directory:
                    # 使用推理次数作为文件名的一部分
                    plot_save_path = os.path.join(save_directory, f"{self.layer_name}_sparsityHist_inference_{i}.png")
                    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
                    print(f"[{self.layer_name}] 第 {i} 次推理sparsity图像已保存到: {plot_save_path}")
                if show_plot:
                    plt.show()
                
                plt.close()
    
    
    def plot_activation_weight_3d(self,save_path: Optional[str] = None,show_plot: bool = True,num_random_samples: int = 1,seed: int = 42,
        # activation 的聚合与下采样
        magnitude: str = "abs",  # 'abs' 或 'abs_mean'（对 batch 维先均值/或绝对值再均值）
        max_tokens: Optional[int] = None,
        token_stride: int = 1,
        channel_stride: int = 4,
        # weight 的下采样
        weight_out_stride: int = 4,
        weight_in_stride: int = 4,
        cmap: str = "coolwarm",       # 伪彩色风格
    ):
        """
        随机抽取若干条 self.sparsity_diff 记录；每条记录保存一张图（含两个 3D 子图）：
          左：activation_odd 的 |幅值|（Token × Channel）
          右：weight_even 的 |W|（Out × In）
        """
        if not self.sparsity_diff:
            print(f"[{self.layer_name}] 没有可用于 3D 可视化的 sparsity_3D 数据")
            return
    
        # 随机选取推理次数
        num_samples = min(num_random_samples, len(self.sparsity_diff))
        random.seed(seed)
        random_indices = random.sample(range(len(self.sparsity_diff)), num_samples)
    
        for idx, i in enumerate(random_indices):
            rec = self.sparsity_diff[i]
    
            # -------- activation_odd -> (T, C) --------
            act = np.asarray(rec["activation_odd"])
            print(f"{act.shape}")
            # 可能是 (B,T,C) / (T,C) / (C,) 三种情形
            if act.ndim == 3:
                # 假设 (B, T, C)
                if magnitude == "abs_mean":
                    Z_act = np.abs(act.mean(axis=0))   # (T, C)
                elif magnitude == "abs":
                    Z_act = np.abs(act).mean(axis=0)   # (T, C)
                else:
                    raise ValueError("magnitude 仅支持 'abs' 或 'abs_mean'")
            elif act.ndim == 2:
                # (T, C)
                Z_act = np.abs(act) if magnitude.startswith("abs") else act
            elif act.ndim == 1:
                # (C,) -> 当作单 token
                Z_act = np.abs(act[None, :]) if magnitude.startswith("abs") else act[None, :]
            else:
                raise ValueError(f"activation_odd 维度不支持: {act.shape}")
            
            T0,C0 = Z_act.shape
            # 下采样
            Z_act = Z_act[::token_stride, ::channel_stride]
            # 清理 NaN/Inf
            Z_act = np.nan_to_num(Z_act, nan=0.0, posinf=0.0, neginf=0.0)
    
            T_ds = np.arange(T0)[::max(1, token_stride)]
            C_ds = np.arange(C0)[::max(1, channel_stride)]
            
            X_act, Y_act = np.meshgrid(C_ds,T_ds)  # X: Channel, Y: Token
    
            # -------- weight_even -> (Out, In) --------
            W = np.asarray(rec["weight_even"])
            print(f"{W.shape}")
            if W.ndim != 2:
                # 若不是二维（比如多头权重），兜底压成二维
                W = W.reshape(W.shape[0], -1)
            O0,I0 = W.shape
            Z_w = np.abs(W)[::weight_out_stride, ::weight_in_stride]
            Z_w = np.nan_to_num(Z_w, nan=0.0, posinf=0.0, neginf=0.0)
    
            O_ds = np.arange(O0)[::max(1, weight_out_stride)]
            I_ds = np.arange(I0)[::max(1, weight_in_stride)]
            
            X_w, Y_w = np.meshgrid(I_ds, O_ds)  # X: In, Y: Out
            
            # ======================= diff_activation -> (T, C) ======================
            dact = np.asarray(rec["diff_activation"])
            if dact.ndim == 3:  # (B, T, C)
                # 和 activation_odd 一样的聚合逻辑
                if magnitude == "abs_mean":
                    Z_dact = np.abs(dact.mean(axis=0))
                elif magnitude == "abs":
                    Z_dact = np.abs(dact).mean(axis=0)
                else:
                    raise ValueError("magnitude 仅支持 'abs' 或 'abs_mean'")
            elif dact.ndim == 2:
                Z_dact = np.abs(dact) if magnitude.startswith("abs") else dact
            elif dact.ndim == 1:
                Z_dact = np.abs(dact[None, :]) if magnitude.startswith("abs") else dact[None, :]
            else:
                raise ValueError(f"diff_activation 维度不支持: {dact.shape}")
    
            if max_tokens is not None:
                Z_dact = Z_dact[:max_tokens, :]
            Td0, Cd0 = Z_dact.shape
            Z_dact = Z_dact[::max(1, token_stride), ::max(1, channel_stride)]
            Z_dact = np.nan_to_num(Z_dact, nan=0.0, posinf=0.0, neginf=0.0)
            Td_idx = np.arange(Td0)[::max(1, token_stride)]
            Cd_idx = np.arange(Cd0)[::max(1, channel_stride)]
            X_dact, Y_dact = np.meshgrid(Cd_idx, Td_idx)
            
            # ======================= diff_weight -> (Out, In) =======================
            W_diff = np.asarray(rec["diff_weight"])
            if W_diff.ndim != 2:
                W_diff = W_diff.reshape(W_diff.shape[0], -1)
            Od0, Id0 = W_diff.shape
            Z_wd = np.abs(W_diff)[::max(1, weight_out_stride), ::max(1, weight_in_stride)]
            Z_wd = np.nan_to_num(Z_wd, nan=0.0, posinf=0.0, neginf=0.0)
            Outd_idx = np.arange(Od0)[::max(1, weight_out_stride)]
            Ind_idx  = np.arange(Id0)[::max(1, weight_in_stride)]
            X_wd, Y_wd = np.meshgrid(Ind_idx, Outd_idx)
            
            # -------- 画一张图：4个 3D 子图 --------
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f'Inference times: {i} - 3D Surfaces - Layer: {self.layer_name}', fontsize=13)
    
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1.plot_surface(X_act, Y_act, Z_act, cmap=cmap, linewidth=0, antialiased=True)
            ax1.set_title("Activation Odd (|value|)")
            ax1.set_xlabel("Channel")
            ax1.set_ylabel("Token")
            ax1.set_zlabel("Magnitude")
    
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            ax2.plot_surface(X_w, Y_w, Z_w, cmap=cmap, linewidth=0, antialiased=True)
            ax2.set_title("Weight Even (|W|)")
            ax2.set_xlabel("In")
            ax2.set_ylabel("Out")
            ax2.set_zlabel("Magnitude")
            
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax3.plot_surface(X_dact, Y_dact, Z_dact, cmap=cmap, linewidth=0, antialiased=True)
            ax3.set_title("Diff Activation (|value|)")
            ax3.set_xlabel("Channel (orig idx)")
            ax3.set_ylabel("Token (orig idx)")
            ax3.set_zlabel("Magnitude")
    
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.plot_surface(X_wd, Y_wd, Z_wd, cmap=cmap, linewidth=0, antialiased=True)
            ax4.set_title("Diff Weight (|ΔW|)")
            ax4.set_xlabel("In (orig idx)")
            ax4.set_ylabel("Out (orig idx)")
            ax4.set_zlabel("Magnitude")
    
            fig.tight_layout()
    
            # 保存文件：一条推理 -> 一张图
            save_directory = os.path.dirname(save_path)
            out_path = os.path.join(
                save_directory,
                f"{self.layer_name}_sparsity3D_inference_{i}.png"
            )
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"[{self.layer_name}] 第 {i} 次推理3D图像已保存到: {out_path}")
            if show_plot:
                plt.show()
                
            plt.close(fig)
    
    
    def plot_validation_errors(self, save_path: Optional[str] = None, show_plot: bool = True,
                             num_random_samples: int = 10):
        """
        Visualize validation errors
        
        Args:
            save_path: 保存图片的路径
            show_plot: 是否显示图片
            num_random_samples: 随机选取的推理次数
        """
        if not self.validation_errors:
            print(f"[{self.layer_name}] No validation data available for visualization")
            return
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'TicToc Computation Validation Results - Layer: {self.layer_name}', fontsize=14)
    
        # Extract data
        inference_counts = [e['inference_count'] for e in self.validation_errors]
        mean_errors = [e['mean_error'] for e in self.validation_errors]
    
        # Mean error (左图)
        ax1.semilogy(inference_counts, mean_errors, 'g-o', markersize=4)
        ax1.set_xlabel('Inference Count')
        ax1.set_ylabel('Mean Error (log scale)')
        ax1.set_title(f'Mean Error Trend - {self.layer_name}')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1e-5, color='r', linestyle='--', label='Tolerance Threshold (1e-5)')
        ax1.legend()
    
        # 随机选取几个推理次数的输出对比（右图）
        ax2.set_title(f'Output Comparison (Random {num_random_samples} Inferences) - {self.layer_name}')
    
        # 随机选取推理次数
        num_samples = min(num_random_samples, len(self.validation_errors))
        if num_samples > 0:
            random.seed(42)   # 设计seed
            random_indices = random.sample(range(len(self.validation_errors)), num_samples)
        
            for idx, i in enumerate(random_indices):
                error_data = self.validation_errors[i]
                expected = error_data['expected_output'].flatten()
                diff = error_data['diff_output'].flatten()
                mse = np.mean((error_data['expected_output'] - error_data['diff_output']) ** 2)   # 计算mse
                
                # 计算相对误差(mean)
                absolute_error = np.abs(error_data['expected_output'] - error_data['diff_output'])
                relative_error = absolute_error / np.abs(error_data['expected_output'])
                mean_relative_error = np.mean(relative_error)
                
                # 计算余弦相似度
                def compute_cosine_similarity(array1, array2):
                    """
                    计算两个numpy数组的余弦相似度
                    """
                    # 展平数组（如果是多维的）
                    flat1 = array1.flatten()
                    flat2 = array2.flatten()
    
                    # 计算点积
                    dot_product = np.dot(flat1, flat2)
    
                    # 计算各自的L2范数
                    norm1 = np.linalg.norm(flat1)
                    norm2 = np.linalg.norm(flat2)
    
                    # 计算余弦相似度
                    cosine_sim = dot_product / (norm1 * norm2)
    
                    return cosine_sim
                cosine_similarity = compute_cosine_similarity(
                                    error_data['expected_output'], 
                                    error_data['diff_output'])

                
                # 输出误差摘要
                # save_directory = os.path.dirname(save_path)
                # errdata_path = os.path.join(save_directory, f"{self.layer_name}_errdata.txt")
                # with open(errdata_path, 'a') as f:
                    # 如果是第一次写入文件，可以选择写入标题（例如）：
                    # if idx == 0:
                        # f.write(f"{self.layer_name}误差数据如下:\n")
                    # f.write(f"推理次数：{i}\n")
                    # f.write(f"最大误差：{error_data['max_error']}\n")
                    # f.write(f"平均误差:{error_data['mean_error']}\n")
                    # f.write(f"扰动矩阵部分的占比：{error_data['weight_part']}\n")
                    # f.write(f"输入差距部分的占比:{error_data['act_part']}\n")
                    # f.write(f"均方误差 (MSE): {mse:.6f}\n")
                    # f.write(f"平均相对误差: {mean_relative_error:.6f}\n")
                    # f.write(f"余弦相似度：{cosine_similarity:.6f}\n")
                    # if cosine_similarity > !!!:
                        # whoisbadlayer_path = os.path.join(save_directory, f"bad_layers.txt")
                        # with open (whoisbadlayer_path,'a') as w:
                            # w.write(f"{self.layer_name}\n")
                    
            
                # 如果输出维度太大，进行下采样显示
                max_points = 30  # 最多显示30个点
                if len(expected) > max_points:
                    step = len(expected) // max_points
                    indices = np.arange(0, len(expected), step)  # 张量扁平化后的索引value
                    expected_sampled = expected[indices]
                    diff_sampled = diff[indices]
                    x = np.arange(len(expected_sampled))  
                else:
                    expected_sampled = expected
                    diff_sampled = diff
                    x = np.arange(len(expected))
            
                inference_num = error_data['inference_count']
                ax2.plot(x, expected_sampled, 'b-',alpha=0.5, linewidth=0.8)
                ax2.plot(x, diff_sampled, 'r--', alpha=0.5, linewidth=0.8)

        ax2.set_xlabel('Output Element Index')
        ax2.set_ylabel('Output Value')
        ax2.legend(['Normal Linear', 'Differential Computation'],loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            # 在文件名中包含层名称
            base, ext = os.path.splitext(save_path)
            # layer_save_path = f"{base}_{self.layer_name.replace('.', '_')}{ext}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[{self.layer_name}] Plot saved to: {save_path}")

        if show_plot:
            plt.show()

        plt.close()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, enable_validation={self.enable_validation}, ' \
               f'layer_name={self.layer_name}'


def replace_linear_layers_with_differential(model: nn.Module, enable_validation: bool = False, enable_accurate_diff: bool = False, max_steps: int = 50):
    """
    将模型中的Linear层替换为DifferentialLinear层
    
    Args:
        model: 要替换的模型
        enable_validation: 是否检验正确性
    """
    def replace_module(module, prefix=""):
        for name, child in module.named_children():
            # 构建完整的层路径名称
            full_name = f"{prefix}.{name}" if prefix else name
            
            print(f'Checking layer: {full_name}')
            
            if isinstance(child, nn.Linear) and not isinstance(child, DifferentialLinear):
                # 创建新的DifferentialLinear层，带有层名称
                differential_layer = DifferentialLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                    enable_validation=enable_validation,
                    enable_accurate_diff=enable_accurate_diff,
                    layer_name=full_name,  # 传入层的完整路径名称
                    max_steps=max_steps
                )
                
                # 复制权重和偏置
                differential_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    differential_layer.bias.data = child.bias.data.clone()
                
                # 替换模块
                setattr(module, name, differential_layer)
                
                print(f"Replace {full_name} with DifferentialLinear")
            else:
                # 递归处理子模块
                replace_module(child, full_name)
    
    replace_module(model)


def get_all_differential_layers(model: nn.Module) -> List[DifferentialLinear]:
    """
    获取模型中的所有DifferentialLinear层
    
    Args:
        model: 模型
        
    Returns:
        DifferentialLinear层列表
    """
    differential_layers = []
    
    def find_layers(module):
        for child in module.children():
            if isinstance(child, DifferentialLinear):
                differential_layers.append(child)
            find_layers(child)
    
    find_layers(model)
    return differential_layers


def reset_all_inference_states(model: nn.Module):
    """重置模型中所有DifferentialLinear层的推理状态"""
    layers = get_all_differential_layers(model)
    for layer in layers:
        layer.reset_inference_state()
    print(f"已重置 {len(layers)} 个DifferentialLinear层的状态")


