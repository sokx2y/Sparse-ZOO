#!/usr/bin/env python3
"""
DifferentialLinear 功能测试脚本

这个脚本用于测试 DifferentialLinear 的基本功能是否正常工作，
包括奇偶次推理的正确性验证。
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

try:
    from src.diff_linear import DifferentialLinear, replace_linear_layers_with_differential, \
                           get_all_differential_layers, reset_all_inference_states
    print("✓ 成功导入 DifferentialLinear 相关模块")
except ImportError as e:
    print(f"✗ 导入 DifferentialLinear 失败: {e}")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 创建DifferentialLinear层
    layer = DifferentialLinear(10, 5, enable_validation=True)
    print(f"✓ 创建 DifferentialLinear 层: {layer.in_features} -> {layer.out_features}")
    
    # 测试前向推理
    input_data = torch.randn(3, 10)
    output = layer(input_data)
    
    print(f"✓ 前向推理成功: 输入形状 {input_data.shape} -> 输出形状 {output.shape}")
    
    # 检查推理计数
    assert layer.inference_count == 1, f"推理计数错误: 期望1, 实际{layer.inference_count}"
    print(f"✓ 推理计数正确: {layer.inference_count}")
    
    # 检查缓存
    assert layer.cached_input_odd is not None, "奇数次输入未缓存"
    assert layer.cached_output_odd is not None, "奇数次输出未缓存"
    print(f"✓ 奇数次缓存正确")
    
    return True


def test_odd_even_inference_correctness():
    """测试奇偶数次推理的正确性"""
    print("\n=== 测试奇偶数次推理正确性 ===")
    
    # 创建带验证的DifferentialLinear层
    layer = DifferentialLinear(8, 4, bias=True, enable_validation=True)
    
    # 创建标准Linear层用于对比
    standard_layer = nn.Linear(8, 4, bias=True)
    
    # 复制权重和偏置，确保两个层参数相同
    standard_layer.weight.data = layer.weight.data.clone()
    standard_layer.bias.data = layer.bias.data.clone()
    
    # 测试数据
    input1 = torch.randn(2, 8)
    input2 = torch.randn(2, 8)
    
    # 第1次推理（奇数次）
    output1_diff = layer(input1)
    output1_standard = standard_layer(input1)
    
    # 检查奇数次推理的输出是否一致
    error1 = torch.max(torch.abs(output1_diff - output1_standard)).item()
    assert error1 < 1e-6, f"奇数次推理输出不一致: 误差={error1}"
    print(f"✓ 第1次推理（奇数次）输出正确: 最大误差={error1:.2e}")
    
    # 第2次推理（偶数次）
    output2_diff = layer(input2)
    output2_standard = standard_layer(input2)
    
    # 检查偶数次推理的输出是否一致
    error2 = torch.max(torch.abs(output2_diff - output2_standard)).item()
    assert error2 < 1e-6, f"偶数次推理输出不一致: 误差={error2}"
    print(f"✓ 第2次推理（偶数次）差分计算正确: 最大误差={error2:.2e}")
    
    # 验证差分计算的逻辑
    expected_diff_output = layer.cached_output_odd + (input2 - layer.cached_input_odd) @ layer.weight.T
    if layer.bias is not None:
        # 偶数次推理时bias已经包含在cached_output_odd中，所以不需要再加
        pass
    
    error_logic = torch.max(torch.abs(output2_diff - expected_diff_output)).item()
    assert error_logic < 1e-6, f"差分计算逻辑错误: 误差={error_logic}"
    print(f"✓ 差分计算逻辑验证通过: 误差={error_logic:.2e}")
    
    return True


def test_multiple_inferences():
    """测试多次推理的正确性"""
    print("\n=== 测试多次推理正确性 ===")
    
    layer = DifferentialLinear(6, 3, enable_validation=True)
    standard_layer = nn.Linear(6, 3)
    
    # 同步权重
    standard_layer.weight.data = layer.weight.data.clone()
    standard_layer.bias.data = layer.bias.data.clone()
    
    # 进行多次推理
    n_tests = 10
    max_errors = []
    
    for i in range(n_tests):
        input_data = torch.randn(2, 6)
        
        output_diff = layer(input_data)
        output_standard = standard_layer(input_data)
        
        error = torch.max(torch.abs(output_diff - output_standard)).item()
        max_errors.append(error)
        
        print(f"  第{i+1}次推理（{'奇数' if (i+1)%2==1 else '偶数'}次）: 最大误差={error:.2e}")
    
    # 检查所有误差都在可接受范围内
    assert all(e < 1e-6 for e in max_errors), f"存在过大的误差: {max(max_errors)}"
    print(f"✓ {n_tests}次推理全部通过，最大误差={max(max_errors):.2e}")
    
    # 检查验证记录
    if layer.validation_errors:
        print(f"✓ 记录了{len(layer.validation_errors)}个验证误差")
    
    return True


def test_model_replacement():
    """测试模型替换功能"""
    print("\n=== 测试模型替换功能 ===")
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 3)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # 保存原始权重
    original_fc1_weight = model.fc1.weight.data.clone()
    original_fc2_weight = model.fc2.weight.data.clone()
    
    # 替换Linear层
    replace_linear_layers_with_differential(model, enable_validation=True)
    
    # 验证替换成功
    assert isinstance(model.fc1, DifferentialLinear), "fc1未成功替换"
    assert isinstance(model.fc2, DifferentialLinear), "fc2未成功替换"
    assert isinstance(model.relu, nn.ReLU), "ReLU层被错误替换"
    print("✓ Linear层替换成功")
    
    # 验证权重保持不变
    assert torch.allclose(model.fc1.weight.data, original_fc1_weight), "fc1权重改变"
    assert torch.allclose(model.fc2.weight.data, original_fc2_weight), "fc2权重改变"
    print("✓ 权重保持不变")
    
    # 获取所有DifferentialLinear层
    diff_layers = get_all_differential_layers(model)
    assert len(diff_layers) == 2, f"DifferentialLinear层数量错误: 期望2, 实际{len(diff_layers)}"
    print(f"✓ 成功获取{len(diff_layers)}个DifferentialLinear层")
    
    return True


def test_reset_functionality():
    """测试重置功能"""
    print("\n=== 测试重置功能 ===")
    
    layer = DifferentialLinear(4, 2, enable_validation=True)
    
    # 进行一些推理
    input1 = torch.randn(1, 4)
    input2 = torch.randn(1, 4)
    
    output1 = layer(input1)
    output2 = layer(input2)
    
    # 验证状态已更新
    assert layer.inference_count == 2, "推理计数错误"
    assert layer.cached_input_odd is not None, "缓存为空"
    assert len(layer.validation_errors) > 0, "验证记录为空"
    
    # 重置状态
    layer.reset_inference_state()
    
    # 验证重置成功
    assert layer.inference_count == 0, "推理计数未重置"
    assert layer.cached_input_odd is None, "输入缓存未清空"
    assert layer.cached_output_odd is None, "输出缓存未清空"
    assert len(layer.validation_errors) == 0, "验证记录未清空"
    print("✓ 重置功能正常")
    
    return True


def test_shape_mismatch_handling():
    """测试形状不匹配的处理"""
    print("\n=== 测试形状不匹配处理 ===")
    
    layer = DifferentialLinear(5, 3)
    
    # 第1次推理
    input1 = torch.randn(2, 5)
    output1 = layer(input1)
    
    # 第2次推理，使用不同的batch size
    input2 = torch.randn(3, 5)  # 不同的batch size
    output2 = layer(input2)
    
    # 应该使用正常的Linear计算，不会报错
    print("✓ 形状不匹配时自动使用正常Linear计算")
    
    return True


def test_validation_errors_recording():
    """测试验证误差记录功能"""
    print("\n=== 测试验证误差记录 ===")
    
    layer = DifferentialLinear(4, 2, enable_validation=True)
    layer.max_validation_records = 5  # 限制记录数量
    
    # 进行多次推理
    for i in range(10):
        input_data = torch.randn(1, 4)
        output = layer(input_data)
    
    # 检查验证记录
    even_count = 10 // 2  # 偶数次推理的数量
    expected_records = min(even_count, layer.max_validation_records)
    
    assert len(layer.validation_errors) == expected_records, \
        f"验证记录数量错误: 期望{expected_records}, 实际{len(layer.validation_errors)}"
    
    print(f"✓ 验证记录数量正确: {len(layer.validation_errors)} (限制={layer.max_validation_records})")
    
    # 检查记录内容
    if layer.validation_errors:
        first_error = layer.validation_errors[0]
        assert 'inference_count' in first_error, "缺少推理计数"
        assert 'max_error' in first_error, "缺少最大误差"
        assert 'mean_error' in first_error, "缺少平均误差"
        print("✓ 验证记录格式正确")
    
    return True


def test_plotting_functionality():
    """测试绘图功能"""
    print("\n=== 测试绘图功能 ===")
    
    layer = DifferentialLinear(3, 2, enable_validation=True)
    
    # 进行一些推理以生成验证数据
    for i in range(6):
        input_data = torch.randn(2, 3)
        output = layer(input_data)
    
    # 测试绘图功能（不显示，只保存）
    try:
        # 创建测试目录
        test_dir = "./test_plots"
        os.makedirs(test_dir, exist_ok=True)
        
        # 测试验证误差图
        plot_path = os.path.join(test_dir, "test_validation_errors.png")
        layer.plot_validation_errors(save_path=plot_path, show_plot=False)
        
        if os.path.exists(plot_path):
            print("✓ 验证误差图生成成功")
        else:
            print("✗ 验证误差图生成失败")
            return False
            
    except Exception as e:
        print(f"✗ 绘图功能测试失败: {e}")
        return False
    
    return True


def test_model_inference_correctness():
    """测试完整模型推理的正确性"""
    print("\n=== 测试完整模型推理正确性 ===")
    
    # 创建两个相同的模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 4)
            self.fc2 = nn.Linear(4, 2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 标准模型
    standard_model = TestModel()
    
    # 差分模型
    diff_model = TestModel()
    # 同步权重
    diff_model.load_state_dict(standard_model.state_dict())
    # 替换为DifferentialLinear
    replace_linear_layers_with_differential(diff_model, enable_validation=True)
    
    # 测试多次推理
    test_inputs = [torch.randn(3, 8) for _ in range(6)]
    
    for i, input_data in enumerate(test_inputs):
        output_standard = standard_model(input_data)
        output_diff = diff_model(input_data)
        
        error = torch.max(torch.abs(output_standard - output_diff)).item()
        print(f"  模型推理{i+1}: 最大误差={error:.2e}")
        
        assert error < 1e-5, f"模型推理误差过大: {error}"
    
    print("✓ 完整模型推理正确性验证通过")
    
    return True


def main():
    """主测试函数"""
    print("开始测试 DifferentialLinear 功能...")
    
    tests = [
        test_basic_functionality,
        test_odd_even_inference_correctness,
        test_multiple_inferences,
        test_model_replacement,
        test_reset_functionality,
        test_shape_mismatch_handling,
        test_validation_errors_recording,
        test_plotting_functionality,
        test_model_inference_correctness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ 测试 {test.__name__} 失败")
        except Exception as e:
            print(f"✗ 测试 {test.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！DifferentialLinear 功能正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)