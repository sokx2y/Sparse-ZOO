#!/usr/bin/env python3
"""
CustomLinear 功能测试脚本

这个脚本用于测试 CustomLinear 的基本功能是否正常工作。
"""

import sys
import os
import torch
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.custom_linear import CustomLinear
    print("✓ 成功导入 CustomLinear")
except ImportError as e:
    print(f"✗ 导入 CustomLinear 失败: {e}")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 创建CustomLinear层
    layer = CustomLinear(10, 5)
    print(f"✓ 创建 CustomLinear 层: {layer.in_features} -> {layer.out_features}")
    
    # 测试前向推理
    input_data = torch.randn(3, 10)
    output = layer(input_data)
    
    print(f"✓ 前向推理成功: 输入形状 {input_data.shape} -> 输出形状 {output.shape}")
    
    # 检查推理计数
    assert layer.inference_count == 1, f"推理计数错误: 期望1, 实际{layer.inference_count}"
    print(f"✓ 推理计数正确: {layer.inference_count}")
    
    # 检查数据记录
    assert len(layer.odd_inputs) == 1, f"奇数次输入记录错误: 期望1, 实际{len(layer.odd_inputs)}"
    assert len(layer.even_inputs) == 0, f"偶数次输入记录错误: 期望0, 实际{len(layer.even_inputs)}"
    print(f"✓ 数据记录正确: 奇数次={len(layer.odd_inputs)}, 偶数次={len(layer.even_inputs)}")
    
    return True


def test_odd_even_recording():
    """测试奇偶数次记录功能"""
    print("\n=== 测试奇偶数次记录功能 ===")
    
    layer = CustomLinear(5, 3)
    
    # 进行多次推理
    input_data = torch.randn(2, 5)
    
    for i in range(5):
        output = layer(input_data)
        print(f"第 {i+1} 次推理: 奇数次记录={len(layer.odd_inputs)}, 偶数次记录={len(layer.even_inputs)}")
    
    # 验证记录数量
    expected_odd = 3  # 第1,3,5次
    expected_even = 2  # 第2,4次
    
    assert len(layer.odd_inputs) == expected_odd, f"奇数次记录数量错误: 期望{expected_odd}, 实际{len(layer.odd_inputs)}"
    assert len(layer.even_inputs) == expected_even, f"偶数次记录数量错误: 期望{expected_even}, 实际{len(layer.even_inputs)}"
    
    print(f"✓ 奇偶数次记录功能正常: 奇数次={len(layer.odd_inputs)}, 偶数次={len(layer.even_inputs)}")
    
    return True


def test_statistics():
    """测试统计功能"""
    print("\n=== 测试统计功能 ===")
    
    layer = CustomLinear(4, 2)
    
    # 进行推理
    input_data = torch.randn(3, 4)
    output = layer(input_data)
    
    # 检查统计信息
    summary = layer.get_data_summary()
    
    assert 'inference_count' in summary, "缺少推理计数"
    assert 'odd_records' in summary, "缺少奇数次记录数"
    assert 'even_records' in summary, "缺少偶数次记录数"
    assert 'stats' in summary, "缺少统计信息"
    
    print(f"✓ 统计功能正常: {summary['inference_count']} 次推理")
    
    # 检查统计值
    stats = summary['stats']
    assert 'odd_input_mean' in stats, "缺少奇数次输入均值"
    assert 'odd_input_std' in stats, "缺少奇数次输入标准差"
    
    print(f"✓ 统计值计算正常: 输入均值={stats['odd_input_mean'][0]:.4f}")
    
    return True


def test_memory_management():
    """测试内存管理功能"""
    print("\n=== 测试内存管理功能 ===")
    
    # 设置较小的最大记录数
    layer = CustomLinear(3, 2)
    layer.max_records = 3
    
    # 进行多次推理
    input_data = torch.randn(1, 3)
    
    for i in range(6):
        output = layer(input_data)
    
    # 验证记录数量不超过限制
    total_records = len(layer.odd_inputs) + len(layer.even_inputs)
    assert total_records <= layer.max_records, f"记录数量超过限制: {total_records} > {layer.max_records}"
    
    print(f"✓ 内存管理正常: 总记录数={total_records} <= 限制={layer.max_records}")
    
    return True


def test_control_functions():
    """测试控制功能"""
    print("\n=== 测试控制功能 ===")
    
    layer = CustomLinear(3, 2)
    
    # 测试禁用记录
    layer.enable_recording = False
    input_data = torch.randn(1, 3)
    output = layer(input_data)
    
    assert layer.inference_count == 0, "禁用记录后仍增加推理计数"
    print("✓ 禁用记录功能正常")
    
    # 测试启用记录
    layer.enable_recording = True
    output = layer(input_data)
    
    assert layer.inference_count == 1, "启用记录后未增加推理计数"
    print("✓ 启用记录功能正常")
    
    # 测试清除记录
    layer.clear_records()
    
    assert len(layer.odd_inputs) == 0, "清除记录后奇数次记录未清空"
    assert len(layer.even_inputs) == 0, "清除记录后偶数次记录未清空"
    print("✓ 清除记录功能正常")
    
    return True


def test_plotting_functions():
    """测试绘图功能"""
    print("\n=== 测试绘图功能 ===")
    
    layer = CustomLinear(3, 2)
    
    # 进行一些推理
    input_data = torch.randn(2, 3)
    for i in range(4):
        output = layer(input_data)
    
    # 测试绘图功能（不显示，只保存）
    try:
        # 创建测试目录
        test_dir = "./test_plots"
        os.makedirs(test_dir, exist_ok=True)
        
        # 测试数据分布图
        dist_path = os.path.join(test_dir, "test_distribution.png")
        layer.plot_data_distribution(save_path=dist_path, show_plot=False)
        
        if os.path.exists(dist_path):
            print("✓ 数据分布图生成成功")
        else:
            print("✗ 数据分布图生成失败")
            return False
        
        # 测试统计趋势图
        trend_path = os.path.join(test_dir, "test_trend.png")
        
        if os.path.exists(trend_path):
            print("✓ 统计趋势图生成成功")
        else:
            print("✗ 统计趋势图生成失败")
            return False
            
    except Exception as e:
        print(f"✗ 绘图功能测试失败: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("开始测试 CustomLinear 功能...")
    
    tests = [
        test_basic_functionality,
        test_odd_even_recording,
        test_statistics,
        test_memory_management,
        test_control_functions,
        test_plotting_functions
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
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！CustomLinear 功能正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 