import torch
import inspect
from mx.specs import finalize_mx_specs

# 1) 首选函数式 API；若无则回退到模块类 API
try:
    from mx.linear import linear as mx_linear_fn
    mx_linear_cls = None
except Exception:
    mx_linear_fn = None
    from mx.linear import Linear as mx_linear_cls

# 2) MX 配置：开启 INT2，关闭 bfloat 逐元素量化，方便对齐手算
mx_specs = {
    "a_elem_format": "int2",
    "w_elem_format": "int2",
    "scale_bits": 8,           # 共享指数位宽（常用 8）
    "block_size": 4,           # 为了好算，正好一块含 4 个元素
    "shared_exp_method": "max",
    "bfloat": 0,               # 逐元素量化关闭（避免额外截断）
    "custom_cuda": False,      # 先走 Python 路；等价正确后再可改 True 测 CUDA
    "round": "nearest",
    "round_mx_output": "nearest",
    "mx_flush_fp32_subnorms": False,
}
mx_specs = finalize_mx_specs(mx_specs)

# 3) 构造一个“可手算”的输入/权重
#    对于 block_size=4，输入块归一化后量化为 [0,0,+1,-1] → 反量化 [0,0,+2,-2]
#    权重同理 → [0,0,-2,+2]
#    点积: 2*(-2) + (-2)*(+2) = -4 - 4 = -8
x = torch.tensor([[0.3, -0.5, 1.2, -3.7]], dtype=torch.float32)     # shape [1, 4]
w = torch.tensor([[0.2,  0.6, -1.1, 3.4]], dtype=torch.float32)     # shape [1, 4]
expected = torch.tensor([[-8.0]], dtype=torch.float32)

# 4) 调用 mx.linear（函数或类，自动适配）
def call_linear_fn(fn, x, w, mx_specs):
    sig = inspect.signature(fn)
    kwargs = {}
    if "bias" in sig.parameters: kwargs["bias"] = None
    if "mx_specs" in sig.parameters: kwargs["mx_specs"] = mx_specs
    if "prequantized_weights" in sig.parameters: kwargs["prequantized_weights"] = False
    if "name" in sig.parameters: kwargs["name"] = "test_int2"
    return fn(x, w, **kwargs)

if mx_linear_fn is not None:
    y = call_linear_fn(mx_linear_fn, x, w, mx_specs)
else:
    layer = mx_linear_cls(in_features=4, out_features=1, bias=False, mx_specs=mx_specs)
    with torch.no_grad():
        layer.weight.copy_(w)
    y = layer(x)

print("y =", y)
print("expected ≈", expected)
print("abs diff =", (y - expected).abs())

# 断言一致（允许极小浮点误差）
assert torch.allclose(y, expected, atol=1e-6), "INT2 quantized linear mismatch!"
print("✅ INT2 linear test passed.")

