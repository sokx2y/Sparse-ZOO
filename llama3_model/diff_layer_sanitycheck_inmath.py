import torch
import torch.nn.functional as F

from diff_fake_quant_mx import diffLinear, diffEmbedding, diffLlamaRMSNorm


torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

RANK = 2
SCALE = 1e-3
BIAS_SCALE = 1e-3


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def rel_err(a, b):
    num = (a.float() - b.float()).norm()
    den = b.float().norm().clamp_min(1e-12)
    return (num / den).item()


print("device =", device)
print("dtype =", dtype)


# ============================================================
# 1. diffLinear math check
# ============================================================

print("\n=== diffLinear math check ===")

in_features = 7
out_features = 5
batch = 2
seq = 3

linear = diffLinear(
    layer_name="test.linear",
    in_features=in_features,
    out_features=out_features,
    bias=True,
    device=device,
    dtype=dtype,
)

with torch.no_grad():
    linear.weight.normal_(mean=0.0, std=0.02)
    linear.bias.normal_(mean=0.0, std=0.02)

x = torch.randn(batch, seq, in_features, device=device, dtype=dtype)
dx = torch.randn_like(x) * 1e-2

u = torch.randn(out_features, RANK, device=device, dtype=dtype)
v = torch.randn(in_features, RANK, device=device, dtype=dtype)
z = torch.randn(out_features, device=device, dtype=dtype)

def uv_provider(param_name, shape, device_, dtype_, inference_count):
    assert param_name == "test.linear.weight"
    assert tuple(shape) == (out_features, in_features)
    return u.to(device_, dtype_), v.to(device_, dtype_), SCALE

def z_provider(param_name, shape, device_, dtype_, inference_count):
    assert param_name == "test.linear.bias"
    assert tuple(shape) == (out_features,)
    return z.to(device_, dtype_), BIAS_SCALE

linear.uv_provider = uv_provider
linear.z_provider = z_provider

base, diff = linear.forward_delta(x, dx)

delta_w = SCALE * (u.float() @ v.float().t())          # [out, in]
delta_b = BIAS_SCALE * z.float()                       # [out]

explicit_base = F.linear(x.float(), linear.weight.float(), linear.bias.float())
explicit_pert = F.linear(
    x.float() + dx.float(),
    linear.weight.float() + delta_w,
    linear.bias.float() + delta_b,
)

print("base vs explicit_base max_abs:", max_abs(base, explicit_base))
print("base+diff vs explicit_pert max_abs:", max_abs(base.float() + diff.float(), explicit_pert))
print("diff vs explicit_pert-base max_abs:", max_abs(diff, explicit_pert - explicit_base))
print("base+diff rel_err:", rel_err(base.float() + diff.float(), explicit_pert))

assert max_abs(base, explicit_base) < 5e-3
assert max_abs(base.float() + diff.float(), explicit_pert) < 5e-3


# ============================================================
# 2. diffEmbedding math check
# ============================================================

print("\n=== diffEmbedding math check ===")

num_embeddings = 11
embedding_dim = 6

emb = diffEmbedding(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    layer_name="test.embed",
    uv_provider=None,
    device=device,
    dtype=dtype,
)

with torch.no_grad():
    emb.weight.normal_(mean=0.0, std=0.02)

input_ids = torch.tensor([[0, 3, 5, 7], [2, 3, 10, 1]], device=device)

u_emb = torch.randn(num_embeddings, RANK, device=device, dtype=dtype)
v_emb = torch.randn(embedding_dim, RANK, device=device, dtype=dtype)

def emb_uv_provider(param_name, shape, device_, dtype_, inference_count):
    assert param_name == "test.embed.weight"
    assert tuple(shape) == (num_embeddings, embedding_dim)
    return u_emb.to(device_, dtype_), v_emb.to(device_, dtype_), SCALE

emb.uv_provider = emb_uv_provider

emb_base, emb_diff = emb.forward_delta(input_ids)

# For embedding, only selected rows matter:
# delta_embedding[input_ids] = scale * u[input_ids] @ v.T
explicit_emb_base = F.embedding(input_ids, emb.weight.float())
explicit_emb_delta = SCALE * torch.matmul(
    u_emb.float()[input_ids],
    v_emb.float().t(),
)
explicit_emb_pert = explicit_emb_base + explicit_emb_delta

print("base vs explicit_base max_abs:", max_abs(emb_base, explicit_emb_base))
print("base+diff vs explicit_pert max_abs:", max_abs(emb_base.float() + emb_diff.float(), explicit_emb_pert))
print("diff vs explicit_delta max_abs:", max_abs(emb_diff, explicit_emb_delta))
print("base+diff rel_err:", rel_err(emb_base.float() + emb_diff.float(), explicit_emb_pert))

assert max_abs(emb_base, explicit_emb_base) < 5e-3
assert max_abs(emb_base.float() + emb_diff.float(), explicit_emb_pert) < 5e-3


# ============================================================
# 3. diffLlamaRMSNorm math check
# ============================================================

print("\n=== diffLlamaRMSNorm math check ===")

hidden_size = 8
eps = 1e-6

norm = diffLlamaRMSNorm(
    hidden_size=hidden_size,
    eps=eps,
    layer_name="test.norm",
    z_provider=None,
    device=device,
    dtype=dtype,
)

with torch.no_grad():
    norm.weight.normal_(mean=1.0, std=0.02)

x_norm = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)
dx_norm = torch.randn_like(x_norm) * 1e-2

z_norm = torch.randn(hidden_size, device=device, dtype=dtype)

def norm_z_provider(param_name, shape, device_, dtype_, inference_count):
    assert param_name == "test.norm.weight"
    assert tuple(shape) == (hidden_size,)
    return z_norm.to(device_, dtype_), BIAS_SCALE

norm.z_provider = norm_z_provider

norm_base, norm_diff = norm.forward_delta(x_norm, dx_norm)

def llama_rmsnorm_ref(hidden_states, weight, eps):
    h = hidden_states.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return h * weight.float()

explicit_norm_base = llama_rmsnorm_ref(x_norm, norm.weight, eps)
explicit_norm_pert = llama_rmsnorm_ref(
    x_norm.float() + dx_norm.float(),
    norm.weight.float() + BIAS_SCALE * z_norm.float(),
    eps,
)

print("base vs explicit_base max_abs:", max_abs(norm_base, explicit_norm_base))
print("base+diff vs explicit_pert max_abs:", max_abs(norm_base.float() + norm_diff.float(), explicit_norm_pert))
print("diff vs explicit_pert-base max_abs:", max_abs(norm_diff, explicit_norm_pert - explicit_norm_base))
print("base+diff rel_err:", rel_err(norm_base.float() + norm_diff.float(), explicit_norm_pert))

assert max_abs(norm_base, explicit_norm_base) < 5e-3
assert max_abs(norm_base.float() + norm_diff.float(), explicit_norm_pert) < 5e-3


print("\n✅ diff layer math checks passed.")