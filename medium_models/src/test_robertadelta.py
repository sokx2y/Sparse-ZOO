import copy
import math
import types

import torch
import torch.nn.functional as F
from torch import nn


# 如果你的 modeling_roberta.py 在当前目录，就直接这样 import。
# 如果它在别的目录，先 sys.path.insert(0, "你的目录")
from modeling_roberta import RobertaConfig, RobertaForCausalLM


class FakeDiffLinear(nn.Linear):
    """
    Test-only diffLinear.

    当前物理权重 = theta + eps * delta
    provider 返回 scale = -2eps
    所以 forward_delta 里的 perturbed 权重 = theta - eps * delta
    """

    def __init__(self, layer_name, in_features, out_features, bias=True, device=None, dtype=None, uv_provider=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.layer_name = layer_name
        self.uv_provider = uv_provider
        self.inference_count = 0

    def forward_delta(self, x, dx):
        self.inference_count += 1
        if dx is None:
            dx = torch.zeros_like(x)

        y = F.linear(x, self.weight, self.bias)

        weight_pert = self.weight
        bias_pert = self.bias

        if self.uv_provider is not None:
            u, v, scale = self.uv_provider(
                self.layer_name + ".weight",
                self.weight.shape,
                self.weight.device,
                self.weight.dtype,
                self.inference_count,
            )
            weight_pert = self.weight + float(scale) * (u @ v.t())

        y_pert = F.linear(x + dx, weight_pert, bias_pert)
        return y, y_pert - y


class FakeDiffEmbedding(nn.Embedding):
    """
    Test-only diffEmbedding.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        layer_name="",
        uv_provider=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.layer_name = layer_name
        self.uv_provider = uv_provider
        self.inference_count = 0

    def forward_delta(self, input_ids):
        self.inference_count += 1

        y = F.embedding(input_ids, self.weight, self.padding_idx)

        weight_pert = self.weight
        if self.uv_provider is not None:
            u, v, scale = self.uv_provider(
                self.layer_name + ".weight",
                self.weight.shape,
                self.weight.device,
                self.weight.dtype,
                self.inference_count,
            )
            weight_pert = self.weight + float(scale) * (u @ v.t())

        y_pert = F.embedding(input_ids, weight_pert, self.padding_idx)
        return y, y_pert - y


class FakeDiffLayerNorm(nn.LayerNorm):
    """
    Test-only diffLayerNorm.

    这里只扰动 LayerNorm.weight，不扰动 bias。
    如果你真实 LOZO 也扰动 bias，后面可以再加强这个 check。
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, layer_name="", z_provider=None, device=None, dtype=None):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )
        self.layer_name = layer_name
        self.z_provider = z_provider
        self.inference_count = 0

    def forward_delta(self, x, dx):
        self.inference_count += 1
        if dx is None:
            dx = torch.zeros_like(x)

        y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        weight_pert = self.weight
        bias_pert = self.bias

        if self.z_provider is not None and self.weight is not None:
            z, scale = self.z_provider(
                self.layer_name + ".weight",
                self.weight.shape,
                self.weight.device,
                self.weight.dtype,
                self.inference_count,
            )
            weight_pert = self.weight + float(scale) * z

        y_pert = F.layer_norm(x + dx, self.normalized_shape, weight_pert, bias_pert, self.eps)
        return y, y_pert - y


def make_lowrank_delta(weight, rank=2, seed=0):
    g = torch.Generator(device=weight.device)
    g.manual_seed(seed)

    out_dim, in_dim = weight.shape
    u = torch.randn(out_dim, rank, generator=g, device=weight.device, dtype=weight.dtype)
    v = torch.randn(in_dim, rank, generator=g, device=weight.device, dtype=weight.dtype)

    u = u / math.sqrt(out_dim)
    v = v / math.sqrt(in_dim)

    delta_w = u @ v.t()
    return u, v, delta_w


def make_vector_delta(vec, seed=0):
    g = torch.Generator(device=vec.device)
    g.manual_seed(seed)
    z = torch.randn(vec.shape, generator=g, device=vec.device, dtype=vec.dtype)
    z = z / math.sqrt(vec.numel())
    return z


def get_submodule(root, dotted_name):
    cur = root
    if dotted_name == "":
        return cur
    for p in dotted_name.split("."):
        cur = getattr(cur, p)
    return cur


def set_submodule(root, dotted_name, value):
    parts = dotted_name.split(".")
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def force_tie_lm_head(model):
    """
    RoBERTa LM head decoder 通常和 word_embeddings tied。
    这里强制 tie，保证显式两次 forward 和 forward_delta 处在同一语义。
    """
    model.lm_head.decoder.weight = model.roberta.embeddings.word_embeddings.weight


def collect_unique_params(model):
    """
    用 named_parameters 收集唯一参数，避免 tied weight 重复。
    """
    matrix_names = []
    vector_names = []

    for name, p in model.named_parameters():
        if p.dim() == 2:
            matrix_names.append(name)
        elif p.dim() == 1 and name.endswith(".weight"):
            vector_names.append(name)

    return matrix_names, vector_names


def perturb_model_in_place(model, matrix_deltas, vector_deltas, scale):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, item in matrix_deltas.items():
            params[name].add_(scale * item["delta_w"])
        for name, item in vector_deltas.items():
            params[name].add_(scale * item["z"])


def replace_embedding(model, module_name, uv_provider):
    src = get_submodule(model, module_name)

    dst = FakeDiffEmbedding(
        num_embeddings=src.num_embeddings,
        embedding_dim=src.embedding_dim,
        padding_idx=src.padding_idx,
        layer_name=module_name,
        uv_provider=uv_provider,
        device=src.weight.device,
        dtype=src.weight.dtype,
    )

    with torch.no_grad():
        dst.weight.copy_(src.weight)

    set_submodule(model, module_name, dst)


def replace_linear(model, module_name, uv_provider, layer_name_override=None):
    src = get_submodule(model, module_name)
    layer_name = layer_name_override if layer_name_override is not None else module_name

    dst = FakeDiffLinear(
        layer_name=layer_name,
        in_features=src.in_features,
        out_features=src.out_features,
        bias=(src.bias is not None),
        uv_provider=uv_provider,
        device=src.weight.device,
        dtype=src.weight.dtype,
    )

    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)

    set_submodule(model, module_name, dst)


def replace_layernorm(model, module_name, z_provider):
    src = get_submodule(model, module_name)

    dst = FakeDiffLayerNorm(
        normalized_shape=src.normalized_shape,
        eps=src.eps,
        elementwise_affine=src.elementwise_affine,
        layer_name=module_name,
        z_provider=z_provider,
        device=src.weight.device if src.weight is not None else None,
        dtype=src.weight.dtype if src.weight is not None else None,
    )

    with torch.no_grad():
        if src.weight is not None:
            dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)

    set_submodule(model, module_name, dst)


def build_diff_model_from_plus_model(model_plus, config, uv_provider, z_provider):
    model_fd = RobertaForCausalLM(config).eval()
    force_tie_lm_head(model_fd)

    model_fd.load_state_dict(model_plus.state_dict(), strict=True)

    # Embeddings
    replace_embedding(model_fd, "roberta.embeddings.word_embeddings", uv_provider)
    replace_embedding(model_fd, "roberta.embeddings.position_embeddings", uv_provider)
    replace_embedding(model_fd, "roberta.embeddings.token_type_embeddings", uv_provider)

    # Linear modules except lm_head.decoder, because decoder is tied to word_embeddings.
    for module_name, module in list(model_fd.named_modules()):
        if isinstance(module, nn.Linear):
            if module_name == "lm_head.decoder":
                continue
            replace_linear(model_fd, module_name, uv_provider)

    # LayerNorm modules
    for module_name, module in list(model_fd.named_modules()):
        if isinstance(module, nn.LayerNorm):
            replace_layernorm(model_fd, module_name, z_provider)

    # Replace lm_head.decoder separately.
    # tied 情况下，它应该使用 word_embeddings 的 provider 名。
    replace_linear(
        model_fd,
        "lm_head.decoder",
        uv_provider,
        layer_name_override="roberta.embeddings.word_embeddings",
    )

    return model_fd


def compute_causal_lm_loss(logits, labels, vocab_size):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return torch.nn.CrossEntropyLoss()(
        shifted_logits.view(-1, vocab_size),
        shifted_labels.view(-1),
    )


def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    eps = 1e-3
    rank = 2

    config = RobertaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        type_vocab_size=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        is_decoder=False,
        add_cross_attention=False,
    )
    config.pad_token_id = 1
    config.use_cache = False

    model_theta = RobertaForCausalLM(config).to(device=device, dtype=dtype).eval()
    force_tie_lm_head(model_theta)

    matrix_names, vector_names = collect_unique_params(model_theta)

    print("matrix params:")
    for n in matrix_names:
        print(" ", n)
    print("vector LayerNorm-like weight params:")
    for n in vector_names:
        print(" ", n)

    params = dict(model_theta.named_parameters())

    matrix_deltas = {}
    for idx, name in enumerate(matrix_names):
        p = params[name]
        u, v, delta_w = make_lowrank_delta(p, rank=rank, seed=10000 + idx)
        matrix_deltas[name] = {"u": u, "v": v, "delta_w": delta_w}

    vector_deltas = {}
    for idx, name in enumerate(vector_names):
        p = params[name]
        z = make_vector_delta(p, seed=20000 + idx)
        vector_deltas[name] = {"z": z}

    model_plus = copy.deepcopy(model_theta).eval()
    model_minus = copy.deepcopy(model_theta).eval()
    force_tie_lm_head(model_plus)
    force_tie_lm_head(model_minus)

    perturb_model_in_place(model_plus, matrix_deltas, vector_deltas, scale=+eps)
    perturb_model_in_place(model_minus, matrix_deltas, vector_deltas, scale=-eps)

    provider_hits = {}

    def uv_provider(param_name, shape, device_, dtype_, inference_count):
        provider_hits[param_name] = provider_hits.get(param_name, 0) + 1
        assert param_name in matrix_deltas, f"unexpected matrix param_name: {param_name}"

        item = matrix_deltas[param_name]
        assert tuple(shape) == tuple(item["delta_w"].shape), (
            f"shape mismatch for {param_name}: got {shape}, expected {item['delta_w'].shape}"
        )

        return (
            item["u"].to(device=device_, dtype=dtype_),
            item["v"].to(device=device_, dtype=dtype_),
            -2.0 * eps,
        )

    def z_provider(param_name, shape, device_, dtype_, inference_count):
        provider_hits[param_name] = provider_hits.get(param_name, 0) + 1
        assert param_name in vector_deltas, f"unexpected vector param_name: {param_name}"

        item = vector_deltas[param_name]
        assert tuple(shape) == tuple(item["z"].shape), (
            f"shape mismatch for {param_name}: got {shape}, expected {item['z'].shape}"
        )

        return item["z"].to(device=device_, dtype=dtype_), -2.0 * eps

    model_fd = build_diff_model_from_plus_model(model_plus, config, uv_provider, z_provider)
    model_fd = model_fd.to(device=device, dtype=dtype).eval()

    input_ids = torch.tensor(
        [
            [0, 5, 6, 7, 8, 1, 1],
            [0, 9, 10, 11, 12, 13, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    attention_mask = (input_ids != config.pad_token_id).long()
    labels = input_ids.clone()

    with torch.no_grad():
        plus_outputs = model_plus(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )

        minus_outputs = model_minus(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )

        fd_outputs = model_fd.forward_delta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )

    # RobertaForCausalLM.forward return_dict=False:
    #   [0] loss
    #   [1] logits
    loss_plus_ref = plus_outputs[0]
    logits_plus_ref = plus_outputs[1]

    loss_minus_ref = minus_outputs[0]
    logits_minus_ref = minus_outputs[1]

    # forward_delta return_dict=False 应该是：
    #   [0] loss_base
    #   [1] loss_perturbed
    #   [2] logits_base
    #   [3] logits_perturbed
    assert isinstance(fd_outputs, tuple), type(fd_outputs)
    assert len(fd_outputs) >= 4, f"forward_delta output too short: len={len(fd_outputs)}"

    loss_plus_fd = fd_outputs[0]
    loss_minus_fd = fd_outputs[1]
    logits_plus_fd = fd_outputs[2]
    logits_minus_fd = fd_outputs[3]

    # 再额外用同一套 CE loss 手算一遍，防止 tuple 误读。
    loss_plus_ref_manual = compute_causal_lm_loss(logits_plus_ref, labels, config.vocab_size)
    loss_minus_ref_manual = compute_causal_lm_loss(logits_minus_ref, labels, config.vocab_size)

    print("\n=== RobertaForCausalLM.forward_delta LOZO-style check ===")
    print("fd tuple len:", len(fd_outputs))

    print("logits plus  max_abs:", (logits_plus_fd - logits_plus_ref).abs().max().item())
    print("logits minus max_abs:", (logits_minus_fd - logits_minus_ref).abs().max().item())

    print("loss plus  fd/ref:", loss_plus_fd.item(), loss_plus_ref.item())
    print("loss minus fd/ref:", loss_minus_fd.item(), loss_minus_ref.item())

    print("loss plus  ref/manual:", loss_plus_ref.item(), loss_plus_ref_manual.item())
    print("loss minus ref/manual:", loss_minus_ref.item(), loss_minus_ref_manual.item())

    print("\nprovider hits:")
    for k in sorted(provider_hits):
        print(f"  {k}: {provider_hits[k]}")

    torch.testing.assert_close(logits_plus_fd, logits_plus_ref, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(logits_minus_fd, logits_minus_ref, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(loss_plus_fd, loss_plus_ref, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(loss_minus_fd, loss_minus_ref, atol=2e-5, rtol=2e-4)

    # tied weight 情况下，lm_head.decoder 应该走 word_embeddings 的 provider 名。
    assert "roberta.embeddings.word_embeddings.weight" in provider_hits
    assert "lm_head.decoder.weight" not in provider_hits, (
        "lm_head.decoder should use layer_name='roberta.embeddings.word_embeddings' when tied."
    )

    print("\nRobertaForCausalLM.forward_delta check OK")


if __name__ == "__main__":
    main()