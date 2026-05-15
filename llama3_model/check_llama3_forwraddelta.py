import inspect
import os
import py_compile
import sys

import torch


def check_compile():
    print("\n[1] py_compile check")
    for fname in ["run_lozo.py", "modeling_llama.py", "diff_fake_quant_mx.py"]:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname} not found in cwd={os.getcwd()}")
        py_compile.compile(fname, doraise=True)
        print(f"  OK: {fname}")


def check_imports():
    print("\n[2] import check")

    import run_lozo
    from modeling_llama import LlamaForCausalLM
    from diff_fake_quant_mx import QuantizeLlamaForLOZO

    print("  OK: import run_lozo")
    print("  OK: import LlamaForCausalLM:", LlamaForCausalLM)
    print("  OK: import QuantizeLlamaForLOZO:", QuantizeLlamaForLOZO)

    return run_lozo, LlamaForCausalLM, QuantizeLlamaForLOZO


def check_run_lozo_source(run_lozo):
    print("\n[3] run_lozo source check")

    src = inspect.getsource(run_lozo)

    checks = {
        "imports LlamaForCausalLM": "LlamaForCausalLM" in src,
        "imports/calls QuantizeLlamaForLOZO": "QuantizeLlamaForLOZO" in src,
        "has llama model_type branch": 'model_type == "llama"' in src or "model_type == 'llama'" in src,
        "has is_llama logic": "is_llama" in src,
    }

    for name, ok in checks.items():
        print(f"  {name}: {ok}")

    if not all(checks.values()):
        print("\n  WARNING: Some source-level checks failed.")
        print("  This does not always mean the code is wrong, but please inspect run_lozo.py manually.")


def make_zero_uv_provider(rank=2, eps=1e-3):
    def uv_provider(param_name, shape, device, dtype, inference_count):
        if len(shape) != 2:
            raise ValueError(f"uv_provider expected 2D shape for {param_name}, got {shape}")

        out_dim, in_dim = int(shape[0]), int(shape[1])
        u = torch.zeros(out_dim, rank, device=device, dtype=dtype)
        v = torch.zeros(in_dim, rank, device=device, dtype=dtype)

        # In real LOZO this is -2 * zo_eps.
        # Here zeros make the perturbed branch numerically identical,
        # which is enough to test wiring and forward_delta execution.
        scale = -2.0 * eps
        return u, v, scale

    return uv_provider


def make_zero_z_provider(eps=1e-3):
    def z_provider(param_name, shape, device, dtype, inference_count):
        z = torch.zeros(shape, device=device, dtype=dtype)
        scale = -2.0 * eps
        return z, scale

    return z_provider


def call_quantize_llama(QuantizeLlamaForLOZO, model):
    print("\n[4] QuantizeLlamaForLOZO call check")

    sig = inspect.signature(QuantizeLlamaForLOZO)
    print("  signature:", sig)

    candidate_kwargs = {
        "model": model,
        "mx_w_elem_format": None,
        "mx_a_elem_format": None,
        "mx_diffw_elem_format": None,
        "mx_diffa_elem_format": None,
        "enable_w": True,
        "enable_x": True,
        "enable_diffx": True,
        "enable_diffw": True,
        "uv_provider": make_zero_uv_provider(),
        "z_provider": make_zero_z_provider(),
    }

    kwargs = {
        k: v
        for k, v in candidate_kwargs.items()
        if k in sig.parameters
    }

    print("  passing kwargs:", sorted(kwargs.keys()))
    QuantizeLlamaForLOZO(**kwargs)
    print("  OK: QuantizeLlamaForLOZO returned")


def check_module_replacement(model):
    print("\n[5] module replacement check")

    checks = {
        "model.forward_delta": hasattr(model, "forward_delta"),
        "model.model.forward_delta": hasattr(model.model, "forward_delta"),
        "layer0.forward_delta": hasattr(model.model.layers[0], "forward_delta"),
        "embed_tokens.forward_delta": hasattr(model.model.embed_tokens, "forward_delta"),
        "q_proj.forward_delta": hasattr(model.model.layers[0].self_attn.q_proj, "forward_delta"),
        "k_proj.forward_delta": hasattr(model.model.layers[0].self_attn.k_proj, "forward_delta"),
        "v_proj.forward_delta": hasattr(model.model.layers[0].self_attn.v_proj, "forward_delta"),
        "o_proj.forward_delta": hasattr(model.model.layers[0].self_attn.o_proj, "forward_delta"),
        "gate_proj.forward_delta": hasattr(model.model.layers[0].mlp.gate_proj, "forward_delta"),
        "up_proj.forward_delta": hasattr(model.model.layers[0].mlp.up_proj, "forward_delta"),
        "down_proj.forward_delta": hasattr(model.model.layers[0].mlp.down_proj, "forward_delta"),
        "input_layernorm.forward_delta": hasattr(model.model.layers[0].input_layernorm, "forward_delta"),
        "post_attention_layernorm.forward_delta": hasattr(model.model.layers[0].post_attention_layernorm, "forward_delta"),
        "final norm.forward_delta": hasattr(model.model.norm, "forward_delta"),
        "lm_head.forward_delta": hasattr(model.lm_head, "forward_delta"),
    }

    for name, ok in checks.items():
        print(f"  {name}: {ok}")

    assert all(checks.values()), "Some modules do not have forward_delta after QuantizeLlamaForLOZO"

    print("  OK: all expected modules have forward_delta")


def check_tied_weight(model):
    print("\n[6] tied weight check")

    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print("  lm_head.weight tied with embed_tokens.weight:", tied)

    if not tied:
        print("  WARNING: lm_head and embed_tokens are not tied.")
        print("  If Step 1 intentionally preserves tying after replacement, this should be True.")


def check_tiny_forward_delta(model):
    print("\n[7] tiny forward_delta runtime check")

    model.eval()

    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9, 10],
        ],
        device=device,
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model.forward_delta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=False,
        )

    assert isinstance(outputs, tuple), type(outputs)
    assert len(outputs) >= 4, f"Expected at least 4 outputs, got len={len(outputs)}"

    loss_plus = outputs[0]
    loss_minus = outputs[1]
    logits_plus = outputs[2]
    logits_minus = outputs[3]

    print("  len(outputs):", len(outputs))
    print("  loss_plus shape:", tuple(loss_plus.shape))
    print("  loss_minus shape:", tuple(loss_minus.shape))
    print("  logits_plus shape:", tuple(logits_plus.shape))
    print("  logits_minus shape:", tuple(logits_minus.shape))
    print("  logits max_abs plus-minus:", (logits_plus - logits_minus).abs().max().item())

    assert loss_plus.dim() == 0
    assert loss_minus.dim() == 0
    assert logits_plus.shape == logits_minus.shape

    print("  OK: tiny forward_delta runtime check passed")


def main():
    print("cwd:", os.getcwd())
    print("python:", sys.executable)
    print("torch:", torch.__version__)

    check_compile()
    run_lozo, LlamaForCausalLM, QuantizeLlamaForLOZO = check_imports()
    check_run_lozo_source(run_lozo)

    print("\n[4-pre] build tiny local LlamaForCausalLM")
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    config.pad_token_id = 0
    config.use_cache = False
    config._attn_implementation = "eager"

    model = LlamaForCausalLM(config)
    model.eval()

    print("  model_type:", model.config.model_type)
    print("  use_cache:", model.config.use_cache)
    print("  has model.forward_delta before quantize:", hasattr(model, "forward_delta"))
    print("  q_proj type before:", type(model.model.layers[0].self_attn.q_proj))

    call_quantize_llama(QuantizeLlamaForLOZO, model)

    print("  q_proj type after:", type(model.model.layers[0].self_attn.q_proj))
    print("  embed_tokens type after:", type(model.model.embed_tokens))
    print("  lm_head type after:", type(model.lm_head))

    check_module_replacement(model)
    check_tied_weight(model)
    check_tiny_forward_delta(model)

    print("\nALL STEP3 LLAMA WIRING CHECKS PASSED")


if __name__ == "__main__":
    main()