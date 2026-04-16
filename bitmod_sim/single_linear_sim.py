import argparse
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from accelerator import Accelerator


def get_default_bitmod_config(
    batch_size: int = 1,
    cxt_len: int = 256,
    is_generation: bool = False,
    is_lossless: bool = False,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be > 0.")

    if is_generation:
        pe_array_dim = [64, 16]
        w_prec = 6.0625 if is_lossless else 3.0625
    else:
        pe_array_dim = [32, 32]
        w_prec = 6.0625 if is_lossless else 4.0625

    return {
        "IS_BIT_SERIAL": True,
        "PE_DP_SIZE": 4,
        "PE_ENERGY": 0.56,
        "PE_AREA": 1507.7,
        "PE_ARRAY_DIM": pe_array_dim,
        "I_PREC": 16,
        "KV_PREC": 8,
        "W_PREC": w_prec,
        "BATCH_SIZE": batch_size,
        "CXT_LEN": cxt_len,
        "IS_GENERATION": is_generation,
    }


def _shape_tuple(tensor_or_shape: Any, tensor_name: str) -> Tuple[int, ...]:
    if hasattr(tensor_or_shape, "shape"):
        shape = tuple(int(dim) for dim in tensor_or_shape.shape)
    elif isinstance(tensor_or_shape, Sequence):
        shape = tuple(int(dim) for dim in tensor_or_shape)
    else:
        raise TypeError(
            f"{tensor_name} must provide a .shape attribute or be a shape-like sequence."
        )

    if len(shape) == 0:
        raise ValueError(f"{tensor_name} must have at least one dimension.")
    return shape


def _ensure_supported_runtime() -> None:
    if os.name != "nt":
        return

    cacti_binary = os.path.join(os.path.dirname(__file__), "mem", "cacti", "cacti")
    if not os.path.exists(cacti_binary):
        return

    with open(cacti_binary, "rb") as handle:
        magic = handle.read(4)

    if magic == b"\x7fELF":
        raise RuntimeError(
            "The repository's CACTI binary is Linux ELF. Full latency/energy simulation "
            "requires a Linux environment; no Windows compatibility changes were made."
        )


class SingleLinearSimulator(Accelerator):
    """
    Simulate one linear layer with the same PE-array / memory model as `Accelerator`.

    Assumed math contract:
        y = x @ w^T (+ bias)

    Shape convention:
        x: [K], [B, K], or [B, T, K]
        w: [N, K]
        bias: [N] (optional)

    The simulator maps shapes to the existing FC/GEMM abstraction used by
    `Accelerator`:
        [K]       -> batch_size=1, num_token=1, M=1,   K=K, N=N
        [B, K]    -> batch_size=B, num_token=1, M=B,   K=K, N=N
        [B, T, K] -> batch_size=B, num_token=T, M=B*T, K=K, N=N

    To mirror the repository's generation-mode convention from `test_bitmod`,
    pass x as [1, batch_size, K]. This keeps the same [batch, token, hidden]
    internal layout that `Accelerator` uses for profiled linear layers.

    Bias is shape-checked but ignored in latency/energy because the current
    model-level simulator only accounts for the GEMM dataflow.
    """

    def __init__(
        self,
        x: Any,
        w: Any,
        bias: Optional[Any] = None,
        i_prec: int = 16,
        kv_prec: int = 8,
        w_prec: int = 8,
        batch_size: int = 1,
        is_bit_serial: bool = False,
        pe_dp_size: int = 1,
        pe_energy: float = 0,
        pe_area: float = 0,
        pe_array_dim: Sequence[int] = (),
        init_mem: bool = True,
        cxt_len: int = 256,
        is_generation: bool = False,
        layer_name: str = "single_linear",
    ):
        if pe_energy == 0:
            raise ValueError("ERROR! You must provide the energy cost of a PE.")
        if len(pe_array_dim) != 2:
            raise ValueError(
                f"ERROR! The dimension of PE array must be 2. But you gave {len(pe_array_dim)}."
            )

        # `Accelerator.__init__` is tied to model_shape_config/*.pickle.
        # For single-layer simulation we initialize the same hardware state
        # manually, then reuse the inherited tile/cycle/energy methods.
        self.model_name = layer_name
        self.i_prec = i_prec
        self.kv_prec = kv_prec
        self.w_prec = w_prec
        self.batch_size = batch_size
        self.is_bit_serial = is_bit_serial

        self.pe_latency = {}
        if is_bit_serial:
            self.pe_latency["attn"] = math.ceil(math.floor(kv_prec) / 2)
            self.pe_latency["linear"] = math.ceil(math.floor(w_prec) / 2)
        else:
            self.pe_latency["attn"] = 1
            self.pe_latency["linear"] = 1
        self.pe_dp_size = pe_dp_size
        self.total_pe_count = math.prod(pe_array_dim)
        self.pe_energy = pe_energy * self.PR_SCALING
        self.pe_area = pe_area * self.PR_SCALING
        self.pe_array_area = pe_area * self.total_pe_count
        self.pe_array_dim = {"h": pe_array_dim[0], "w": pe_array_dim[1]}
        self.cxt_len = cxt_len
        self.is_generation = is_generation

        (
            self.input_shape,
            self.weight_shape,
            self.output_shape,
            self.gemm_shape,
        ) = self._build_single_linear_profile(x, w, bias, layer_name)

        self.cycle_compute = None
        self.mem_initialized = False
        if init_mem:
            _ensure_supported_runtime()
            self._init_mem()
            self._check_layer_mem_size()
            self._calc_num_mem_refetch()
            self.mem_initialized = True

    def _build_single_linear_profile(
        self,
        x: Any,
        w: Any,
        bias: Optional[Any],
        layer_name: str,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Dict[str, int]]:
        x_shape = _shape_tuple(x, "x")
        w_shape = _shape_tuple(w, "w")
        if len(w_shape) != 2:
            raise ValueError(f"w must have shape [N, K], but got {w_shape}.")
        if len(x_shape) not in (1, 2, 3):
            raise ValueError(
                f"x must have shape [K], [B, K], or [B, T, K], but got {x_shape}."
            )

        out_features, in_features = w_shape
        if x_shape[-1] != in_features:
            raise ValueError(
                f"x and w are incompatible: x last dim is {x_shape[-1]}, w expects K={in_features}."
            )

        if len(x_shape) == 1:
            batch_dim = 1
            token_dim = 1
            user_output_shape = (out_features,)
        elif len(x_shape) == 2:
            batch_dim = x_shape[0]
            token_dim = 1
            user_output_shape = (batch_dim, out_features)
        else:
            batch_dim = x_shape[0]
            token_dim = x_shape[1]
            user_output_shape = (batch_dim, token_dim, out_features)

        if bias is not None:
            bias_shape = _shape_tuple(bias, "bias")
            if bias_shape != (out_features,):
                raise ValueError(
                    f"bias must have shape [{out_features}], but got {bias_shape}."
                )

        weight_dim = [1, out_features, in_features]
        input_dim = [batch_dim, token_dim, in_features]
        output_dim = [batch_dim, token_dim, out_features]

        self.weight_dim = {layer_name: weight_dim}
        self.input_dim = {layer_name: input_dim}
        self.output_dim = {layer_name: output_dim}
        self.layer_name_list = [layer_name]

        return (
            x_shape,
            w_shape,
            user_output_shape,
            {
                "m": batch_dim * token_dim,
                "n": out_features,
                "k": in_features,
                "batch_size": batch_dim,
                "num_token": token_dim,
            },
        )

    def simulate(self) -> Dict[str, Any]:
        if not self.mem_initialized:
            raise RuntimeError(
                "Memory is not initialized. Set init_mem=True to simulate full latency/energy."
            )

        cycle_compute, cycle_total = self.calc_cycle()
        compute_energy = self.calc_compute_energy()
        sram_rd_energy = self.calc_sram_rd_energy()
        sram_wr_energy = self.calc_sram_wr_energy()
        dram_energy = self.calc_dram_energy()
        layer_name = self.layer_name_list[0]

        return {
            "layer_name": layer_name,
            "x_shape": self.input_shape,
            "w_shape": self.weight_shape,
            "y_shape": self.output_shape,
            "gemm_shape": self.gemm_shape,
            "tile_count": self._calc_tile_fc(
                self.weight_dim[layer_name], self.output_dim[layer_name]
            ),
            "num_mem_refetch": {
                "weight": self._layer_mem_refetch[layer_name][0],
                "input": self._layer_mem_refetch[layer_name][1],
            },
            "memory_bytes": {
                "weight": self._w_mem_required[layer_name],
                "input": self._i_mem_required[layer_name],
                "output": self._o_mem_required[layer_name],
            },
            "cycle": {
                "compute": cycle_compute,
                "dram": self._layer_cycle_dram[layer_name],
                "total": cycle_total,
            },
            "energy_pj": {
                "compute": compute_energy,
                "sram_rd": sram_rd_energy,
                "sram_wr": sram_wr_energy,
                "dram": dram_energy,
                "onchip": compute_energy + sram_rd_energy + sram_wr_energy,
                "total": compute_energy + sram_rd_energy + sram_wr_energy + dram_energy,
            },
            "area_mm2": {
                "pe_array": self.pe_array_area / 1e6,
                "weight_buffer": self.w_sram.area,
                "input_buffer": self.i_sram.area,
            },
        }


def _print_summary(result: Dict[str, Any]) -> None:
    energy_uJ = {name: value / 1e6 for name, value in result["energy_pj"].items()}

    print(f'layer: {result["layer_name"]}')
    print(
        "shape: "
        f'x={result["x_shape"]}, '
        f'w={result["w_shape"]}, '
        f'y={result["y_shape"]}, '
        f'gemm(M,N,K)=({result["gemm_shape"]["m"]}, {result["gemm_shape"]["n"]}, {result["gemm_shape"]["k"]})'
    )
    print(f'total cycle:        ({result["cycle"]["compute"]}, {result["cycle"]["total"]})')
    print(f'dram cycle:         {result["cycle"]["dram"]}')
    print(f'tile count:         {result["tile_count"]}')
    print(f'num mem refetch:    {result["num_mem_refetch"]}')
    print(f'pe array area:      {result["area_mm2"]["pe_array"]} mm2')
    print(f'weight buffer area: {result["area_mm2"]["weight_buffer"]} mm2')
    print(f'input buffer area:  {result["area_mm2"]["input_buffer"]} mm2')
    print(f'dram energy:        {energy_uJ["dram"]} uJ')
    print(f'on-chip energy:     {energy_uJ["onchip"]} uJ')
    print(f'total energy:       {energy_uJ["total"]} uJ')


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_generation", action="store_true", help="If enabled, then evaluate generation mode.")
    parser.add_argument("--is_lossless", action="store_true", help="If enabled, use lossless BitMoD precision defaults.")
    parser.add_argument("--batch_size", type=int, default=1, help="The input batch size used by the default example.")
    parser.add_argument("--cxt_len", type=int, default=256, help="The context length used by the default example.")
    parser.add_argument("--in_features", type=int, default=4096, help="The K dimension of the example linear.")
    parser.add_argument("--out_features", type=int, default=11008, help="The N dimension of the example linear.")
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    bitmod_cfg = get_default_bitmod_config(
        batch_size=args.batch_size,
        cxt_len=args.cxt_len,
        is_generation=args.is_generation,
        is_lossless=args.is_lossless,
    )

    if args.is_generation:
        x = (1, bitmod_cfg["BATCH_SIZE"], args.in_features)
    else:
        x = (bitmod_cfg["BATCH_SIZE"], bitmod_cfg["CXT_LEN"], args.in_features)
    w = (args.out_features, args.in_features)

    sim = SingleLinearSimulator(
        x=x,
        w=w,
        i_prec=bitmod_cfg["I_PREC"],
        kv_prec=bitmod_cfg["KV_PREC"],
        w_prec=bitmod_cfg["W_PREC"],
        batch_size=bitmod_cfg["BATCH_SIZE"],
        is_bit_serial=bitmod_cfg["IS_BIT_SERIAL"],
        pe_dp_size=bitmod_cfg["PE_DP_SIZE"],
        pe_energy=bitmod_cfg["PE_ENERGY"],
        pe_area=bitmod_cfg["PE_AREA"],
        pe_array_dim=bitmod_cfg["PE_ARRAY_DIM"],
        cxt_len=bitmod_cfg["CXT_LEN"],
        is_generation=bitmod_cfg["IS_GENERATION"],
    )
    _print_summary(sim.simulate())