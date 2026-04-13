import argparse
import math
from typing import Any, Dict, Optional, Sequence

from single_linear_sim import (
    SingleLinearSimulator,
    _build_argparser as _build_base_argparser,
    _ensure_supported_runtime,
    _print_summary as _print_base_summary,
    get_default_bitmod_config,
)


DEFAULT_OUTPUT_PRECISION_BITS = 16
DEFAULT_BASE_ACTIVATION_PRECISION_BITS = 16
DEFAULT_BASE_WEIGHT_PRECISION_BITS = 16


def get_default_bitmodbb_config(
    batch_size: int = 1,
    cxt_len: int = 256,
    is_generation: bool = False,
    is_lossless: bool = False,
    output_prec: float = DEFAULT_OUTPUT_PRECISION_BITS,
) -> Dict[str, Any]:
    config = get_default_bitmod_config(
        batch_size=batch_size,
        cxt_len=cxt_len,
        is_generation=is_generation,
        is_lossless=is_lossless,
    )
    config["OUTPUT_PREC"] = output_prec
    config["BASE_ACTIVATION_PREC"] = DEFAULT_BASE_ACTIVATION_PRECISION_BITS
    config["BASE_WEIGHT_PREC"] = DEFAULT_BASE_WEIGHT_PRECISION_BITS
    return config


class SingleLinearSimulatorBB(SingleLinearSimulator):
    """
    A `single_linear_sim` variant with two explicit modeling changes:

    1. Output precision is modeled separately from input precision. By default
       the output is bf16 (16 bits), which better matches mx_linear-style use.
    2. In non-bit-serial mode, lower precisions increase effective throughput
       using a simple BitFusion-like packing model around a base 8b x 8b FU.

    Modeling assumption for non-bit-serial mode:
        One FU is fully occupied by one (base_activation_prec x base_weight_prec)
        MAC per cycle. If runtime precisions are smaller, multiple independent
        MAC lanes can be packed into the same FU proportionally. If runtime
        precisions are larger, the same work consumes more FU capacity and
        throughput drops proportionally.

    This is intentionally a coarse throughput model. It does not attempt to
    capture packing fragmentation, alignment losses, or array remapping limits.
    """

    def __init__(
        self,
        x: Any,
        w: Any,
        bias: Optional[Any] = None,
        i_prec: float = 16,
        kv_prec: float = 8,
        w_prec: float = 8,
        output_prec: float = DEFAULT_OUTPUT_PRECISION_BITS,
        batch_size: int = 1,
        is_bit_serial: bool = False,
        pe_dp_size: int = 1,  # 考虑到这个以bit为level， 一个FU处理的是8bit*8bit的GEMM， 按照一个16*16为一个点积，本来应该是0.25；我们这里按照一个FU有8*8个BB来仿。
        pe_energy: float = 0,
        pe_area: float = 0,
        pe_array_dim: Sequence[int] = (),
        init_mem: bool = True,
        cxt_len: int = 256,
        is_generation: bool = False,
        layer_name: str = "single_linear",
        base_activation_prec: float = DEFAULT_BASE_ACTIVATION_PRECISION_BITS,
        base_weight_prec: float = DEFAULT_BASE_WEIGHT_PRECISION_BITS,
    ):
        self.output_prec = self._validate_positive_precision(output_prec, "output_prec")
        self.base_activation_prec = self._validate_positive_precision(
            base_activation_prec, "base_activation_prec"
        )
        self.base_weight_prec = self._validate_positive_precision(
            base_weight_prec, "base_weight_prec"
        )
        self._layer_effective_parallelism = {}

        super().__init__(
            x=x,
            w=w,
            bias=bias,
            i_prec=i_prec,
            kv_prec=kv_prec,
            w_prec=w_prec,
            batch_size=batch_size,
            is_bit_serial=is_bit_serial,
            pe_dp_size=pe_dp_size,
            pe_energy=pe_energy,
            pe_area=pe_area,
            pe_array_dim=pe_array_dim,
            init_mem=False,
            cxt_len=cxt_len,
            is_generation=is_generation,
            layer_name=layer_name,
        )

        self.mem_initialized = False
        if init_mem:
            _ensure_supported_runtime()
            self._init_mem()
            self._check_layer_mem_size()
            self._calc_num_mem_refetch()
            self.mem_initialized = True

    @staticmethod
    def _validate_positive_precision(precision: float, name: str) -> float:
        precision = float(precision)
        if precision <= 0:
            raise ValueError(f"{name} must be > 0, but got {precision}.")
        return precision

    def _get_weight_precision(self, layer_name: str) -> float:
        if ("attn_qk" in layer_name) or ("attn_v" in layer_name):
            return float(self.kv_prec)
        return float(self.w_prec)

    def _get_output_precision(self, layer_name: str) -> float:
        del layer_name
        return float(self.output_prec)

    def get_precision_speedup(self, activation_prec: float, weight_prec: float) -> float:
        """
        Return the idealized non-bit-serial throughput scaling relative to a
        base FU that performs one (base_activation_prec x base_weight_prec) MAC
        per cycle.

        Example with the default base (8b x 8b):
            8 x 8 -> 1x
            8 x 4 -> 2x
            4 x 4 -> 4x
            2 x 2 -> 16x

        For precisions above the base, the returned factor is < 1, meaning the
        same work takes proportionally more cycles.
        """

        activation_prec = self._validate_positive_precision(
            activation_prec, "activation_prec"
        )
        weight_prec = self._validate_positive_precision(weight_prec, "weight_prec")

        activation_lane_scaling = self.base_activation_prec / activation_prec
        weight_lane_scaling = self.base_weight_prec / weight_prec
        return activation_lane_scaling * weight_lane_scaling

    def get_effective_parallelism(self, layer_name: str) -> float:
        if self.is_bit_serial:
            return 1.0
        return self.get_precision_speedup(
            activation_prec=float(self.i_prec),
            weight_prec=self._get_weight_precision(layer_name),
        )

    def _calc_compute_cycle(self):
        self._layer_cycle_compute = {}
        self._layer_effective_parallelism = {}
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            if ("attn_qk" in name) or ("attn_v" in name):
                pe_latency = self.pe_latency["attn"]
            else:
                pe_latency = self.pe_latency["linear"]

            if w_dim is None:
                continue

            tile_layer = self._calc_tile_fc(w_dim, o_dim)
            effective_parallelism = self.get_effective_parallelism(name)
            cycle_layer_compute = math.ceil(tile_layer * pe_latency / effective_parallelism)

            self._layer_effective_parallelism[name] = effective_parallelism
            self._layer_cycle_compute[name] = max(1, int(cycle_layer_compute))

    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}

        for name in self.layer_name_list:
            i_prec = float(self.i_prec)
            o_prec = self._get_output_precision(name)
            w_prec = self._get_weight_precision(name)

            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]

            batch_kv, cout_w, cin_w = w_dim
            batch_size_in, num_token_in, cin_i = i_dim
            batch_size_out, num_token_out, cin_o = o_dim

            assert cin_w == cin_i, (
                f"The last dimension of weight and input matrices, {cin_w} and {cin_i}, "
                "do not match."
            )
            assert cout_w == cin_o, (
                f"The output dimension of weight and output matrices, {cout_w} and {cin_o}, "
                "do not match."
            )
            assert num_token_in == num_token_out, (
                f"The num_token of input and output matrices, {num_token_in} and {num_token_out}, "
                "do not match."
            )
            assert batch_size_in == batch_size_out, (
                f"The batch_size of input and output matrices, {batch_size_in} and {batch_size_out}, "
                "do not match."
            )

            self._w_mem_required[name] = math.ceil(cin_w * w_prec / 8) * cout_w * batch_kv
            self._i_mem_required[name] = (
                math.ceil(cin_i * i_prec / 8) * num_token_in * batch_size_in
            )
            self._o_mem_required[name] = (
                math.ceil(cin_o * o_prec / 8) * num_token_out * batch_size_out
            )

    def _calc_sram_wr_energy_fc(self, layer_name):
        w_dim = self.weight_dim[layer_name]
        i_dim = self.input_dim[layer_name]
        o_dim = self.output_dim[layer_name]

        i_prec = float(self.i_prec)
        o_prec = self._get_output_precision(layer_name)
        w_prec = self._get_weight_precision(layer_name)

        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        batch_kv, cout_w, cin_w = w_dim
        batch_size_in, num_token_in, cin_i = i_dim
        batch_size_out, num_token_out, cin_o = o_dim

        num_w_sram_wr = math.ceil(cin_w * w_prec / w_sram_min_wr_bw) * cout_w * batch_kv
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w
        num_i_sram_wr = (
            math.ceil(cin_i * i_prec / i_sram_min_wr_bw) * num_token_in * batch_size_in
        )
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i
        num_o_sram_wr = (
            math.ceil(cin_o * o_prec / i_sram_min_wr_bw) * num_token_out * batch_size_out
        )
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        return energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr

    def collect_modeling_snapshot(self) -> Dict[str, Any]:
        """
        Gather a small set of stats that do not require SRAM/DRAM initialization.
        This is useful for quick sanity checks on precision scaling trends.
        """

        self._check_layer_mem_size()
        self._calc_compute_cycle()
        layer_name = self.layer_name_list[0]
        return {
            "layer_name": layer_name,
            "compute_cycles": self._layer_cycle_compute[layer_name],
            "tile_count": self._calc_tile_fc(
                self.weight_dim[layer_name], self.output_dim[layer_name]
            ),
            "precision_bits": {
                "input": float(self.i_prec),
                "weight": self._get_weight_precision(layer_name),
                "output": self._get_output_precision(layer_name),
            },
            "effective_parallelism": self._layer_effective_parallelism[layer_name],
            "memory_bytes": {
                "weight": self._w_mem_required[layer_name],
                "input": self._i_mem_required[layer_name],
                "output": self._o_mem_required[layer_name],
            },
        }

    def simulate(self) -> Dict[str, Any]:
        result = super().simulate()
        layer_name = self.layer_name_list[0]
        result["precision_bits"] = {
            "input": float(self.i_prec),
            "weight": self._get_weight_precision(layer_name),
            "output": self._get_output_precision(layer_name),
        }
        result["compute_model"] = {
            "is_bit_serial": self.is_bit_serial,
            "base_activation_prec": self.base_activation_prec,
            "base_weight_prec": self.base_weight_prec,
            "effective_parallelism": self._layer_effective_parallelism.get(
                layer_name, self.get_effective_parallelism(layer_name)
            ),
        }
        return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = _build_base_argparser()
    parser.description = (
        "single_linear_simBB: explicit output precision plus BitFusion-like "
        "precision-sensitive throughput for non-bit-serial mode."
    )
    parser.add_argument(
        "--output_prec",
        type=float,
        default=DEFAULT_OUTPUT_PRECISION_BITS,
        help="Output precision in bits. Default is bf16 = 16.",
    )
    parser.add_argument(
        "--base_activation_prec",
        type=float,
        default=DEFAULT_BASE_ACTIVATION_PRECISION_BITS,
        help="Base activation precision used by the non-bit-serial FU model.",
    )
    parser.add_argument(
        "--base_weight_prec",
        type=float,
        default=DEFAULT_BASE_WEIGHT_PRECISION_BITS,
        help="Base weight precision used by the non-bit-serial FU model.",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run a minimal trend check without initializing CACTI memories.",
    )
    return parser


def _print_summary(result: Dict[str, Any]) -> None:
    _print_base_summary(result)
    precision_bits = result.get("precision_bits", {})
    if precision_bits:
        print(f"precision bits:    {precision_bits}")

    compute_model = result.get("compute_model", {})
    if compute_model:
        print(f"compute model:     {compute_model}")


def run_minimal_sanity_check() -> Dict[str, Dict[str, Any]]:
    common_kwargs = {
        "x": (1, 64, 128),
        "w": (256, 128),
        "output_prec": DEFAULT_OUTPUT_PRECISION_BITS,
        "batch_size": 1,
        "is_bit_serial": False,
        "pe_dp_size": 4,
        "pe_energy": 0.56,
        "pe_area": 1507.7,
        "pe_array_dim": (32, 32),
        "init_mem": False,
        "cxt_len": 64,
        "is_generation": False,
        "base_activation_prec": DEFAULT_BASE_ACTIVATION_PRECISION_BITS,
        "base_weight_prec": DEFAULT_BASE_WEIGHT_PRECISION_BITS,
    }

    sim_8x8 = SingleLinearSimulatorBB(i_prec=8, w_prec=8, **common_kwargs)
    sim_4x4 = SingleLinearSimulatorBB(i_prec=4, w_prec=4, **common_kwargs)

    return {
        "8x8": sim_8x8.collect_modeling_snapshot(),
        "4x4": sim_4x4.collect_modeling_snapshot(),
    }


def _print_sanity_check(result: Dict[str, Dict[str, Any]]) -> None:
    stat_8x8 = result["8x8"]
    stat_4x4 = result["4x4"]

    for label in ("8x8", "4x4"):
        stat = result[label]
        print(f"{label} precision snapshot:")
        print(f"  compute cycles:         {stat['compute_cycles']}")
        print(f"  effective parallelism:  {stat['effective_parallelism']}")
        print(f"  memory bytes:           {stat['memory_bytes']}")

    print("sanity trend:")
    print(f"  8x8 slower than 4x4:    {stat_8x8['compute_cycles'] > stat_4x4['compute_cycles']}")
    print(
        "  output bytes stay bf16: "
        f"{stat_8x8['memory_bytes']['output']} == {stat_4x4['memory_bytes']['output']}"
    )


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    if args.sanity_check:
        _print_sanity_check(run_minimal_sanity_check())
    else:
        bitmod_cfg = get_default_bitmodbb_config(
            batch_size=args.batch_size,
            cxt_len=args.cxt_len,
            is_generation=args.is_generation,
            is_lossless=args.is_lossless,
            output_prec=args.output_prec,
        )

        if args.is_generation:
            x = (1, bitmod_cfg["BATCH_SIZE"], args.in_features)
        else:
            x = (bitmod_cfg["BATCH_SIZE"], bitmod_cfg["CXT_LEN"], args.in_features)
        w = (args.out_features, args.in_features)

        sim = SingleLinearSimulatorBB(
            x=x,
            w=w,
            i_prec=bitmod_cfg["I_PREC"],
            kv_prec=bitmod_cfg["KV_PREC"],
            w_prec=bitmod_cfg["W_PREC"],
            output_prec=bitmod_cfg["OUTPUT_PREC"],
            batch_size=bitmod_cfg["BATCH_SIZE"],
            is_bit_serial=bitmod_cfg["IS_BIT_SERIAL"],
            pe_dp_size=bitmod_cfg["PE_DP_SIZE"],
            pe_energy=bitmod_cfg["PE_ENERGY"],
            pe_area=bitmod_cfg["PE_AREA"],
            pe_array_dim=bitmod_cfg["PE_ARRAY_DIM"],
            cxt_len=bitmod_cfg["CXT_LEN"],
            is_generation=bitmod_cfg["IS_GENERATION"],
            base_activation_prec=args.base_activation_prec,
            base_weight_prec=args.base_weight_prec,
        )
        _print_summary(sim.simulate())
