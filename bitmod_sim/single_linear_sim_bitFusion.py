import argparse
import math
from typing import Any, Dict, Optional, Sequence

from single_linear_sim import (
    SingleLinearSimulator,
    _build_argparser as _build_base_argparser,
    _ensure_supported_runtime,
    get_default_bitmod_config,
)


DEFAULT_OUTPUT_PRECISION_BITS = 16
DEFAULT_BASE_ACTIVATION_PRECISION_BITS = 8
DEFAULT_BASE_WEIGHT_PRECISION_BITS = 8
DEFAULT_DRAM_EFFECTIVE_BW_BITS_PER_CYCLE = 256.0
DEFAULT_DRAM_ENERGY_PJ_PER_BIT = 18.75

DRAM_PROFILE_DATA_RATE_MTPS = {
    "ddr4-3200": 3200.0,
    "ddr5-4800": 4800.0,
    "ddr5-5600": 5600.0,
    "ddr5-6400": 6400.0,
}


def resolve_dram_effective_bw_bits_per_cycle(
    profile: str,
    data_rate_mtps: Optional[float],
    bus_width_bits: float,
    channels: int,
    accelerator_freq_mhz: float,
) -> float:
    if data_rate_mtps is None and profile == "legacy":
        return DEFAULT_DRAM_EFFECTIVE_BW_BITS_PER_CYCLE
    if data_rate_mtps is None:
        data_rate_mtps = DRAM_PROFILE_DATA_RATE_MTPS[profile]
    if data_rate_mtps <= 0 or bus_width_bits <= 0 or channels <= 0 or accelerator_freq_mhz <= 0:
        raise ValueError("DRAM data rate, bus width, channels, and accelerator frequency must be > 0")
    return float(data_rate_mtps) * float(bus_width_bits) * int(channels) / float(accelerator_freq_mhz)


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
        pe_dp_size: int = 1,
        pe_energy: float = 0,
        pe_area: float = 0,
        pe_array_dim: Sequence[int] = (),
        init_mem: bool = True,
        cxt_len: int = 256,
        is_generation: bool = False,
        layer_name: str = "single_linear",
        base_activation_prec: float = DEFAULT_BASE_ACTIVATION_PRECISION_BITS,
        base_weight_prec: float = DEFAULT_BASE_WEIGHT_PRECISION_BITS,
        dram_effective_bw_bits_per_cycle: float = DEFAULT_DRAM_EFFECTIVE_BW_BITS_PER_CYCLE,
        dram_energy_pj_per_bit: float = DEFAULT_DRAM_ENERGY_PJ_PER_BIT,
    ):
        self.output_prec = self._validate_positive_precision(output_prec, "output_prec")
        self.base_activation_prec = self._validate_positive_precision(
            base_activation_prec, "base_activation_prec"
        )
        self.base_weight_prec = self._validate_positive_precision(
            base_weight_prec, "base_weight_prec"
        )
        self._layer_effective_parallelism = {}
        self.dram_effective_bw_bits_per_cycle = self._validate_positive_precision(
            dram_effective_bw_bits_per_cycle, "dram_effective_bw_bits_per_cycle"
        )
        self.dram_energy_pj_per_bit = self._validate_positive_precision(
            dram_energy_pj_per_bit, "dram_energy_pj_per_bit"
        )

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
            # Accelerator models DDR as rw_bw * 2. Store half the requested
            # effective bandwidth while preserving an explicit pJ/bit cost.
            self.dram.rw_bw = self.dram_effective_bw_bits_per_cycle / 2.0
            self.dram.r_cost = self.dram.rw_bw * self.dram_energy_pj_per_bit
            self.dram.w_cost = self.dram.rw_bw * self.dram_energy_pj_per_bit
            self.dram.r_bw_min = self.dram.rw_bw
            self.dram.w_bw_min = self.dram.rw_bw
            self.dram.r_cost_min = self.dram.r_cost
            self.dram.w_cost_min = self.dram.w_cost
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

    def _get_activation_lane_scaling(self) -> float:
        return self.base_activation_prec / float(self.i_prec)

    def _get_weight_lane_scaling(self, layer_name: str) -> float:
        return self.base_weight_prec / self._get_weight_precision(layer_name)

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

    def calc_sram_rd_energy(self):
        total_energy = 0.0
        activation_lane_scaling = self._get_activation_lane_scaling()

        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            total_tile = self._calc_tile_fc(w_dim, o_dim)
            weight_lane_scaling = self._get_weight_lane_scaling(name)

            w_rd_energy = math.ceil(total_tile / weight_lane_scaling) * self.w_sram.r_cost
            i_rd_energy = math.ceil(total_tile / activation_lane_scaling) * self.i_sram.r_cost
            total_energy += w_rd_energy + i_rd_energy

        return total_energy

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
        result["dram_model"] = {
            "effective_bandwidth_bits_per_cycle": self.dram_effective_bw_bits_per_cycle,
            "energy_pj_per_bit": self.dram_energy_pj_per_bit,
        }
        return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = _build_base_argparser()
    parser.description = (
        "single_linear_simBB: explicit output precision plus BitFusion-like "
        "precision-sensitive throughput for non-bit-serial mode."
    )
    parser.add_argument(
        "--i_prec",
        type=float,
        default=None,
        help="Activation precision in bits. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--kv_prec",
        type=float,
        default=None,
        help="KV-cache precision in bits. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--w_prec",
        type=float,
        default=None,
        help="Weight precision in bits. Overrides the default config when provided.",
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
        "--pe_dp_size",
        type=int,
        default=None,
        help="PE dot-product size. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--pe_energy",
        type=float,
        default=None,
        help="PE energy in pJ. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--pe_area",
        type=float,
        default=None,
        help="PE area. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--pe_array_h",
        type=int,
        default=None,
        help="PE array height. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--pe_array_w",
        type=int,
        default=None,
        help="PE array width. Overrides the default config when provided.",
    )
    parser.add_argument(
        "--is_bit_serial",
        action="store_true",
        help="Enable bit-serial mode. Default BitFusion-like runs should leave this disabled.",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run a minimal trend check without initializing CACTI memories.",
    )
    parser.add_argument(
        "--dram_profile",
        choices=("legacy", *DRAM_PROFILE_DATA_RATE_MTPS.keys()),
        default="legacy",
        help="DRAM bandwidth profile. legacy preserves the original 256 effective bits/cycle.",
    )
    parser.add_argument(
        "--dram_data_rate_mtps",
        type=float,
        default=None,
        help="Custom DRAM data rate in MT/s. Overrides --dram_profile when provided.",
    )
    parser.add_argument("--dram_bus_width_bits", type=float, default=64.0)
    parser.add_argument("--dram_channels", type=int, default=1)
    parser.add_argument("--accelerator_freq_mhz", type=float, default=1000.0)
    parser.add_argument(
        "--dram_energy_pj_per_bit",
        type=float,
        default=DEFAULT_DRAM_ENERGY_PJ_PER_BIT,
        help="DRAM transfer energy. Defaults to the legacy model's 18.75 pJ/bit.",
    )
    return parser


def _apply_optional_overrides(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    overridden = dict(config)

    if args.i_prec is not None:
        overridden["I_PREC"] = args.i_prec
    if args.kv_prec is not None:
        overridden["KV_PREC"] = args.kv_prec
    if args.w_prec is not None:
        overridden["W_PREC"] = args.w_prec
    if args.pe_dp_size is not None:
        overridden["PE_DP_SIZE"] = args.pe_dp_size
    if args.pe_energy is not None:
        overridden["PE_ENERGY"] = args.pe_energy
    if args.pe_area is not None:
        overridden["PE_AREA"] = args.pe_area
    if args.pe_array_h is not None or args.pe_array_w is not None:
        pe_h, pe_w = overridden["PE_ARRAY_DIM"]
        overridden["PE_ARRAY_DIM"] = [
            args.pe_array_h if args.pe_array_h is not None else pe_h,
            args.pe_array_w if args.pe_array_w is not None else pe_w,
        ]

    overridden["IS_BIT_SERIAL"] = bool(args.is_bit_serial)
    overridden["OUTPUT_PREC"] = args.output_prec
    overridden["BASE_ACTIVATION_PREC"] = args.base_activation_prec
    overridden["BASE_WEIGHT_PREC"] = args.base_weight_prec
    return overridden


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
    print(f'compute latency:    {result["cycle"]["compute"]} cycles')
    print(f'dram latency:       {result["cycle"]["dram"]} cycles')
    print(f'total latency:      {result["cycle"]["total"]} cycles')
    print(
        "total cycle:        "
        f'({result["cycle"]["compute"]}, {result["cycle"]["total"]})'
    )
    print(f'tile count:         {result["tile_count"]}')
    print(f'num mem refetch:    {result["num_mem_refetch"]}')
    print(f'memory bytes:       {result["memory_bytes"]}')
    print(f'pe array area:      {result["area_mm2"]["pe_array"]} mm2')
    print(f'weight buffer area: {result["area_mm2"]["weight_buffer"]} mm2')
    print(f'input buffer area:  {result["area_mm2"]["input_buffer"]} mm2')
    print(f'compute energy:     {energy_uJ["compute"]} uJ')
    print(f'sram rd energy:     {energy_uJ["sram_rd"]} uJ')
    print(f'sram wr energy:     {energy_uJ["sram_wr"]} uJ')
    print(f'dram energy:        {energy_uJ["dram"]} uJ')
    print(f'on-chip energy:     {energy_uJ["onchip"]} uJ')
    print(f'total energy:       {energy_uJ["total"]} uJ')

    precision_bits = result.get("precision_bits", {})
    if precision_bits:
        print(f"precision bits:    {precision_bits}")

    compute_model = result.get("compute_model", {})
    if compute_model:
        print(f"compute model:     {compute_model}")
    dram_model = result.get("dram_model", {})
    if dram_model:
        print(f"dram model:        {dram_model}")


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
        bitmod_cfg = _apply_optional_overrides(bitmod_cfg, args)

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
            dram_effective_bw_bits_per_cycle=resolve_dram_effective_bw_bits_per_cycle(
                args.dram_profile,
                args.dram_data_rate_mtps,
                args.dram_bus_width_bits,
                args.dram_channels,
                args.accelerator_freq_mhz,
            ),
            dram_energy_pj_per_bit=args.dram_energy_pj_per_bit,
        )
        _print_summary(sim.simulate())