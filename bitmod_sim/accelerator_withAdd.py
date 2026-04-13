import math
import os

from typing import Any, Dict, List, Optional, Sequence, Tuple

from accelerator import Accelerator


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


def _shape_tuple(shape_like: Any, tensor_name: str) -> Tuple[int, ...]:
    if hasattr(shape_like, "shape"):
        shape = tuple(int(dim) for dim in shape_like.shape)
    elif isinstance(shape_like, Sequence):
        shape = tuple(int(dim) for dim in shape_like)
    else:
        raise TypeError(f"{tensor_name} must provide a .shape attribute or be a shape-like sequence.")

    if len(shape) == 0:
        return (1,)
    return shape


def _numel_from_shape(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


class AcceleratorWithAdd(Accelerator):
    """
    Extend the repository's GEMM simulator with a parameter-update pass model.

    GEMM statistics reuse the original `Accelerator` equations as closely as
    possible, but the model profile is extracted directly from a PyTorch-like
    `model` via `named_modules()` and `named_parameters()` instead of
    `model_shape_config/*.pickle`.

    The added parameter-update pass is modeled as a streaming memory operation:
        read param
        read update
        write updated param

    Add compute cost is intentionally ignored by default because this pass is
    treated as memory-bound.
    """

    def __init__(
        self,
        model: Any,
        model_name: Optional[str] = None,
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
    ):
        if pe_energy == 0:
            raise ValueError("ERROR! You must provide the energy cost of a PE.")
        if len(pe_array_dim) != 2:
            raise ValueError(
                f"ERROR! The dimension of PE array must be 2. But you gave {len(pe_array_dim)}."
            )
        if not hasattr(model, "named_modules"):
            raise TypeError("model must provide a named_modules() method.")
        if not hasattr(model, "named_parameters"):
            raise TypeError("model must provide a named_parameters() method.")

        self.model = model
        self.model_name = model_name or model.__class__.__name__
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

        self.profile_notes = []
        self._build_model_profile(model)
        self._parameter_shapes = self._collect_parameter_shapes(model)

        self.cycle_compute = None
        self.mem_initialized = False
        if init_mem:
            _ensure_supported_runtime()
            self._init_mem()
            self._check_layer_mem_size()
            self._calc_num_mem_refetch()
            self.mem_initialized = True

    def _build_model_profile(self, model: Any) -> None:
        batch_size = self.batch_size
        cxt_len = self.cxt_len
        is_generation = self.is_generation

        weight_dim = {}
        input_dim = {}
        output_dim = {}

        for name, module in model.named_modules():
            if not self._is_linear_module(module):
                continue

            weight_shape = _shape_tuple(module.weight, f"{name}.weight")
            if len(weight_shape) != 2:
                continue

            out_features, in_features = weight_shape
            weight_dim[name] = [1, out_features, in_features]
            if is_generation:
                input_dim[name] = [1, batch_size, in_features]
                output_dim[name] = [1, batch_size, out_features]
            else:
                input_dim[name] = [batch_size, cxt_len, in_features]
                output_dim[name] = [batch_size, cxt_len, out_features]

        attention_layers_added = self._append_attention_ops(weight_dim, input_dim, output_dim)
        if attention_layers_added == 0:
            self.profile_notes.append(
                "Attention GEMM ops were not added because the model config did not expose the "
                "required hidden-size / head-count fields."
            )

        self.weight_dim = weight_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_name_list = list(weight_dim.keys())

    def _append_attention_ops(
        self,
        weight_dim: Dict[str, List[int]],
        input_dim: Dict[str, List[int]],
        output_dim: Dict[str, List[int]],
    ) -> int:
        config = self._extract_model_config()
        required_keys = ["num_hidden_layers", "hidden_size", "num_attention_heads"]
        if any(key not in config for key in required_keys):
            return 0

        num_hidden_layers = int(config["num_hidden_layers"])
        hidden_size = int(config["hidden_size"])
        num_attention_heads = int(config["num_attention_heads"])
        num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
        if num_attention_heads == 0 or num_key_value_heads == 0:
            return 0

        if hidden_size % num_attention_heads == 0:
            head_size = hidden_size // num_attention_heads
        else:
            head_size = hidden_size / num_attention_heads
        query_share_factor = num_attention_heads / num_key_value_heads
        batch_size = self.batch_size
        cxt_len = self.cxt_len
        is_generation = self.is_generation

        for layer_idx in range(num_hidden_layers):
            op_name = f"model.layers.{layer_idx}.self_attn.attn_qk"
            if is_generation:
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size]
                input_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, head_size]
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, cxt_len]
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size]
                input_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size]
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len]

            op_name = f"model.layers.{layer_idx}.self_attn.attn_v"
            if is_generation:
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len]
                input_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, cxt_len]
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, head_size]
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len]
                input_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len]
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size]

        return num_hidden_layers * 2

    def _extract_model_config(self) -> Dict[str, Any]:
        config = getattr(self.model, "config", None)
        if config is None:
            return {}
        if hasattr(config, "to_dict"):
            return dict(config.to_dict())
        if isinstance(config, dict):
            return dict(config)

        config_dict = {}
        for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads"):
            if hasattr(config, key):
                config_dict[key] = getattr(config, key)
        return config_dict

    def _is_linear_module(self, module: Any) -> bool:
        if not hasattr(module, "weight"):
            return False

        class_name = module.__class__.__name__.lower()
        if "linear" not in class_name:
            return False

        try:
            weight_shape = _shape_tuple(module.weight, f"{module.__class__.__name__}.weight")
        except (TypeError, ValueError):
            return False
        return len(weight_shape) == 2

    def _collect_parameter_shapes(self, model: Any) -> List[Dict[str, Any]]:
        parameter_shapes = []
        for name, parameter in model.named_parameters():
            shape = _shape_tuple(parameter, name)
            parameter_shapes.append(
                {
                    "name": name,
                    "shape": shape,
                    "numel": _numel_from_shape(shape),
                    "ndim": len(shape),
                    "prec_bits": self._infer_parameter_precision_bits(parameter),
                }
            )
        return parameter_shapes

    def _infer_parameter_precision_bits(self, parameter: Any) -> int:
        if hasattr(parameter, "element_size"):
            try:
                return int(parameter.element_size()) * 8
            except TypeError:
                pass

        dtype = getattr(parameter, "dtype", None)
        if dtype is not None:
            dtype_str = str(dtype).lower()
            if "16" in dtype_str or "half" in dtype_str or "bfloat16" in dtype_str:
                return 16
            if "32" in dtype_str or "float" in dtype_str:
                return 32
            if "8" in dtype_str:
                return 8

        return int(math.ceil(self.w_prec))

    def _calc_gemm_cycle_quiet(self) -> Dict[str, int]:
        self._calc_compute_cycle()
        self._calc_dram_cycle()

        total_cycle = 0
        total_cycle_compute = 0
        total_cycle_dram = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[name]
            cycle_layer_dram = self._layer_cycle_dram[name]
            total_cycle_compute += cycle_layer_compute
            total_cycle_dram += cycle_layer_dram
            total_cycle += max(cycle_layer_compute, cycle_layer_dram)

        self.cycle_compute = total_cycle_compute
        return {
            "compute": int(total_cycle_compute),
            "dram": int(total_cycle_dram),
            "total": int(total_cycle),
        }

    def _bits_to_bytes(self, numel: int, prec_bits: int) -> int:
        return int(math.ceil(numel * prec_bits / 8))

    def _calc_stream_sram_access_energy(self, bits: int, is_weight_buffer: bool, is_write: bool) -> float:
        if bits == 0:
            return 0.0

        if is_weight_buffer:
            sram = self.w_sram
        else:
            sram = self.i_sram

        if is_write:
            bw = sram.w_bw_min
            cost = sram.w_cost_min
        else:
            bw = sram.r_bw_min
            cost = sram.r_cost_min
        num_access = int(math.ceil(bits / bw))
        return num_access * cost

    def calc_add_stats(self) -> Dict[str, Any]:
        if not self.mem_initialized:
            raise RuntimeError(
                "Memory is not initialized. Set init_mem=True to simulate parameter-update latency/energy."
            )

        dram_bandwidth = self.dram.rw_bw * 2
        bus_width = self.dram.rw_bw
        dram_rd_cost = self.dram.r_cost
        dram_wr_cost = self.dram.w_cost

        total_read_bytes = 0
        total_write_bytes = 0
        total_dram_cycle = 0
        total_dram_energy = 0.0
        total_sram_energy = 0.0
        parameter_count = 0
        matrix_like_parameter_count = 0
        vector_like_parameter_count = 0
        total_numel = 0

        for parameter_info in self._parameter_shapes:
            parameter_count += 1
            total_numel += parameter_info["numel"]
            if parameter_info["ndim"] >= 2:
                matrix_like_parameter_count += 1
            else:
                vector_like_parameter_count += 1

            bytes_per_tensor = self._bits_to_bytes(parameter_info["numel"], parameter_info["prec_bits"])
            read_bytes = bytes_per_tensor     # low-rank uv read is ignored
            write_bytes = bytes_per_tensor
            total_bits = (read_bytes + write_bytes) * 8

            total_read_bytes += read_bytes
            total_write_bytes += write_bytes
            if total_bits > 0:
                total_dram_cycle += max(1, int(math.ceil(total_bits / dram_bandwidth)))

            total_dram_energy += read_bytes * 8 / bus_width * dram_rd_cost
            total_dram_energy += write_bytes * 8 / bus_width * dram_wr_cost

            tensor_bits = bytes_per_tensor * 8
            total_sram_energy += self._calc_stream_sram_access_energy(
                tensor_bits, is_weight_buffer=True, is_write=True
            )
            total_sram_energy += self._calc_stream_sram_access_energy(
                tensor_bits, is_weight_buffer=True, is_write=False
            )
            total_sram_energy += self._calc_stream_sram_access_energy(
                tensor_bits, is_weight_buffer=False, is_write=True
            )
            total_sram_energy += self._calc_stream_sram_access_energy(
                tensor_bits, is_weight_buffer=False, is_write=False
            )
            total_sram_energy += self._calc_stream_sram_access_energy(
                tensor_bits, is_weight_buffer=False, is_write=True
            )

        total_energy = total_dram_energy + total_sram_energy
        total_traffic = total_read_bytes + total_write_bytes

        return {
            "latency": int(total_dram_cycle),
            "energy_pj": {
                "dram": total_dram_energy,
                "sram": total_sram_energy,
                "compute": 0.0,
                "total": total_energy,
            },
            "memory_bytes": {
                "read": int(total_read_bytes),
                "write": int(total_write_bytes),
                "total": int(total_traffic),
            },
            "parameter_stats": {
                "count": int(parameter_count),
                "matrix_like": int(matrix_like_parameter_count),
                "vector_like": int(vector_like_parameter_count),
                "elements": int(total_numel),
            },
            "assumptions": [
                "Each parameter update is modeled as one streaming pass: read param, read update, write param.",
                "Update generation (for example u @ v^T or random vector sampling) is intentionally excluded.",
                "Element-wise add compute cost is set to zero and the pass is treated as memory-bound.",
                "Streaming over large parameters does not add DRAM refetch traffic because there is no data reuse."
            ],
        }

    def calc_gemm_stats(self) -> Dict[str, Any]:
        if not self.mem_initialized:
            raise RuntimeError(
                "Memory is not initialized. Set init_mem=True to simulate GEMM latency/energy."
            )

        cycle = self._calc_gemm_cycle_quiet()
        compute_energy = self.calc_compute_energy()
        sram_rd_energy = self.calc_sram_rd_energy()
        sram_wr_energy = self.calc_sram_wr_energy()
        dram_energy = self.calc_dram_energy()

        return {
            "cycle": cycle,
            "energy_pj": {
                "compute": compute_energy,
                "sram_rd": sram_rd_energy,
                "sram_wr": sram_wr_energy,
                "dram": dram_energy,
                "onchip": compute_energy + sram_rd_energy + sram_wr_energy,
                "total": compute_energy + sram_rd_energy + sram_wr_energy + dram_energy,
            },
            "layer_count": len(self.layer_name_list),
            "profile_notes": list(self.profile_notes),
        }

    def simulate(self) -> Dict[str, Any]:
        gemm_stats = self.calc_gemm_stats()
        add_stats = self.calc_add_stats()

        return {
            "model_name": self.model_name,
            "gemm_latency": gemm_stats["cycle"]["total"],
            "gemm_energy": gemm_stats["energy_pj"]["total"],
            "add_latency": add_stats["latency"],
            "add_energy": add_stats["energy_pj"]["total"],
            "add_memory_read": add_stats["memory_bytes"]["read"],
            "add_memory_write": add_stats["memory_bytes"]["write"],
            "gemm": gemm_stats,
            "add": add_stats,
        }
