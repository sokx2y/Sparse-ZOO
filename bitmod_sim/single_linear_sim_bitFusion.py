import math

from accelerator import Accelerator
from mem.mem_instance import MemoryInstance


MODEL_NAME = "single_linear_bitfusion"
LAYER_NAME = "single_linear"

# INPUT_SHAPE is interpreted as [BT, Cin], where BT = batch_size * tokens_per_sample.
INPUT_SHAPE = [1024, 1024]
WEIGHT_SHAPE = [1024, 1024]

# Bit Fusion paper:
# - one Fusion Unit (FU) contains 16 BitBricks
# - the FU spatially supports up to 8-bit operands in one cycle
# - 16-bit support is obtained temporally over 4 cycles
FUSION_UNIT_BITBRICKS = 16
FUSION_BUFFER_ACCESS_BITS = 32
FUSION_UNIT_AREA = 1394.0
FUSION_UNIT_POWER_NW = 538.0
BIT_FUSION_FREQ_MHZ = 500.0
FUSION_UNIT_ENERGY = FUSION_UNIT_POWER_NW / BIT_FUSION_FREQ_MHZ

IS_BIT_SERIAL = False
PE_DP_SIZE = 1            # 在bitfusion里这个这是保留接口，不参与实际的cycle计算
PE_ENERGY = FUSION_UNIT_ENERGY
PE_AREA = FUSION_UNIT_AREA
PE_ARRAY_DIM = [32, 16]    # bitfusion论文说512个FU，我们这里将整个array假定为32*16
I_PREC = 16
KV_PREC = 8     # 由于此代码仿真的是单层linear，这个attention计算的KV精度知识保留接口
W_PREC = 4
BATCH_SIZE = 16
CXT_LEN = INPUT_SHAPE[0] // BATCH_SIZE
IS_GENERATION = False

W_SRAM_SIZE_KB = 56
I_SRAM_SIZE_KB = 56
SRAM_READ_PJ_PER_BIT = 0.15      # 读写能耗是模仿45nm 下 CACTI-P 估出来的量级
SRAM_WRITE_PJ_PER_BIT = 0.20     # 读写能耗是模仿45nm 下 CACTI-P 估出来的量级
DRAM_BANDWIDTH_BITS = 128


class SingleLinearBitFusionAccelerator(Accelerator):
    SUPPORTED_PRECISIONS = (2, 4, 8, 16)
    BITBRICKS_PER_FU = FUSION_UNIT_BITBRICKS
    BUFFER_ACCESS_BITS = FUSION_BUFFER_ACCESS_BITS

    def _init_model_profiler(self, model_name, cxt_len=256, is_generation=False):
        bt, cin = INPUT_SHAPE
        cout, weight_cin = WEIGHT_SHAPE
        assert cin == weight_cin
        assert bt % self.batch_size == 0, "INPUT_SHAPE[0] must equal batch_size * tokens_per_sample."

        num_token = bt // self.batch_size
        self.weight_dim = {LAYER_NAME: [1, cout, cin]}
        self.input_dim = {LAYER_NAME: [self.batch_size, num_token, cin]}
        self.output_dim = {LAYER_NAME: [self.batch_size, num_token, cout]}
        self.layer_name_list = [LAYER_NAME]
    
    # BitFusion 只支持 24 8 16 的混合精度乘法
    def _round_to_supported_precision(self, precision):
        requested_precision = max(2, math.ceil(precision))
        for supported_precision in self.SUPPORTED_PRECISIONS:
            if requested_precision <= supported_precision:
                return supported_precision
        raise ValueError(
            f"Bit Fusion in this simulator only supports precisions up to "
            f"{self.SUPPORTED_PRECISIONS[-1]} bits, got {precision}."
        )

    def _weight_precision_for_layer(self, layer_name):
        if ("attn_qk" in layer_name) or ("attn_v" in layer_name):
            return self.kv_prec
        return self.w_prec
    
    # 将乘法递归分解成 2-bit 基本乘法 （bitbricks）
    def _macs_per_fu_per_cycle(self, input_precision, weight_precision):
        input_chunks = math.ceil(input_precision / 2)
        weight_chunks = math.ceil(weight_precision / 2)
        bitbrick_groups_per_mac = input_chunks * weight_chunks
        return self.BITBRICKS_PER_FU / bitbrick_groups_per_mac

    def _calc_compute_cycle(self):
        self._layer_cycle_compute = {}
        input_precision = self._round_to_supported_precision(self.i_prec)

        for name in self.layer_name_list:
            _, cout, cin = self.weight_dim[name]
            batch_size, num_token, _ = self.output_dim[name]
            weight_precision = self._round_to_supported_precision(self._weight_precision_for_layer(name))

            total_mac = batch_size * num_token * cout * cin
            total_parallel_mac_per_cycle = (
                self.total_pe_count * self._macs_per_fu_per_cycle(input_precision, weight_precision)
            )
            self._layer_cycle_compute[name] = math.ceil(total_mac / total_parallel_mac_per_cycle)

    def calc_sram_rd_energy(self):
        total_energy = 0
        for name in self.layer_name_list:
            num_fetch_w, num_fetch_i = self._layer_mem_refetch[name]
            w_access = math.ceil(self._w_mem_required[name] * 8 / self.w_sram.rw_bw) * num_fetch_w
            i_access = math.ceil(self._i_mem_required[name] * 8 / self.i_sram.rw_bw) * num_fetch_i
            total_energy += w_access * self.w_sram.r_cost
            total_energy += i_access * self.i_sram.r_cost
        return total_energy

    def _init_mem(self):
        w_bandwidth = self.BUFFER_ACCESS_BITS * self.pe_array_dim["h"]
        i_bandwidth = self.BUFFER_ACCESS_BITS * self.pe_array_dim["w"]
        w_sram_config = {
            "technology": 0.045,
            "mem_type": "ram",
            "size": W_SRAM_SIZE_KB * 1024 * 8,
            "bank_count": 8,
            "rw_bw": w_bandwidth,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
        }
        self.w_sram = MemoryInstance(
            w_sram_config,
            r_cost=w_bandwidth * SRAM_READ_PJ_PER_BIT,
            w_cost=w_bandwidth * SRAM_WRITE_PJ_PER_BIT,
            latency=1,
            min_r_granularity=self.BUFFER_ACCESS_BITS,
            min_w_granularity=self.BUFFER_ACCESS_BITS,
            get_cost_from_cacti=False,
        )

        i_sram_config = {
            "technology": 0.045,
            "mem_type": "ram",
            "size": I_SRAM_SIZE_KB * 1024 * 8,
            "bank_count": 8,
            "rw_bw": i_bandwidth,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
        }
        self.i_sram = MemoryInstance(
            i_sram_config,
            r_cost=i_bandwidth * SRAM_READ_PJ_PER_BIT,
            w_cost=i_bandwidth * SRAM_WRITE_PJ_PER_BIT,
            latency=1,
            min_r_granularity=self.BUFFER_ACCESS_BITS,
            min_w_granularity=self.BUFFER_ACCESS_BITS,
            get_cost_from_cacti=False,
        )

        dram_config = {
            "technology": 0.045,
            "mem_type": "dram",
            "size": int(1e9 * 8),
            "bank_count": 1,
            "rw_bw": DRAM_BANDWIDTH_BITS,
            "r_port": 0,
            "w_port": 0,
            "rw_port": 1,
        }
        dram_cost = DRAM_BANDWIDTH_BITS / 64 * 1200
        self.dram = MemoryInstance(
            dram_config,
            r_cost=dram_cost,
            w_cost=dram_cost,
            latency=1,
            min_r_granularity=DRAM_BANDWIDTH_BITS,
            min_w_granularity=DRAM_BANDWIDTH_BITS,
            get_cost_from_cacti=False,
        )


SingleLinearAccelerator = SingleLinearBitFusionAccelerator


if __name__ == "__main__":
    acc = SingleLinearBitFusionAccelerator(
        model_name=MODEL_NAME,
        i_prec=I_PREC,
        kv_prec=KV_PREC,
        w_prec=W_PREC,
        batch_size=BATCH_SIZE,
        is_bit_serial=IS_BIT_SERIAL,
        pe_dp_size=PE_DP_SIZE,
        pe_energy=PE_ENERGY,
        pe_area=PE_AREA,
        pe_array_dim=PE_ARRAY_DIM,
        cxt_len=CXT_LEN,
        is_generation=IS_GENERATION,
    )

    total_cycle = acc.calc_cycle()
    compute_energy = acc.calc_compute_energy() / 1e6
    sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
    sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
    dram_energy = acc.calc_dram_energy() / 1e6
    total_energy = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

    print(f"batch_size: {BATCH_SIZE}")
    print(f"tokens_per_sample: {INPUT_SHAPE[0] // BATCH_SIZE}")
    print(f"fusion_units: {PE_ARRAY_DIM[0] * PE_ARRAY_DIM[1]}")
    print(f"total_cycle: {total_cycle}")
    print(f"compute_energy: {compute_energy} uJ")
    print(f"sram_rd_energy: {sram_rd_energy} uJ")
    print(f"sram_wr_energy: {sram_wr_energy} uJ")
    print(f"dram_energy: {dram_energy} uJ")
    print(f"total_energy: {total_energy} uJ")

