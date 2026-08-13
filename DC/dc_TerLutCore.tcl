# environment variables are setted in .synopsys_dc.setup
# This file is for module level synthesis
set LOG_DATE [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
set_host_options -max_cores 16
set DESIGN_NAME TerLutCore
set SUFFIX 800MHz_PTPX

# Setup the library
set NVLM_PATH /capsule/home/hzzhu/techlib/tn12ffc/stdcell/TSMCHOME/digital/Front_End/timing_power_noise/NLDM
set PAD_PATH /capsule/home/hzzhu/techlib/tn12ffc/io/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tphn12ffcllgv18e_140a
# set SRAM_PATH /capsule/home/linyuzheng/workspace/Gaura/memory/20250210

#set_units -current mA
set target_library "$NVLM_PATH/tcbn12ffcllbwp6t20p96cpd_120a/tcbn12ffcllbwp6t20p96cpdtt0p8v25c.db \
$PAD_PATH/tphn12ffcllgv18ett0p8v1p8v25c.db "

set symbol_library "$NVLM_PATH/tcbn12ffcllbwp6t20p96cpd_120a/tcbn12ffcllbwp6t20p96cpdtt0p8v25c.db \
$PAD_PATH/tphn12ffcllgv18ett0p8v1p8v25c.db "

set link_library "* $NVLM_PATH/tcbn12ffcllbwp6t20p96cpd_120a/tcbn12ffcllbwp6t20p96cpdtt0p8v25c.db \
$PAD_PATH/tphn12ffcllgv18ett0p8v1p8v25c.db "
file mkdir ./work
define_design_lib WORK -path ./work
set_svf $DESIGN_NAME.svf

set_app_var verilogout_show_unconnected_pins true

###############################################  input RTL files  ##############################################


#####################  Top Generate Files   #####################
# analyze -format sverilog /capsule/home/linyuzheng/workspace/Gaura/syn_single_logicCore_v5_20250317/src/CoreWrapper.sv
analyze -format sverilog /capsule/home/ZR_Huang/hzr_workspace/TACKLE/src/ChipConfig.sv
analyze -format sverilog /capsule/home/ZR_Huang/hzr_workspace/TACKLE/src/DataInterface.sv
analyze -format sverilog /capsule/home/ZR_Huang/hzr_workspace/TACKLE/src/LutCore/Dynamic_decoder.v
analyze -format sverilog /capsule/home/ZR_Huang/hzr_workspace/TACKLE/src/LutCore/TerLutCore.sv
# analyze -format sverilog -define { FLOW_ASIC } /capsule/home/linyuzheng/workspace/Gaura/syn_single_logicCore/src/TopChip.sv

elaborate $DESIGN_NAME
current_design ${DESIGN_NAME}
link 



# set_dont_touch [get_cells uChip/ClkSel/i_clk*]
# set_clock_group -logically_exclusive -group {clock} -group {clock_f}
# set_clock_group -physically_exclusive -group {CLK0} -group {CLK_F}
# set_clock_groups -asynchronous -name func_async -group {clock clock_f} -group {Rx_clock}
create_clock -name clock -period 1.25 [get_ports clk]
create_clock -name clock_slow -period 20 [get_ports clk_slow]
set_clock_groups -async -group clk -group clk_slow

# ## TO DELETE!!!!!!!!
# set_dont_touch [get_cells uChip/uScorpioCore/uBackend/cimCoreArray/cimCore/uBbox]

# set_false_path -from [get_ports pad_reset]

set_false_path -fall_from [get_clocks *]
set_clock_uncertainty -setup 0.125 [all_clocks]
set_clock_uncertainty -hold 0.05 [all_clocks]
set_clock_transition 0.1 [all_clocks]
set_clock_latency 0.2 [all_clocks]
# set_propagated_clock [all_clocks]
# set_timing_derate -early 0.9
# set_timing_derate -late 1.1

# set_input_delay -max 0.2 -clock clock [remove_from_collection [all_inputs] [get_ports "i_clock i_reset"]]
# set_output_delay -max 0.2 -clock clock [all_outputs]
# set_load 0.2 [get_ports [all_outputs]]  

# set false paths
# set_dont_touch_network [get_ports reset]

# driving and loading # TODO: CHECK HERE !!!
# set driving_cell "BUFFD8BWP6T20P96CPD"
# set_driving_cell -lib_cell $driving_cell [remove_from_collection [all_inputs] [get_ports {pad_clock pad_reset pad_spi_ssn pad_spi_sck pad_spi_mosi pad_d2d_rx_clock pad_d2d_rx_flit_valid pad_d2d_rx_flit_* pad_d2d_rx_creditFree pad_d2d_rx_replayPkgID}]]

#design rule constraints
# set_clock_gating_style -control_point before -control_signal scan_enable -positive_edge_logic {integrated:CKLNQD12BWP30P140HVT} -max_fanout 32 -minimum_bitwidth 4
set_clock_gating_style -control_point before -control_signal scan_enable -positive_edge_logic {integrated:CKLNQD12BWP6T20P96CPD} -max_fanout 32 -minimum_bitwidth 4
set_max_transition 0.4 [current_design]
# set_max_capacitance 0.9 [current_design]
set_max_fanout 40 [current_design]
set_max_area 0

set_fix_hold [all_clocks]
set_fix_multiple_port_nets -buffer_constants -all

compile_ultra -no_autoungroup -gate_clock

##############################################  output files  ##############################################
set REPORTS reports_${DESIGN_NAME}/${LOG_DATE}_${SUFFIX}
set SYN_FILES syn_files_${DESIGN_NAME}/${LOG_DATE}_${SUFFIX}
file delete -force $SYN_FILES
file delete -force $REPORTS
file mkdir $SYN_FILES
file mkdir $REPORTS

report_port -verbose > $REPORTS/${DESIGN_NAME}_port_verbose.rpt
report_clock  > $REPORTS/${DESIGN_NAME}_clock.rpt
check_design > $REPORTS/${DESIGN_NAME}_check.rpt
report_power -hierarchy > $REPORTS/$DESIGN_NAME.hierarchy.power
report_power > $REPORTS/$DESIGN_NAME.power
report_area  -hierarchy > $REPORTS/$DESIGN_NAME.area
report_timing -significant_digits 5 > $REPORTS/$DESIGN_NAME.max.timing
report_timing -significant_digits 5 -delay_type min > $REPORTS/$DESIGN_NAME.min.timing
report_constraint -verbose > $REPORTS/$DESIGN_NAME.constraint
report_constraint -all_violators > $REPORTS/$DESIGN_NAME.violation
report_clock_gating -verbose > $REPORTS/$DESIGN_NAME.icg
report_clock_gating -gating_elements > $REPORTS/$DESIGN_NAME.elements.icg

# Ouptut the results
write -h $DESIGN_NAME -output ./$SYN_FILES/$DESIGN_NAME.db
write_file -format ddc -hierarchy -output ./$SYN_FILES/$DESIGN_NAME.ddc
# Delays in SDF format for Verilog simulation
write_sdf -context verilog -version 1.0 ./$SYN_FILES/$DESIGN_NAME.syn.sdf
# The post-syn Verilog netlist
write -h -f verilog $DESIGN_NAME -output ./$SYN_FILES/$DESIGN_NAME.syn.v -pg
# Constraints in SDC format, for APR
write_sdc ./$SYN_FILES/$DESIGN_NAME.sdc

set_svf -off
