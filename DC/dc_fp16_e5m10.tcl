remove_design -all
#################################### set library ########################################

# set search_path "$search_path ../rtl ../scripts /home/lincx/library/tsmc28/lib /home/LIBRARY/TSMC28HPC/TSMCHOME/digital/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp30p140ulvt_140a /home/ranxiangyu/siliconsmart/convert"
# set target_library   "tcbn28hpcplusbwp30p140ulvttt0p9v25c_ccs.db"
# #set target_library   "tcbn28_ulvt_900.db"
# set synthetic_library   "dw_foundation.sldb"
# set link_library     " $target_library $synthetic_library"


set search_path "$search_path ../rtl ../scripts"


set NVLM_PATH /capsule/pdk/tsmcN28/TSMCHOME/digital/Front_End/timing_power_noise/NLDM
set TSMC_MEM_PATH /capsule/home/mjli/work/memory/tsmc-5716c1
# set NEW_PATH /lamport/shared/hyzhang/Tianyuan/custom_lib/db
set NEW_PATH_TEST /capsule/home/bcb/workspace/zhy/liberate/LIBRARY
set PAD_PATH /lamport/shared/hzzhu/techlib/tn22ull/stdcell/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tphn22ullgv18e_110c
# set SRAM_PATH /capsule/home/skyhe/WorkSpace/Puti/memory
set BLOCK_DB /capsule/home/bcb/workspace/zhy/Public/0326_1139/CimBlock.db

set_units -current mA
set synthetic_library   "dw_foundation.sldb"
set target_library "$NVLM_PATH/tcbn28hpcplusbwp30p140_180a/tcbn28hpcplusbwp30p140tt0p9v25c.db "
set symbol_library "$NVLM_PATH/tcbn28hpcplusbwp30p140_180a/tcbn28hpcplusbwp30p140tt0p9v25c.db "
set link_library "* $NVLM_PATH/tcbn28hpcplusbwp30p140_180a/tcbn28hpcplusbwp30p140tt0p9v25c.db  $synthetic_library"


#################################### set location #######################################
set top_design M4N4K4_fp16_E5_M10_array_k4
# set top_design_file fp8_E5_M2_array_k512
# set top_design lut_16 
# set top_design LUT_Array_B_BIT2_1Cycle_opt

# set top_design lut_16

set outdir ..

###################################### initial ##########################################
analyze -format sverilog  ../rtl/k_outk_adder.sv
analyze -format sverilog  ../rtl/${top_design}.sv

elaborate $top_design
link

################################## add constraint ######################################
source constraint.tcl
source namingrules.tcl

#################################### compile #######################################
#set_flatten true
compile
# compile_ultra -retime
#compile_ultra -timing_high_effort_script

############################ save results ##################################

# write -format verilog -hier -out $outdir/netlists/${top_design}.nl.v
# write_sdf $outdir/reports/fp16_e5m10/${top_design}.sdf
# write_sdc $outdir/reports/fp16_e5m10/${top_design}.sdc
# write -format ddc -hier -o $outdir/netlists/${top_design}.ddc
report_area -hier > $outdir/reports/fp16_e5m10_new/area_${top_design}.rpt
report_power > $outdir/reports/fp16_e5m10_new/power_${top_design}.rpt
report_constraint -all_violators > $outdir/reports/fp16_e5m10_new/violation_${top_design}.rpt
report_timing -delay max > $outdir/reports/fp16_e5m10_new/timing_${top_design}.rpt
report_timing -delay min >> $outdir/reports/fp16_e5m10_new/timing_${top_design}.rpt

#check_design

#write -format verilog -hier -out $outdir/netlists/${top_design}.v
##write_sdf $outdir/reports/fp16_e5m10/${top_design}.sdf
##write_sdc $outdir/reports/fp16_e5m10/${top_design}.sdc
##report_area -hier > $outdir/reports/fp16_e5m10/area.rpt
##report_power > $outdir/reports/fp16_e5m10/power.rpt
##report_constraint -all_violators > $outdir/reports/fp16_e5m10/violation.rpt
##report_timing -delay max > $outdir/reports/fp16_e5m10/timing.rpt
##report_timing -delay min >> $outdir/reports/fp16_e5m10/timing.rpt
#
######################################## quit ##########################################
#
