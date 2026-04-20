#!/bin/bash
# 最终稳定版 - m=2
# 基础命令: 10

main_dir=$FLEXQ_NDP_DIR
log_dir="$main_dir/scripts/final/3_single_op_with_predictor/log_rebuttal_mm_new"

# 创建日志目录
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
cd $main_dir

nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 4096 -n 4096 > $log_dir/7B_MVM1_B2_2_4096_4096_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 4096 -n 4096 > $log_dir/7B_MVM1_B2_2_4096_4096_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 4096 -n 11008 > $log_dir/7B_MVM2_B2_2_4096_11008_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 4096 -n 11008 > $log_dir/7B_MVM2_B2_2_4096_11008_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 11008 -n 4096 > $log_dir/7B_MVM3_B2_2_11008_4096_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 11008 -n 4096 > $log_dir/7B_MVM3_B2_2_11008_4096_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 5120 -n 5120 > $log_dir/13B_MVM1_B2_2_5120_5120_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 5120 -n 5120 > $log_dir/13B_MVM1_B2_2_5120_5120_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 5120 -n 13824 > $log_dir/13B_MVM2_B2_2_5120_13824_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 5120 -n 13824 > $log_dir/13B_MVM2_B2_2_5120_13824_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 13824 -n 5120 > $log_dir/13B_MVM3_B2_2_13824_5120_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 13824 -n 5120 > $log_dir/13B_MVM3_B2_2_13824_5120_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 6656 -n 6656 > $log_dir/34B_MVM1_B2_2_6656_6656_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 6656 -n 6656 > $log_dir/34B_MVM1_B2_2_6656_6656_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 6656 -n 832 > $log_dir/34B_MVM2_B2_2_6656_832_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 6656 -n 832 > $log_dir/34B_MVM2_B2_2_6656_832_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 6656 -n 20480 > $log_dir/34B_MVM3_B2_2_6656_20480_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 6656 -n 20480 > $log_dir/34B_MVM3_B2_2_6656_20480_w4s8_q1_wonly.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wa.yaml -m 2 -k 20480 -n 6656 > $log_dir/34B_MVM4_B2_2_20480_6656_w4s8_q1_wa.log 2>&1 &
nohup python -OO ./aim_rebuttal.py -t single_op_with_predictor_part2 --num-processes 1 --total_buffer_size 20 --qconfig ./scripts/final/3_single_op_with_predictor/workload/detail/w4s8_q1_wonly.yaml -m 2 -k 20480 -n 6656 > $log_dir/34B_MVM4_B2_2_20480_6656_w4s8_q1_wonly.log 2>&1 &

wait
