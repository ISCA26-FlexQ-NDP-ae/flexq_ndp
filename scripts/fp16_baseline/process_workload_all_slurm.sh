#!/bin/bash

# refresh the /home/xietongxin/nfs/PNMulator/exp_trial

main_dir=$FLEXQ_NDP_DIR
fp16_baseline_dir="$main_dir/scripts/fp16_baseline"
cd $fp16_baseline_dir

bash ./process_workload_slurm.sh llama2_34B_decode_tk32.csv aim 30 for_quant
bash ./process_workload_slurm.sh llama2_34B_prefill.csv aim 30 for_quant
bash ./process_workload_slurm.sh llama2_7B_decode_tk32.csv aim 30 for_quant
bash ./process_workload_slurm.sh llama2_13B_decode_tk32.csv aim 30 for_quant
bash ./process_workload_slurm.sh llama2_7B_prefill.csv aim 30 for_quant
bash ./process_workload_slurm.sh llama2_13B_prefill.csv aim 30 for_quant