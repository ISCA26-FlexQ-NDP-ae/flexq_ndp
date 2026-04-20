# run on eva7

main_dir=$FLEXQ_NDP_DIR
log_dir="$main_dir/scripts/final/3_single_op_with_predictor/log_optimal"

# make dir if not exist
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
cd $main_dir

workload_dir="$main_dir/scripts/final/3_single_op_with_predictor/workload"
workloads=(
    "mvm_7B.csv"
    "mvm_all.csv"
    "mm_7B.csv"
    "mm_13_34B.csv"
)
quant_configs=(
    "quant_all.yaml"
    "quant_e2e.yaml"
    "quant_7B_mm.yaml"
    "quant_e2e.yaml"
)

cmd=""
for i in "${!workloads[@]}"; do
    workload="$workload_dir/${workloads[$i]}"
    quant_config="$workload_dir/${quant_configs[$i]}"

    IFS=','
    while read name type M K N B
    do
        if [ -z "$name" ] || [ -z "$type" ]; then
            continue
        fi

        name=${name#﻿}
        name=${name#*/}

        name=$(echo "$name" | tr -d '\r\n')
        type=$(echo "$type" | tr -d '\r\n')
        M=$(echo "$M" | tr -d '\r\n')
        K=$(echo "$K" | tr -d '\r\n')
        N=$(echo "$N" | tr -d '\r\n')
        B=$(echo "$B" | tr -d '\r\n')

        log_filename="${name}_${M}_${K}_${N}.log"
        nohup python -OO $main_dir/aim_rebuttal.py -t get_ideal_latency --total_buffer_size 20 --qconfig $quant_config -m $M -k $K -n $N > $log_dir/$log_filename 2>&1 &
    done < $workload
done

wait
