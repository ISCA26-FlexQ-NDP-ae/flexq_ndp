# run on eva7
main_dir=$FLEXQ_NDP_DIR
log_csv="$main_dir/scripts/final/3_single_op_with_predictor/best_partition_strategy.csv"
log_dir="$main_dir/scripts/final/3_single_op_with_predictor/quant_latency/log"

# make dir if not exist
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
cd $main_dir

cmd=""
IFS=','
while read name M K N S
do
    if [ -z "$name" ]; then
        continue
    fi
    log_filename="${name}.log"
    # 去除BOM字符（UTF-8 BOM是 EF BB BF，在bash中表现为不可见字符）
    name=${name#﻿}  # 去除BOM字符
    name=${name#*/}  # 去除name中开头文件起始符
    
    # 去除可能的回车符和换行符
    name=$(echo "$name" | tr -d '\r\n')
    M=$(echo "$M" | tr -d '\r\n')
    K=$(echo "$K" | tr -d '\r\n')
    N=$(echo "$N" | tr -d '\r\n')
    S=$(echo "$S" | tr -d '\r\n')
    # echo $name $M $K $N $S
    nohup python $main_dir/aim_rebuttal.py -t test_quant_latency --total_buffer_size 20 -m $M -k $K -n $N --SS "$S" > $log_dir/$log_filename 2>&1 &
done < $log_csv
