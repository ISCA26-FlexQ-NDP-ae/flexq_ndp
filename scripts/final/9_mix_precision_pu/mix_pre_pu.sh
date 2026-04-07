# run on eva6
main_dir=$FLEXQ_NDP_DIR
workload="$main_dir/scripts/final/9_mix_precision_pu/workload/mm.csv"
quant_config="$main_dir/scripts/final/9_mix_precision_pu/workload/quant.yaml"
log_dir="$main_dir/scripts/final/9_mix_precision_pu/log"

# make dir if not exist
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
cd $main_dir

cmd=""
IFS=','
while read name type M K N B
do
    if [ -z "$name" ] || [ -z "$type" ]; then
        continue
    fi
    
    # 去除BOM字符（UTF-8 BOM是 EF BB BF，在bash中表现为不可见字符）
    name=${name#﻿}  # 去除BOM字符
    name=${name#*/}  # 去除name中开头文件起始符
    
    # 去除可能的回车符和换行符
    name=$(echo "$name" | tr -d '\r\n')
    type=$(echo "$type" | tr -d '\r\n')
    M=$(echo "$M" | tr -d '\r\n')
    K=$(echo "$K" | tr -d '\r\n')
    N=$(echo "$N" | tr -d '\r\n')
    B=$(echo "$B" | tr -d '\r\n')
    
    log_filename="${name}_${M}_${K}_${N}.log"
    cmd+="python -OO $main_dir/aim_rebuttal.py -t test_mix_precision_pu --num-processes 128 --total_buffer_size 20 --qconfig $quant_config -m $M -k $K -n $N > $log_dir/$log_filename 2>&1 && "
done < $workload

echo $cmd

cmd=${cmd%&& }
nohup bash -c "$cmd" &
