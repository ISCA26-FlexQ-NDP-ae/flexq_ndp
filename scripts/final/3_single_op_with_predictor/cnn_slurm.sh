# run on eva7

main_dir=$FLEXQ_NDP_DIR
workload="$main_dir/scripts/final/3_single_op_with_predictor/workload/cnn.csv"
log_dir="$main_dir/scripts/final/3_single_op_with_predictor/log_cnn"

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
    
    # 去除B中最后一个字符（如果需要的话）
    # B=${B%?}
    
    log_filename="${name}_${M}_${K}_${N}.log"
    cmd+="python $main_dir/aim_rebuttal.py -t test_cnn --num-processes 128 --total_buffer_size 20 -m $M -k $K -n $N > $log_dir/$log_filename 2>&1 && "
done < $workload

echo $cmd

cmd=${cmd%&& }
bash -c "$cmd"
