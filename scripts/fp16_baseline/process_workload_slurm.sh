#!/bin/bash

# refresh the /home/xietongxin/nfs/PNMulator/exp_trial

# 去除文件后缀，获得文件名
workload=${1%.*}
filename=workload/${1}
echo "workload: ${workload}"
IFS=','
while read name type M K N B
do
    # 去除name中开头文件起始符
    name=${name#*/}
    # 去除N中最后一个字符
    B=${B%?}
    # echo "name: ${name}, M: ${M}, K: ${K}, N: ${N}"
    mkdir -p ./${4}/${workload}/progress
    nohup python compile.py -A ${2} -W ${type} -N ${name} -S ${M} ${K} ${N} ${B} -O ${workload} -Q -K ${3} -WS ${4} > ./${4}/${workload}/progress/${name} 2>&1 &
done < $filename

wait
