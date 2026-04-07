# Setup

Clone the repo and install dependencies.

```bash
git clone
cd flexq_ndp
pip install -r requirements.txt
```

Set project dir, recommand to add this command in `~/.bashrc` or `~/.zshrc`.

```bash
export FLEXQ_NDP_DIR="PATH_TO_FLEXQ_NDP"
```

# A. GEMM Operator

!!! please run A before other experiments !!!

## A.1 Speedups (Figure 8)

All latencies are reported in cycles, to get the real time, please multiply tCK in `flexq_ndp/config`

### Part 1: Search compilation strategies for different cases (~ 1 day)

```bash
# run the script (in the background)
bash ./scripts/final/3_single_op_with_predictor/part_1.sh
# monitor the process to see if it is finished
ps -aux | grep -c single_op_with_predictor_part1 # Num of processes
ps -aux | grep single_op_with_predictor_part1 | grep -v grep # List of processes
```

### Part 2: Get Baseline & FlexQ-NDP Latency (all MVM < 1h, big MM (34B) takes ~ 1 day)

```bash
# run the script (in the background)
bash ./scripts/final/3_single_op_with_predictor/part2_m1.sh # 94 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m2.sh # 20 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m4.sh # 86 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m8.sh # 20 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m16.sh # 94 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m32.sh # 20 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m64.sh # 86 configs (each is launched to a process)
bash ./scripts/final/3_single_op_with_predictor/part2_m4096.sh # 46 configs (each is launched to a process)
```

- NOTE: During Part 2, you can monitor the num of existing process, and run the next bash if there're free cpu cores:

    ```
    ps -aux | grep -c single_op_with_predictor_part2 # Num of processes
    ps -aux | grep single_op_with_predictor_part2 | grep -v grep # List of processes
    ```

### Part 3: Get Theroratical Latency Lower Bound & Extract CSV for Figure 8 (< 1 hour)

```bash
# get optimal latency to ./scripts/final/3_single_op_with_predictor/log_optimal
bash ./scripts/final/3_single_op_with_predictor/mm_speedup_optimal.sh
# extract latency csv to ./scripts/final/3_single_op_with_predictor/fig8
python ./scripts/final/3_single_op_with_predictor/extract_fig8.py
```

## A.2 Energy (Table VI) (~ 1 hour)

The energy consumption is calculated from the trace collected from
simulation and power parameters, following the equations in
[DRAMSim3](https://github.com/umd-memsys/DRAMsim3).
In configs files, the units of voltage, current, and tCK is V, mA, and ns.
As a result, the unit of reported energy is pJ.
The detail process of energy estimation is in `flexq_ndp/tools/share.py:L39-L54`.

```bash
# run the script (in the background)
bash ./scripts/final/8_energy/bash_energy.sh

# extract results (in pJ)
python ./scripts/final/8_energy/extract_energy.py
```

## A.3 Comparison between quantization precisions and QConfigs (Figure 9) (< 1 minute)

```bash
python ./scripts/final/3_single_op_with_predictor/extract_fig9.py
```
Results in `./scripts/final/3_single_op_with_predictor/fig9`.

# B. CNN (Table VIII) (~ 1 hour)

```bash
bash ./scripts/final/3_single_op_with_predictor/cnn.sh
```
Results in `./scripts/final/3_single_op_with_predictor/log_cnn`.

# C. End-to-End Result (Fig.10) (~ 0.5 hour)

```bash
# 1. Extract all single op data (Ensure that you have finished #A)
python ./scripts/final/3_single_op_with_predictor/extract_all.py
# 2. Run FP16 baseline in background, results will be reported in flexq_ndp/scripts/fp16_baseline/for_quant
bash ./scripts/fp16_baseline/process_workload_all.sh
ps -aux | grep for_quant | grep -v grep # please ensures that this step has finished
# 3. Get Online Quantization Latency in background
bash ./scripts/final/3_single_op_with_predictor/quant_latency/quant_lat.sh
ps -aux | grep test_quant_latency | grep -v grep # please ensures that this step has finished
# 4. Calculate end to end latency & Draw Picture
python ./scripts/final/4_e2e/calculate_e2e_lat.py
```

- NOTE: Our estimation of weight rewrite of 34B model 
    - under FP16 precision, the weight of 34B model ~ 68GB
    - the capacity of our assumed NDP arch is 32GB (32 AiM chips, 2 channels per chip)
    - Total bursts to write = {(68-32)*1024^3 Byte} / 64Channels / {32Byte/Burst} = 18874368
    - Considering switching between bank groups during write, Total latency = 18874368 * tBL(2 cycle in GDDR6) = `37748736 cycles`

# D. Ablation (Table IX, ~5h)

```bash
# run in 32 processes (maybe create more on the peak, do not run together with other program)
bash ./scripts/final/7_ablation/ablation.sh
#
python ./scripts/final/7_ablation/extract_ablation.py
```

# F. Various Hardware Configurations

## F.1 Buffer Size (Fig.11, ~8h)

```bash
# run in 128 processes (maybe create more on the peak, do not run together with other program)
# if the program crashes, you can change `--num-processes` option in the bash script.
bash ./scripts/final/5_buffer_change/buffer_change.sh
#
python ./scripts/final/5_buffer_change/extract_buffer_change.py
```

## F.2 FP32 Throughput (Fig.12, ~1 day)

```bash
# run in 128 processes (maybe create more on the peak, do not run together with other program)
# if the program crashes, you can change `--num-processes` option in the bash script.
bash ./scripts/final/6_throughput_change/throughput_change.sh
# 
python ./scripts/final/6_throughput_change/extract_throughput_speedup.py
```

## F.3 Compatibility with Mix-precision PU (Fig.13, ~0.5h)

```bash
# run in 128 processes (maybe create more on the peak, do not run together with other program)
# if the program crashes, you can change `--num-processes` option in the bash script.
bash ./scripts/final/9_mix_precision_pu/mix_pre_pu.sh
# extract result
python ./scripts/final/9_mix_precision_pu/extract_mix_precision_pu.py
```

# NOTE: How to monitor and terminate background commands

To monitor the issued compile commands, you can use commands like, the `[KEYWORD]` should be replaced to words contained in the background commands. For example, `aim`, `test_quant_latency`, `for_quant`.
```bash
ps -aux | grep [KEYWORD] # command text
ps -aux | grep -c [KEYWORD] # command num
```

To kill certain background process, you can use commands like
```bash
kill -9 $(ps -aux | grep [KEYWORD]| grep -v grep | awk '{print $2}')
```