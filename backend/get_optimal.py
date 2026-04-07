import tqdm
import time
import math
from tools import *
from midend import *
from backend import *
from tools import *
from sim import sim
import math
from multiprocessing import Pool
# mute tqdm
def tqdm_replacement(iterable_object, *args, **kwargs):
    return iterable_object

# MARK: W + A

def get_partition_space(mm_size):
    # A. get hw partition space
    partition_tool = Partition(require_power_of_2 = True)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    partition_space = filtered_partition_space
    return partition_space

def get_optimal_latency_w_a(args, batch_size=1, 
                       operand_A_info={
                           'matrix_size': (1, 5120), # (m, k)
                           'scale_group': (1, 128),
                           'matrix_precision': 8,
                           'scale_offset_precision': 16,
                           'with_scale': True,
                           'with_offset': False,
                       }, 
                       operand_B_info={
                           'matrix_size': (5120, 5120), # (n, k)
                           'scale_group': (128, 128),
                           'matrix_precision': 8,
                           'scale_offset_precision': 16,
                           'with_scale': True,
                           'with_offset': False,
                       },
                       result_precision=16,
                       accumulation_precision=32,
                       log_file=None,
                       consider_scale_buffer=False,
                       consider_data_read=False,
                       consider_result_buffer=False,
                       breakdown_analysis=False,
                       silent = False,
                       buffer_specify = None,
                       buffer_b = False,
                       k_tile = 0,
                       loop_friendly_mapping = False,
                       mix_scale_mapping = False,
                       new_partition = False,
                       reorder = False,
                       partition_specify = None
                       ):
    
    if silent:
        tqdm.tqdm = tqdm_replacement

    """
    NOTE: 0. Get design space
    """
    # get mm size
    m_size = operand_A_info['matrix_size'][0]
    k_size = operand_A_info['matrix_size'][1]
    assert k_size == operand_B_info['matrix_size'][1], f"Invalid matrix size: {operand_A_info['matrix_size']} * {operand_B_info['matrix_size']}"
    n_size = operand_B_info['matrix_size'][0]
    mm_size = (m_size, k_size, n_size, batch_size)
    # get group size
    s_group_m = operand_A_info['scale_group'][0] if operand_A_info['with_scale'] else 0
    if operand_A_info['with_scale'] and operand_B_info['with_scale']:
        s_group_k = operand_A_info['scale_group'][1]
        scale_precision = operand_A_info['scale_offset_precision']
        assert s_group_k == operand_B_info['scale_group'][1], f"Invalid scale group: {operand_A_info['scale_group']} * {operand_B_info['scale_group']}"
    elif operand_A_info['with_scale'] and not operand_B_info['with_scale']:
        s_group_k = operand_A_info['scale_group'][1]
        scale_precision = operand_A_info['scale_offset_precision']
    elif not operand_A_info['with_scale'] and operand_B_info['with_scale']:
        s_group_k = operand_B_info['scale_group'][1]
        scale_precision = operand_B_info['scale_offset_precision']
    else:
        s_group_k = 0
    s_group_n = operand_B_info['scale_group'][0] if operand_B_info['with_scale'] else 0
    scale_group = (s_group_m, s_group_k, s_group_n)

    """
    1. buffer 分配
    """
    # Update log file with buffer size information after allocation
    if log_file is None:
        workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}x{operand_B_info['matrix_size'][0]}"
        # Add precision and scale group info
        a_prec = operand_A_info['matrix_precision']
        b_prec = operand_B_info['matrix_precision']
        # a_scale_group = operand_A_info['scale_group'][1] if operand_A_info['with_scale'] else 0
        b_scale_group = operand_B_info['scale_group'][1] if operand_B_info['with_scale'] else 0
        a_scale_group = operand_B_info['scale_group'][0] if operand_B_info['with_scale'] else 0
        function_desc = f"W{b_prec}A{a_prec}S{scale_precision}_sg{a_scale_group}x{b_scale_group}"
        # Auto-detect function name for directory, use function_desc in filename
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}")
        should_close_files = True
    else:
        should_close_files = False
    
    print("MM Size: ", mm_size, file=log_file)
    print("Scale Group: ", scale_group, file=log_file)
    
    # A. get hw partition space
    partition_tool = Partition(require_power_of_2 = False, log_file=log_file)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    partition_space = filtered_partition_space

    print(f"partition_space: {partition_space}", file=log_file)

    # partition_space = get_partition_space(mm_size)
    # print(f"partition_space: {partition_space}", file=log_file)
    compute_level, pu_num, _ = partition_space[0]
    
    # 如果配置改变，交换 A/B 信息

    """
    NOTE: 2. get the baseline mapping
    """
    baseline_partition = None
    if partition_specify is not None:
        baseline_partition = partition_specify
    else:
        if new_partition:
            for compute_level, pu_num, partition in partition_space:
                # m,k,l,b: only partition on l, and k = 8
                if partition[3][0] * partition[3][1] * partition[3][3] * \
                    partition[2][0] * partition[2][1] * partition[2][3] * \
                    partition[1][0] * partition[1][1] * partition[1][3] == 1 and \
                    partition[0][2] == 8: # ch
                    baseline_partition = partition
                    # baseline = compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping
                    print(f"strategy: {baseline_partition}", file=log_file)
                    break
            # for compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping in design_space:
        else:
            for compute_level, pu_num, partition in partition_space:
            # for compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping in design_space:
                # m,k,l,b: only partition on l, and k = 8
                if partition[3][0] * partition[3][1] * partition[3][3] * \
                    partition[2][0] * partition[2][1] * partition[2][3] * \
                    partition[1][0] * partition[1][1] * partition[1][3] * \
                    partition[0][0] * partition[0][1] * partition[0][3] == 1:
                    # baseline = compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping
                    baseline_partition = partition
                    print(f"strategy: {baseline_partition}", file=log_file)
                    break
    
    if baseline_partition == None: # corner case
        baseline_partition = partition_space[0][2]
        print(f"strategy: {baseline_partition}", file=log_file)

    # design_space = []
    # NOTE: get size after mapping & adjust mapping through scale group
    # compute_level, pu_num, _ = partition_space[index]
    mm_size_per_pu = partition_tool.mem_partition_mm(mm_size, baseline_partition, scale_group)
    # partition_space[index] = (compute_level, pu_num, partition, mm_size_per_pu)
    A_size = (mm_size_per_pu[0], mm_size_per_pu[1])
    B_size = (mm_size_per_pu[2], mm_size_per_pu[1])
    k_col = math.ceil(mm_size_per_pu[1] * b_prec / SimConfig.co_w)
    data_optimal_col = k_col * mm_size_per_pu[0] * mm_size_per_pu[2]
    k_group_size = math.ceil(mm_size_per_pu[1] / b_scale_group)
    dequant_optimal_col = math.ceil(k_group_size * mm_size_per_pu[0] * mm_size_per_pu[2] * accumulation_precision / SimConfig.co_w) * 2
    print(f"data_optimal_col: {data_optimal_col}, dequant_optimal_col: {dequant_optimal_col}", file=log_file)
    return (dequant_optimal_col + data_optimal_col) * SimConfig.col_change_apox

# MARK: Weight Only
def get_optimal_latency_w_only(args, batch_size=1, 
                       operand_A_info={
                           'matrix_size': (1, 5120), # (m, k)
                           'scale_group': (1, 128),
                           'matrix_precision': 8,
                           'scale_offset_precision': 16,
                           'with_scale': True,
                           'with_offset': False,
                       }, 
                       operand_B_info={
                           'matrix_size': (5120, 5120), # (n, k)
                           'scale_group': (128, 128),
                           'matrix_precision': 8,
                           'scale_offset_precision': 16,
                           'with_scale': True,
                           'with_offset': False,
                       },
                       result_precision=16,
                       accumulation_precision=32,
                       log_file=None,
                       consider_scale_buffer=False,
                       consider_data_read=False,
                       consider_result_buffer=False,
                       breakdown_analysis=False,
                       silent = False,
                       buffer_specify = None,
                       buffer_b = False,
                       k_tile = 0,
                       loop_friendly_mapping = False,
                       mix_scale_mapping = False,
                       new_partition = False,
                       reorder = False,
                       partition_specify = None,
                       buffer_b_old = False
                       ):
    a_prec = operand_A_info['matrix_precision']
    b_prec = operand_B_info['matrix_precision']
    scale_precision = operand_B_info['scale_offset_precision']
    if silent:
        tqdm.tqdm = tqdm_replacement
    assert not operand_A_info['with_scale'] and not operand_A_info['with_offset'], "A 必须有 Scale 且没有 Offset" 
    assert operand_B_info['with_scale'] and not operand_B_info['with_offset'], "B 必须没有 Scale 且没有 Offset" 
    """
    NOTE: 1. Get design space
    """
    # get mm size
    m_size = operand_A_info['matrix_size'][0]
    k_size = operand_A_info['matrix_size'][1]
    assert k_size == operand_B_info['matrix_size'][1], f"Invalid matrix size: {operand_A_info['matrix_size']} * {operand_B_info['matrix_size']}"
    n_size = operand_B_info['matrix_size'][0]
    mm_size = (m_size, k_size, n_size, batch_size)
    # get group size
    s_group_m = operand_A_info['scale_group'][0] if operand_A_info['with_scale'] else 0
    if operand_A_info['with_scale'] and operand_B_info['with_scale']:
        s_group_k = operand_A_info['scale_group'][1]
        assert s_group_k == operand_B_info['scale_group'][1], f"Invalid scale group: {operand_A_info['scale_group']} * {operand_B_info['scale_group']}"
    elif operand_A_info['with_scale'] and not operand_B_info['with_scale']:
        s_group_k = operand_A_info['scale_group'][1]
    elif not operand_A_info['with_scale'] and operand_B_info['with_scale']:
        s_group_k = operand_B_info['scale_group'][1]
    else:
        s_group_k = 0
    s_group_n = operand_B_info['scale_group'][0] if operand_B_info['with_scale'] else 0
    scale_group = (s_group_m, s_group_k, s_group_n)

    """
    buffer size 分配，后续可以优化
    - Data A Buffer
    - Data B Buffer 暂时认为不需要
    - Scale B Buffer
    - Result Buffer
    """

    if log_file is None:
        workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}x{operand_B_info['matrix_size'][0]}"
        # Add precision and scale group info for weight-only quantization
        function_desc = f"W{b_prec}A{a_prec}S{scale_precision}_sg{operand_B_info['scale_group'][0]}x{operand_B_info['scale_group'][1]}"
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}")
        should_close_files = True
    else:
        should_close_files = False

    print("MM Size: ", mm_size, file=log_file)
    print("A info", operand_A_info, file=log_file)
    print("B info", operand_B_info, file=log_file)
    
    # A. get hw partition space
    partition_tool = Partition(require_power_of_2 = False, log_file=log_file)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    partition_space = filtered_partition_space
    print(f"partition_space: {partition_space}", file=log_file)    
    
    compute_level, pu_num, _ = partition_space[0]
    
    """
    NOTE: 2. get the baseline mapping
    """
    baseline_partition = None
    if partition_specify is not None:
        baseline_partition = partition_specify
    else:
    # baseline = None
        # for compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping in design_space:
        for compute_level, pu_num, partition in partition_space:
            # m,k,l,b: only partition on l, and k = 8
            if partition[3][0] * partition[3][1] * partition[3][3] * \
                partition[2][0] * partition[2][1] * partition[2][3] * \
                partition[1][0] * partition[1][1] * partition[1][3] * \
                partition[0][0] * partition[0][1] * partition[0][3] == 1:
                # baseline = compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping
                baseline_partition = partition
                print(f"strategy: {baseline_partition}", file=log_file)
                break
    
    if baseline_partition == None: # corner case
        baseline_partition = partition_space[0][2]
        print(f"strategy: {baseline_partition}", file=log_file)
    
    # compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping = baseline

    # NOTE: get size after mapping & adjust mapping through scale group
    # for index in tqdm.tqdm(range(len(partition_space))):
        # compute_level, pu_num, partition = partition_space[index]
    mm_size_per_pu = partition_tool.mem_partition_mm(mm_size, baseline_partition, scale_group)
    # partition_space[index] = (compute_level, pu_num, partition, mm_size_per_pu)
    A_size = (mm_size_per_pu[0], mm_size_per_pu[1])
    B_size = (mm_size_per_pu[2], mm_size_per_pu[1])

    k_col = math.ceil(mm_size_per_pu[1] * a_prec / SimConfig.co_w)
    data_optimal_col = k_col * mm_size_per_pu[0] * mm_size_per_pu[2]
    B_col = math.ceil(mm_size_per_pu[1] * b_prec / SimConfig.co_w) * mm_size_per_pu[2]

    B_dequant_col = B_col * scale_precision / b_prec
    print(f"data_optimal_col: {data_optimal_col}, B_dequant_col: {B_dequant_col}", file=log_file)
    return (data_optimal_col + B_dequant_col) * SimConfig.col_change_apox