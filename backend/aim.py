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

# MARK: Process single ir_list
def process_single_ir_list(args_tuple):
    """
    处理单个 ir_list 的优化、代码生成和模拟
    
    Args:
        args_tuple: (ir_list, reorder, idx) 元组
        
    Returns:
        tuple: (commands, latency, log_content, idx)
    """
    import sys
    ir_list, reorder, idx = args_tuple
    
    # print(f"\n ======== origin ir list (#{idx}) =========")
    # ir_list.print()
    
    if reorder:
        # print(f"\n ======== performing 1st pass (#{idx}) =========")
        ir_list.first_optimize_pass()
        # print(f"\n ======== after 1st pass (#{idx}) ========")
        # ir_list.print()
        
        # print(f"\n ======== performing 2nd pass (#{idx}) =========")
        ir_list.second_optimize_pass()
        # print(f"\n ======== after 2nd pass (#{idx}) =========")
        # ir_list.print()
        
        # print(f"\n ======== performing 3rd pass (#{idx}) =========")
        ir_list.third_optimize_pass()
        # print(f"\n ======== after 3rd pass (#{idx}) =========")
        # ir_list.print()
    
    commands = ir_list.codegen()
    lat = sim([
        (0, [], commands)
    ], silent=True, use_tqdm=False)
    
    return lat

# MARK: Parse DRAM Row Change
def parse_row_change(cmd_list):
    """
    解析命令列表中的DRAM行访问情况，记录每个行访问对应的列数
    
    Args:
        cmd_list: 命令列表，每个命令的格式为 (LEVEL, OPTYPE, ...)
        
    Returns:
        list: [(row_id, col_len), ...] 记录访问的行和对应的列数
    """
    if not cmd_list:
        return []
    
    # 用于存储访问记录 (row_id, col_len)
    row_access_records = []
    
    for cmd in cmd_list:
        if len(cmd) < 3:
            continue
            
        level, optype = cmd[0], cmd[1]
        
        # 解析不同类型命令中的行索引和列数
        if optype.name == 'pu':
            # PU命令只关注op1的row_id和col_num
            # DE级别: (LEVEL, OPTYPE, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, op2, col_num, auto_precharge)
            # RA级别: (LEVEL, OPTYPE, ch_id, ra_id, (pu_num, pu_mask), op1, op2, col_num, auto_precharge)
            if level.name == 'DE' and len(cmd) >= 9:
                op1 = cmd[6]  # op1: (bank_id, row_id, col_offset)
                col_num = cmd[8]  # 列数
                assert len(op1) >= 2, f"PU DE command op1 format error: {op1}"
                row_access_records.append((op1[1], col_num))  # (row_id, col_len)
            elif level.name == 'RA' and len(cmd) >= 8:
                op1 = cmd[5]  # op1: (bank_id, row_id, col_offset)
                col_num = cmd[7]  # 列数
                assert len(op1) >= 2, f"PU RA command op1 format error: {op1}"
                row_access_records.append((op1[1], col_num))  # (row_id, col_len)
                    
        elif optype.name in ['buf2bk', 'bk2buf']:
            # Buffer和Bank之间的传输: op1包含row_id
            # (LEVEL, OPTYPE, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, buf, auto_precharge)
            if len(cmd) >= 8:
                op1 = cmd[6]  # op1: (bank_id, row_id, col_offset)
                buf = cmd[7]   # buf: (is_input, buf_addr, col_len)
                assert len(op1) >= 2, f"buf2bk/bk2buf command op1 format error: {op1}"
                assert len(buf) >= 3, f"buf2bk/bk2buf command buf format error: {buf}"
                row_access_records.append((op1[1], buf[2]))  # (row_id, col_len)

        elif optype.name in ['bk2gb', 'gb2bk']:
            # Bank和GlobalBuffer之间的传输: op1包含row_id, col_num在固定位置
            # (LEVEL, OPTYPE, ch_id, ra_id, de_id, bank_id/bank_mask, op1, gb_col_offset, col_num, auto_precharge)
            if len(cmd) >= 9:
                op1 = cmd[6]  # op1: (row_id, col_offset)
                col_num = cmd[8]  # 列数
                assert len(op1) >= 1, f"bk2gb/gb2bk command op1 format error: {op1}"
                row_access_records.append((op1[0], col_num))  # (row_id, col_len)
                    
        elif optype.name in ['host_read', 'host_write']:
            # Host读写: row_id在固定位置, col_num也在固定位置
            # (LEVEL, OPTYPE, ch_id, ra_id, de_mask, bank_id/bank_mask, row_id, col_offset, col_num, auto_precharge)
            if len(cmd) >= 9:
                row_id = cmd[6]
                col_num = cmd[8]
                row_access_records.append((row_id, col_num))  # (row_id, col_len)
        
        elif optype.name in ['reg2buf']:
            row_access_records.append((-1, cmd[7]))

    return row_access_records

# MARK: W + A

def get_partition_space(mm_size):
    # A. get hw partition space
    partition_tool = Partition(require_power_of_2 = True)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    partition_space = filtered_partition_space
    return partition_space

def test_baseline_with_buffer(args, batch_size=1, 
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
                       sequential_gen = False,
                       report_energy = False,
                       old_scale_mix = False,
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
    # aim 使用 global buffer 存储数据，当前配置对应 64 col
    available_input_buffer_in_col = SimConfig.de_gb // SimConfig.co_w
    # 分配 Scale 与数据 Buffer
    bk_per_pu = SimConfig.bg * SimConfig.ba // SimConfig.de_pu[0]
    assert bk_per_pu == 1, "这是 1 BK 架构"
    # 分配 A / B Scale Buffer，当前策略为平分
    scale_a_buf_col = 4
    scale_b_buf_col = 4
    # NOTE: 可能不能存在 Global Buffer 中
    if args.use_global_buffer:
        scale_b_buf_col = 0
    data_a_buf_col = available_input_buffer_in_col - scale_a_buf_col - scale_b_buf_col
    # quant_group_result_buf_col = 100000
    # extra_scale_b_buf_col = 100000
    # 最新实验中都通过这个参数指定数据 Buffer 大小
    if buffer_specify is not None:
        # if len(buffer_specify) >= 3:
        #     data_a_buf_col, scale_a_buf_col, scale_b_buf_col = buffer_specify[:3]
        # if len(buffer_specify) >= 5:
            data_a_buf_col, scale_a_buf_col, scale_b_buf_col, quant_group_result_buf_col, extra_scale_b_buf_col = buffer_specify[:5]
    if quant_group_result_buf_col < 1: return math.inf
    if not reorder and extra_scale_b_buf_col > 0: # reduce searching space
        return math.inf
    # 计算quant group result buffer 大小
    max_quant_group_result_size = quant_group_result_buf_col * SimConfig.co_w // accumulation_precision
    
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
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}", 
                                                           data_buf_size=data_a_buf_col, 
                                                           scale_a_buf_size=scale_a_buf_col,
                                                           scale_b_buf_size=scale_b_buf_col,
                                                           quant_group_result_buf_size=quant_group_result_buf_col,
                                                           extra_scale_b_buf_size=extra_scale_b_buf_col)
        should_close_files = True
    else:
        should_close_files = False
    
    # 分配结果 Buffer
    available_output_buffer_in_col = SimConfig.de_pu_bf // SimConfig.co_w
    assert available_output_buffer_in_col > 1, "Output Buffer 大小无法容纳一次计算"
    partial_buf_col = 1
    result_buf_col = available_output_buffer_in_col - partial_buf_col
    # NOTE: 可能不能存在 Global Buffer 中
    if args.use_global_buffer:
        partial_buf_col = 0
        result_buf_col = 0

    

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
    # map A
    A_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=A_size,
        scale_group=operand_A_info['scale_group'],
        matrix_precision=operand_A_info['matrix_precision'],
        scale_offset_precision=operand_A_info['scale_offset_precision'],
        with_scale=operand_A_info['with_scale'],
        with_offset=operand_A_info['with_offset'],
        buffer_guidance=None,
    )
    # map B
    B_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=B_size,
        scale_group=operand_B_info['scale_group'],
        matrix_precision=operand_B_info['matrix_precision'],
        scale_offset_precision=operand_B_info['scale_offset_precision'],
        with_scale=operand_B_info['with_scale'],
        with_offset=operand_B_info['with_offset'],
        buffer_guidance=None,
    )
    # map result
    result_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=(mm_size_per_pu[0], mm_size_per_pu[2]),
        scale_group=None,
        matrix_precision=result_precision,
        scale_offset_precision=accumulation_precision,
        with_scale=False,
        with_offset=False,
        buffer_guidance=None,
    )
    # design_space.append((compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping))
    
    baseline = (compute_level, pu_num, baseline_partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping)
    
    
    # compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping = baseline
    pu_mask = [True for _ in range(pu_num)]
    corner_case_analysis(baseline, operand_A_info, operand_B_info, result_precision, log_file)
    print(f"divieded mm size: {mm_size_per_pu}", file=log_file)

    # 在内存中的结构：
    # A_data # A_scale # B_data # B_scale # Result
    
    cmd_addr_list = []
    commands = []
    codegen_tool = BaseCodegen(require_power_of_2 = False)
    # MARK: W + A LoopGen
    # 尝试调整循环顺序，最优化 data 读取
    gen = LoopGenerator(mm_size_per_pu[0], mm_size_per_pu[2], mm_size_per_pu[1])
    # if mm_size_per_pu[0] >= 8:
    #     gen = LoopGenerator(1, mm_size_per_pu[2], mm_size_per_pu[1])
    if buffer_b:
        k_tile_size_data = data_a_buf_col*(SimConfig.co_w//operand_B_info['matrix_precision'])
        k_tile_size_scale = scale_a_buf_col*operand_B_info['scale_group'][1]*(SimConfig.co_w//operand_B_info['scale_offset_precision'])
        gen.set_loop_order(['n0', 'n1', 'n2', 'k0', 'k1', 'm0', 'm1', 'm2', 'k2'])
    else:
        k_tile_size_data = data_a_buf_col*(SimConfig.co_w//operand_A_info['matrix_precision'])
        k_tile_size_scale = scale_a_buf_col*operand_A_info['scale_group'][1]*(SimConfig.co_w//operand_A_info['scale_offset_precision'])
        gen.set_loop_order(['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2'])
    if k_tile_size_scale == 0: k_tile_size_scale = math.inf
    if k_tile > 0:
        gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
    else:
        if k_tile_size_data > k_tile_size_scale:
            k_tile = min(k_tile_size_scale, mm_size_per_pu[1])
            gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
        else:
            k_tile = min(k_tile_size_data, mm_size_per_pu[1])
            gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
    
    # MARK: mx gran

    mx_gran = max(scale_b_buf_col, 1)
    
    _row_offset = 0
    if mix_scale_mapping:
        if buffer_b:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
        else:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            try:
                if old_scale_mix:
                    B_addr = ScaleMixAddr(operand_B_info, gen, mx_gran, True, _row_offset)
                else:
                    B_addr = ScaleMixAddr_v2(operand_B_info, gen, mx_gran, True, _row_offset) # _v2
            except Exception as e:
                # print(f"Error: {e}")
                return math.inf
            # if breakdown_analysis:
            #     print(f"B_addr: {B_addr.sg_k0}, {B_addr.k_block_new}, {B_addr.actual_db_col_num}, {B_addr.db_col_num}", file=log_file)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    elif loop_friendly_mapping:
        if buffer_b:
            A_addr = LoopfriendlyAddr(operand_A_info, gen, A_mapping, False, _row_offset)
            # A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
        else:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    else:
        A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
        _row_offset = A_addr.get_end_row()
        B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
        _row_offset = B_addr.get_end_row()
        result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    
    # Use parallel address generation 
    if sequential_gen:
        cmd_addr_list = generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, 1, silent = silent)
    else:
        cmd_addr_list = generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, args.num_processes, silent = silent)
    # cmd_addr_list = generate_cmd_addr_list_sequential(gen, A_addr, B_addr, result_addr)

    # 使用 CmdAddrList 包装地址列表，简化访问
    cmd_list = CmdAddrList(cmd_addr_list)
    
    # 提取缩放轨迹数据 - 使用便捷方法替代硬编码索引
    a_data_trace = cmd_list.extract_a_data_trace()
    b_data_trace = cmd_list.extract_b_data_trace()
    a_scale_trace = cmd_list.extract_a_scale_trace()
    b_scale_trace = cmd_list.extract_b_scale_trace()
    
    A_data_buffer = DataBuffer(buffer_capacity=data_a_buf_col, data_trace=a_data_trace, buffer_name="A_DataBuffer")
    B_data_buffer = DataBuffer(buffer_capacity=data_a_buf_col, data_trace=b_data_trace, buffer_name="A_DataBuffer")
    if buffer_b: data_buffer = B_data_buffer
    else: data_buffer = A_data_buffer
    A_scale_buffer = ScaleBuffer(buffer_capacity=scale_a_buf_col, scale_trace=a_scale_trace, buffer_name="A_ScaleBuffer")
    
    if scale_b_buf_col > 0:
        if mix_scale_mapping:
            scale_b_prefetch_col = min(scale_b_buf_col, mx_gran)
        else:
            scale_b_prefetch_col = scale_b_buf_col
        B_scale_buffer = ScaleBuffer(buffer_capacity=scale_b_prefetch_col, scale_trace=b_scale_trace, buffer_name="B_ScaleBuffer")
    else:
        B_scale_buffer = None

    # 结果 Buffer 初始化
    max_result_num = result_buf_col * SimConfig.co_w // accumulation_precision
    max_result_num = max(1, max_result_num)
    result_buffer = ResultBuffer(max_result_capacity=max_result_num, buffer_name="ResultBuffer")
    
    # MARK: max partial num
    if mix_scale_mapping:
        max_partial_num = max(1, partial_buf_col) * SimConfig.co_w // accumulation_precision
    else:
        max_partial_num = max(1, partial_buf_col) * SimConfig.co_w // accumulation_precision

    # A -> BK 0, Scale A -> BK 0
    # B -> BK 1, Scale B -> BK 1
    config = (
        buffer_b, consider_data_read, consider_scale_buffer, consider_result_buffer, \
        pu_num, pu_mask, \
        mix_scale_mapping, accumulation_precision, mm_size_per_pu, \
        max_partial_num, max_quant_group_result_size, scale_b_buf_col, result_buf_col, extra_scale_b_buf_col, \
        codegen_tool, log_file
    )
    print("start code gen", file=log_file)
    performance_metrics, ir_list_list = aim_w_a(config, cmd_list, args,
        data_buffer, A_scale_buffer, B_scale_buffer, result_buffer,
        result_addr, B_addr,
    )
    print("start sim and reorder", file=log_file)
    (data_a_read_col_num, data_a_read_row_num,
        scale_a_read_col_num, scale_a_read_row_num,
        scale_b_read_col_num, scale_b_read_row_num,
        data_b_read_col_num, data_b_read_row_num,
    ) = performance_metrics
    all_commands = []
    # assert len(ir_list_list) == mm_size_per_pu[0], f"Invalid ir_list_list length: {len(ir_list_list)} != {mm_size_per_pu[0]}"
    for ir_list in ir_list_list:
        print("\n ======== origin ir list =========", file=log_file)
        ir_list.print()
        if report_energy:
            ir_list.log_energy(mix_scale_mapping, pu_num)
        if reorder:
            print("\n ======== performing 1st pass =========", file=log_file)
            ir_list.first_optimize_pass()
            print("\n ======== after 1st pass ========", file=log_file)
            ir_list.print()
            print("\n ======== performing 2nd pass =========", file=log_file)
            ir_list.second_optimize_pass()
            print("\n ======== after 2nd pass =========", file=log_file)
            ir_list.print()
            print("\n ======== performing 3rd pass =========", file=log_file)
            ir_list.third_optimize_pass()
            print("\n ======== after 3rd pass =========", file=log_file)
            ir_list.print()
        commands = ir_list.codegen()
        all_commands.extend(commands)
        # if mm_size_per_pu[0] >= 8: break
    if report_energy:
        print(SimConfig.calculate_energy())
        SimConfig.reset_energy()
    if breakdown_analysis:
        access_list = parse_row_change(all_commands)
        row_num = 0
        col_num = 0
        dequant_num = 0
        dequant_in_row_change = 0
        dequant_between_row_change = 0
        past_row = -1
        for id, (row, col_len) in enumerate(access_list):
            if row == -1:
                dequant_num += 1
                dequant_between_row_change += 1
                if id < len(access_list) - 1 and past_row != access_list[id+1][0]:
                    dequant_in_row_change += 1
            else:
                col_num += col_len
                if row != past_row:
                    row_num += 1
                    past_row = row
                    if dequant_between_row_change > 0:
                        # print(dequant_between_row_change)
                        dequant_between_row_change = 0
        print("=================================================")
        print("|   指标            |    行数     |   列数      |")
        print("=================================================")
        # print(f"| Latency           |   {lat:<9} |             |")
        print(f"| 读 A Scale        |   {scale_a_read_row_num:<9} |   {scale_a_read_col_num:<9} |")
        print(f"| 读 B Scale        |   {scale_b_read_row_num:<9} |   {scale_b_read_col_num:<9} |")
        print(f"| 读 A Data         |   {data_a_read_row_num:<9} |   {data_a_read_col_num:<9} |")
        print(f"| (DRAM) 读 B Data  |   {data_b_read_row_num:<9} |   {data_b_read_col_num:<9} |")
        print(f"| (DRAM) 整体       |   {row_num:<9} |   {col_num:<9} |")
        print(f"| (DRAM) Dequant    |   {dequant_num:<9} |   {dequant_in_row_change:<9} |")
        print("=================================================")

    # 按照原始顺序合并结果
    lat = sim([
        (0, [], all_commands)
    ], silent=True, use_tqdm=False)
    # if mm_size_per_pu[0] >= 8: lat = lat * mm_size_per_pu[0]
    if should_close_files:
        log_file.close()
        csvfile.close()
    return lat

def test_baseline_with_buffer_mixprepu(args, batch_size=1, 
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
                       sequential_gen = False,
                       report_energy = False,
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
    # aim 使用 global buffer 存储数据，当前配置对应 64 col
    available_input_buffer_in_col = SimConfig.de_gb // SimConfig.co_w
    # 分配 Scale 与数据 Buffer
    bk_per_pu = SimConfig.bg * SimConfig.ba // SimConfig.de_pu[0]
    assert bk_per_pu == 1, "这是 1 BK 架构"
    # 分配 A / B Scale Buffer，当前策略为平分
    scale_a_buf_col = 4
    scale_b_buf_col = 4
    # NOTE: 可能不能存在 Global Buffer 中
    if args.use_global_buffer:
        scale_b_buf_col = 0
    data_a_buf_col = available_input_buffer_in_col - scale_a_buf_col - scale_b_buf_col
    # quant_group_result_buf_col = 100000
    # extra_scale_b_buf_col = 100000
    # 最新实验中都通过这个参数指定数据 Buffer 大小
    if buffer_specify is not None:
        data_a_buf_col, scale_a_buf_col, scale_b_buf_col, quant_group_result_buf_col, extra_scale_b_buf_col = buffer_specify[:5]
    data_a_buf_col += scale_a_buf_col
    if quant_group_result_buf_col < 1: return math.inf
    if not reorder and extra_scale_b_buf_col > 0: # reduce searching space
        return math.inf
    # 计算quant group result buffer 大小
    max_quant_group_result_size = quant_group_result_buf_col * SimConfig.co_w // accumulation_precision
    
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
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}", 
                                                           data_buf_size=data_a_buf_col, 
                                                           scale_a_buf_size=scale_a_buf_col,
                                                           scale_b_buf_size=scale_b_buf_col,
                                                           quant_group_result_buf_size=quant_group_result_buf_col,
                                                           extra_scale_b_buf_size=extra_scale_b_buf_col)
        should_close_files = True
    else:
        should_close_files = False
    
    # 分配结果 Buffer
    available_output_buffer_in_col = SimConfig.de_pu_bf // SimConfig.co_w
    assert available_output_buffer_in_col > 1, "Output Buffer 大小无法容纳一次计算"
    partial_buf_col = 1
    result_buf_col = available_output_buffer_in_col - partial_buf_col
    # NOTE: 可能不能存在 Global Buffer 中
    if args.use_global_buffer:
        partial_buf_col = 0
        result_buf_col = 0

    

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
    # map A
    A_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=A_size,
        scale_group=operand_A_info['scale_group'],
        matrix_precision=operand_A_info['matrix_precision'],
        scale_offset_precision=operand_A_info['scale_offset_precision'],
        with_scale=operand_A_info['with_scale'],
        with_offset=operand_A_info['with_offset'],
        buffer_guidance=None,
    )
    # map B
    B_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=B_size,
        scale_group=operand_B_info['scale_group'],
        matrix_precision=operand_B_info['matrix_precision'],
        scale_offset_precision=operand_B_info['scale_offset_precision'],
        with_scale=operand_B_info['with_scale'],
        with_offset=operand_B_info['with_offset'],
        buffer_guidance=None,
    )
    # map result
    result_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=(mm_size_per_pu[0], mm_size_per_pu[2]),
        scale_group=None,
        matrix_precision=result_precision,
        scale_offset_precision=accumulation_precision,
        with_scale=False,
        with_offset=False,
        buffer_guidance=None,
    )
    # design_space.append((compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping))
    
    baseline = (compute_level, pu_num, baseline_partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping)
    
    
    # compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping = baseline
    pu_mask = [True for _ in range(pu_num)]
    corner_case_analysis(baseline, operand_A_info, operand_B_info, result_precision, log_file)
    print(f"divieded mm size: {mm_size_per_pu}", file=log_file)

    # 在内存中的结构：
    # A_data # A_scale # B_data # B_scale # Result
    
    cmd_addr_list = []
    commands = []
    codegen_tool = BaseCodegen(require_power_of_2 = False)
    # MARK: W + A LoopGen
    # 尝试调整循环顺序，最优化 data 读取
    gen = LoopGenerator(mm_size_per_pu[0], mm_size_per_pu[2], mm_size_per_pu[1])
    # if mm_size_per_pu[0] >= 8:
    #     gen = LoopGenerator(1, mm_size_per_pu[2], mm_size_per_pu[1])
    if buffer_b:
        k_tile_size_data = data_a_buf_col*(SimConfig.co_w//operand_B_info['matrix_precision'])
        # k_tile_size_scale = scale_a_buf_col*operand_B_info['scale_group'][1]*(SimConfig.co_w//operand_B_info['scale_offset_precision'])
        gen.set_loop_order(['n0', 'n1', 'n2', 'k0', 'k1', 'm0', 'm1', 'm2', 'k2'])
    else:
        k_tile_size_data = data_a_buf_col*(SimConfig.co_w//operand_A_info['matrix_precision'])
        # k_tile_size_scale = scale_a_buf_col*operand_A_info['scale_group'][1]*(SimConfig.co_w//operand_A_info['scale_offset_precision'])
        gen.set_loop_order(['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2'])
    # if k_tile_size_scale == 0: k_tile_size_scale = math.inf
    if k_tile > 0:
        gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
    else:
        # process loop tile size, in mix precision pu
        k_tile = (SimConfig.co_w//operand_B_info['matrix_precision']) * (k_tile_size_data // (SimConfig.co_w//operand_B_info['matrix_precision']))
        if k_tile == 0:
            return math.inf
        k_tile = min(k_tile_size_data, mm_size_per_pu[1])
        gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
    
    # MARK: mx gran

    mx_gran = max(scale_b_buf_col, 1)
    
    _row_offset = 0
    if mix_scale_mapping:
        if buffer_b:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
        else:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            try:
                B_addr = ScaleMixAddr_v2(operand_B_info, gen, mx_gran, True, _row_offset) # _v2
            except Exception as e:
                # print(f"Error: {e}")
                return math.inf
            # if breakdown_analysis:
            #     print(f"B_addr: {B_addr.sg_k0}, {B_addr.k_block_new}, {B_addr.actual_db_col_num}, {B_addr.db_col_num}", file=log_file)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    elif loop_friendly_mapping:
        if buffer_b:
            A_addr = LoopfriendlyAddr(operand_A_info, gen, A_mapping, False, _row_offset)
            # A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
        else:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    else:
        A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
        _row_offset = A_addr.get_end_row()
        B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
        _row_offset = B_addr.get_end_row()
        result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    
    # Use parallel address generation 
    if sequential_gen:
        cmd_addr_list = generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, 1, silent = silent)
    else:
        cmd_addr_list = generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, args.num_processes, silent = silent)
    # cmd_addr_list = generate_cmd_addr_list_sequential(gen, A_addr, B_addr, result_addr)

    # 使用 CmdAddrList 包装地址列表，简化访问
    cmd_list = CmdAddrList(cmd_addr_list)
    # print(f"cmd_list: {cmd_list.get_pair(0)[0].B_scale_group_b_id}")
    
    # 提取缩放轨迹数据 - 使用便捷方法替代硬编码索引
    a_data_trace = cmd_list.extract_a_data_trace()
    b_data_trace = cmd_list.extract_b_data_trace()
    a_scale_trace = cmd_list.extract_a_scale_trace()
    b_scale_trace = cmd_list.extract_b_scale_trace()
    
    A_data_buffer = DataBuffer(buffer_capacity=data_a_buf_col, data_trace=a_data_trace, buffer_name="A_DataBuffer")
    B_data_buffer = DataBuffer(buffer_capacity=data_a_buf_col, data_trace=b_data_trace, buffer_name="A_DataBuffer")
    if buffer_b: data_buffer = B_data_buffer
    else: data_buffer = A_data_buffer
    A_scale_buffer = ScaleBuffer(buffer_capacity=1, scale_trace=a_scale_trace, buffer_name="A_ScaleBuffer")
    
    if scale_b_buf_col > 0:
        if mix_scale_mapping:
            scale_b_prefetch_col = min(scale_b_buf_col, mx_gran)
        else:
            scale_b_prefetch_col = scale_b_buf_col
        B_scale_buffer = ScaleBuffer(buffer_capacity=scale_b_prefetch_col, scale_trace=b_scale_trace, buffer_name="B_ScaleBuffer")
    else:
        B_scale_buffer = None

    # 结果 Buffer 初始化
    max_result_num = result_buf_col * SimConfig.co_w // accumulation_precision
    max_result_num = max(1, max_result_num)
    result_buffer = ResultBuffer(max_result_capacity=max_result_num, buffer_name="ResultBuffer")
    
    # MARK: max partial num
    if mix_scale_mapping:
        max_partial_num = max(1, partial_buf_col) * SimConfig.co_w // accumulation_precision
    else:
        max_partial_num = max(1, partial_buf_col) * SimConfig.co_w // accumulation_precision

    # A -> BK 0, Scale A -> BK 0
    # B -> BK 1, Scale B -> BK 1
    config = (
        buffer_b, consider_data_read, consider_scale_buffer, consider_result_buffer, \
        pu_num, pu_mask, \
        mix_scale_mapping, accumulation_precision, mm_size_per_pu, \
        max_partial_num, max_quant_group_result_size, scale_b_buf_col, result_buf_col, extra_scale_b_buf_col, \
        codegen_tool, log_file
    )
    print("start code gen", file=log_file)
    performance_metrics, ir_list_list = aim_mixprepu(config, cmd_list, args,
        data_buffer, A_scale_buffer, B_scale_buffer, result_buffer,
        result_addr, B_addr,
    )
    print("start sim and reorder", file=log_file)
    (data_a_read_col_num, data_a_read_row_num,
        scale_a_read_col_num, scale_a_read_row_num,
        scale_b_read_col_num, scale_b_read_row_num,
        data_b_read_col_num, data_b_read_row_num,
    ) = performance_metrics
    all_commands = []
    # assert len(ir_list_list) == mm_size_per_pu[0], f"Invalid ir_list_list length: {len(ir_list_list)} != {mm_size_per_pu[0]}"
    for ir_list in ir_list_list:
        print("\n ======== origin ir list =========", file=log_file)
        ir_list.print()
        if report_energy:
            ir_list.log_energy(mix_scale_mapping, pu_num)
        if reorder:
            print("\n ======== performing 1st pass =========", file=log_file)
            ir_list.first_optimize_pass()
            print("\n ======== after 1st pass ========", file=log_file)
            ir_list.print()
            print("\n ======== performing 2nd pass =========", file=log_file)
            ir_list.second_optimize_pass()
            print("\n ======== after 2nd pass =========", file=log_file)
            ir_list.print()
            print("\n ======== performing 3rd pass =========", file=log_file)
            ir_list.third_optimize_pass()
            print("\n ======== after 3rd pass =========", file=log_file)
            ir_list.print()
        commands = ir_list.codegen()
        all_commands.extend(commands)
        # if mm_size_per_pu[0] >= 8: break
    if report_energy:
        print(SimConfig.calculate_energy())
        SimConfig.reset_energy()
    if breakdown_analysis:
        access_list = parse_row_change(all_commands)
        row_num = 0
        col_num = 0
        dequant_num = 0
        dequant_in_row_change = 0
        dequant_between_row_change = 0
        past_row = -1
        for id, (row, col_len) in enumerate(access_list):
            if row == -1:
                dequant_num += 1
                dequant_between_row_change += 1
                if id < len(access_list) - 1 and past_row != access_list[id+1][0]:
                    dequant_in_row_change += 1
            else:
                col_num += col_len
                if row != past_row:
                    row_num += 1
                    past_row = row
                    if dequant_between_row_change > 0:
                        # print(dequant_between_row_change)
                        dequant_between_row_change = 0
        print("=================================================")
        print("|   指标            |    行数     |   列数      |")
        print("=================================================")
        # print(f"| Latency           |   {lat:<9} |             |")
        print(f"| 读 A Scale        |   {scale_a_read_row_num:<9} |   {scale_a_read_col_num:<9} |")
        print(f"| 读 B Scale        |   {scale_b_read_row_num:<9} |   {scale_b_read_col_num:<9} |")
        print(f"| 读 A Data         |   {data_a_read_row_num:<9} |   {data_a_read_col_num:<9} |")
        print(f"| (DRAM) 读 B Data  |   {data_b_read_row_num:<9} |   {data_b_read_col_num:<9} |")
        print(f"| (DRAM) 整体       |   {row_num:<9} |   {col_num:<9} |")
        print(f"| (DRAM) Dequant    |   {dequant_num:<9} |   {dequant_in_row_change:<9} |")
        print("=================================================")

    # 按照原始顺序合并结果
    lat = sim([
        (0, [], all_commands)
    ], silent=True, use_tqdm=False)
    # if mm_size_per_pu[0] >= 8: lat = lat * mm_size_per_pu[0]
    if should_close_files:
        log_file.close()
        csvfile.close()
    return lat

# MARK: Weight Only
def test_baseline_w_only_with_buffer(args, batch_size=1, 
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
                       buffer_b_old = False,
                       sequential_gen = False,
                       ):
    a_prec = operand_A_info['matrix_precision']
    b_prec = operand_B_info['matrix_precision']
    scale_precision = operand_B_info['scale_offset_precision']
    if silent:
        tqdm.tqdm = tqdm_replacement
    # if log_file is None:
    #     workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}x{operand_B_info['matrix_size'][0]}"
    #     # Add precision and scale group info for weight-only quantization
    #     b_scale_group = operand_B_info['scale_group'][1] if operand_B_info['with_scale'] else 0
    #     function_desc = f"W{b_prec}A{a_prec}S{scale_precision}_sg{b_scale_group}"
    #     # Auto-detect function name for directory, use function_desc in filename
    #     log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}")
    #     should_close_files = True
    # else:
    #     should_close_files = False

    # This case is for W only
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

    # Input Buffer分配
    available_input_buffer_in_col = SimConfig.de_gb // SimConfig.co_w
    bk_per_pu = SimConfig.bg * SimConfig.ba // SimConfig.de_pu[0]
    # 1 BK 架构
    assert bk_per_pu == 1, "这是 1 BK 架构"

    scale_buf_col = 4 # 4 col 的 Scale B 缓冲
    # scale_buf_col = 30
    # Data B 有可能需要预留一列的中间缓冲？
    data_buf_col = available_input_buffer_in_col - scale_buf_col - 1 # 59 col
    # data_buf_col = 32
    if buffer_specify is not None:
        data_buf_col, dequant_buf_col, scale_buf_col, _, _ = buffer_specify
    if log_file is None:
        workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}x{operand_B_info['matrix_size'][0]}"
        # Add precision and scale group info for weight-only quantization
        function_desc = f"W{b_prec}A{a_prec}S{scale_precision}_sg{operand_B_info['scale_group'][0]}x{operand_B_info['scale_group'][1]}"
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}", 
                                                    data_buf_size=data_buf_col,
                                                    scale_b_buf_size=scale_buf_col,
                                                    quanted_buf_size=dequant_buf_col)
        should_close_files = True
    else:
        should_close_files = False
    if buffer_b and not buffer_b_old:
        assert dequant_buf_col > 0 # and dequant_buf_col % ( a_prec / b_prec ) == 0
        dequant_buf_col = int(dequant_buf_col // ( a_prec / b_prec ))
        if dequant_buf_col == 0: return math.inf
        if data_buf_col % dequant_buf_col != 0: return math.inf
    data_buf_col += dequant_buf_col # dequant 后的数据无需存一份原数据

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
    if buffer_b and mm_size_per_pu[0] == 4096:
        latency_scale = mm_size_per_pu[2]
        mm_size_per_pu = (mm_size_per_pu[0], mm_size_per_pu[1], 1)
    else:
        latency_scale = 1
    A_size = (mm_size_per_pu[0], mm_size_per_pu[1])
    B_size = (mm_size_per_pu[2], mm_size_per_pu[1])
    # map A
    A_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=A_size,
        scale_group=operand_A_info['scale_group'],
        matrix_precision=operand_A_info['matrix_precision'],
        scale_offset_precision=operand_A_info['scale_offset_precision'],
        with_scale=operand_A_info['with_scale'],
        with_offset=operand_A_info['with_offset'],
        buffer_guidance=None,
    )
    # map B
    B_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=B_size,
        scale_group=operand_B_info['scale_group'],
        matrix_precision=operand_B_info['matrix_precision'],
        scale_offset_precision=operand_B_info['scale_offset_precision'],
        with_scale=operand_B_info['with_scale'],
        with_offset=operand_B_info['with_offset'],
        buffer_guidance=None,
    )
    # map result
    result_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=(mm_size_per_pu[0], mm_size_per_pu[2]),
        scale_group=None,
        matrix_precision=result_precision,
        scale_offset_precision=accumulation_precision,
        with_scale=False,
        with_offset=False,
        buffer_guidance=None,
    )
    
    baseline = (compute_level, pu_num, baseline_partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping)

    pu_mask = [True for _ in range(pu_num)]
    corner_case_analysis(baseline, operand_A_info, operand_B_info, result_precision, log_file)

    cmd_addr_list = []
    commands = []
    codegen_tool = BaseCodegen(require_power_of_2 = False)
    
    # 结果 Buffer 分配
    result_buf_col = SimConfig.de_pu_bf // SimConfig.co_w
    
    # gen = LoopGenerator(mm_size_per_pu[0], mm_size_per_pu[2], mm_size_per_pu[1])
    # gen.set_tile_sizes(m=1,n=1,k=data_buf_col*SimConfig.co_w//operand_A_info['matrix_precision']) # 设置内层 tile size
    # gen.set_loop_order(['m0', 'm1', 'k0', 'n0', 'n1', 'k1'])
    
    gen = LoopGenerator(mm_size_per_pu[0], mm_size_per_pu[2], mm_size_per_pu[1])
    k_tile_size_scale = scale_buf_col*operand_B_info['scale_group'][1]*(SimConfig.co_w//operand_B_info['scale_offset_precision'])
    if buffer_b:
        k_tile_size_data = min(data_buf_col, dequant_buf_col)*(SimConfig.co_w//operand_B_info['matrix_precision'])
        if buffer_b_old:
            k_tile_size_data = data_buf_col*(SimConfig.co_w//operand_B_info['matrix_precision'])
        # print(f"k_tile_size_data: {k_tile_size_data}, k_tile_size_scale: {k_tile_size_scale}")
        gen.set_loop_order(['n0', 'n1', 'n2', 'k0', 'k1', 'm0', 'm1', 'm2', 'k2'])
    else:
        k_tile_size_data = data_buf_col*(SimConfig.co_w//operand_A_info['matrix_precision'])
        gen.set_loop_order(['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2'])
        # NOTE: 在bufferA时，暂时不考虑更改遍历顺序
        # gen.set_block_sizes(k_block0=k_tile_size_data, k_block1=k_tile_size_data)
    if k_tile > 0:
        gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
    else:
        if k_tile_size_data > k_tile_size_scale:
            k_tile = min(k_tile_size_scale, mm_size_per_pu[1])
            gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
        else:
            k_tile = min(k_tile_size_data, mm_size_per_pu[1])
            gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)

    # 指令地址映射
    # DRAM 中的数据排布
        # Data A
        # Data B
        # Scale B
        # Result
    
    # _offset = 0
    # A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _offset)
    # _offset = A_addr.get_end_row()
    # B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _offset)
    # _offset = B_addr.get_end_row()
    # result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _offset)

    _row_offset = 0
    if mix_scale_mapping:
        raise NotImplementedError("Loop-friendly mapping is not implemented for B buffer")
    elif loop_friendly_mapping:
        if buffer_b:
            A_addr = LoopfriendlyAddr(operand_A_info, gen, A_mapping, False, _row_offset)
            # A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            # B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
            B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset, mm_size_per_pu[1])
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
        else:
            A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
            _row_offset = A_addr.get_end_row()
            B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset)
            _row_offset = B_addr.get_end_row()
            result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    else:
        A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
        _row_offset = A_addr.get_end_row()
        B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
        _row_offset = B_addr.get_end_row()
        result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)


    # Use parallel address generation for W-only case with parallel loop generator
    print(f"Generating W-only addresses using parallel loop generator and {args.num_processes} processes...", file=log_file)
    start_time = time.time()
    if sequential_gen:
        cmd_addr_list = generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, 1, silent = silent)
    else:
        cmd_addr_list = generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, args.num_processes, silent)
    end_time = time.time()
    print(f"W-only address generation completed in {end_time - start_time:.2f} seconds", file=log_file)
    
    # 使用 CmdAddrList 包装地址列表，简化W-only模式的访问
    cmd_list_w = CmdAddrList(cmd_addr_list)
    
    # 提取轨迹数据 - 使用便捷方法替代硬编码索引
    b_scale_trace = cmd_list_w.extract_b_scale_trace()
    a_data_trace = cmd_list_w.extract_a_data_trace()
    b_data_trace = cmd_list_w.extract_b_data_trace()

    # 初始化 Buffer 实例
    B_scale_buffer = ScaleBuffer(
        buffer_capacity=scale_buf_col,
        scale_trace=b_scale_trace,
        buffer_name="B_ScaleBuffer"
    )
    A_data_buffer = DataBuffer(
        buffer_capacity=data_buf_col,
        data_trace=a_data_trace,
        buffer_name="A_DataBuffer"
    )
    B_data_buffer = DataBuffer(
        buffer_capacity=data_buf_col,
        data_trace=b_data_trace,
        buffer_name="B_DataBuffer"
    )
    B_dequant_buffer = DataBuffer(
        buffer_capacity=dequant_buf_col,
        data_trace=b_data_trace,
        buffer_name="B_DequantBuffer"
    )
    if buffer_b:
        data_buffer = B_data_buffer
    else:
        data_buffer = A_data_buffer
    # 结果 Buffer 初始化
    max_result_num = result_buf_col * SimConfig.co_w // accumulation_precision
    assert max_result_num > 0, f"max_result_num: {max_result_num}"
    result_buffer = ResultBuffer(
        max_result_capacity=max_result_num,
        buffer_name="ResultBuffer"
    )

    # 统计指标
    # buffer_read_num = 0]
    config = (
    buffer_b, consider_data_read, consider_scale_buffer, consider_result_buffer, \
    pu_num, pu_mask, a_prec, b_prec, scale_precision,\
    mix_scale_mapping, accumulation_precision, \
    data_buf_col, data_buf_col, dequant_buf_col, result_buf_col, \
    codegen_tool, log_file
    )
    if buffer_b:
        if buffer_b_old:
            performance_metrics, commands = aim_w(config, cmd_list_w, args,
                data_buffer, B_scale_buffer, B_dequant_buffer, result_buffer,
                result_addr, B_addr,
            )
        else:
            performance_metrics, commands = aim_w_buf_b(config, cmd_list_w, args,
                data_buffer, B_scale_buffer, B_dequant_buffer, result_buffer,
                result_addr, B_addr,
            )
    else:
        performance_metrics, commands = aim_w(config, cmd_list_w, args,
            data_buffer, B_scale_buffer, B_dequant_buffer, result_buffer,
            result_addr, B_addr,
        )
    (
        data_read_col_num, data_read_row_num,
        buffer_read_col_num, buffer_read_row_num,
        scale_read_col_num, scale_read_row_num,
    ) = performance_metrics
    # 运行仿真
    lat = sim([
        (0, [], commands)
    ], silent=True, use_tqdm=False)
    if breakdown_analysis:
        print("=================================================")
        print("|   指标           |    行数     |   列数      |")
        print("=================================================")
        # print(f"| Latency           |   {lat:<9} |             |")
        print(f"| Scale Buf Read   |   {scale_read_row_num:<9} |   {scale_read_col_num:<9} |")
        print(f"| Data Buf Read    |   {buffer_read_row_num:<9} |   {buffer_read_col_num:<9} |")
        print(f"| Compute Read     |   {data_read_row_num:<9} |   {data_read_col_num:<9} |")
        print("=================================================")

    if should_close_files:
        log_file.close()
        csvfile.close()
    
    return lat * latency_scale

""" 量化函数
- 如果需要和激活函数做融合，应当优先做 Re-Layout，否则感觉有点怪
- 量化函数的输入应当是一个已有的 Layout 分布：

    调用流程，以 MM 函数为例，其输入中已知上一个算子输出的 Layout
    a. 分割算子，确定当前算子输入 Layout
    b. 根据输入 Layout，应当反推对应的重排后量化前的 Layout
    c. 根据上一个算子的输出 Layout / 重排后量化前的 Layout / 输入 Layout，需要的激活函数类型，调用
        - re-layout: 上一个算子的输出 Layout + 分割方案 -> 重排后量化前的 Layout + 新分割方案
        - quantize & activation 函数：重排后量化前的 Layout -> 输入 Layout

    因此量化函数的输入应该为：
    1. 当前 Bank 分到的 Size（重排后量化前的 Layout）
    2. 是否需要做最大值的同步，和哪些部分做最大值的同步（如果已有分布中，一个 scale group 被切开到多个 Bank 存放了，那么最大值可能需要和其他部分一起确定）
        - 关于这一点，我想先考虑不需要的情况，即如果分组数很小，其实问题不会很大
"""

def test_baseline_quant(args,batch_size=1, 
                       operand_A_info={
                           'matrix_size': (1, 5120), # (m, k)
                           'scale_group': (1, 128),
                           'original_matrix_precision': 16,
                           'matrix_precision': 8,
                           'scale_offset_precision': 16,
                           'with_scale': True,
                           'with_offset': False,
                       },
                       log_file=None,
                       ):

    if log_file is None:
        workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}"
        # Add precision and scale group info for quantization
        orig_prec = operand_A_info['original_matrix_precision']
        quant_prec = operand_A_info['matrix_precision']
        scale_group = operand_A_info['scale_group'][1] if operand_A_info['with_scale'] else 0
        function_desc = f"quant_{orig_prec}to{quant_prec}_sg{scale_group}"
        # Auto-detect function name for directory, use function_desc in filename
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"{workload_str}_{function_desc}")
        should_close_files = True
    else:
        should_close_files = False

    # 暂时不考虑 offset 的实现
    assert operand_A_info['with_offset'] == False, "暂不考虑 offset 的实现"
    """
    0. parse hardware information
    """
    available_input_buffer_in_col = SimConfig.de_gb // SimConfig.co_w
    actual_group_k = min(operand_A_info['matrix_size'][1], operand_A_info['scale_group'][1])
    actual_group_m = min(operand_A_info['matrix_size'][0], operand_A_info['scale_group'][0])

    # 分配时保证 Scale Buffer 中的数据使用时，Data Buffer 一定能存下？
    data_in_scale_group = actual_group_k * actual_group_m * operand_A_info['matrix_precision']
    scale_in_scale_group = operand_A_info['scale_offset_precision']
    # data_buf_col = math.ceil(available_input_buffer_in_col * data_in_scale_group / (data_in_scale_group + scale_in_scale_group))
    data_buf_col = 32
    scale_buf_col = 1000000
    assert scale_buf_col > 0, f"scale_buf_col: {scale_buf_col}"
    print(f"data_buf_col: {data_buf_col}, scale_buf_col: {scale_buf_col}", file=log_file)
    
    # 分配 Scale 与数据 Buffer
    bk_per_pu = SimConfig.bg * SimConfig.ba // SimConfig.de_pu[0]   
    assert bk_per_pu == 1

    # # 分配 A / B Scale Buffer，当前策略为平分
    # # 分配结果 Buffer
    # available_output_buffer_in_col = SimConfig.de_pu_bf // SimConfig.co_w
    # assert available_output_buffer_in_col > 1, "Output Buffer 大小无法容纳一次计算"
    # partial_buf_col = 1
    # result_buf_col = available_output_buffer_in_col - partial_buf_col

    """
    NOTE: 1. Get design space
    """
    # get mm size
    m_size = operand_A_info['matrix_size'][0]
    k_size = operand_A_info['matrix_size'][1]
    A_size = (m_size, k_size)

    # get group size
    assert operand_A_info['with_scale']
    s_group_m = operand_A_info['scale_group'][0]
    s_group_k = operand_A_info['scale_group'][1]
    
    print("Input info", operand_A_info, file=log_file)
    
    # A. get hw partition space
    partition_tool = Partition(require_power_of_2 = args.po2)
    A_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=A_size,
        scale_group=operand_A_info['scale_group'],
        matrix_precision=operand_A_info['matrix_precision'],
        scale_offset_precision=operand_A_info['scale_offset_precision'],
        with_scale=operand_A_info['with_scale'],
        with_offset=operand_A_info['with_offset'],
        buffer_guidance=None,
    )
    original_A_mapping = partition_tool.mem_mapping_matrix_baseline(
        matrix_size=A_size,
        scale_group=None,
        matrix_precision=operand_A_info['original_matrix_precision'],
        scale_offset_precision=None,
        with_scale=False,
        with_offset=False,
        buffer_guidance=None,
    )
    pu_num = SimConfig.de_pu[0]
    pu_mask = [True for _ in range(pu_num)]
    # corner_case_analysis(baseline, operand_A_info, operand_B_info, result_precision)

    # 生成指令地址列表
    _OFFSET = 0
    A_addr = TensorAddr(operand_A_info, A_size, A_mapping, _OFFSET) # BK 0
    _OFFSET = A_addr.get_end_row()
    orginal_A_addr = TensorAddr(operand_A_info, A_size, original_A_mapping, _OFFSET) # BK 1
    
    cmd_addr_list = []
    commands = []
    codegen_tool = BaseCodegen(require_power_of_2 = False)

    for m in range(A_size[0]):
        for k in range(A_size[1]):
            A_row, A_col = A_addr.get_tensor_addr(m, k)
            original_A_row, original_A_col = orginal_A_addr.get_tensor_addr(m, k)
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = A_addr.get_scale_addr(m, k)
            if cmd_addr_list == []:
                cmd_addr_list.append([A_row, A_col, original_A_row, original_A_col, A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col])
            else:
                if [A_row, A_col, original_A_row, original_A_col, A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col] != cmd_addr_list[-1]:
                    cmd_addr_list.append([A_row, A_col, original_A_row, original_A_col, A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col])
    cmd_addr_list = CmdAddrList(cmd_addr_list, CmdAddrConfig.A_MATRIX_FORMAT)  
    # scale_trace = cmd_addr_list.extract_a_scale_trace()
    A_scale_buffer = ResultBuffer(
        max_result_capacity=scale_buf_col,
        buffer_name="A_ScaleBuffer"
    )

    # original A -> BK 0
    # A -> BK 1, Scale A -> BK 1
    scale_a_row_offset = A_mapping['data_mapping']['a_row_num'] * A_mapping['data_mapping']['b_row_num']
    
    # 生成计算指令
    commands = []
    commands2 = []
    
    """
    1. 求最大值直到scale buffer填满（优先读一个量化分组的）
    2. 利用这个数据去反过来量化原数据，并存入另一个bank
    3. 存 scale buffer
    """
    # get write latency
    data_col = math.inf
    AB_pos = []
    for cmd_addr_idx, cmd_addr in enumerate(cmd_addr_list):
        """
        如果当前结果需要使用到新的 Scale 数据，则需要处理读入
        """
        current_scale = cmd_addr.A_scale_pos
        current_cmd: CmdAddr
        next_cmd: Optional[CmdAddr]
        current_cmd, next_cmd = cmd_addr_list.get_pair(cmd_addr_idx)
        if next_cmd is None or not current_cmd.same_AB_pos(next_cmd):
            AB_pos.append(current_cmd.AB_pos)
        
        # 只在 scale buffer 满的时候，触发量化操作，注意，此处需要补充完前序的求最大值过程
        # 当前只考虑 m 维度组数为 1 的情况
        assert operand_A_info['scale_group'][0] == 1, "当前只考虑 m 维度组数为 1 的情况"
        
        A_scale_buffer.add_result(current_scale, 1)
        if next_cmd is None:
            # 4. 存 scale buffer
            scale_dict: dict = A_scale_buffer.get_writeback_data()
            for scale_row, scale_col in scale_dict.keys():
                print(f"write scale to bank 0, r{scale_row}, c{scale_col}", file=log_file)
                commands.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, scale_row, scale_col),
                    buf = (True, 0, 1),
                    auto_precharge = False,
                ))
            A_scale_buffer.clear_buffer()
            # 暂时不用 Model data buffer 的行为
            # 1. 存储 data 到 dram
            for A_row, A_col, original_A_row, original_A_col in AB_pos:
                print(f"get max value from bank0, r{original_A_row}, c{original_A_col}", file=log_file)
                # commands.append(
                #     codegen_tool.create_device_bk2buf(
                #         ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                #         op1 = (0, original_A_row, original_A_col),
                #         buf = (True, 0, 1),
                #         auto_precharge = False,
                #     )
                # )
                commands.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, original_A_row, original_A_col),
                    buf = (True, 0, 1),
                    auto_precharge = False,
                ))
                commands2.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, original_A_row, original_A_col),
                    buf = (True, 0, 1),
                    auto_precharge = False,
                ))
            # 2. 开始量化原数据，并存入 Buffer
            A_pos = []
            for A_row, A_col, original_A_row, original_A_col in AB_pos:
                # 收集 A 信息，以供后续使用
                if (A_row, A_col) not in A_pos:
                    A_pos.append((A_row, A_col))
                print(f"quantize from bank0, r{original_A_row}, c{original_A_col} to output buffer", file=log_file)
                commands.append(codegen_tool.create_device_pu(
                        ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                        op1 = (0, original_A_row, original_A_col),
                        op2 = (0, 1, A_col),
                        col_num = 1, 
                        auto_precharge = False,
                    ))
                if len(A_pos) >= data_buf_col:
                    for A_row, A_col in A_pos:
                        print(f"write data from output buffer, to bank0, r{A_row}, c{A_col}", file=log_file)
                        commands.append(codegen_tool.create_device_buf2bk(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            op1 = (0, A_row, A_col),
                            buf = (True, 0, 1),
                            auto_precharge = False,
                        ))
                    A_pos = []
            # # 3. 存量化结果数据
            # assert len(A_pos) <= data_buf_col, f"A_pos: {A_pos}, data_buf_col: {data_buf_col}"
            
            
            # print(f"A_scale_buffer needs writeback: {A_scale_buffer.get_buffer_status()}", file=log_file)
            # 如果需要写回，则需要处理写回
            # print(f"AB_pos: {AB_pos}", file=log_file)
            AB_pos = []

    # NOTE: sim 1
    lat1 = sim([
        (0, [], commands)
    ], silent=True, use_tqdm=False)
    lat2 = sim([
        (0, [], commands2)
    ], silent=True, use_tqdm=False)
    if should_close_files:
        log_file.close()
        csvfile.close()
    
    return lat1, lat2