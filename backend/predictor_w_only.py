import tqdm
import time
import math
from tools import *
from midend import *
from backend import *
from tools import *
from sim import sim
import math

def predict_w_only_rebuttal(args, batch_size=1, 
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

    assert buffer_specify is not None
    data_buf_col, dequant_buf_col, scale_buf_col, _, _ = buffer_specify
    if log_file is None:
        workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}x{operand_B_info['matrix_size'][0]}"
        # Add precision and scale group info for weight-only quantization
        function_desc = f"W{b_prec}A{a_prec}S{scale_precision}_sg{operand_B_info['scale_group'][0]}x{operand_B_info['scale_group'][1]}"
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"Predict_{workload_str}_{function_desc}", 
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
    assert partition_specify is not None
    baseline_partition = partition_specify
    
    # compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping = baseline

    # NOTE: get size after mapping & adjust mapping through scale group
    # for index in tqdm.tqdm(range(len(partition_space))):
        # compute_level, pu_num, partition = partition_space[index]
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
    
    baseline = (compute_level, pu_num, baseline_partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping)

    pu_mask = [True for _ in range(pu_num)]
    # corner_case_analysis(baseline, operand_A_info, operand_B_info, result_precision, log_file)    
    # 结果 Buffer 分配
    result_buf_col = SimConfig.de_pu_bf // SimConfig.co_w
    # 结果 Buffer 初始化
    max_result_num = result_buf_col * SimConfig.co_w // accumulation_precision
    assert max_result_num > 0, f"max_result_num: {max_result_num}"
    
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
    assert k_tile == 0
    if k_tile_size_data > k_tile_size_scale:
        return math.inf
        k_tile = min(k_tile_size_scale, mm_size_per_pu[1])
        gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)
    else:
        k_tile = min(k_tile_size_data, mm_size_per_pu[1])
        gen.set_block_sizes(k_block0=k_tile, k_block1=k_tile)

    _row_offset = 0
    assert loop_friendly_mapping
    if buffer_b:
        A_addr = LoopfriendlyAddr(operand_A_info, gen, A_mapping, False, _row_offset)
        # A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
        _row_offset = A_addr.get_end_row()
        # B_addr = TensorAddr(operand_B_info, (mm_size_per_pu[2], mm_size_per_pu[1]), B_mapping, _row_offset)
        B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset)
        B_addr.k_block = mm_size_per_pu[1]
        _row_offset = B_addr.get_end_row()
        result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    else:
        A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
        _row_offset = A_addr.get_end_row()
        _offset1 = _row_offset
        B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset)
        _row_offset = B_addr.get_end_row()
        B_row_num = _row_offset - _offset1
        result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)


    b_data_in_col = SimConfig.co_w // operand_B_info['matrix_precision']
    a_data_in_col = SimConfig.co_w // operand_A_info['matrix_precision']
    scale_in_col = SimConfig.co_w // operand_B_info['scale_offset_precision']

    
    def whether_is_a_row_change(k_id):
        k_iter_id = k_id // k_tile
        k_in_iter_id = k_id % k_tile
        col_id = k_iter_id * math.ceil(k_tile/a_data_in_col) * mm_size_per_pu[0] + k_in_iter_id // a_data_in_col
        if col_id % SimConfig.co == 0:
            return True
        else:
            return False
    # 大胆预测，n group > 1 时，B scale 需要可以跨 k 大小
    group_size_n = min(mm_size_per_pu[2], operand_B_info['scale_group'][0])
    group_size_k = min(mm_size_per_pu[1], operand_B_info['scale_group'][1])
    # if group_size_n > 1 and ceil(mm_size_per_pu[1]/group_size_k) > scale_buf_col * scale_in_col:
    #     return math.inf
    dequant_a_col_dict = {}
    scale_read_col_dict = {}
    data_read_col_dict = {}
    data_read_col = 0
    scale_read_col = 0
    k_col = math.ceil(mm_size_per_pu[1] / b_data_in_col)
    row_change_overhead = SimConfig.read_row_change_apox - SimConfig.col_change_apox

    if buffer_b:
        scale_list = []
        scale_col_left = 0
        extra_row_change_num = 0
        # k_iter = math.ceil(mm_size_per_pu[1] / k_tile)
        # get dequant k id
        # k_col = math.ceil(mm_size_per_pu[1] / b_data_in_col)
        for n in range(mm_size_per_pu[2]):
            for k_col_id in range(k_col):
                flat_col_id = k_col_id + n * k_col
                k_id = k_col_id * b_data_in_col
                scale_addr = B_addr.get_scale_addr(n, k_id)
                scale_addr = (scale_addr[2],scale_addr[3])
                if scale_addr not in scale_list:
                    # if not silent:
                    #     print(f"scale_addr: {scale_addr}")
                    scale_read_col += 1
                    # change buffer
                    if scale_col_left == 0:
                        if not silent:
                            print(f"flat_col_id: {flat_col_id}, scale_read_col_dict: {scale_list}")
                        scale_list = [scale_addr]
                        scale_col_left = scale_buf_col - 1
                        scale_read_col_dict[flat_col_id] = whether_is_a_row_change(k_id)
                        
                    else:
                        scale_list.append(scale_addr)
                        scale_col_left -= 1
                # dequant 是否发生
                if flat_col_id % dequant_buf_col == 0:
                    dequant_a_col_dict[flat_col_id] = whether_is_a_row_change(k_id)
                if flat_col_id % data_buf_col == 0:
                    real_read_col = min(data_buf_col, mm_size_per_pu[2]*k_col-flat_col_id)
                    start_b_row_id = flat_col_id // SimConfig.co
                    end_b_row_id = (flat_col_id + real_read_col - 1) // SimConfig.co
                    if end_b_row_id != start_b_row_id: extra_row_change_num += 1
                    data_read_col += real_read_col
                    data_read_col_dict[flat_col_id] = whether_is_a_row_change(k_id)
        if not silent:
            print(f"dequant_a_col_dict: {dequant_a_col_dict}")
            print(f"scale_read_col_dict: {scale_read_col_dict}")
            print(f"data_read_col_dict: {data_read_col_dict}")
        overhead_col = dequant_buf_col * scale_precision / b_prec
        overhead_cycle = overhead_col * SimConfig.col_change_apox
        total_dequant_cycle = 0

        # get read b data extra rows

        # dequant 开销
        for col_id in dequant_a_col_dict.keys():
            tmp_overhead_cycle = overhead_cycle
            if dequant_a_col_dict[col_id] or col_id in data_read_col_dict.keys() or col_id in scale_read_col_dict.keys(): 
                tmp_overhead_cycle -= SimConfig.read_row_change_apox
            if col_id in data_read_col_dict.keys():
                tmp_overhead_cycle -= data_buf_col * SimConfig.col_change_apox
            if col_id in scale_read_col_dict.keys():
                tmp_overhead_cycle -= scale_buf_col * SimConfig.col_change_apox
            if tmp_overhead_cycle > 0:
                total_dequant_cycle += tmp_overhead_cycle
        processed_col_flat_id = data_read_col_dict.keys()
        
        for col_id in data_read_col_dict.keys():
            if data_read_col_dict[col_id]:
                extra_row_change_num += 1
            else:
                extra_row_change_num += 2
        for col_id in scale_read_col_dict.keys():
            if not scale_read_col_dict[col_id] and col_id not in processed_col_flat_id:
                extra_row_change_num += 2
            else:
                extra_row_change_num += 1

        # read a data overhead
        a_col, a_row = A_addr.get_meta()

        
        read_a_data_cycle = a_row * row_change_overhead + a_col * SimConfig.col_change_apox
        
        # result data latency
        result_count = max_result_num
        result_row = 0
        result_col = 0
        last_result_row = 0
        for n_id in range(mm_size_per_pu[2]):
            for m_id in range(mm_size_per_pu[0]):
                result_row_id, _ = result_addr.get_tensor_addr(m_id, n_id)
                if result_count == max_result_num: # 此时必定有一次额外的换行
                    result_row += 1
                    result_count = 0
                else:
                    if result_row_id == last_result_row:
                        result_col += 1
                    else:
                        result_row += 1
                last_result_row = result_row_id
                result_count += 1
        result_cycle = result_row * row_change_overhead + result_col * SimConfig.col_change_apox        
        read_b_data_and_scale_cycle = (data_read_col + scale_read_col) * SimConfig.col_change_apox + extra_row_change_num * row_change_overhead
        final_cycle = total_dequant_cycle + read_a_data_cycle * mm_size_per_pu[2] + read_b_data_and_scale_cycle + result_cycle
        if not silent: 
            print(f"dequant overhead_cycle: {total_dequant_cycle}")
            print(f"data_read_col: {data_read_col}")
            print(f"data_read_times: {len(data_read_col_dict.keys())}")
            print(f"scale_read_col: {scale_read_col}")
            print(f"scale_read_times: {len(scale_read_col_dict.keys())}")
            print(f"compute_col: {a_col} * {mm_size_per_pu[2]}")
            print(f"final_cycle: {final_cycle}")
    else:
        additional = scale_precision / a_prec
        if not (k_tile % b_data_in_col == 0 and k_tile >= b_data_in_col): return math.inf
        k_col_tile = k_tile // b_data_in_col
        a_col_left = 0
        a_data_list = []
        a_data_read_col = 0
        a_data_read_row = 0
        scale_col_left = 0
        scale_list = []
        scale_read_col = 0
        scale_read_row = 0
        for k_outer in range(math.ceil(k_col / k_col_tile)):
            real_k_col_tile = min(k_col_tile, k_col - k_outer * k_col_tile)
            for k_inner in range(real_k_col_tile):
                k_id = (k_outer * k_col_tile + k_inner) * b_data_in_col
                a_addr = A_addr.get_tensor_addr(0, k_id)
                if a_addr not in a_data_list:
                    a_data_read_col += 1
                    if a_col_left == 0:
                        a_data_read_row += 1
                        a_data_list = [a_addr]
                        a_col_left = data_buf_col - 1
                    else:
                        a_data_list.append(a_addr)
                        a_col_left -= 1
            # a_data_read_row += 1
            # a_data_read_col += real_k_col_tile * mm_size_per_pu[2]
            for n_id in range(mm_size_per_pu[2]):
                for k_inner in range(real_k_col_tile):
                    flat_col_id = k_outer * mm_size_per_pu[2] * k_col_tile + n_id * real_k_col_tile + k_inner
                    k_id = (k_outer * k_col_tile + k_inner) * b_data_in_col
                    scale_addr = B_addr.get_scale_addr(n_id, k_id)
                    scale_addr = (scale_addr[2],scale_addr[3])
                    if scale_addr not in scale_list:
                            # if not silent:
                        #     print(f"scale_addr: {scale_addr}")
                        scale_read_col += 1
                        # change buffer
                        if scale_col_left == 0:
                            scale_read_row += 1
                            if not silent:
                                print(f"flat_col_id: {flat_col_id}, scale_read_col_dict: {scale_list}")
                            scale_list = [scale_addr]
                            scale_col_left = scale_buf_col - 1                            
                        else:
                            scale_list.append(scale_addr)
                            scale_col_left -= 1
        # k tile is 
        row_num = B_row_num + scale_read_row + a_data_read_row
        col_num = mm_size_per_pu[2] * k_col * (1 + additional) + a_data_read_col + scale_read_col
        final_cycle = row_num * row_change_overhead + col_num * SimConfig.col_change_apox

        # 
        if not silent:
            print(f"a_data_read_col: {a_data_read_col}")
            print(f"a_data_read_row: {a_data_read_row}")
            print(f"scale_read_col: {scale_read_col}")
            print(f"scale_read_row: {scale_read_row}")
            print(f"row_num: {row_num}")
            print(f"col_num: {col_num}")
            print(f"final_cycle: {final_cycle}")

    return final_cycle