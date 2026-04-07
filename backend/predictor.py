import tqdm
import time
import math
from tools import *
from midend import *
from backend import *
from tools import *
from sim import sim
import math

def predict_w_a(args, batch_size=1, 
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
    0. buffer 分配
    """
    # 分配 Scale 与数据 Buffer
    bk_per_pu = SimConfig.bg * SimConfig.ba // SimConfig.de_pu[0]
    assert bk_per_pu == 1, "这是 1 BK 架构"

    # quant_group_result_buf_col = 100000
    # extra_scale_b_buf_col = 100000
    # 最新实验中都通过这个参数指定数据 Buffer 大小
    assert buffer_specify is not None
    data_a_buf_col, scale_a_buf_col, scale_b_buf_col, quant_group_result_buf_col, extra_scale_b_buf_col = buffer_specify[:5]
    # NOTE: Manual Hints 1
    if quant_group_result_buf_col < 1: return math.inf
    if not reorder and extra_scale_b_buf_col > 0: # reduce searching space
        return math.inf
    if data_a_buf_col + scale_a_buf_col + scale_b_buf_col + quant_group_result_buf_col + extra_scale_b_buf_col >= 20:
        if data_a_buf_col < 2 or data_a_buf_col + quant_group_result_buf_col <= 8: return math.inf
    
    # 计算quant group result buffer 大小
    max_quant_group_result_size = quant_group_result_buf_col * SimConfig.co_w // accumulation_precision
    
    # 分配结果 Buffer
    available_output_buffer_in_col = SimConfig.de_pu_bf // SimConfig.co_w
    assert available_output_buffer_in_col > 1, "Output Buffer 大小无法容纳一次计算"
    partial_buf_col = 1
    result_buf_col = available_output_buffer_in_col - partial_buf_col

    """
    NOTE: 1. Get design space
    """
    # get mm size
    m_size = operand_A_info['matrix_size'][0]
    k_size = operand_A_info['matrix_size'][1]
    n_size = operand_B_info['matrix_size'][0]
    mm_size = (m_size, k_size, n_size, batch_size)
    # get group size
    s_group_m = operand_A_info['scale_group'][0] if operand_A_info['with_scale'] else 0
    s_group_k = operand_A_info['scale_group'][1]
    scale_precision = operand_A_info['scale_offset_precision']
    assert s_group_k == operand_B_info['scale_group'][1], f"Invalid scale group: {operand_A_info['scale_group']} * {operand_B_info['scale_group']}"
    s_group_n = operand_B_info['scale_group'][0] if operand_B_info['with_scale'] else 0
    scale_group = (s_group_m, s_group_k, s_group_n)
    """
    NOTE: 2. get the baseline mapping
    """
    # A. get hw partition space
        # Update log file with buffer size information after allocation
    a_prec = operand_A_info['matrix_precision']
    b_prec = operand_B_info['matrix_precision']
    if log_file is None:
        workload_str = f"{operand_A_info['matrix_size'][0]}x{operand_A_info['matrix_size'][1]}x{operand_B_info['matrix_size'][0]}"
        # Add precision and scale group info
        
        # a_scale_group = operand_A_info['scale_group'][1] if operand_A_info['with_scale'] else 0
        b_scale_group = operand_B_info['scale_group'][1] if operand_B_info['with_scale'] else 0
        a_scale_group = operand_B_info['scale_group'][0] if operand_B_info['with_scale'] else 0
        function_desc = f"W{b_prec}A{a_prec}S{scale_precision}_sg{a_scale_group}x{b_scale_group}"
        # Auto-detect function name for directory, use function_desc in filename
        log_file, csvfile, writer = create_function_log_csv(workloadsize=f"Predict_{workload_str}_{function_desc}", 
                                                           data_buf_size=data_a_buf_col, 
                                                           scale_a_buf_size=scale_a_buf_col,
                                                           scale_b_buf_size=scale_b_buf_col,
                                                           quant_group_result_buf_size=quant_group_result_buf_col,
                                                           extra_scale_b_buf_size=extra_scale_b_buf_col)
        should_close_files = True
    else:
        should_close_files = False

    partition_tool = Partition(require_power_of_2 = False, log_file=log_file)
    compute_level = LEVEL.DE
    pu_num = 16
    # 如果配置改变，交换 A/B 信息
    mm_size_per_pu = partition_tool.mem_partition_mm(mm_size, partition_specify, scale_group)
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
    pu_mask = [True for _ in range(pu_num)]
    print(f"divieded mm size: {mm_size_per_pu}", file=log_file)
    # MARK: W + A LoopGen
    # 尝试调整循环顺序，最优化 data 读取
    gen = LoopGenerator(mm_size_per_pu[0], mm_size_per_pu[2], mm_size_per_pu[1])
    assert not buffer_b, "Buffer B is not supported for predictor"
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
        A_addr = TensorAddr(operand_A_info, (mm_size_per_pu[0], mm_size_per_pu[1]), A_mapping, _row_offset)
        _row_offset = A_addr.get_end_row()
        try:
            B_addr = ScaleMixAddr_v2(operand_B_info, gen, mx_gran, True, _row_offset) # _v2
        except Exception as e:
            return math.inf
        if breakdown_analysis:
            print(f"B_addr: {B_addr.sg_k0}, {B_addr.k_block_new}, {B_addr.actual_db_col_num}, {B_addr.db_col_num}", file=log_file)
        _row_offset = B_addr.get_end_row()
        result_addr = TensorAddr(result_precision, (mm_size_per_pu[0], mm_size_per_pu[2]), result_mapping, _row_offset)
    else:
        raise NotImplementedError("Only support mix scale mapping for predictor")
    
    scale_in_col, data_in_col, k_block_new = B_addr.scale_in_col, B_addr.data_in_col, B_addr.k_block_new
    

    # 全可以存下：
    # 与 scale_per_col 求余后，剩下的部分                    
    # 实际列数小于等于 scale buffer

    # NOTE: HINT, A scale buffer 可能的来回横跳
    loop_k = gen.block_sizes['k']['block1']
    max_repeat_time = mm_size_per_pu[1] // loop_k
    repeat_corner = mm_size_per_pu[1] - loop_k * max_repeat_time
    repeat_corner = repeat_corner % (s_group_k * scale_in_col)
    mod_loop_k = loop_k % (s_group_k * scale_in_col)
    therotical_scale_buf_col = math.ceil(loop_k / (s_group_k * scale_in_col))
    if mod_loop_k > 0 \
        and (s_group_k * scale_in_col) % mod_loop_k > 0 \
        and max_repeat_time * mod_loop_k + repeat_corner > (s_group_k * scale_in_col) \
        and scale_a_buf_col < therotical_scale_buf_col + 1: return math.inf
    # NOTE: HINT, B DATA BLOCK 可能的来回横跳
    # 循环一定对应 1 db
    # if mm_size_per_pu[1] > k_block_new and k_block_new % loop_k > 0: return math.inf
    if mm_size_per_pu[1] > k_block_new and k_block_new > loop_k: return math.inf

    """
    开始评估延迟，此处可以包含kn的复杂度，但是不能包含 m 的复杂度
    """
    # 观察法，限定 A data 和 scale buffer 比例，从而让循环大小只和 A data 有关，方便后续推导
    if k_tile_size_data > k_tile_size_scale: return math.inf
    if mm_size_per_pu[1] > k_tile_size_scale and k_tile_size_scale % loop_k > 0: return math.inf
    # 如果再武断一点，我觉得好的策略就不应该产生 k 遍历顺序的差别
    # 先从读取量开始
    # B data
    final_data_col_id, bound_group_n_id, n_group_in_db, group_size, final_db_id, actual_db_col_num = B_addr.get_meta()
    # A data
    k_col_in_db = int(k_block_new / data_in_col)
    k_db_iter = math.ceil(mm_size_per_pu[1] / k_block_new)
    # print(f"mm_size_per_pu[1] / k_block_new / loop_k: {mm_size_per_pu[1]} / {k_block_new} / {loop_k}")
    a_data_change_log = []
    a_scale_change_log = []
    current_scale_bound = 0
    potential_slot = {}
    row_change_num = 0
    for iter in range(k_db_iter):
        k_scale_num = B_addr.sg_k0 * (iter+1)
        need_scale = False
        if k_scale_num > current_scale_bound:
            current_scale_bound += scale_a_buf_col * scale_in_col
            need_scale = True
        contained_scale_col = math.ceil(iter * (bound_group_n_id+1) / n_group_in_db) * mx_gran
        contained_data_col = k_col_in_db * (iter * (bound_group_n_id+1) * group_size[0])
        row_id = (contained_scale_col + contained_data_col) // SimConfig.co
        col_id = (contained_scale_col + contained_data_col) % SimConfig.co
        flat_col_id = contained_scale_col + contained_data_col
        a_data_change_log.append((row_id, col_id))
        slot_cycle = 2 * SimConfig.read_row_change_apox + k_col_in_db * SimConfig.col_change_apox
        # 正常换行
        # 一定会在前后触发换行
        if need_scale:
            a_scale_change_log.append((row_id, col_id))
            # slot, whether block
            potential_slot[flat_col_id] = (slot_cycle, True)
            row_change_num += 3
        else:
            potential_slot[flat_col_id] = (slot_cycle, False)
            row_change_num += 2

    # print(f"a_data_change_log: {a_data_change_log}")
    # print(f"a_scale_change_log: {a_scale_change_log}")
    max_partial_sum = SimConfig.co_w//accumulation_precision
    dequant_num = math.ceil(B_addr.sg_k0 / max_partial_sum) * k_db_iter * mm_size_per_pu[2]
    # add row change slot
    data_b_row_change_count = int(final_data_col_id // SimConfig.co + 1)
    for row_id in range(data_b_row_change_count):
        slot_cycle = SimConfig.read_row_change_apox
        if row_id * SimConfig.co not in potential_slot.keys():
            potential_slot[row_id * SimConfig.co] = (slot_cycle, False)
            row_change_num += 1
    # print(f"potential_slot: {potential_slot}")
    last_scale_b_absorb = True
    last_scale_b_left_col = mx_gran
    dequant_num = 0
    hide_dequant_num = 0
    # iterate data blocks
    for db_id in reversed(range(final_db_id+1)):
        start_id = (actual_db_col_num + mx_gran) * db_id
        end_id = start_id + actual_db_col_num + mx_gran
        # 找出potential_slot字典中在两个id间的 key
        keys_in_range = [key for key in potential_slot.keys() if start_id < key <= end_id]
        keys_in_range = sorted(keys_in_range, reverse=True)
        keys_zuse = [key for key in keys_in_range if potential_slot[key][1] == True]
        if not silent: print(f"keys_in_range for db_id={db_id}: {keys_in_range}")
        dequant_list = [] 
        # 这里存在一些 corner，k到边缘可能不太一样
        # 找出中途的 dequant 在哪里？
        for n_id in range(n_group_in_db*group_size[0]):
            for k in range(ceil(B_addr.sg_k0 / max_partial_sum)):
                group_n_id = n_id // group_size[0]
                group_k_inblock_id = min(B_addr.sg_k0, (k+1) * max_partial_sum) - 1
                group_flat_id = group_k_inblock_id + B_addr.sg_k0 * group_n_id
                # indb_id = group_flat_id % B_addr.actual_db_scale_group_num
                indb_col_id = group_flat_id // scale_in_col
                scale_col_id = start_id + indb_col_id
                col_id = start_id + mx_gran + \
                    ceil(min(mm_size_per_pu[1], min(B_addr.sg_k0, (k+1) * max_partial_sum) * group_size[1]) * b_prec / SimConfig.co_w) +\
                        n_id * k_col_in_db
                buffer_left = max_quant_group_result_size - min(max_partial_sum, B_addr.sg_k0-k*max_partial_sum)
                # print(B_addr.sg_k0-k*(SimConfig.co_w//accumulation_precision))
                dequant_list.append((int(col_id), buffer_left, int(scale_col_id)))
                dequant_num += 1
        dequant_list.reverse()
        dequant_list_copy = dequant_list.copy()
        if not silent: print(f"dequant_list: {dequant_list}")
        
        hard_dequant_list = []
        # pass 1
        for slot_col_id in sorted(keys_in_range, reverse=True):
            slot_cycle = potential_slot[slot_col_id][0]
            if slot_col_id in keys_zuse:
                keys_zuse.remove(slot_col_id)
            zuse_id = keys_zuse[0] if keys_zuse != [] else 0
            while dequant_list != []:
                dequant_col_id, buffer_left, _ = dequant_list[0]
                if dequant_col_id > slot_col_id: 
                    hard_dequant_list.append(dequant_list.pop(0))
                    continue
                if dequant_col_id <= zuse_id or slot_cycle <= 0:
                    break
                # move 
                move_partial_sum_num = ceil((slot_col_id - dequant_col_id) * data_in_col / group_size[1])
                if buffer_left >= move_partial_sum_num:
                    slot_cycle -= 2 * SimConfig.col_change_apox
                    # print(f"move dequant {dequant_col_id} to {slot_col_id}, buffer_left: {buffer_left-move_partial_sum_num}, slot_left: {slot_cycle}")
                    if not silent:
                        print(f"move dequant {dequant_col_id} to {slot_col_id}, buffer_left: {buffer_left-move_partial_sum_num}, slot_left: {slot_cycle}")
                    hide_dequant_num += 1
                    dequant_list.pop(0)
                else:
                    break
        # pass 2: scale b 读取有可能被分为两条命令执行
        hard_dequant_list += dequant_list
        tmp = []
        for aaa in reversed(dequant_list_copy):
            if aaa in hard_dequant_list:
                tmp.append(aaa)
            else:
                break
        hard_dequant_list = tmp
        row_id = (start_id + mx_gran - 1) // SimConfig.co
        if row_id > start_id // SimConfig.co:
            this_scale_b_col = start_id + mx_gran - (start_id // SimConfig.co + 1) * SimConfig.co
            this_scale_b_left_col = mx_gran - this_scale_b_col
        else:
            this_scale_b_col = mx_gran
            this_scale_b_left_col = mx_gran
        hard_dequant_list.reverse()
        to_cover_col = this_scale_b_col
        while hard_dequant_list != []:
            dequant_col_id, buffer_left, _ = hard_dequant_list[0]
            if (dequant_col_id-1) // SimConfig.co != row_id: break
            if to_cover_col <= 0: break
            else:
                to_cover_col -= 2
                hard_dequant_list.pop(0)
                hide_dequant_num += 1
        if to_cover_col == this_scale_b_col: this_scale_b_absorb = True
        else: this_scale_b_absorb = False
        # pass 3: 
        if not (last_scale_b_absorb and last_scale_b_left_col == mx_gran):
            third_dequant_list = []
            for aaa in dequant_list_copy:
                if aaa in hard_dequant_list:
                    third_dequant_list.append(aaa)
                else:
                    break
            to_cover_col = last_scale_b_left_col
            extra_scale_pos = []
            while third_dequant_list != []:
                dequant_col_id, buffer_left, scale_col_id = third_dequant_list[0]
                if to_cover_col <= 0: break
                if len(extra_scale_pos) >= extra_scale_b_buf_col: break
                move_partial_sum_num = ceil((end_id - dequant_col_id) * data_in_col / group_size[1])
                if buffer_left >= move_partial_sum_num:
                    to_cover_col -= 2
                    # if not silent:
                    #     print(f"move dequant {dequant_col_id} to {slot_col_id}, buffer_left: {buffer_left-move_partial_sum_num}, slot_left: {slot_cycle}")
                    # print(f"move dequant {dequant_col_id} to {slot_col_id}, buffer_left: {buffer_left-move_partial_sum_num}, slot_left: {slot_cycle}")
                    hide_dequant_num += 1
                    third_dequant_list.pop(0)
                    if scale_col_id not in extra_scale_pos:
                        extra_scale_pos.append(scale_col_id)
                else:
                    break
        last_scale_b_absorb = this_scale_b_absorb
        last_scale_b_left_col = this_scale_b_left_col
            # if dequant_col_id // 
    

        # dequant before which?
    actual_read_b_col = math.ceil(mm_size_per_pu[1] / data_in_col) * mm_size_per_pu[2]
    read_b_scale_col = (final_db_id+1) * mx_gran
    read_a_data_col = math.ceil(mm_size_per_pu[1] / data_in_col)
    read_a_scale_col = math.ceil(math.ceil(mm_size_per_pu[1]/group_size[1])/scale_in_col)
    predict_lat = (actual_read_b_col + read_b_scale_col + read_a_data_col + read_a_scale_col) * SimConfig.col_change_apox\
         + row_change_num * SimConfig.read_row_change_apox + (dequant_num - hide_dequant_num) * 2 * SimConfig.col_change_apox
    predict_lat = predict_lat * mm_size_per_pu[0]
    # A Scale 只有 A data 更换时发生
    # 算的结果是在哪一列之前进行读取操作
    # k维度：修正后的循环顺序，或者等效为 1 db 的宽度
    if not silent:
        print(f"dequant_num: {dequant_num}")
        print(f"hide_dequant_num: {hide_dequant_num}")
        print(f"actual_read_b_col: {actual_read_b_col}")
        print(f"read_b_scale_col: {read_b_scale_col}")
        print(f"read_a_data_col: {read_a_data_col}")
        print(f"read_a_scale_col: {read_a_scale_col}")
        print(f"row_change_num: {row_change_num}")
        print(f"predict_lat: {predict_lat}")
    # A scale
    return predict_lat

def predict_w_only(args, batch_size=1, 
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
        gen.set_block_sizes(k_block0=k_tile_size_data, k_block1=k_tile_size_data)
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
        B_addr = LoopfriendlyAddr(operand_B_info, gen, B_mapping, True, _row_offset)
        _row_offset = B_addr.get_end_row()
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
    if buffer_b:
        scale_list = []
        scale_col_left = 0
        extra_row_change_num = 0
        # k_iter = math.ceil(mm_size_per_pu[1] / k_tile)
        # get dequant k id
        k_col = math.ceil(mm_size_per_pu[1] / b_data_in_col)
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

        row_change_overhead = SimConfig.read_row_change_apox - SimConfig.col_change_apox
        read_a_data_cycle = a_row * row_change_overhead + a_col * SimConfig.col_change_apox
        read_b_data_and_scale_cycle = (data_read_col + scale_read_col) * SimConfig.col_change_apox + extra_row_change_num * row_change_overhead
        final_cycle = total_dequant_cycle + read_a_data_cycle * mm_size_per_pu[2] + read_b_data_and_scale_cycle
        if not silent: 
            print(f"dequant overhead_cycle: {total_dequant_cycle}")
            print(f"data_read_col: {data_read_col}")
            print(f"data_read_times: {len(data_read_col_dict.keys())}")
            print(f"scale_read_col: {scale_read_col}")
            print(f"scale_read_times: {len(scale_read_col_dict.keys())}")
            print(f"compute_col: {a_col} * {mm_size_per_pu[2]}")
            print(f"final_cycle: {final_cycle}")
    return final_cycle

    # Use parallel address generation for W-only case with parallel loop generator
    print(f"Generating W-only addresses using parallel loop generator and {args.num_processes} processes...", file=log_file)
    start_time = time.time()
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
    
    return lat
    pass