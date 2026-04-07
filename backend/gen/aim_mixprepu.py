import tqdm
from tools import SimConfig
from midend import *
from backend import *
from tools import *
import math

def convert_abcol(abcol):
    new_abcol = []
    b_col_last = None
    for a_col, b_col in abcol:
        if b_col_last != b_col:
            new_abcol.append((a_col, b_col))
            b_col_last = b_col
    return new_abcol

def aim_mixprepu(config, cmd_list, args,
    data_buffer, A_scale_buffer, B_scale_buffer, result_buffer,
    result_addr, B_addr,
):
    
    buffer_b, consider_data_read, consider_scale_buffer, consider_result_buffer, \
    pu_num, pu_mask, \
    mix_scale_mapping, accumulation_precision, mm_size_per_pu, \
    max_partial_num, max_quant_group_result_size, scale_b_buf_col, result_buf_col, extra_scale_b_buf_col, \
    codegen_tool, log_file = config
    
    # 生成计算指令
    buffered_partial = None
    buffered_k_scale = []
    
    # 统计 Scale A 信息
    scale_a_read_col_num = 0
    scale_a_read_row_num = 0
    scale_b_read_col_num = 0
    scale_b_read_row_num = 0
    data_a_read_col_num = 0
    data_a_read_row_num = 0
    data_b_read_col_num = 0
    data_b_read_row_num = 1

    # # 获取最后一个命令的缩放组信息 - 使用属性访问替代硬编码索引
    # last_cmd = CmdAddr(cmd_addr_list[-1])
    # required_scale_group_num_k = last_cmd.B_scale_group_b_id + 1
    
    AB_col = []
    AB_col_rst = []
    # 5col - 5*256/fp16 = 80
    # ！max partial num 为容纳多少个组的部分结果，和 scale k group 对应
    ir_list_list = []
    ir_list = IR_List((pu_num, pu_mask, mix_scale_mapping, SimConfig.col_change_apox, SimConfig.read_row_change_apox), 
    # 100*max_partial_num, # 以部分和的个数为单位
    max_quant_group_result_size, # 以部分和的个数为单位
    # 100, # 以 Col 为单位
    extra_scale_b_buf_col,
    codegen_tool,
    log_file)
    m_start_id = 0
    # 使用 CmdAddrList 进行迭代，简化地址访问
    for i, cmd in tqdm.tqdm(enumerate(cmd_list), total=len(cmd_list), desc="Processing commands"):
        # 检查是否需要新的 scale 数据
        if buffer_b:
            data_row = cmd.A_row
        else:
            data_row = cmd.B_row
        # 先检查 data
        if consider_data_read:
            current_a_data = cmd.get_current_a_data()
            current_b_data = cmd.get_current_b_data()
            if buffer_b:
                current_data = current_b_data
            else:
                current_data = current_a_data
            require_new_data = not data_buffer.is_data_available(current_data)
            if require_new_data:
                # 清空当前的计算指令
                if len(AB_col) > 0:

                    ir_list.append(
                        [
                            "compute",
                            data_row,
                            convert_abcol(AB_col),
                            AB_col_rst,
                        ]
                    )
                    AB_col = []
                    AB_col_rst = []
                
                # 第二步，更新buffer并获取需要读取的新数据
                A_new, A_new_contents = data_buffer.update_buffer({current_data})
                # 第三步，生成 A 读取指令（使用update_buffer返回的A_new）
                if len(A_new) > 0:
                    A_data_row_dict = data_buffer.get_data_row_dict(A_new)
                    for a_row in A_data_row_dict.keys():
                        data_a_read_row_num += 1
                        data_a_read_col_num += len(A_data_row_dict[a_row])
                        if not args.silent:
                            sorted_cols = sorted(A_data_row_dict[a_row])
                            print(f"Read A Data: r{a_row}, c{sorted_cols}", file=log_file)
                        real_row = a_row
                        # gb for a / scale a buf
                        ir_list.append(
                            [
                                "read_a",
                                real_row,
                                A_data_row_dict[a_row],
                            ]
                        )
                    data_buffer.actual_update_buffer(A_new_contents)
                    if not args.silent:
                        print(f"Read A Data, current buffer: {A_new_contents}", file=log_file)
        # 再检查 scale
        if consider_scale_buffer:
            current_a_scale = cmd.A_scale_pos
            current_b_scale = cmd.B_scale_pos
            require_new_scale = (scale_b_buf_col > 0 and not B_scale_buffer.is_data_available(current_b_scale))
            if require_new_scale:
                # 第一步，清空当前的计算指令
                if len(AB_col) > 0:
                    ir_list.append(
                        [
                            "compute",
                            data_row,
                            convert_abcol(AB_col),
                            AB_col_rst,
                        ]
                    )
                    AB_col = []
                    AB_col_rst = []
            
            # # 第二步，更新buffer并获取需要读取的新数据
            # if not A_scale_buffer.is_data_available(current_a_scale):
            #     A_new, A_new_contents = A_scale_buffer.update_buffer({current_a_scale})
            #     if len(A_new) > 0:
            #         if not args.silent:
            #             print(f"Read A Scale: {list(A_new)}, current buffer: {A_new_contents}", file=log_file)
            #         A_scale_row_dict = A_scale_buffer.get_scale_row_dict(A_new)
            #         for a_row in A_scale_row_dict.keys():
            #             scale_a_read_row_num += 1
            #             scale_a_read_col_num += len(A_scale_row_dict[a_row])
            #             real_row = a_row
                        
            #             # ir_list.append(
            #             #     [
            #             #         "scale_a_read",
            #             #         real_row,
            #             #         A_scale_row_dict[a_row],
            #             #         # NOTE: optional for debug
            #             #         list(A_new),
            #             #         A_new_contents,
            #             #     ]
            #             # )
            #         A_scale_buffer.actual_update_buffer(A_new_contents)
            
            if scale_b_buf_col > 0 and not B_scale_buffer.is_data_available(current_b_scale):
                B_new, B_new_contents = B_scale_buffer.update_buffer({current_b_scale})
                if len(B_new) > 0:
                    if not args.silent:
                        print(f"Read B Scale: {list(B_new)}, current buffer: {B_new_contents}", file=log_file)
                    B_scale_row_dict = B_scale_buffer.get_scale_row_dict(B_new)
                    
                    for b_row in sorted(B_scale_row_dict.keys()):
                        scale_b_read_row_num += 1
                        scale_b_read_col_num += len(B_scale_row_dict[b_row])
                        real_row = b_row
                        ir_list.append(
                            [
                                "scale_b_read",
                                real_row,
                                B_scale_row_dict[b_row],
                                # NOTE: optional for debug
                                list(B_new),
                                B_new_contents,
                            ]
                        )
                    B_scale_buffer.actual_update_buffer(B_new_contents)
                    
        # 获取当前和下一个命令对 - 使用便捷方法替代大量变量解包
        current, next_cmd = cmd_list.get_pair(i)
        # if next_cmd is not None:
        #     m_id = next_cmd.m
        #     if m_id != m_start_id:
        #         ir_list_list.append(ir_list)
        #         # if mm_size_per_pu[0] >= 8: 
        #         #     performance_metrics = (
        #         #         data_a_read_col_num, data_a_read_row_num,
        #         #         scale_a_read_col_num, scale_a_read_row_num,
        #         #         scale_b_read_col_num, scale_b_read_row_num,
        #         #         data_b_read_col_num, data_b_read_row_num,
        #         #     )
        #         #     ir_list_list.extend([1] * (mm_size_per_pu[0] - 1))
        #         #     return performance_metrics, ir_list_list
        #         ir_list = IR_List((pu_num, pu_mask, mix_scale_mapping, SimConfig.col_change_apox, SimConfig.read_row_change_apox), 
        #         # 100*max_partial_num, # 以部分和的个数为单位
        #         max_quant_group_result_size, # 以部分和的个数为单位
        #         # 100, # 以 Col 为单位
        #         extra_scale_b_buf_col,
        #         codegen_tool,
        #         log_file)
        #         m_start_id = m_id
        # 结果记录
        if next_cmd is None or next_cmd.matrix_coord != current.matrix_coord or next_cmd.B_scale_group_b_id != current.B_scale_group_b_id:
            # print(f"compute A{current.A_pos}_s{current.A_scale_pos} B{current.B_pos}_s{current.B_scale_pos} Result{current.matrix_coord} k_scale_group_id({current.B_scale_group_b_id})")
            AB_col_rst.append(current.matrix_coord+(current.B_scale_group_b_id,))

        # Col 处理 - 使用便捷的比较方法
        if next_cmd is None or not current.same_AB_col(next_cmd) or \
        (not buffer_b and not current.same_B_row(next_cmd)) or \
            (buffer_b and not current.same_A_row(next_cmd)):
            AB_col.append(current.AB_col)
            data_b_read_col_num += 1
            if not args.silent:
                print(f"compute A{current.A_pos}_s{current.A_scale_pos} B{current.B_pos}_s{current.B_scale_pos} Result{current.matrix_coord} k_scale_group_id({current.B_scale_group_b_id})", file=log_file)
        
        # Partial Buffer 维护 - 使用属性访问替代硬编码索引
        # assert next_cmd.B_scale_group_b_id == next_cmd.B_scale_group_b_id, f"B_scale_group_b_id, B_scale_group_b_id mismatch: {next_cmd.B_scale_group_b_id}, {next_cmd.B_scale_group_b_id}"
        new_added = None
        if next_cmd is None or next_cmd.matrix_coord != current.matrix_coord or next_cmd.B_scale_group_b_id != current.B_scale_group_b_id:
            # 先处理当前结果的汇入
            if len(buffered_k_scale) == 0:
                assert buffered_partial == None, "buffered_partial 应该为 None"
                buffered_partial = current.matrix_coord
                a_scale_pos = current.A_scale_pos
                b_scale_pos = current.B_scale_pos
            buffered_k_scale.append(current.B_scale_group_b_id)
            # 再预测下一次循环，会不会超过容量限制
            if next_cmd is None or buffered_partial != next_cmd.matrix_coord or len(buffered_k_scale) == max_partial_num:
                # 累加部分结果到结果 Buffer
                new_added = (current.matrix_coord, buffered_k_scale, b_scale_pos, a_scale_pos)
                if not args.silent:
                    print(f"Reduce, partial buffer: {buffered_partial}, buffered_k_scale: {buffered_k_scale}", file=log_file)
                buffered_partial = None
                buffered_k_scale = []
            else:
                if mix_scale_mapping:
                    try:
                        assert next_cmd.B_scale_pos == b_scale_pos, f"B_scale_pos mismatch: {next_cmd.B_scale_pos}, {b_scale_pos}"
                    except AssertionError as e:
                        print(f"WARNING: B_scale_pos mismatch detected. Expected {b_scale_pos}, got {next_cmd.B_scale_pos}", file=log_file)
                        # return float('inf')
                        b_scale_pos = next_cmd.B_scale_pos

        # 如果需要 Scale 计算 / 换行，则也需要把之前累加的 Col 给发出
        # 因此此处 A 的数据是通过 GB 访问的，因此不需要检测 A 的换行
        if next_cmd is not None:
            if buffer_b:
                if not current.same_A_row(next_cmd):
                    data_b_read_row_num += 1
            else:
                if not current.same_B_row(next_cmd):
                    data_b_read_row_num += 1
            # data_b_read_row_num += 1
        if next_cmd is None or \
            (not buffer_b and not current.same_B_row(next_cmd)) or \
            (buffer_b and not current.same_A_row(next_cmd)) or \
                new_added is not None:
            if len(AB_col) > 0:
                ir_list.append([
                        "compute",
                        data_row,
                        convert_abcol(AB_col),
                        AB_col_rst,
                    ]
                )
                AB_col = []
                AB_col_rst = []

        # 使用 ResultBuffer 处理结果管理
        if new_added is not None and consider_result_buffer:
            # 实际实现 partial buffer 延迟
            if scale_b_buf_col > 0:
                compute_col = math.ceil(len(new_added[1]) * accumulation_precision / SimConfig.co_w)
                ir_list.append(["dequant", compute_col, 0, new_added])
            else:
                # FIXME: 暂时没有做 ir-list 的处理，如果无 Scale B Buffer，此时 Scale B 需要从 DRAM 中读取
                dequant_b_scale_dict = {}
                (_, n), k_scale_list = new_added
                for k_scale_group_id in k_scale_list:
                    B_scale_row, B_scale_col = B_addr.get_scale_addr_by_b_group_id(n, k_scale_group_id)
                    if B_scale_row not in dequant_b_scale_dict.keys():
                        dequant_b_scale_dict[B_scale_row] = [B_scale_col]
                    else:
                        if B_scale_col not in dequant_b_scale_dict[B_scale_row]:
                            dequant_b_scale_dict[B_scale_row].append(B_scale_col)
                # for B_scale_row in dequant_b_scale_dict.keys():
                #     commands.append(codegen_tool.create_device_pu(
                #         ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                #         op1 = (0, B_scale_row, 0),
                #         op2 = (0, 1, 0), # scale A from GB
                #         col_num = 2 * len(dequant_b_scale_dict[B_scale_row]), # * scale A * scale B
                #         auto_precharge = False,
                #     ))
            
            result_buffer.add_result(new_added[0], len(new_added[1]))
            if not args.silent:
                print(f"New Added {new_added[0]} with {new_added[1]}", file=log_file)
            
            if next_cmd is None or result_buffer.needs_writeback(next_cmd.matrix_coord):
                buffered_results = result_buffer.get_writeback_data()
                if result_buf_col > 0:
                    # Write Back Addr Dict (Row: [Col])
                    write_back_addr_dict = {}
                    for m, n in buffered_results.keys():
                        result_row, result_col = result_addr.get_tensor_addr(m, n)
                        if result_row not in write_back_addr_dict.keys():
                            write_back_addr_dict[result_row] = [result_col]
                        else:
                            if result_col not in write_back_addr_dict[result_row]:
                                write_back_addr_dict[result_row].append(result_col)
                    if not args.silent:
                        print(f"write_back_addr_dict: {write_back_addr_dict}", file=log_file)
                
                    for idx, row in enumerate(write_back_addr_dict.keys()):
                        if next_cmd is None and idx == len(write_back_addr_dict.keys()) - 1:
                            auto_precharge = True
                        else:
                            auto_precharge = False
                        ir_list.append(
                            [
                                "write_back",
                                row,
                                write_back_addr_dict[row],
                                auto_precharge
                            ]
                        )
                else:
                    pass
                    # FIXME: 这个 case 也没有做 ir-list 的处理，部分结果直接通过 Host 维护
                    # TODO: 可能精度不同，不一定通过一个命令可以完成，但是根据经验，这部分的延迟影响不大
                    # assert len(buffered_results) == 1
                    # commands.append(codegen_tool.create_host_read_mac_reg(
                    #     ch_id=0, ra_id=0, device_mask=[True for _ in range(SimConfig.de)], pu_mask=pu_mask, 
                    # ))
                result_buffer.clear_buffer()
    performance_metrics = (
        data_a_read_col_num, data_a_read_row_num,
        scale_a_read_col_num, scale_a_read_row_num,
        scale_b_read_col_num, scale_b_read_row_num,
        data_b_read_col_num, data_b_read_row_num,
    )
    ir_list_list.append(ir_list)
    return performance_metrics, ir_list_list
