import tqdm
from tools import SimConfig
from midend import *
from backend import *
from tools import *
import math

# 定义计算函数
# B 从 DRAM 读入，A 从 GB 读入
def gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, consider_scale_buffer=False, additional=1.0):
    # 做延迟分解，看计算前乘 Scale 的消耗有多大
    if consider_scale_buffer:
        col_num = ceil(len(AB_col) * (1+additional)) # 同时，当时需要考虑计算两次
    else:
        col_num = len(AB_col)
    if col_num > 0:
        A_col_offset, B_col_offset = AB_col[0]
        commands.append(codegen_tool.create_device_pu(
            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
            op1 = (0, data_row, B_col_offset),
            op2 = (0, 1, A_col_offset),
            col_num = col_num,
            auto_precharge = False,
        ))
        print(f"[Compute   ] row {data_row} for {col_num} cols: {AB_col}", file=log_file)
        # clear AB_col
        AB_col.clear()

def aim_w(config, cmd_list, args,
    data_buffer, B_scale_buffer, B_dequant_buffer, result_buffer,
    result_addr, B_addr,
):
    buffer_b, consider_data_read, consider_scale_buffer, consider_result_buffer, \
    pu_num, pu_mask, a_prec, b_prec, scale_precision,\
    mix_scale_mapping, accumulation_precision, \
    data_buf_col, data_buf_col, dequant_buf_col, result_buf_col, \
    codegen_tool, log_file = config
    
    additional = scale_precision / a_prec
    # 生成计算指令
    buffered_partial = None
    buffered_k_scale = []
    
    commands = []
    AB_col = []
    data_read_col_num = 1
    data_read_row_num = 1
    buffer_read_col_num = 0
    buffer_read_row_num = 0
    scale_read_col_num = 0
    scale_read_row_num = 0

    # # 获取最后一个命令的缩放组信息 - 使用属性访问替代硬编码索引
    # last_cmd = CmdAddr(cmd_addr_list[-1])
    # required_scale_group_num_k = last_cmd.A_scale_group_b_id + 1
    
    AB_col = []
    AB_col_rst = []
    for i, cmd in enumerate(cmd_list):
        if buffer_b:
            data_row = cmd.A_row
        else:
            data_row = cmd.B_row
        # 使用 ScaleBuffer 处理 B scale 读入
        if consider_scale_buffer:
            current_b_scale = cmd.B_scale_pos
            require_new_scale = not B_scale_buffer.is_data_available(current_b_scale)
            if require_new_scale:
                # 清空当前的计算指令
                gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, consider_result_buffer, additional)
                B_new, B_new_contents = B_scale_buffer.update_buffer({current_b_scale})
                if len(B_new) > 0:
                    print(f"[Read B Scale]: {list(B_new)}, current B_scale_buf: {B_new_contents}", file=log_file)
                    B_scale_row_dict = B_scale_buffer.get_scale_row_dict(B_new)
                    for b_row in B_scale_row_dict.keys():
                        real_row = b_row
                        commands.append(codegen_tool.create_device_bk2buf(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            op1 = (0, real_row, 0),
                            buf = (True, 0, len(B_scale_row_dict[b_row])),
                            auto_precharge = False,
                        ))
                        scale_read_col_num += len(B_scale_row_dict[b_row])
                        scale_read_row_num += 1
                    B_scale_buffer.actual_update_buffer(B_new_contents)
        # 使用 DataBuffer 处理 A data 读入
        if consider_data_read:
            # 获取B数据信息 - 使用便捷方法替代硬编码索引
            current_a_data = cmd.get_current_a_data()
            current_b_data = cmd.get_current_b_data()
            if buffer_b:
                current_data = current_b_data
            else:
                current_data = current_a_data
                print(f"current_data: {current_data}", file=log_file)
            # 检查是否需要新的数据
            require_new_data = not data_buffer.is_data_available(current_data)
            # require_new_data = not A_data_buffer.is_data_available(current_a_data)
            
            if require_new_data:
                # 清空当前的计算指令
                gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, consider_result_buffer, additional)
                
                # 更新 A Data Buffer
                A_data_new, A_data_new_contents = data_buffer.update_buffer({current_data})
                                
                # Bank -> A Buffer
                if len(A_data_new) > 0:
                    # 生成 data 读取指令
                    A_data_row_dict = data_buffer.get_data_row_dict(A_data_new)
                    # buffer_read_num += 1
                    for a_row in A_data_row_dict.keys():
                        buffer_read_row_num += 1
                        sorted_cols = sorted(A_data_row_dict[a_row])
                        print(f"[Read A Data] r{a_row}, c{sorted_cols}", file=log_file)
                        commands.append(codegen_tool.create_device_bk2buf(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            op1 = (0, a_row, 0),
                            buf = (True, 0, len(A_data_row_dict[a_row])),
                            auto_precharge = False,
                        ))
                        buffer_read_col_num += len(A_data_row_dict[a_row])
                    data_buffer.actual_update_buffer(A_data_new_contents)

        # 获取当前和下一个命令对 - 使用便捷方法替代大量变量解包
        current, next_cmd = cmd_list.get_pair(i)
        current: CmdAddr
        next_cmd: Optional[CmdAddr]
        # 非最后一次循环
        if next_cmd is not None:
            # Col 处理 - 使用便捷的比较方法
            row_change = (buffer_b and not current.same_A_row(next_cmd)) or (not buffer_b and not current.same_B_row(next_cmd))
            if row_change:
                data_read_row_num += 1
            if not current.same_AB_col(next_cmd) or row_change:
                AB_col.append(current.AB_col)
                data_read_col_num += 1
                print(f"+ A{current.A_pos} B{current.B_pos}_s{current.B_scale_pos} Result{current.matrix_coord}", file=log_file)

            # 使用 ResultBuffer 管理结果 - 使用属性访问替代硬编码索引
            write_back_addr_dict = {}
            if consider_result_buffer:
                # 先把现有结果做插入
                current_result = current.matrix_coord
                if not result_buffer.can_add_result(current_result):
                    # 需要先写回
                    buffered_results = result_buffer.get_writeback_data()
                    for m, n in buffered_results:
                        result_row, result_col = result_addr.get_tensor_addr(m, n)
                        if result_row not in write_back_addr_dict.keys():
                            write_back_addr_dict[result_row] = [result_col]
                        else:
                            if result_col not in write_back_addr_dict[result_row]:
                                write_back_addr_dict[result_row].append(result_col)
                    print(f"[Write Back Result]: {write_back_addr_dict}", file=log_file)
                    result_buffer.clear_buffer()
                
                # 添加当前结果
                result_buffer.add_result(current_result)

            # 结果写回，则也需要把之前累加的 Col 给发出
            if row_change or len(write_back_addr_dict) > 0:
                gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, consider_result_buffer, additional)

            # issue Result Buffer -> DRAM 的换出指令
            for row in write_back_addr_dict.keys():
                commands.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, row, 0),
                    buf = (True, 0, len(write_back_addr_dict[row])),
                    auto_precharge = False,
                ))
            
        # 最后一次循环
        else:
            # Col 处理 - 使用属性访问替代硬编码索引
            AB_col.append(current.AB_col)
            print(f"+ A{current.A_pos} B{current.B_pos}_s{current.B_scale_pos} Result{current.matrix_coord}", file=log_file)

            # 发出最后的计算指令
            gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, consider_result_buffer, additional)

            # 先把现有结果做插入
            current_result = current.matrix_coord
            result_buffer.add_result(current_result)
            # print(f"buffered_output: {result_buffer.get_buffer_status()}", file=log_file)
            
            # 最终写回
            buffered_results = result_buffer.get_writeback_data()
            write_back_addr_dict = {}
            for m, n in buffered_results:
                result_row, result_col = result_addr.get_tensor_addr(m, n)
                if result_row not in write_back_addr_dict:
                    write_back_addr_dict[result_row] = [result_col]
                else:
                    if result_col not in write_back_addr_dict[result_row]:
                        write_back_addr_dict[result_row].append(result_col)
            print(f"[Write Back Result]: {write_back_addr_dict}", file=log_file)
            
            for idx, row in enumerate(write_back_addr_dict.keys()):
                if idx == len(write_back_addr_dict.keys()) - 1:
                    auto_precharge = True
                else:
                    auto_precharge = False
                commands.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, row, 0),
                    buf = (True, 0, len(write_back_addr_dict[row])),
                    auto_precharge = auto_precharge,
                ))
    performance_metrics = (
        data_read_col_num, data_read_row_num,
        buffer_read_col_num, buffer_read_row_num,
        scale_read_col_num, scale_read_row_num,
    )
    return performance_metrics, commands

def aim_w_buf_b(config, cmd_list, args,
    data_buffer, B_scale_buffer, B_dequant_buffer, result_buffer,
    result_addr, B_addr,
):

    buffer_b, consider_data_read, consider_scale_buffer, consider_result_buffer, \
    pu_num, pu_mask, a_prec, b_prec, scale_precision,\
    mix_scale_mapping, accumulation_precision, \
    data_buf_col, data_buf_col, dequant_buf_col, result_buf_col, \
    codegen_tool, log_file = config
    
    # 生成计算指令
    buffered_partial = None
    buffered_k_scale = []
    
    commands = []
    AB_col = []
    data_read_col_num = 1
    data_read_row_num = 1
    buffer_read_col_num = 0
    buffer_read_row_num = 0
    scale_read_col_num = 0
    scale_read_row_num = 0

    # # 获取最后一个命令的缩放组信息 - 使用属性访问替代硬编码索引
    # last_cmd = CmdAddr(cmd_addr_list[-1])
    # required_scale_group_num_k = last_cmd.A_scale_group_b_id + 1
    
    AB_col = []
    AB_col_rst = []
    for i, cmd in enumerate(cmd_list):
        data_row = cmd.A_row
        # 使用 ScaleBuffer 处理 B scale 读入
        if consider_scale_buffer:
            current_b_scale = cmd.B_scale_pos
            require_new_scale = not B_scale_buffer.is_data_available(current_b_scale)
            if require_new_scale:
                # 清空当前的计算指令
                gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, False)
                B_new, B_new_contents = B_scale_buffer.update_buffer({current_b_scale})
                if len(B_new) > 0:
                    print(f"[ Read B Scale ]: {list(B_new)}, current B_scale_buf: {B_new_contents}", file=log_file)
                    B_scale_row_dict = B_scale_buffer.get_scale_row_dict(B_new)
                    for b_row in B_scale_row_dict.keys():
                        real_row = b_row
                        commands.append(codegen_tool.create_device_bk2buf(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            op1 = (0, real_row, 0),
                            buf = (True, 0, len(B_scale_row_dict[b_row])),
                            auto_precharge = False,
                        ))
                        scale_read_col_num += len(B_scale_row_dict[b_row])
                        scale_read_row_num += 1
                    B_scale_buffer.actual_update_buffer(B_new_contents)
        # 使用 DataBuffer 处理 A data 读入
        if consider_data_read:
            # 获取B数据信息 - 使用便捷方法替代硬编码索引
            current_b_data = cmd.get_current_b_data()
            current_data = current_b_data
            print(f"current_data: {current_data}", file=log_file)
            # 检查是否需要新的数据
            require_dequant = not B_dequant_buffer.is_data_available(current_data)
            # require_new_data = not A_data_buffer.is_data_available(current_a_data)
            if require_dequant:
                # 清空当前的计算指令
                gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, False)
                # 更新 Data Buffer
                data_new, data_new_contents = B_dequant_buffer.update_buffer({current_data})
                # 需要做 dequant
                assert len(data_new) > 0
                # compute_col = int(len(data_new) * a_prec / b_prec)
                compute_col = int(len(data_new) * scale_precision / b_prec)
                read_col = 0
                if not data_buffer.is_list_available(data_new):
                    # 生成 data 读取指令
                    data_new_before_dequant, data_new_before_dequant_contents = data_buffer.update_buffer(data_new)
                    read_col = len(data_new_before_dequant)
                    assert read_col > 0
                    data_before_dequant_row_dict = data_buffer.get_data_row_dict(data_new_before_dequant)
                    # buffer_read_num += 1
                    for row in data_before_dequant_row_dict.keys():
                        buffer_read_row_num += 1
                        sorted_cols = sorted(data_before_dequant_row_dict[row])
                        print(f"[ Read B Data ] r{row}, c{sorted_cols}", file=log_file)
                        commands.append(codegen_tool.create_device_bk2buf(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            op1 = (0, row, 0),
                            buf = (True, 0, len(data_before_dequant_row_dict[row])),
                            auto_precharge = False,
                        ))
                        buffer_read_col_num += len(data_before_dequant_row_dict[row])
                    data_buffer.actual_update_buffer(data_new_before_dequant_contents)
                if compute_col > read_col and consider_result_buffer:
                    commands.append(codegen_tool.create_device_reg2buf(
                        ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                        buffer_addr = 0, end_point = (compute_col-read_col) * SimConfig.col_change_apox,
                    ))
                # print dequant information
                data_row_dict = B_dequant_buffer.get_data_row_dict(data_new)
                for row in data_row_dict.keys():
                    sorted_cols = sorted(data_row_dict[row])
                    print(f"[ Dequant   ] r{row}, c{sorted_cols}, real extra = {compute_col-read_col}", file=log_file)

                B_dequant_buffer.actual_update_buffer(data_new_contents)
        # 获取当前和下一个命令对 - 使用便捷方法替代大量变量解包
        current, next_cmd = cmd_list.get_pair(i)
        current: CmdAddr
        next_cmd: Optional[CmdAddr]
        # 非最后一次循环
        if next_cmd is not None:
            # Col 处理 - 使用便捷的比较方法
            row_change = not current.same_A_row(next_cmd)
            if row_change:
                data_read_row_num += 1
            if not current.same_AB_col(next_cmd) or row_change:
                AB_col.append(current.AB_col)
                data_read_col_num += 1
                print(f"+ A{current.A_pos} B{current.B_pos}_s{current.B_scale_pos} Result{current.matrix_coord}", file=log_file)

            # 使用 ResultBuffer 管理结果 - 使用属性访问替代硬编码索引
            write_back_addr_dict = {}
            if consider_result_buffer:
                # 先把现有结果做插入
                current_result = current.matrix_coord
                if not result_buffer.can_add_result(current_result):
                    # 需要先写回
                    buffered_results = result_buffer.get_writeback_data()
                    for m, n in buffered_results:
                        result_row, result_col = result_addr.get_tensor_addr(m, n)
                        if result_row not in write_back_addr_dict.keys():
                            write_back_addr_dict[result_row] = [result_col]
                        else:
                            if result_col not in write_back_addr_dict[result_row]:
                                write_back_addr_dict[result_row].append(result_col)
                    print(f"[ Write Back Result ]: {write_back_addr_dict}", file=log_file)
                    result_buffer.clear_buffer()
                
                # 添加当前结果
                result_buffer.add_result(current_result)

            # 结果写回，则也需要把之前累加的 Col 给发出
            if row_change or len(write_back_addr_dict) > 0:
                gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, False)

            # issue Result Buffer -> DRAM 的换出指令
            for row in write_back_addr_dict.keys():
                commands.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, row, 0),
                    buf = (True, 0, len(write_back_addr_dict[row])),
                    auto_precharge = False,
                ))
            
        # 最后一次循环
        else:
            # Col 处理 - 使用属性访问替代硬编码索引
            AB_col.append(current.AB_col)
            print(f"+ A{current.A_pos} B{current.B_pos}_s{current.B_scale_pos} Result{current.matrix_coord}", file=log_file)

            # 发出最后的计算指令
            gen_compute_cmd(AB_col, codegen_tool, commands, pu_num, pu_mask, log_file, data_row, False)

            # 先把现有结果做插入
            current_result = current.matrix_coord
            result_buffer.add_result(current_result)
            # print(f"buffered_output: {result_buffer.get_buffer_status()}", file=log_file)
            
            # 最终写回
            buffered_results = result_buffer.get_writeback_data()
            write_back_addr_dict = {}
            for m, n in buffered_results:
                result_row, result_col = result_addr.get_tensor_addr(m, n)
                if result_row not in write_back_addr_dict:
                    write_back_addr_dict[result_row] = [result_col]
                else:
                    if result_col not in write_back_addr_dict[result_row]:
                        write_back_addr_dict[result_row].append(result_col)
            print(f"[ Write Back Result ]: {write_back_addr_dict}", file=log_file)
            
            for idx, row in enumerate(write_back_addr_dict.keys()):
                if idx == len(write_back_addr_dict.keys()) - 1:
                    auto_precharge = True
                else:
                    auto_precharge = False
                commands.append(codegen_tool.create_device_buf2bk(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, row, 0),
                    buf = (True, 0, len(write_back_addr_dict[row])),
                    auto_precharge = auto_precharge,
                ))

    performance_metrics = (
        data_read_col_num, data_read_row_num,
        buffer_read_col_num, buffer_read_row_num,
        scale_read_col_num, scale_read_row_num,
    )
    return performance_metrics, commands