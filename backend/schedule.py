from tools import SimConfig

class IR_List:
    def __init__(self, meta_data, max_partial_size, extra_scale_b_buffer,codegen_tool,log_file):
        self.meta_data = meta_data
        self.max_partial_size = max_partial_size
        self.ir_list = []
        self.log_file = log_file
        self.codegen_tool = codegen_tool
        self.slot_size = {}
        self.valid_slot_in_order = []
        self.scale_b_read_id_list = []
        self.uncovered_dequant_id_list = [] # 未被覆盖的 dequant 指令 id 列表
        self.optimal_col = 0
        self.extra_scale_b_buffer = extra_scale_b_buffer
    
    def merge_ranges(self, sorted_list):
        """Merge adjacent numbers in a sorted list into ranges like '1-3,5,10'"""
        if not sorted_list:
            return ""
        
        result = []
        start = sorted_list[0]
        end = sorted_list[0]
        
        for i in range(1, len(sorted_list)):
            if sorted_list[i] == end + 1:
                end = sorted_list[i]
            else:
                if start == end:
                    result.append(str(start))
                else:
                    result.append(f"{start}-{end}")
                start = end = sorted_list[i]
        
        if start == end:
            result.append(str(start))
        else:
            result.append(f"{start}-{end}")
        
        return ",".join(result)

    # 将不同类型的ir打印至logfile, debug 时才会使用
    def __print(self, ir, idx):
        type = ir[0]
        if type == "compute":
            data_row, AB_col, AB_col_rst = ir[1:4]
            A_col = list(set([col for col, _ in AB_col]))
            sorted_a_col = sorted(A_col)
            B_col = list(set([col for _, col in AB_col]))
            sorted_b_col = sorted(B_col)
            # NOTE: Only valid for debug
            o = {}
            for m,n,k_scale_group_b_id in AB_col_rst:
                if (m,n) not in o.keys():
                    o[(m,n)] = [k_scale_group_b_id]
                else:
                    o[(m,n)].append(k_scale_group_b_id)
            for (m,n) in o.keys():
                o[(m,n)] = self.merge_ranges(o[(m,n)])
            print(f"{idx:>4}[ Compute      ] row {data_row}, col {self.merge_ranges(sorted_b_col)}, i_col {self.merge_ranges(sorted_a_col)} o{o}", file=self.log_file)
        elif type == "read_a":
            row, cols = ir[1:3]
            print(f"{idx:>4}[ Read A Data  ] row {row}, col {self.merge_ranges(sorted(cols))}", file=self.log_file)
        elif type == "scale_a_read":
            row, cols, new, new_contents = ir[1:5]
            print(f"{idx:>4}[ Read A Scale ] row {row}, col {self.merge_ranges(sorted(cols))}, read {new}, new {new_contents}", file=self.log_file)
        elif type == "scale_b_read":
            row, cols, new, new_contents = ir[1:5]
            print(f"{idx:>4}[ Read B Scale ] row {row}, col {self.merge_ranges(sorted(cols))}, read {new}, new {new_contents}", file=self.log_file)
        elif type == "dequant":
            compute_col, read_col, new_added = ir[1:4]
            if read_col > 0:
                print(f"{idx:>4}[ Dequant  r+d ] compute {compute_col} col, read {read_col} col, (m,n)={new_added[0]}, k_scale_id {self.merge_ranges(new_added[1])}, b scale pos: {new_added[2]}, a scale pos: {new_added[3]}", file=self.log_file)
            elif idx in self.uncovered_dequant_id_list:
                print(f"{idx:>4}[ Dequant  --- ] compute {compute_col} col, read {read_col} col, (m,n)={new_added[0]}, k_scale_id {self.merge_ranges(new_added[1])}, b scale pos: {new_added[2]}, a scale pos: {new_added[3]}", file=self.log_file)
            else:
                print(f"{idx:>4}[ Dequant      ] compute {compute_col} col, read {read_col} col, (m,n)={new_added[0]}, k_scale_id {self.merge_ranges(new_added[1])}, b scale pos: {new_added[2]}, a scale pos: {new_added[3]}", file=self.log_file)
        elif type == "write_back":
            row, cols, auto_precharge = ir[1:4]
            print(f"{idx:>4}[ Write Back   ] row {row}, col {self.merge_ranges(sorted(cols)) if isinstance(cols, list) else cols}, auto_precharge {auto_precharge}", file=self.log_file)
        else:
            pass
    
    # 将不同类型的ir打印至logfile, debug 时才会使用
    def log_energy(self, mix_scale_mapping, pu_num):
        last_row = -1
        for ir in self.ir_list:
            type = ir[0]
            if type == "compute":
                row, AB_col, AB_col_rst = ir[1:4]
                SimConfig.rd_num += len(AB_col) * pu_num
                SimConfig.compute_num += len(AB_col) * pu_num
                SimConfig.act_cycle += len(AB_col) * SimConfig.col_change_apox
            elif type == "read_a":
                row, cols = ir[1:3]
                SimConfig.rd_num += len(cols) * pu_num
                SimConfig.act_cycle += len(cols) * SimConfig.col_change_apox
            elif type == "scale_a_read":
                row, cols, new, new_contents = ir[1:5]
                SimConfig.rd_num += len(cols) * pu_num
                SimConfig.act_cycle += len(cols) * SimConfig.col_change_apox
            elif type == "scale_b_read":
                row, cols, new, new_contents = ir[1:5]
                SimConfig.rd_num += len(cols) * pu_num
                SimConfig.act_cycle += len(cols) * SimConfig.col_change_apox
            elif type == "dequant":
                compute_col, read_col, new_added = ir[1:4]
                scale_pos = new_added[2]
                if read_col > 0:
                    row = scale_pos[0]
                    SimConfig.rd_num += read_col * pu_num
                    SimConfig.compute_num += read_col * pu_num
                    SimConfig.act_cycle += read_col * SimConfig.col_change_apox
                else:
                    row = last_row
                if compute_col - read_col > 0:
                    SimConfig.compute_num += (compute_col - read_col) * pu_num
            elif type == "write_back":
                row, cols, auto_precharge = ir[1:4]
                SimConfig.wr_num += len(cols) * pu_num
                SimConfig.act_cycle += len(cols) * SimConfig.col_change_apox
            
            if row != last_row:
                SimConfig.act_num += pu_num
                SimConfig.pre_num += pu_num
                last_row = row

    # codegen
    def __codegen(self, ir):
        pu_num, pu_mask, mix_scale_mapping, col_change_apox, row_change_apox = self.meta_data
        type = ir[0]
        if type == "compute":
            data_row, AB_col, AB_col_rst = ir[1:4]
            col_num = len(AB_col)
            if col_num > 0:
                A_col_offset, B_col_offset = AB_col[0]
                return [self.codegen_tool.create_device_pu(
                    ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                    op1 = (0, data_row, B_col_offset),
                    op2 = (0, 1, A_col_offset), # from GB
                    col_num = col_num,
                    auto_precharge = False,
                )]
        elif type == "read_a":
            row, cols = ir[1:3]
            return [self.codegen_tool.create_device_bk2gb(
                ch_id=0, ra_id=0, de_id=0, bank_id = 0,
                op1 = (row, 0),
                gb_col_offset=0,
                col_num = len(cols),
                auto_precharge = False,
            )]
        elif type == "scale_a_read":
            row, cols, new, new_contents = ir[1:5]
            return [self.codegen_tool.create_device_bk2buf(
                ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                op1 = (0, row, 0),
                buf = (True, 0, len(cols)),
                auto_precharge = False,
            )]
        elif type == "scale_b_read":
            row, cols, new, new_contents = ir[1:5]
            return [self.codegen_tool.create_device_bk2buf(
                ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                op1 = (0, row, 0),
                buf = (True, 0, len(cols)),
                auto_precharge = False,
            )]
        elif type == "dequant":
            compute_col, read_col, new_added = ir[1:4]
            scale_pos = new_added[2]
            if mix_scale_mapping:
                return_list = []
                if read_col > 0:
                    return_list.append(
                        self.codegen_tool.create_device_pu(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            op1 = (0, scale_pos[0], scale_pos[1]),
                            op2 = (0, 1, 0),
                            col_num = read_col,
                            auto_precharge = False,
                        )
                    )
                if compute_col-read_col > 0:
                    return_list.append(
                        self.codegen_tool.create_device_reg2buf(
                            ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                            buffer_addr = 0, end_point = (compute_col-read_col) * col_change_apox,
                        )
                    )
                return return_list
            else:
                return [
                    self.codegen_tool.create_device_reg2buf(
                        ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                        buffer_addr = 0, end_point = (compute_col-read_col) * col_change_apox,
                    )
                ]
        elif type == "write_back":
            row, cols, auto_precharge = ir[1:4]
            return [self.codegen_tool.create_device_buf2bk(
                ch_id=0, ra_id=0, de_id=0, pu_num=pu_num, pu_mask=pu_mask, 
                op1 = (0, row, 0),
                buf = (True, 0, len(cols)),
                auto_precharge = auto_precharge,
            )]
        else:
            raise NotImplementedError(f"codegen for {type} is not implemented")

    def append(self, ir):
        if ir[0] == "dequant":
            if ir[2] == 0:
                self.uncovered_dequant_id_list.append(len(self.ir_list))
            self.optimal_col += ir[1]
        elif ir[0] == "compute":
            self.optimal_col += len(ir[2])
        self.ir_list.append(ir)

    def print(self):
        for idx, ir in enumerate(self.ir_list):
            if ir != []:
                self.__print(ir, idx)

    def __dequant_pass_check(self, dequant_buffer_left, another_ir):
        if another_ir == []:
            return dequant_buffer_left
        elif another_ir[0] == 'compute':
            _, _, compute_AB_col_rst = another_ir[1:4]
            burden = len(compute_AB_col_rst)
        elif another_ir[0] in ['scale_a_read', 'dequant', 'write_back', 'scale_b_read']:
            burden = self.max_partial_size + 1
        elif another_ir[0] == 'read_a':
            burden = 0
        else:
            raise NotImplementedError(f"dequant pass check for {another_ir[0]} is not implemented")
        return dequant_buffer_left - burden

    def __move_ir(self, ir_id, new_id):
        tmp = self.ir_list[ir_id]
        for id in range(ir_id, new_id):
            self.ir_list[id] = self.ir_list[id+1]
        self.ir_list[new_id] = tmp
    
    # pass 1: moving dequant
    def __pre_scan(self):
        pu_num, pu_mask, mix_scale_mapping, col_change_apox, row_change_apox = self.meta_data
        potantial_slot = 0
        print(f"optimal cycle {self.optimal_col*col_change_apox}", file = self.log_file)
        # slot 为可以容纳的 col 数量
        open_row = -1
        for id, ir in enumerate(self.ir_list):
            # row_change_slot
            if ir[0] != "dequant":
                row = ir[1]
                slot_id = id - 1
                # row change slot
                if row != open_row:
                    if open_row != -1:
                        self.slot_size[slot_id] = row_change_apox
                        self.valid_slot_in_order.append(slot_id)
                        potantial_slot += row_change_apox
                        print(f"row_change_slot {slot_id} {self.ir_list[id-1][0]}->{ir[0]} {row_change_apox}", file = self.log_file)
                    open_row = row
                # dequant slot
                if ir[0] == "read_a":
                    if slot_id in self.slot_size.keys():
                        self.slot_size[slot_id] += len(ir[2])*col_change_apox
                        potantial_slot += len(ir[2])*col_change_apox
                    else:
                        self.slot_size[slot_id] = len(ir[2])*col_change_apox
                        self.valid_slot_in_order.append(slot_id)
                        potantial_slot += len(ir[2])*col_change_apox
                    print(f"read_a_slot {slot_id} {ir[0]} {len(ir[2])*col_change_apox}", file = self.log_file)
            else:
                # add dequant info
                dequant_compute_col, dequant_read_col, dequant_new_added = ir[1:4]
                # 计算剩余的 Buffer 容量
                dequant_buffer_left = self.max_partial_size - len(dequant_new_added[1])
                # 计算
                self.ir_list[id] = ["dequant", dequant_compute_col, dequant_read_col, dequant_new_added, dequant_buffer_left]
        print(f"potantial_slot {potantial_slot}", file = self.log_file)   
    
    def first_optimize_pass(self):
        self.__pre_scan()
        pu_num, pu_mask, mix_scale_mapping, col_change_apox, row_change_apox = self.meta_data
        reverse_dequant_id_list = self.uncovered_dequant_id_list[::-1]
        print(f"start from {len(reverse_dequant_id_list)} uncovered dequant id", file = self.log_file)
        for dequant_id in reverse_dequant_id_list:
            # 获取当前 dequant 指令的信息
            dequant_ir = self.ir_list[dequant_id]
            _, _, _, _, dequant_buffer_left = dequant_ir
            
            # 尝试将 dequant 指令向后移动
            current_pos = dequant_id
            possible_choice = []
            while current_pos < len(self.ir_list) - 1:
                # 先判断当前位置是否为一个有效的 slot
                if current_pos in self.valid_slot_in_order:
                    slot_left = self.slot_size[current_pos]
                    possible_choice.append((current_pos, dequant_buffer_left, slot_left))
                next_ir = self.ir_list[current_pos + 1]
                dequant_buffer_left = self.__dequant_pass_check(dequant_buffer_left, next_ir)
                if dequant_buffer_left >= 0:
                    current_pos += 1
                else:
                    break
            
            # 如果可以交换，我会优先选择最靠后的位置
            if len(possible_choice) > 0:
                switch_pos, dequant_buffer_left, slot_left = possible_choice[-1]
                # if switch_pos != dequant_id: continue
                print(f"dequant_id {dequant_id} switch_pos {switch_pos} dequant_buffer_left {dequant_buffer_left} slot_left {slot_left}", file = self.log_file)
                # update dequant_buffer_left
                self.ir_list[dequant_id][4] = dequant_buffer_left
                # mark dequant_id as covered
                self.uncovered_dequant_id_list.remove(dequant_id)
                # FIXME: 如果之后的计算并行度/计算频率发生变化，这个值可能会变
                slot_left -= 2 * col_change_apox # 
                for _switch_pos, _, _slot_left in possible_choice[:-1]:
                    tmp_slot_left = self.slot_size.pop(_switch_pos)
                    self.slot_size[_switch_pos-1] = tmp_slot_left
                    self.valid_slot_in_order.remove(_switch_pos)
                    self.valid_slot_in_order.append(_switch_pos-1)
                # 维护过程中受影响的 slot 位置
                if slot_left > 0:
                    # 成功交换后，首先 slot_id 会往前走
                    new_slot_id = switch_pos - 1
                    self.slot_size.pop(switch_pos)
                    self.slot_size[new_slot_id] = slot_left
                    self.valid_slot_in_order.remove(switch_pos)
                    self.valid_slot_in_order.append(new_slot_id)
                else: # 此处已经不是一个有效的 slot 了，需要删除
                    self.slot_size.pop(switch_pos)
                    self.valid_slot_in_order.remove(switch_pos)
                # 实际交换指令
                self.__move_ir(dequant_id, switch_pos)
        print(f"end with {len(self.uncovered_dequant_id_list)} uncovered dequant id", file = self.log_file)

    # second pass: scale b absorb dequant
    
    def __scale_b_scan(self):
        for id, ir in enumerate(self.ir_list):
            if ir[0] == "scale_b_read":
                self.scale_b_read_id_list.append(id)
                self.ir_list[id].append([]) # 额外添加一个空位，用于吸收 dequant 指令

    # 吸收到未优化的 dequant 是有效的，被吸收的前几个指令会以空位代替，最后一个被吸收的指令会被替换为 dequant 和 scale b read 的混合
    # 遇到已优化的 dequant 指令需要停止
    # 后续的compute指令，如果在同一行可以互换
    def second_optimize_pass(self):
        print(f"start from {len(self.uncovered_dequant_id_list)} uncovered dequant id", file = self.log_file)
        self.__scale_b_scan()
        scale_b_read_id_list = self.scale_b_read_id_list.copy()

        for id in scale_b_read_id_list:
            ir = self.ir_list[id]
            row, cols, new, new_contents = ir[1:5]
            current_pos = id
            to_cover_col = len(cols)
            effective_dequant_id = []
            while current_pos < len(self.ir_list) - 1:
                if self.ir_list[current_pos + 1][0] == "dequant":
                    dequant_ir = self.ir_list[current_pos + 1]
                    if current_pos + 1 in self.uncovered_dequant_id_list:
                        effective_dequant_id.append(current_pos + 1)
                        current_pos += 1
                        dequant_compute_col, dequant_read_col, dequant_new_added = dequant_ir[1:4]
                        to_cover_col -= dequant_compute_col
                        if to_cover_col <= 0:
                            break
                    else:
                        break
                elif self.ir_list[current_pos + 1][0] == "compute":
                    row = self.ir_list[current_pos + 1][1]
                    if row == ir[1]:
                        current_pos += 1
                    else:
                        break
                elif self.ir_list[current_pos + 1][0] == "scale_b_read":
                    break
                else:
                    current_pos += 1
            if len(effective_dequant_id) > 0:
                self.scale_b_read_id_list.remove(id)
                compute_col = len(cols) - to_cover_col
                read_col = len(cols)
                # col_num = len(cols) - min(to_cover_col, 0)
                b_scale_pos = new[0]
                mn = ""
                scale_group_id = []
                self.ir_list[id] = []
                for dequant_id in effective_dequant_id:
                    self.uncovered_dequant_id_list.remove(dequant_id)
                    mn += str(self.ir_list[dequant_id][3][0])
                    scale_group_id += self.ir_list[dequant_id][3][1]
                    self.ir_list[dequant_id] = []
                self.ir_list[effective_dequant_id[-1]] = ["dequant", compute_col, read_col, (mn, scale_group_id, b_scale_pos, 0), 0]
                print(f"read b scale {id} + dequant {effective_dequant_id}-> dequant {effective_dequant_id[-1]}", file = self.log_file)
        print(f"end with {len(self.uncovered_dequant_id_list)} uncovered dequant id", file = self.log_file)

    # 如果 scale b buffer 有更多的容量，则允许将先前的 dequant 指令吸收进来
    def third_optimize_pass(self):
        for id in self.scale_b_read_id_list:
            self.ir_list[id].append(len(self.ir_list[id][2]))
        # scale b format, 'scale_b_read', row, cols, new, new_contents, [], to_cover_col
        reverse_dequant_id_list = self.uncovered_dequant_id_list[::-1]
        print(f"start from {len(reverse_dequant_id_list)} uncovered dequant id", file = self.log_file)
        for dequant_id in reverse_dequant_id_list:
            # 获取当前 dequant 指令的信息
            dequant_ir = self.ir_list[dequant_id]
            _, compute_col, read_col, new_added, dequant_buffer_left = dequant_ir
            scale_b_pos = new_added[2]
            # 尝试将 dequant 指令向后移动
            current_pos = dequant_id
            while current_pos < len(self.ir_list) - 1:
                # 先判断当前位置是否为一个有效的 slot
                if (current_pos + 1) in self.scale_b_read_id_list:
                    if scale_b_pos in self.ir_list[current_pos + 1][5] or len(self.ir_list[current_pos + 1][5]) < self.extra_scale_b_buffer:
                        self.uncovered_dequant_id_list.remove(dequant_id)
                        self.ir_list[dequant_id] = []
                        self.ir_list[current_pos + 1][6] -= compute_col
                        if scale_b_pos not in self.ir_list[current_pos + 1][5]:
                            self.ir_list[current_pos + 1][5].append(scale_b_pos)
                        if self.ir_list[current_pos + 1][6] <= 0:
                            self.scale_b_read_id_list.remove(current_pos + 1)
                            self.ir_list[current_pos + 1] = [
                                'dequant',
                                len(self.ir_list[current_pos + 1][2])-self.ir_list[current_pos + 1][6],
                                len(self.ir_list[current_pos + 1][2]),
                                ('nan', [], (self.ir_list[current_pos + 1][1],0), 'nan'),
                                0
                            ]
                    break
                next_ir = self.ir_list[current_pos + 1]
                dequant_buffer_left = self.__dequant_pass_check(dequant_buffer_left, next_ir)
                if dequant_buffer_left >= 0:
                    current_pos += 1
                else:
                    break
            
        # scan other modified scale b read
        scale_b_read_id_list = self.scale_b_read_id_list.copy()
        for id in scale_b_read_id_list:
            if self.ir_list[id][5] != []:
                self.scale_b_read_id_list.remove(id)
                self.ir_list[id] = [
                    'dequant',
                    len(self.ir_list[id][2])-self.ir_list[id][6],
                    len(self.ir_list[id][2]),
                    ('nan', [], (self.ir_list[id][1],0), 'nan'),
                    0
                ]
            
        print(f"end with {len(self.uncovered_dequant_id_list)} uncovered dequant id", file = self.log_file)

    def codegen(self):
        commands = []
        for ir in self.ir_list:
            if ir != []:
                commands.extend(self.__codegen(ir))
        return commands
    