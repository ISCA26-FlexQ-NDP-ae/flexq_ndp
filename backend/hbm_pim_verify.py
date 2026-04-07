from tools import *
from backend.base import BaseCodegen
import numpy as np
from math import ceil

class hbmpim_verify(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(hbmpim_verify, self).__init__(require_power_of_2)
    
    def mm_micro(self, mm_schedule, base_group_id,
                    channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row, 
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b,
                    pu_list, performance_threshold):
        # pu, pu_col, pu_row_change, host_write_pu_inbuf_col
        tmp_inst_groups = []
        cmd_left = performance_threshold
        group_id = base_group_id
        # the verification only perform on MVM
        assert m_block == 1 and m_row == 1 and om_block == 1 and om_row == 1
        assert l_block == 4 # 32 / 8 = 4
        
        weight_bank_list = [pu_id * self.bank_num // pu_num + weight_bank for pu_id in pu_list]
        rank_id = 0 # hbm only have one pim rank
        tmp_inst_list = []
        # NOTE: 1. weight load, not considered in the PIMSimulator, so we also don't consider it
        # for device_id in device_list:
        #     device_mask = [i==device_id for i in range(SimConfig.de)]
        #     for weight_bank_id in weight_bank_list:
        #         bank_mask = [i==weight_bank_id for i in range(self.bank_num)]
        #         for l_row_id in range(l_row):
        #             l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
        #             for k_row_id in range(k_row):
        #                 k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
        #                 col_len = k_block_real * l_block_real
        #                 tmp_inst_list.append(self.create_host_write(
        #                     channel_id, rank_id, device_mask, bank_mask, weight_row_offset, 0, col_len, True
        #                 ))
        # NOTE: 2. PIM computation
        device_mask = [i in device_list for i in range(SimConfig.de)]
        pu_mask = [(i in pu_list) for i in range(pu_num)]
        # cut block = 8 * 8 col = 2 row
        out_loop = ceil(l_row / 2)
        last_l_row = l_row - (out_loop - 1) * 2
        in_loop = k_row
        # outer loop
        for out_loop_id in range(out_loop):
            l_row_inner = 2 if out_loop_id < out_loop - 1 else last_l_row
            # inner loop
            for k_row_id in range(in_loop):
                k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                col_len = k_block_real
                pu_mask = [True for _ in range(pu_num)]
                # rewrite the input buffer
                for channel_id in channel_list: # hbm-pim manner: the sychronization is performed every computation
                    tmp_inst_list.append(
                        self.create_host_write_pu_inbuf(
                            channel_id, rank_id, device_mask, pu_mask, 0, col_len
                        )
                    )
                # inner slice
                for l_row_id in range(out_loop_id * 2, out_loop_id * 2 + l_row_inner):
                    l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                    weight_row_id = l_row_id + k_row_id * l_row
                    for l_block_id in range(l_block_real):
                        input_col_offset = 0
                        weight_col_offset = l_block_id * k_block_real
                        weight_rowchange = (l_block_id == l_block_real - 1)
                        # compute 
                        for channel_id in channel_list:
                            for device_id in device_list:
                                tmp_inst_list.append(self.create_device_pu(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (weight_bank, weight_row_offset + weight_row_id, weight_col_offset), 
                                    (weight_bank, 0, input_col_offset), 
                                    col_len, weight_rowchange
                        ))
                # NOTE: 3. result move to bank
                for channel_id in channel_list:
                    for device_id in device_list:
                        tmp_inst_list.append(self.create_device_buf2bk(
                            channel_id, rank_id, device_id, pu_num, pu_mask, 
                            (output_bank, output_row_offset + l_row_id, 0), 
                            (False, 0, 0), True
                        ))
        tmp_inst_groups.append((group_id, [], tmp_inst_list))
        group_id += 1
        cmd_left -= len(tmp_inst_list)
        # break

        # if performance_threshold < inf:
        return tmp_inst_groups, performance_threshold - cmd_left