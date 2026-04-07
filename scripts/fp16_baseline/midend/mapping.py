from tools import *
from math import ceil, floor

class Mapping(HW_info):
    def __init__(self, require_power_of_2):
        super(Mapping, self).__init__(require_power_of_2)

    def assign_hw(self, hw_partition):
        if SimConfig.pu_level == LEVEL.DE:
            channel_list = range(mul(hw_partition[0]))
            rank_list = range(mul(hw_partition[1]))
            device_list = range(mul(hw_partition[2]))
            pu_list = range(mul(hw_partition[3]))
        elif SimConfig.pu_level == LEVEL.RA:
            channel_list = range(mul(hw_partition[0]))
            rank_list = range(mul(hw_partition[1]))
            device_list = []
            pu_list = range(mul(hw_partition[2]))
        return channel_list, rank_list, device_list, pu_list
    
    def assign_dram(self, pu_num, mkl_Input_to_row, ml_Out_to_row, hw_partition):
        if SimConfig.pu_level == LEVEL.DE:
            bank_per_pu = self.bank_num // pu_num
            # input decode
            in_block, in_row, in_corner = mkl_Input_to_row
            m_block, k_block, l_block, b_block = in_block
            m_row, k_row, l_row, b_row = in_row
            m_block_corner, k_block_corner, l_block_corner, b_block_corner = in_corner
            # output decode
            out_block, out_row, out_corner = ml_Out_to_row
            om_block, ol_block, ob_block = out_block
            om_row, ol_row, ob_row = out_row
            om_block_corner, ol_block_corner, ob_block_corner = out_corner
            # PU decode
            pu_m, pu_k, pu_l, pu_b = hw_partition[3]

            if bank_per_pu == 1:
                # 可以access的Bank只有一个
                # 同时，有一些Bank需要写入多个input数据
                # 我觉得一个思路是，共享输入的PU对应的Bank分担输入数据，这样就能占用尽量少的行数
                assert SimConfig.de_gb > 0 or SimConfig.de_pu_inbuf > 0
                # weight
                weight_bank = 0
                weight_row_offset = 0
                # input
                input_bank = 0 # indicate that the input should be shared using global buffer
                input_row_offset = k_row * l_row * b_row
                input_share_num = pu_l # the number of banks that share the same input
                input_row_after_div = ceil( m_row * k_row * b_row / input_share_num )
                # output
                output_bank = 0
                output_row_offset = input_row_offset + input_row_after_div
            else:
                # input = bank 0
                # weight = bank 1
                # output = bank 2 (if there's another bank)/ 0,1 (or choose the less one)
                input_bank = 0
                weight_bank = 1
                if bank_per_pu > 2:
                    output_bank = 2
                    output_row_offset = 0
                else:
                    if m_row > l_row:
                        output_bank = 1
                        output_row_offset = k_row * l_row * b_row
                    else:
                        output_bank = 0
                        output_row_offset = m_row * k_row * b_row
                # row list
                input_row_offset = 0
                weight_row_offset = 0
        elif SimConfig.pu_level == LEVEL.RA:
            device_per_pu = self.device_num // pu_num
            # input decode
            in_block, in_row, in_corner = mkl_Input_to_row
            m_block, k_block, l_block, b_block = in_block
            m_row, k_row, l_row, b_row = in_row
            m_block_corner, k_block_corner, l_block_corner, b_block_corner = in_corner
            # output decode
            out_block, out_row, out_corner = ml_Out_to_row
            om_block, ol_block, ob_block = out_block
            om_row, ol_row, ob_row = out_row
            om_block_corner, ol_block_corner, ob_block_corner = out_corner
            pu_m, pu_k, pu_l, pu_b = hw_partition[2]

            if device_per_pu == 1:
                assert SimConfig.ra_gb > 0 or SimConfig.ra_pu_inbuf > 0
                # weight
                weight_bank = 0
                weight_row_offset = 0
                # input
                input_bank = 0 # indicate that the input should be shared using global buffer
                input_row_offset = k_row * l_row * b_row
                input_div_num = pu_l
                input_row_after_div = ceil(m_row*k_row*b_row/input_div_num) 
                # output
                output_bank = 0
                output_row_offset = input_row_offset + input_row_after_div
            else:
                # input = bank 0
                # weight = bank 1
                # output = bank 2 (if there's another bank)/ 0,1 (or choose the less one)
                input_bank = 0
                weight_bank = 1
                if device_per_pu > 2:
                    output_bank = 2
                    output_row_offset = 0
                else:
                    if m_row > l_row:
                        output_bank = 1
                        output_row_offset = k_row * l_row * b_row
                    else:
                        output_bank = 0
                        output_row_offset = m_row * k_row * b_row
                # row list
                input_row_offset = 0
                weight_row_offset = 0
        return input_bank, input_row_offset, \
            weight_bank, weight_row_offset, \
                output_bank, output_row_offset
