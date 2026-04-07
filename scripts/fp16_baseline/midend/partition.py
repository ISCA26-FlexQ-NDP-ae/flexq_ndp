import random
from tools import *
from functools import reduce
import math

# Tools Functions
# mul = lambda x: reduce(lambda x,y:x*y,x)

class Partition(HW_info):

    def __init__(self, require_power_of_2):
        # HW information
        super(Partition, self).__init__(require_power_of_2)
        self.require_power_of_2 = require_power_of_2

    def get_partition_space_mm(self, mm_size=[1000,1000,1000,1], bitwidth=16):
        """
        compute-level, pu-mode
        mm partition point= public: m(CH, RA, DE, PU),  k(CH, RA, DE, PU),  l(CH, RA, DE, PU)
                            input:  m(RO, CO, SIMD=0),  k(RO, CO, SIMD  ),  
                            weight:                     k(RO, CO, SIMD  ),  l(RO, CO, SIMD=0)
                            output: m(RO, CO, SIMD  ),                      l(RO, CO, SIMD  )                                          
        """
        m, k, l, b = mm_size # b = batch dimension, for Activation-Activation computation
        # device-level
        level = SimConfig.pu_level
        possible_div = []
        if level == LEVEL.DE:
            for device_pu_num in self.de_pu_num_list:
                # 获取可能的分割方式
                ch_divide = [(ch_m, ch_k, ch_l, ch_b) for ch_m in range(1, self.channel_num+1) 
                            for ch_k in range(1, self.channel_num+1) 
                            for ch_l in range(1, self.channel_num+1)
                            for ch_b in range(1, self.channel_num+1)
                            if ch_m * ch_k * ch_l * ch_b <= self.channel_num 
                            and self.powerof2(ch_m) and self.powerof2(ch_k) and self.powerof2(ch_l) and self.powerof2(ch_b)]
                ra_divide = [(ra_m, ra_k, ra_l, ra_b) for ra_m in range(1, self.rank_num+1) 
                            for ra_k in range(1, self.rank_num+1) 
                            for ra_l in range(1, self.rank_num+1) 
                            for ra_b in range(1, self.rank_num+1)
                            if ra_m * ra_k * ra_l * ra_b <= self.rank_num 
                            and self.powerof2(ra_m) and self.powerof2(ra_k) and self.powerof2(ra_l) and self.powerof2(ra_b)]
                de_divide = [(de_m, de_k, de_l, de_b) 
                            for de_m in range(1, self.device_num+1) 
                            for de_k in range(1, self.device_num+1) 
                            for de_l in range(1, self.device_num+1) 
                            for de_b in range(1, self.device_num+1)
                            if de_m * de_k * de_l * de_b <= self.device_num 
                            and self.powerof2(de_m) and self.powerof2(de_k) and self.powerof2(de_l) and self.powerof2(de_b)]
                pu_divide = [(pu_m, pu_k, pu_l, pu_b) for pu_m in range(1, device_pu_num+1) 
                            for pu_k in range(1, device_pu_num+1) 
                            for pu_l in range(1, device_pu_num+1) 
                            for pu_b in range(1, device_pu_num+1)
                            if pu_m * pu_k * pu_l * pu_b <= device_pu_num 
                            and self.powerof2(pu_m) and self.powerof2(pu_k) and self.powerof2(pu_l) and self.powerof2(pu_b)]
                # #  m_first
                # ch_divide = [(ch_m, ch_k, ch_l) for ch_k in range(1, self.channel_num+1) 
                #             for ch_l in range(1, self.channel_num+1) 
                #             for ch_m in range(1, self.channel_num+1) 
                #             if ch_m * ch_k * ch_l <= self.channel_num 
                #             and self.powerof2(ch_m) and self.powerof2(ch_k) and self.powerof2(ch_l)]
                # ra_divide = [(ra_m, ra_k, ra_l) for ra_k in range(1, self.rank_num+1) 
                #             for ra_l in range(1, self.rank_num+1) 
                #             for ra_m in range(1, self.rank_num+1) 
                #             if ra_m * ra_k * ra_l <= self.rank_num 
                #             and self.powerof2(ra_m) and self.powerof2(ra_k) and self.powerof2(ra_l)]
                # de_divide = [(de_m, de_k, de_l) 
                #             for de_k in range(1, self.device_num+1) 
                #             for de_l in range(1, self.device_num+1) 
                #             for de_m in range(1, self.device_num+1) 
                #             if de_m * de_k * de_l <= self.device_num 
                #             and self.powerof2(de_m) and self.powerof2(de_k) and self.powerof2(de_l)]
                # pu_divide = [(pu_m, pu_k, pu_l) for pu_m in range(1, device_pu_num+1) 
                #             for pu_k in range(1, device_pu_num+1) 
                #             for pu_l in range(1, device_pu_num+1) 
                #             if pu_m * pu_k * pu_l <= device_pu_num 
                #             and self.powerof2(pu_m) and self.powerof2(pu_k) and self.powerof2(pu_l)]
                # divide = [(ch, ra, de, pu) for ch in ch_divide 
                #                 for ra in ra_divide for de in de_divide for pu in pu_divide
                
                # NOTE: if no good baseline is found, then you should first garunteed the first dimension is the best
                divide = [(ch, ra, de, pu) for pu in pu_divide 
                                for de in de_divide for ra in ra_divide for ch in ch_divide 
                                if ((ch[0] * ra[0] * de[0] * pu[0] <= m 
                                        or (self.require_power_of_2  and max(1, ch[0] * ra[0] * de[0] * pu[0] / 2) < m) 
                                        or (not self.require_power_of_2 and min(max(1,(ch[0]-1))*ra[0]*de[0]*pu[0], ch[0]*max(1,(ra[0]-1))*de[0]*pu[0], ch[0]*ra[0]*max(1,(de[0]-1))*pu[0], ch[0]*ra[0]*de[0]*max(1,(pu[0]-1))) < m))
                                    and (ch[1] * ra[1] * de[1] * pu[1] <= k 
                                        or (self.require_power_of_2  and max(1, ch[1] * ra[1] * de[1] * pu[1] / 2) < k)
                                        or (not self.require_power_of_2 and min(max(1,(ch[1]-1))*ra[1]*de[1]*pu[1], ch[1]*max(1,(ra[1]-1))*de[1]*pu[1], ch[1]*ra[1]*max(1,(de[1]-1))*pu[1], ch[1]*ra[1]*de[1]*max(1,(pu[1]-1))) < k))
                                    and (ch[2] * ra[2] * de[2] * pu[2] <= l
                                        or (self.require_power_of_2  and max(1, ch[2] * ra[2] * de[2] * pu[2] / 2) < l)
                                        or (not self.require_power_of_2 and min(max(1,(ch[2]-1))*ra[2]*de[2]*pu[2], ch[2]*max(1,(ra[2]-1))*de[2]*pu[2], ch[2]*ra[2]*max(1,(de[2]-1))*pu[2], ch[2]*ra[2]*de[2]*max(1,(pu[2]-1))) < l))   
                                    and (ch[3] * ra[3] * de[3] * pu[3] <= b
                                        or (self.require_power_of_2  and max(1, ch[3] * ra[3] * de[3] * pu[3] / 2) < b)
                                        or (not self.require_power_of_2 and min(max(1,(ch[3]-1))*ra[3]*de[3]*pu[3], ch[3]*max(1,(ra[3]-1))*de[3]*pu[3], ch[3]*ra[3]*max(1,(de[3]-1))*pu[3], ch[3]*ra[3]*de[3]*max(1,(pu[3]-1))) < b))
                                )]
                possible_div.extend(
                    [(level, device_pu_num, divide) for divide in divide]
                )
                print(f"channel={self.channel_num}, rank={self.rank_num}, device={self.device_num}, device_pu={device_pu_num}, divide_num={len(divide)}")
        else:
            # 获取可能的分割方式
            assert SimConfig.pu_level == LEVEL.RA
            rank_pu_num = SimConfig.ra_pu
            # ch_divide = [(ch_m, ch_k, ch_l) for ch_k in range(1, self.channel_num+1) 
            #             for ch_l in range(1, self.channel_num+1) 
            #             for ch_m in range(1, self.channel_num+1) 
            #             if ch_m * ch_k * ch_l <= self.channel_num 
            #             and self.powerof2(ch_m) and self.powerof2(ch_k) and self.powerof2(ch_l)]
            # ra_divide = [(ra_m, ra_k, ra_l) for ra_k in range(1, self.rank_num+1) 
            #             for ra_l in range(1, self.rank_num+1) 
            #             for ra_m in range(1, self.rank_num+1) 
            #             if ra_m * ra_k * ra_l <= self.rank_num 
            #             and self.powerof2(ra_m) and self.powerof2(ra_k) and self.powerof2(ra_l)]
            # pu_divide = [(pu_m, pu_k, pu_l) for pu_m in range(1, rank_pu_num+1) 
            #             for pu_k in range(1, rank_pu_num+1) 
            #             for pu_l in range(1, rank_pu_num+1) 
            #             if pu_m * pu_k * pu_l <= rank_pu_num 
            #             and self.powerof2(pu_m) and self.powerof2(pu_k) and self.powerof2(pu_l)]
            ch_divide = [(ch_m, ch_k, ch_l, ch_b) for ch_m in range(1, self.channel_num+1) 
                        for ch_k in range(1, self.channel_num+1) 
                        for ch_l in range(1, self.channel_num+1)
                        for ch_b in range(1, self.channel_num+1)
                        if ch_m * ch_k * ch_l * ch_b <= self.channel_num 
                        and self.powerof2(ch_m) and self.powerof2(ch_k) and self.powerof2(ch_l) and self.powerof2(ch_b)]
            ra_divide = [(ra_m, ra_k, ra_l, ra_b) for ra_m in range(1, self.rank_num+1) 
                        for ra_k in range(1, self.rank_num+1) 
                        for ra_l in range(1, self.rank_num+1) 
                        for ra_b in range(1, self.rank_num+1)
                        if ra_m * ra_k * ra_l * ra_b <= self.rank_num 
                        and self.powerof2(ra_m) and self.powerof2(ra_k) and self.powerof2(ra_l) and self.powerof2(ra_b)]
            pu_divide = [(pu_m, pu_k, pu_l, pu_b) for pu_m in range(1, rank_pu_num+1) 
                        for pu_k in range(1, rank_pu_num+1) 
                        for pu_l in range(1, rank_pu_num+1) 
                        for pu_b in range(1, rank_pu_num+1)
                        if pu_m * pu_k * pu_l * pu_b <= rank_pu_num 
                        and self.powerof2(pu_m) and self.powerof2(pu_k) and self.powerof2(pu_l) and self.powerof2(pu_b)]
            divide = [(ch, ra, pu) for ch in ch_divide 
                            for ra in ra_divide for pu in pu_divide
                            if ((ch[0] * ra[0] * pu[0] <= m 
                                    or (self.require_power_of_2  and max(1, ch[0] * ra[0] * pu[0] / 2) < m) 
                                    or (not self.require_power_of_2 and min(max(1,(ch[0]-1))*ra[0]*pu[0], ch[0]*max(1,(ra[0]-1))*pu[0], ch[0]*ra[0]*max(1,(pu[0]-1))) < m))
                                and (ch[1] * ra[1] * pu[1] <= k 
                                    or (self.require_power_of_2  and max(1, ch[1] * ra[1] * pu[1] / 2) < k)
                                    or (not self.require_power_of_2 and min(max(1,(ch[1]-1))*ra[1]*pu[1], ch[1]*max(1,(ra[1]-1))*pu[1], ch[1]*ra[1]*max(1,(pu[1]-1))) < k))
                                and (ch[2] * ra[2] * pu[2] <= l
                                    or (self.require_power_of_2  and max(1, ch[2] * ra[2] * pu[2] / 2) < l)
                                    or (not self.require_power_of_2 and min(max(1,(ch[2]-1))*ra[2]*pu[2], ch[2]*max(1,(ra[2]-1))*pu[2], ch[2]*ra[2]*max(1,(pu[2]-1))) < l))
                                and (ch[3] * ra[3] * pu[3] <= b
                                    or (self.require_power_of_2  and max(1, ch[3] * ra[3] * pu[3] / 2) < b)
                                    or (not self.require_power_of_2 and min(max(1,(ch[3]-1))*ra[3]*pu[3], ch[3]*max(1,(ra[3]-1))*pu[3], ch[3]*ra[3]*max(1,(pu[3]-1))) < b))
                            )]
            possible_div.extend(
                [(level, rank_pu_num, divide) for divide in divide]
            )
            print(f"channel={self.channel_num}, rank={self.rank_num}, rank_pu={rank_pu_num}, divide_num={len(divide)}")

        return possible_div

    def choose_from_partition_space_mm(self, possible_divide):
        # 这块代码的目的是把硬件尽可能用满的项提取出来
        filtered_divide = []
        if SimConfig.pu_level == LEVEL.DE:
            for divide in possible_divide:
                level, pu_num, div = divide
                ch, ra, de, pu = div
                if(mul(ch) == self.channel_num and mul(ra) == self.rank_num and mul(de) == self.device_num and mul(pu) == pu_num):
                    filtered_divide.append(divide)
                elif(
                    ( mul([ch[0]+1, ch[1], ch[2], ch[3]])>self.channel_num and mul([ch[0], ch[1]+1, ch[2], ch[3]])>self.channel_num and mul([ch[0], ch[1], ch[2]+1, ch[3]])>self.channel_num and mul([ch[0], ch[1], ch[2], ch[3]+1])>self.channel_num ) and
                    ( mul([ra[0]+1, ra[1], ra[2], ra[3]])>self.rank_num and mul([ra[0], ra[1]+1, ra[2], ra[3]])>self.rank_num and mul([ra[0], ra[1], ra[2]+1, ra[3]])>self.rank_num and mul([ra[0], ra[1], ra[2], ra[3]+1])>self.rank_num ) and
                    ( mul([de[0]+1, de[1], de[2], de[3]])>self.device_num and mul([de[0], de[1]+1, de[2], de[3]])>self.device_num and mul([de[0], de[1], de[2]+1, de[3]])>self.device_num and mul([de[0], de[1], de[2], de[3]+1])>self.device_num ) and
                    ( mul([pu[0]+1, pu[1], pu[2], pu[3]])>pu_num and mul([pu[0], pu[1]+1, pu[2], pu[3]])>pu_num and mul([pu[0], pu[1], pu[2]+1, pu[3]])>pu_num and mul([pu[0], pu[1], pu[2], pu[3]+1])>pu_num )
                ):
                    filtered_divide.append(divide)
        elif SimConfig.pu_level == LEVEL.RA:
            for divide in possible_divide:
                level, pu_num, div = divide
                ch, ra, pu = div
                if(mul(ch) == self.channel_num and mul(ra) == self.rank_num and mul(pu) == pu_num):
                    filtered_divide.append(divide)
                elif(
                    ( mul([ch[0]+1, ch[1], ch[2], ch[3]])>self.channel_num and mul([ch[0], ch[1]+1, ch[2], ch[3]])>self.channel_num and mul([ch[0], ch[1], ch[2]+1, ch[3]])>self.channel_num and mul([ch[0], ch[1], ch[2], ch[3]+1])>self.channel_num ) and
                    ( mul([ra[0]+1, ra[1], ra[2], ra[3]])>self.rank_num and mul([ra[0], ra[1]+1, ra[2], ra[3]])>self.rank_num and mul([ra[0], ra[1], ra[2]+1, ra[3]])>self.rank_num and mul([ra[0], ra[1], ra[2], ra[3]+1])>self.rank_num ) and
                    ( mul([pu[0]+1, pu[1], pu[2], pu[3]])>pu_num and mul([pu[0], pu[1]+1, pu[2], pu[3]])>pu_num and mul([pu[0], pu[1], pu[2]+1, pu[3]])>pu_num and mul([pu[0], pu[1], pu[2], pu[3]+1])>pu_num )
                ):
                    filtered_divide.append(divide)
        else:
            raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
        return filtered_divide
        
    def mem_partition_mm(self, mm_size, compute_divide):
        # 计算分块后各个PU负责处理的矩阵大小，需要处理一些corner case、、、
        m, k, l, b = mm_size
        if SimConfig.pu_level == LEVEL.DE:
            ch, ra, de, pu = compute_divide
            ch_m, ch_k, ch_l, ch_b = ch
            ra_m, ra_k, ra_l, ra_b = ra
            de_m, de_k, de_l, de_b = de
            pu_m, pu_k, pu_l, pu_b = pu
            div_for_m = mul([ch_m, ra_m, de_m, pu_m])
            div_for_k = mul([ch_k, ra_k, de_k, pu_k])
            div_for_l = mul([ch_l, ra_l, de_l, pu_l])
            div_for_b = mul([ch_b, ra_b, de_b, pu_b])
            ch_num = mul(ch); ra_num = mul(ra); de_num = mul(de); pu_num = mul(pu)
            parrallel_num = div_for_m * div_for_k * div_for_l * div_for_b
            # m,k,l 划分后的大小
            m_after_div = math.ceil( m / div_for_m )
            m_corner_case = m_after_div * div_for_m - m
            m_normal_case = div_for_m - m_corner_case
            k_after_div = math.ceil( k / div_for_k )
            k_corner_case = k_after_div * div_for_k - k
            k_normal_case = div_for_k - k_corner_case
            l_after_div = math.ceil( l / div_for_l )
            l_corner_case = l_after_div * div_for_l - l
            l_normal_case = div_for_l - l_corner_case
            b_after_div = math.ceil( b / div_for_b )
            b_corner_case = b_after_div * div_for_b - b
            b_normal_case = div_for_b - b_corner_case
            # 已知分块，生成各实例具体的分块大小
            # indexes = [
            #     self.get_div_index(id, [ch_num, ra_num, de_num, pu_num])
            #     for id in range(parrallel_num)
            # ]
            # mkl = []
            # for index in indexes:
            #     ch_mkl = self.get_div_index(index[0], [ch_m, ch_k, ch_l])
            #     ra_mkl = self.get_div_index(index[1], [ra_m, ra_k, ra_l])
            #     de_mkl = self.get_div_index(index[2], [de_m, de_k, de_l])
            #     pu_mkl = self.get_div_index(index[3], [pu_m, pu_k, pu_l])
            #     # M index
            #     m_index = self.get_div_id([ch_mkl[0], ra_mkl[0], de_mkl[0], pu_mkl[0]], [ch_m, ra_m, de_m, pu_m])
            #     # K index
            #     k_index = self.get_div_id([ch_mkl[1], ra_mkl[1], de_mkl[1], pu_mkl[1]], [ch_k, ra_k, de_k, pu_k])
            #     # L index
            #     l_index = self.get_div_id([ch_mkl[2], ra_mkl[2], de_mkl[2], pu_mkl[2]], [ch_l, ra_l, de_l, pu_l])
            #     # 分块大小
            #     mmkkll = [m_after_div if m_index < m_normal_case else m_after_div - 1, 
            #             k_after_div if k_index < k_normal_case else k_after_div - 1, 
            #             l_after_div if l_index < l_normal_case else l_after_div - 1]
            #     mkl.append(mmkkll)
                # print(f"index={index}, mmkkll={mmkkll}")
        elif SimConfig.pu_level == LEVEL.RA:
            ch, ra, pu = compute_divide
            ch_m, ch_k, ch_l, ch_b = ch
            ra_m, ra_k, ra_l, ra_b = ra
            pu_m, pu_k, pu_l, pu_b = pu
            div_for_m = mul([ch_m, ra_m, pu_m])
            div_for_k = mul([ch_k, ra_k, pu_k])
            div_for_l = mul([ch_l, ra_l, pu_l])
            div_for_b = mul([ch_b, ra_b, pu_b])
            # m,k,l 划分后的大小
            m_after_div = math.ceil( m / div_for_m )
            m_corner_case = m_after_div * div_for_m - m
            k_after_div = math.ceil( k / div_for_k )
            k_corner_case = k_after_div * div_for_k - k
            l_after_div = math.ceil( l / div_for_l )
            l_corner_case = l_after_div * div_for_l - l
            b_after_div = math.ceil( b / div_for_b )
            b_corner_case = b_after_div * div_for_b - b
        else:
            raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
        # 计算分块后的存储空间
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = self.mem_mapping_mm((m_after_div, k_after_div, l_after_div, b_after_div))
        return simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
    
    # 已知各PU负责的分块大小，生成各实例具体的分块大小
    # 当然，按照较大值处理我觉得也完全没有问题，就是可能会有一些冗余计算
    # 这边还是暂时忽略padding后的索引管理问题
    def mem_mapping_mm(self, mm_size):
        """
        input: mm * kk * bb
        weight: kk * ll * bb
        两者的k维度在分配时需要对应
        """
        mm, kk, ll, bb = mm_size
        # Step 1: 列空间分配
        simd_k = min(kk, self.simd) # K在列中的分块大小
        k_col = math.ceil(kk / simd_k) # k占用的列数，最后一列直接做padding，只不过可能吃不满计算并行度
        # Step 2: 行空间分配
        # 一个Row中存放了()的矩阵分块
        unified_col_num = self.col_num if SimConfig.pu_level == LEVEL.DE else self.bank_num * self.col_num # 兼容不同的PU
        max_k_in_a_row = min(k_col, unified_col_num)
        max_m_in_a_row = min(mm, unified_col_num)
        max_l_in_a_row = min(ll, unified_col_num)
        max_b_in_a_row = min(bb, unified_col_num)

        mkl_Input_to_row = [ # 放置时，k维度变化在内层；如果要求2倍数切分、、、
            (
                (
                    min(unified_col_num//k_row, max_m_in_a_row), # m_block in a row
                    k_row, # k_block in a row, 其实应当允许切分，只不过可能大小比较麻烦
                    min(unified_col_num//k_row, max_l_in_a_row), # l_block in a row
                    b_row # b_block in a row
                ),
                (
                    math.ceil(mm/min(unified_col_num//(k_row*b_row), max_m_in_a_row)), # row num for m dimension
                    math.ceil(k_col/k_row), # row num for k dimension
                    math.ceil(ll/min(unified_col_num//(k_row*b_row), max_l_in_a_row)), # row num for l dimension
                    math.ceil(bb/b_row) # row num for b dimension
                ),
                (
                    mm - (math.ceil(mm/min(unified_col_num//(k_row*b_row), max_m_in_a_row))-1) * min(unified_col_num//k_row, max_m_in_a_row), # corner case
                    k_col - (math.ceil(k_col/k_row)-1) * k_row, # corner case
                    ll - (math.ceil(ll/min(unified_col_num//(k_row*b_row), max_l_in_a_row))-1) * min(unified_col_num//k_row, max_l_in_a_row), # corner case
                    bb - (math.ceil(bb/b_row)-1) * b_row # corner case
                )
             ) for k_row in range(1, max_k_in_a_row+1) if (self.powerof2(k_row)) 
             for b_row in range(1, min(max_b_in_a_row, unified_col_num//k_row)+1) if (self.powerof2(b_row))
        ]
        # 筛选k的取值，从而尽量把空间占满
        mkl_Input_to_row = list(filter(lambda x: x[0][1] == max_k_in_a_row or x[0][0]*x[0][3]*(x[0][1]+1) > unified_col_num or x[0][2]*x[0][3]*(x[0][1]+1) > unified_col_num, mkl_Input_to_row))        
        """
        output: mm * ll * bb, col只分配l维度较为合理，因为l会是下一次的reduce维度
        """
        simd_l = min(ll, self.simd)
        l_col = math.ceil(ll / simd_l)
        ml_Out_to_row = [
            (
                (
                    min(unified_col_num//l_row, max_m_in_a_row), # m_block in a row
                    l_row, # l_block in a row
                    b_row # bb_block in a row
                ),
                (
                    math.ceil(mm/min(unified_col_num//l_row, max_m_in_a_row)), # row num for m dimension
                    math.ceil(l_col/l_row), # row num for l dimension,
                    math.ceil(bb/b_row) # row num for bb dimension
                ),
                (
                    mm - (math.ceil(mm/min(unified_col_num//l_row, max_m_in_a_row))-1) * min(unified_col_num//l_row, max_m_in_a_row), # corner case
                    l_col - (math.ceil(l_col/l_row)-1) * l_row, # corner case
                    bb - (math.ceil(bb/b_row)-1) * b_row # corner case
                ),
             ) for l_row in range(1, min(l_col, unified_col_num)+1) if ( self.powerof2(l_row) )
             for b_row in range(1, min(max_b_in_a_row, unified_col_num//l_row)+1) if (self.powerof2(b_row))
        ]
        # 筛选l的取值，从而尽量把输出空间占满
        ml_Out_to_row = list(filter(lambda x: x[0][1] == min(l_col, unified_col_num) or x[0][0]*x[0][2]*(x[0][1]+1) > unified_col_num, ml_Out_to_row))
        return simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
    
    def choose_from_mem_space(self, input_space, out_space):
        # 选择k最大的方式
        current_row = 0
        return_input = None
        return_output = None
        for input_mapping in input_space:
            _row = input_mapping[1]
            if _row > current_row:
                current_row = _row
                return_input = input_mapping
        current_row = 0
        for out_mapping in out_space:
            _row = out_mapping[1]
            if _row > current_row:
                current_row = _row
                return_output = out_mapping
        return return_input, return_output