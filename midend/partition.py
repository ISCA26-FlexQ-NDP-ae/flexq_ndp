import random
from tools import *
from functools import reduce
import math

# Tools Functions
# mul = lambda x: reduce(lambda x,y:x*y,x)

class Partition(HW_info):

    def __init__(self, require_power_of_2, log_file=None):
        # HW information
        super(Partition, self).__init__(require_power_of_2)
        self.require_power_of_2 = require_power_of_2
        # channel number
        self.ch_num = SimConfig.ch
        # bank number
        self.bank_num = SimConfig.ba * SimConfig.bg
        # row number
        self.row_num = SimConfig.ro
        # col number
        self.col_num = SimConfig.co
        # col width in bit
        self.col_width = SimConfig.co_w # 会遇到 16/32 * 4 bit的情况，怎么办
        self.log_file = log_file
        if self.log_file is not None:
            # self.log_print = lambda x: print(x, file=self.log_file)
            self.log_print = lambda x: None
        else:
            self.log_print = lambda x: print(x)

    def get_partition_space_mm(self, mm_size=[1000,1000,1000,1], bitwidth=16):
        """
        compute-level, pu-mode
        mm partition point= public: m(CH, RA, DE, PU),  k(CH, RA, DE, PU),  n(CH, RA, DE, PU)
                            input:  m(RO, CO, SIMD=0),  k(RO, CO, SIMD  ),  
                            weight:                     k(RO, CO, SIMD  ),  n(RO, CO, SIMD=0)
                            output: m(RO, CO, SIMD  ),                      n(RO, CO, SIMD  )                                          
        """
        m, k, n, b = mm_size # b = batch dimension, for Activation-Activation computation
        # device-level
        level = SimConfig.pu_level
        possible_div = []
        if level == LEVEL.DE:
            for device_pu_num in self.de_pu_num_list:
                # 获取可能的分割方式
                ch_divide = [(ch_m, ch_k, ch_n, ch_b) for ch_m in range(1, self.channel_num+1) 
                            for ch_k in range(1, self.channel_num+1) 
                            for ch_n in range(1, self.channel_num+1)
                            for ch_b in range(1, self.channel_num+1)
                            if ch_m * ch_k * ch_n * ch_b <= self.channel_num 
                            and self.powerof2(ch_m) and self.powerof2(ch_k) and self.powerof2(ch_n) and self.powerof2(ch_b)]
                ra_divide = [(ra_m, ra_k, ra_n, ra_b) for ra_m in range(1, self.rank_num+1) 
                            for ra_k in range(1, self.rank_num+1) 
                            for ra_n in range(1, self.rank_num+1) 
                            for ra_b in range(1, self.rank_num+1)
                            if ra_m * ra_k * ra_n * ra_b <= self.rank_num 
                            and self.powerof2(ra_m) and self.powerof2(ra_k) and self.powerof2(ra_n) and self.powerof2(ra_b)]
                de_divide = [(de_m, de_k, de_n, de_b) 
                            for de_m in range(1, self.device_num+1) 
                            for de_k in range(1, self.device_num+1) 
                            for de_n in range(1, self.device_num+1) 
                            for de_b in range(1, self.device_num+1)
                            if de_m * de_k * de_n * de_b <= self.device_num 
                            and self.powerof2(de_m) and self.powerof2(de_k) and self.powerof2(de_n) and self.powerof2(de_b)]
                pu_divide = [(pu_m, pu_k, pu_n, pu_b) for pu_m in range(1, device_pu_num+1) 
                            for pu_k in range(1, device_pu_num+1) 
                            for pu_n in range(1, device_pu_num+1) 
                            for pu_b in range(1, device_pu_num+1)
                            if pu_m * pu_k * pu_n * pu_b <= device_pu_num 
                            and self.powerof2(pu_m) and self.powerof2(pu_k) and self.powerof2(pu_n) and self.powerof2(pu_b)]
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
                                    and (ch[2] * ra[2] * de[2] * pu[2] <= n
                                        or (self.require_power_of_2  and max(1, ch[2] * ra[2] * de[2] * pu[2] / 2) < n)
                                        or (not self.require_power_of_2 and min(max(1,(ch[2]-1))*ra[2]*de[2]*pu[2], ch[2]*max(1,(ra[2]-1))*de[2]*pu[2], ch[2]*ra[2]*max(1,(de[2]-1))*pu[2], ch[2]*ra[2]*de[2]*max(1,(pu[2]-1))) < n))   
                                    and (ch[3] * ra[3] * de[3] * pu[3] <= b
                                        or (self.require_power_of_2  and max(1, ch[3] * ra[3] * de[3] * pu[3] / 2) < b)
                                        or (not self.require_power_of_2 and min(max(1,(ch[3]-1))*ra[3]*de[3]*pu[3], ch[3]*max(1,(ra[3]-1))*de[3]*pu[3], ch[3]*ra[3]*max(1,(de[3]-1))*pu[3], ch[3]*ra[3]*de[3]*max(1,(pu[3]-1))) < b))
                                )]
                possible_div.extend(
                    [(level, device_pu_num, divide) for divide in divide]
                )
                self.log_print(f"channel={self.channel_num}, rank={self.rank_num}, device={self.device_num}, device_pu={device_pu_num}, divide_num={len(divide)}")
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
            ch_divide = [(ch_m, ch_k, ch_n, ch_b) for ch_m in range(1, self.channel_num+1) 
                        for ch_k in range(1, self.channel_num+1) 
                        for ch_n in range(1, self.channel_num+1)
                        for ch_b in range(1, self.channel_num+1)
                        if ch_m * ch_k * ch_n * ch_b <= self.channel_num 
                        and self.powerof2(ch_m) and self.powerof2(ch_k) and self.powerof2(ch_n) and self.powerof2(ch_b)]
            ra_divide = [(ra_m, ra_k, ra_n, ra_b) for ra_m in range(1, self.rank_num+1) 
                        for ra_k in range(1, self.rank_num+1) 
                        for ra_n in range(1, self.rank_num+1) 
                        for ra_b in range(1, self.rank_num+1)
                        if ra_m * ra_k * ra_n * ra_b <= self.rank_num 
                        and self.powerof2(ra_m) and self.powerof2(ra_k) and self.powerof2(ra_n) and self.powerof2(ra_b)]
            pu_divide = [(pu_m, pu_k, pu_n, pu_b) for pu_m in range(1, rank_pu_num+1) 
                        for pu_k in range(1, rank_pu_num+1) 
                        for pu_n in range(1, rank_pu_num+1) 
                        for pu_b in range(1, rank_pu_num+1)
                        if pu_m * pu_k * pu_n * pu_b <= rank_pu_num 
                        and self.powerof2(pu_m) and self.powerof2(pu_k) and self.powerof2(pu_n) and self.powerof2(pu_b)]
            divide = [(ch, ra, pu) for ch in ch_divide 
                            for ra in ra_divide for pu in pu_divide
                            if ((ch[0] * ra[0] * pu[0] <= m 
                                    or (self.require_power_of_2  and max(1, ch[0] * ra[0] * pu[0] / 2) < m) 
                                    or (not self.require_power_of_2 and min(max(1,(ch[0]-1))*ra[0]*pu[0], ch[0]*max(1,(ra[0]-1))*pu[0], ch[0]*ra[0]*max(1,(pu[0]-1))) < m))
                                and (ch[1] * ra[1] * pu[1] <= k 
                                    or (self.require_power_of_2  and max(1, ch[1] * ra[1] * pu[1] / 2) < k)
                                    or (not self.require_power_of_2 and min(max(1,(ch[1]-1))*ra[1]*pu[1], ch[1]*max(1,(ra[1]-1))*pu[1], ch[1]*ra[1]*max(1,(pu[1]-1))) < k))
                                and (ch[2] * ra[2] * pu[2] <= n
                                    or (self.require_power_of_2  and max(1, ch[2] * ra[2] * pu[2] / 2) < n)
                                    or (not self.require_power_of_2 and min(max(1,(ch[2]-1))*ra[2]*pu[2], ch[2]*max(1,(ra[2]-1))*pu[2], ch[2]*ra[2]*max(1,(pu[2]-1))) < n))
                                and (ch[3] * ra[3] * pu[3] <= b
                                    or (self.require_power_of_2  and max(1, ch[3] * ra[3] * pu[3] / 2) < b)
                                    or (not self.require_power_of_2 and min(max(1,(ch[3]-1))*ra[3]*pu[3], ch[3]*max(1,(ra[3]-1))*pu[3], ch[3]*ra[3]*max(1,(pu[3]-1))) < b))
                            )]
            possible_div.extend(
                [(level, rank_pu_num, divide) for divide in divide]
            )
            self.log_print(f"channel={self.channel_num}, rank={self.rank_num}, rank_pu={rank_pu_num}, divide_num={len(divide)}")

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
        filter_filter_divide = []
        for divide in filtered_divide:
            level, pu_num, div = divide
            ch, ra, de, pu = div
            if pu[0] * pu[1] * pu[3] == 1:
                filter_filter_divide.append(divide)
        # filter_filter_divide = filtered_divide
        self.log_print(f"divide_num={len(filtered_divide)}->{len(filter_filter_divide)}")
        return filter_filter_divide
        
    def mem_partition_mm(self, mm_size, compute_divide, scale_group=(1,128,128)):
        # 计算分块后各个PU负责处理的矩阵大小，需要处理一些corner case、、、
        m, k, n, b = mm_size
        # print(f"\nmm_size: {mm_size}")
        s_group_m, s_group_k, s_group_n = scale_group # batch 维度默认 scale 不同
        if SimConfig.pu_level == LEVEL.DE:
            ch, ra, de, pu = compute_divide
            ch_m, ch_k, ch_n, ch_b = ch
            ra_m, ra_k, ra_n, ra_b = ra
            de_m, de_k, de_n, de_b = de
            pu_m, pu_k, pu_n, pu_b = pu
            div_for_m = mul([ch_m, ra_m, de_m, pu_m])
            div_for_k = mul([ch_k, ra_k, de_k, pu_k])
            div_for_n = mul([ch_n, ra_n, de_n, pu_n])
            div_for_b = mul([ch_b, ra_b, de_b, pu_b])
            ch_num = mul(ch); ra_num = mul(ra); de_num = mul(de); pu_num = mul(pu)
            parrallel_num = div_for_m * div_for_k * div_for_n * div_for_b
            # m,k,n 划分后的大小
            m_after_div = math.ceil( m / div_for_m )
            m_corner_case = m_after_div * div_for_m - m
            m_normal_case = div_for_m - m_corner_case
            k_after_div = math.ceil( k / div_for_k )
            k_corner_case = k_after_div * div_for_k - k
            k_normal_case = div_for_k - k_corner_case
            n_after_div = math.ceil( n / div_for_n )
            n_corner_case = n_after_div * div_for_n - n
            n_normal_case = div_for_n - n_corner_case
            b_after_div = math.ceil( b / div_for_b )
            b_corner_case = b_after_div * div_for_b - b
            b_normal_case = div_for_b - b_corner_case
        elif SimConfig.pu_level == LEVEL.RA:
            ch, ra, pu = compute_divide
            ch_m, ch_k, ch_n, ch_b = ch
            ra_m, ra_k, ra_n, ra_b = ra
            pu_m, pu_k, pu_n, pu_b = pu
            div_for_m = mul([ch_m, ra_m, pu_m])
            div_for_k = mul([ch_k, ra_k, pu_k])
            div_for_n = mul([ch_n, ra_n, pu_n])
            div_for_b = mul([ch_b, ra_b, pu_b])
            # m,k,n 划分后的大小
            m_after_div = math.ceil( m / div_for_m )
            m_corner_case = m_after_div * div_for_m - m
            k_after_div = math.ceil( k / div_for_k )
            k_corner_case = k_after_div * div_for_k - k
            n_after_div = math.ceil( n / div_for_n )
            n_corner_case = n_after_div * div_for_n - n
            b_after_div = math.ceil( b / div_for_b )
            b_corner_case = b_after_div * div_for_b - b
        else:
            raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
        # 计算分块后的存储空间
        # print(f"div_for_m: {div_for_m}, div_for_k: {div_for_k}, div_for_n: {div_for_n}, div_for_b: {div_for_b}")
        # print(f"m_after_div: {m_after_div}, k_after_div: {k_after_div}, n_after_div: {n_after_div}, b_after_div: {b_after_div}")

        # NOTE:针对量化scale进行修正, 0代表该维度无需修正

        # 若一个PU负责多组，那么必须保证其包含整数个组数
        if s_group_m == 0:
            m_after_scale_group = m_after_div
            m_actual_parrallel_num = math.ceil(m/m_after_scale_group)
        else:
            m_group_num = math.ceil(m / s_group_m)
            m_group_part1 = m_group_num // div_for_m
            m_part1 = m_group_part1 * s_group_m
            m_group_part2 = m_group_num % div_for_m
            m_part2 = 0
            if m_group_part2 > 0:
                m_pu_per_group = div_for_m // m_group_part2
                m_part2 = math.ceil(s_group_m / m_pu_per_group)
            m_after_scale_group = m_part1 + m_part2
            m_actual_parrallel_num = math.ceil(m/m_after_scale_group)
        
        # print(f"k_after_div: {k_after_div}, s_group_k: {s_group_k}")
        if s_group_k == 0:
            k_after_scale_group = k_after_div
        else:
            k_group_num = math.ceil(k / s_group_k)
            k_group_part1 = k_group_num // div_for_k
            k_part1 = k_group_part1 * s_group_k
            k_group_part2 = k_group_num % div_for_k
            k_part2 = 0
            if k_group_part2 > 0:
                k_pu_per_group = div_for_k // k_group_part2 
                k_part2 = math.ceil(s_group_k / k_pu_per_group)
            k_after_scale_group = k_part1 + k_part2
        k_actual_parrallel_num = math.ceil(k/k_after_scale_group)
        
        if s_group_n == 0:
            n_after_scale_group = n_after_div
            n_actual_parrallel_num = math.ceil(n/n_after_scale_group)
        else:
            n_group_num = math.ceil(n / s_group_n)
            n_group_part1 = n_group_num // div_for_n
            n_part1 = n_group_part1 * s_group_n
            n_group_part2 = n_group_num % div_for_n
            n_part2 = 0
            if n_group_part2 > 0:
                n_pu_per_group = div_for_n // n_group_part2
                n_part2 = math.ceil(s_group_n / n_pu_per_group)
            n_after_scale_group = n_part1 + n_part2
            n_actual_parrallel_num = math.ceil(n/n_after_scale_group)
        
        b_after_scale_group = b_after_div
        b_actual_parrallel_num = math.ceil(b/b_after_scale_group)

        # print(f"m_after_scale_group: {m_after_scale_group}, k_after_scale_group: {k_after_scale_group}, n_after_scale_group: {n_after_scale_group}, b_after_scale_group: {b_after_scale_group}")
        # print(f"m_actual_parrallel_num: {m_actual_parrallel_num}, k_actual_parrallel_num: {k_actual_parrallel_num}, n_actual_parrallel_num: {n_actual_parrallel_num}, b_actual_parrallel_num: {b_actual_parrallel_num}")

        # 重新计算现在需要多少并行度
        # Actual_parrallel_num = m_actual_parrallel_num * k_actual_parrallel_num * n_actual_parrallel_num * b_actual_parrallel_num
        # ultilization = Actual_parrallel_num / parrallel_num
        # print(f"actual pu ultilization: {Actual_parrallel_num} / {parrallel_num} = {ultilization}")
        
        return m_after_scale_group, k_after_scale_group, n_after_scale_group, b_after_scale_group
        # simd_k, mkn_Input_to_row, simd_n, mn_Out_to_row = self.mem_mapping_mm((m_after_div, k_after_div, n_after_div, b_after_div))
        # return simd_k, mkn_Input_to_row, simd_n, mn_Out_to_row
    
    """
    特异点：MM 的输出一定是无 Scale 数据，量化核输出一定是 Scale 数据
    中间重新分配流程
    - 起点：上一个算子的输出数据 Layout（一定无 Scale）
    |   Re-Layout
    |   Quant (different group size / precision ) ( rotation / scale / offset )
    |   Activation( RoPE / Attention ) / Normalization( LayerNorm / RMSNorm )
    - 终点：当前算子的输入数据 Layout（考虑可能的 Scale）

    带 Scale 的编码要怎么做？
    """
    # 先考虑在不同行做存储
    def mem_mapping_matrix_baseline(self, matrix_size, scale_group, matrix_precision, scale_offset_precision, with_scale = True, with_offset=False, buffer_guidance=None):
        """
        matrix_size: (a, b), b是需要紧密排列的维度
        scale_group: (a_group, b_group), a_group 和 b_group 是对应 scale 的组数
        """
        a_size, b_size = matrix_size    
        assert with_offset == False, "Offset is not supported currently"
        # map data, b 紧密排列
        b_per_col = min(int(self.col_width / matrix_precision), b_size)
        b_col_num = math.ceil(b_size / b_per_col)
        
        # 一列中 a索引必须相同，即全部由不同的 b 占据
        
        if buffer_guidance is None:
            # 如果没有 buffer 的限制，那么尽量让 a 紧密排列在一行
            b_col_per_row = min(self.col_num, b_col_num)
            a_col_per_row = min(a_size, max(1, self.col_num // b_col_per_row))
            # row_utilization = a_col_per_row * b_col_per_row / (self.col_num)
            # print(f"a_col_per_row: {a_col_per_row}, b_col_per_row: {b_col_per_row}, row_utilization: {row_utilization}")
            a_row_num = math.ceil(a_size / a_col_per_row)
            b_row_num = math.ceil(b_col_num / b_col_per_row)
            a_corner_col_per_row = a_size - (a_row_num-1) * a_col_per_row
            b_corner_col_per_row = b_col_num - (b_row_num-1) * b_col_per_row
            # print(f"a_row_num: {a_row_num}, b_row_num: {b_row_num}, a_corner_col_per_row: {a_corner_col_per_row}, b_corner_col_per_row: {b_corner_col_per_row}")
            theoretical_utilization = (a_size * b_size * matrix_precision) / (a_row_num * b_row_num * self.col_num * self.col_width)
            # NOTE: analyse corner case            
        else:
            assert 0, "buffer_guidance is not supported currently"

        if with_scale or with_offset:
            a_group, b_group = scale_group
            scale_num_a = math.ceil(a_size / a_group)
            scale_num_b = math.ceil(b_size / b_group)
            # print(f"scale_num_a: {scale_num_a}, scale_num_b: {scale_num_b}")
            # map scale & offset
            scale_b_per_col = min(int(self.col_width / scale_offset_precision), scale_num_b)
            scale_a_per_col = min(max(1, int(self.col_width / scale_offset_precision / scale_num_b)), scale_num_a)
            # print(f"scale_a_per_col: {scale_a_per_col}, scale_b_per_col: {scale_b_per_col}")
            scale_b_col_num = math.ceil(scale_num_b / scale_b_per_col)
            scale_a_col_num = math.ceil(scale_num_a / scale_a_per_col)
            scale_b_col_per_row = min(self.col_num, scale_b_col_num)
            scale_a_col_per_row = min(scale_a_col_num, max(1, self.col_num // scale_b_col_per_row))
            scale_row_utilization = scale_a_col_per_row * scale_b_col_per_row / (self.col_num)
            # print(f"scale_a_col_per_row: {scale_a_col_per_row}, scale_b_col_per_row: {scale_b_col_per_row}, scale_row_utilization: {scale_row_utilization}")
            scale_a_row_num = math.ceil(scale_a_col_num / scale_a_col_per_row)
            scale_b_row_num = math.ceil(scale_b_col_num / scale_b_col_per_row)
            scale_a_corner_col_per_row = scale_a_col_num - (scale_a_row_num-1) * scale_a_col_per_row
            scale_b_corner_col_per_row = scale_b_col_num - (scale_b_row_num-1) * scale_b_col_per_row
            # print(f"scale_a_row_num: {scale_a_row_num}, scale_b_row_num: {scale_b_row_num}, scale_a_corner_col_per_row: {scale_a_corner_col_per_row}, scale_b_corner_col_per_row: {scale_b_corner_col_per_row}")  
            
            return {
                "data_mapping": {
                    "b_per_col": b_per_col,
                    "a_col_per_row": a_col_per_row,
                    "b_col_per_row": b_col_per_row,
                    "a_row_num": a_row_num,
                    "b_row_num": b_row_num,
                    "a_corner_col_per_row": a_corner_col_per_row,
                    "b_corner_col_per_row": b_corner_col_per_row,
                },
                "scale_mapping": {
                    "scale_b_per_col": scale_b_per_col,
                    "scale_a_per_col": scale_a_per_col,
                    "scale_b_col_per_row": scale_b_col_per_row,
                    "scale_a_col_per_row": scale_a_col_per_row,
                    "scale_a_row_num": scale_a_row_num,
                    "scale_b_row_num": scale_b_row_num,
                    "scale_a_corner_col_per_row": scale_a_corner_col_per_row,
                    "scale_b_corner_col_per_row": scale_b_corner_col_per_row,
                }
            }
        else: # 无 Scale 的编码
            return {
                "data_mapping": {
                    "b_per_col": b_per_col,
                    "a_col_per_row": a_col_per_row,
                    "b_col_per_row": b_col_per_row,
                    "a_row_num": a_row_num,
                    "b_row_num": b_row_num,
                    "a_corner_col_per_row": a_corner_col_per_row,
                    "b_corner_col_per_row": b_corner_col_per_row,
                },
            }

    def mem_mapping_mm(self, mm_size, ):
        """
        input: mm * kk * bb
        weight: kk * nn * bb
        两者的k维度在分配时需要对应
        """
        mm, kk, nn, bb = mm_size
        # Step 1: 列空间分配
        simd_k = min(kk, self.simd) # K在列中的分块大小
        k_col = math.ceil(kk / simd_k) # k占用的列数，最后一列直接做padding，只不过可能吃不满计算并行度
        # Step 2: 行空间分配
        # 一个Row中存放了()的矩阵分块
        unified_col_num = self.col_num if SimConfig.pu_level == LEVEL.DE else self.bank_num * self.col_num # 兼容不同的PU
        max_k_in_a_row = min(k_col, unified_col_num)
        max_m_in_a_row = min(mm, unified_col_num)
        max_n_in_a_row = min(nn, unified_col_num)
        max_b_in_a_row = min(bb, unified_col_num)

        mkn_Input_to_row = [ # 放置时，k维度变化在内层；如果要求2倍数切分、、、
            (
                (
                    min(unified_col_num//k_row, max_m_in_a_row), # m_block in a row
                    k_row, # k_block in a row, 其实应当允许切分，只不过可能大小比较麻烦
                    min(unified_col_num//k_row, max_n_in_a_row), # n_block in a row
                    b_row # b_block in a row
                ),
                (
                    math.ceil(mm/min(unified_col_num//(k_row*b_row), max_m_in_a_row)), # row num for m dimension
                    math.ceil(k_col/k_row), # row num for k dimension
                    math.ceil(nn/min(unified_col_num//(k_row*b_row), max_n_in_a_row)), # row num for n dimension
                    math.ceil(bb/b_row) # row num for b dimension
                ),
                (
                    mm - (math.ceil(mm/min(unified_col_num//(k_row*b_row), max_m_in_a_row))-1) * min(unified_col_num//k_row, max_m_in_a_row), # corner case
                    k_col - (math.ceil(k_col/k_row)-1) * k_row, # corner case
                    nn - (math.ceil(nn/min(unified_col_num//(k_row*b_row), max_n_in_a_row))-1) * min(unified_col_num//k_row, max_n_in_a_row), # corner case
                    bb - (math.ceil(bb/b_row)-1) * b_row # corner case
                )
             ) for k_row in range(1, max_k_in_a_row+1) if (self.powerof2(k_row)) 
             for b_row in range(1, min(max_b_in_a_row, unified_col_num//k_row)+1) if (self.powerof2(b_row))
        ]
        # 筛选k的取值，从而尽量把空间占满
        mkn_Input_to_row = list(filter(lambda x: x[0][1] == max_k_in_a_row or x[0][0]*x[0][3]*(x[0][1]+1) > unified_col_num or x[0][2]*x[0][3]*(x[0][1]+1) > unified_col_num, mkn_Input_to_row))        
        """
        output: mm * nn * bb, col只分配n维度较为合理，因为n会是下一次的reduce维度
        """
        simd_n = min(nn, self.simd)
        n_col = math.ceil(nn / simd_n)
        mn_Out_to_row = [
            (
                (
                    min(unified_col_num//n_row, max_m_in_a_row), # m_block in a row
                    n_row, # n_block in a row
                    b_row # bb_block in a row
                ),
                (
                    math.ceil(mm/min(unified_col_num//n_row, max_m_in_a_row)), # row num for m dimension
                    math.ceil(n_col/n_row), # row num for n dimension,
                    math.ceil(bb/b_row) # row num for bb dimension
                ),
                (
                    mm - (math.ceil(mm/min(unified_col_num//n_row, max_m_in_a_row))-1) * min(unified_col_num//n_row, max_m_in_a_row), # corner case
                    n_col - (math.ceil(n_col/n_row)-1) * n_row, # corner case
                    bb - (math.ceil(bb/b_row)-1) * b_row # corner case
                ),
             ) for n_row in range(1, min(n_col, unified_col_num)+1) if ( self.powerof2(n_row) )
             for b_row in range(1, min(max_b_in_a_row, unified_col_num//n_row)+1) if (self.powerof2(b_row))
        ]
        # 筛选n的取值，从而尽量把输出空间占满
        mn_Out_to_row = list(filter(lambda x: x[0][1] == min(n_col, unified_col_num) or x[0][0]*x[0][2]*(x[0][1]+1) > unified_col_num, mn_Out_to_row))
        return simd_k, mkn_Input_to_row, simd_n, mn_Out_to_row
    
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