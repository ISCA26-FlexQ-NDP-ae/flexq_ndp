from tools import *
import copy
import math

# MARK: 简单 Mapping
class TensorAddr():
    """
    matrix_size: (a, b), b是需要紧密排列的维度
    scale_group: (a_group, b_group), a_group 和 b_group 是对应 scale 的组数
    """
    def __init__(self, operand_info, tensor_size, tensor_mapping, row_offset=0):
        self.operand_info = operand_info
        self.tensor_size = tensor_size
        self.tensor_mapping = tensor_mapping
        self.row_offset = row_offset
        self.scale_row_offset = tensor_mapping['data_mapping']['a_row_num'] * tensor_mapping['data_mapping']['b_row_num'] + row_offset
        self.data_mapping = tensor_mapping['data_mapping']
        if 'scale_mapping' in self.tensor_mapping.keys():
            self.with_scale = True
            self.scale_mapping = tensor_mapping['scale_mapping']
            self.end_row = self.scale_row_offset + tensor_mapping['scale_mapping']['scale_b_row_num'] * tensor_mapping['scale_mapping']['scale_a_row_num']
        else:
            self.with_scale = False
            self.end_row = self.scale_row_offset
    
    def get_end_row(self):
        return self.end_row
        
    def get_meta(self):
        a = self.tensor_size[0] - 1
        b = self.tensor_size[1] - 1
        scale_group_a_id = a // self.operand_info['scale_group'][0]
        scale_group_b_id = b // self.operand_info['scale_group'][1]
        scale_b_col_num_id = scale_group_b_id // self.scale_mapping['scale_b_per_col']
        scale_a_col_num_id = scale_group_a_id // self.scale_mapping['scale_a_per_col']
        scale_b_row_id = scale_b_col_num_id // self.scale_mapping['scale_b_col_per_row']
        scale_a_row_id = scale_a_col_num_id // self.scale_mapping['scale_a_col_per_row']
        scale_b_col_id = scale_b_col_num_id % self.scale_mapping['scale_b_col_per_row']
        scale_a_col_id = scale_a_col_num_id % self.scale_mapping['scale_a_col_per_row']
        if scale_b_row_id == self.scale_mapping['scale_b_row_num'] - 1:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_corner_col_per_row']
        else:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_col_per_row']
        return real_col_id + 1
    
    def get_tensor_addr(self, a, b):
        assert a < self.tensor_size[0] and b < self.tensor_size[1], f"a or b out of range, a: {a}, b: {b}, tensor_size: {self.tensor_size}"
        b_col_num_id = b // self.data_mapping['b_per_col']
        b_row_id = b_col_num_id // self.data_mapping['b_col_per_row']
        b_col_id = b_col_num_id % self.data_mapping['b_col_per_row']
        a_row_id = a // self.data_mapping['a_col_per_row']
        a_col_id = a % self.data_mapping['a_col_per_row']
        real_row_id = b_row_id + a_row_id * self.data_mapping['b_row_num'] + self.row_offset
        if b_row_id == self.data_mapping['b_row_num'] - 1:
            real_col_id = b_col_id + a_col_id * self.data_mapping['b_corner_col_per_row'] 
        else:
            real_col_id = b_col_id + a_col_id * self.data_mapping['b_col_per_row']
        return real_row_id, real_col_id
    
    def get_scale_addr(self, a, b):
        assert self.with_scale, "No scale mapping"
        scale_group_a_id = a // self.operand_info['scale_group'][0]
        scale_group_b_id = b // self.operand_info['scale_group'][1]
        scale_b_col_num_id = scale_group_b_id // self.scale_mapping['scale_b_per_col']
        scale_a_col_num_id = scale_group_a_id // self.scale_mapping['scale_a_per_col']
        scale_b_row_id = scale_b_col_num_id // self.scale_mapping['scale_b_col_per_row']
        scale_a_row_id = scale_a_col_num_id // self.scale_mapping['scale_a_col_per_row']
        scale_b_col_id = scale_b_col_num_id % self.scale_mapping['scale_b_col_per_row']
        scale_a_col_id = scale_a_col_num_id % self.scale_mapping['scale_a_col_per_row']
        if scale_b_row_id == self.scale_mapping['scale_b_row_num'] - 1:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_corner_col_per_row']
        else:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_col_per_row']
        real_row_id = scale_b_row_id + scale_a_row_id * self.scale_mapping['scale_b_row_num'] + self.scale_row_offset
        return scale_group_a_id, scale_group_b_id, real_row_id, real_col_id

    def get_scale_addr_by_b_group_id(self, a, scale_group_b_id):
        assert self.with_scale, "No scale mapping"
        scale_group_a_id = a // self.operand_info['scale_group'][0]
        scale_b_col_num_id = scale_group_b_id // self.scale_mapping['scale_b_per_col']
        scale_a_col_num_id = scale_group_a_id // self.scale_mapping['scale_a_per_col']
        scale_b_row_id = scale_b_col_num_id // self.scale_mapping['scale_b_col_per_row']
        scale_a_row_id = scale_a_col_num_id // self.scale_mapping['scale_a_col_per_row']
        scale_b_col_id = scale_b_col_num_id % self.scale_mapping['scale_b_col_per_row']
        scale_a_col_id = scale_a_col_num_id % self.scale_mapping['scale_a_col_per_row']
        if scale_b_row_id == self.scale_mapping['scale_b_row_num'] - 1:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_corner_col_per_row']
        else:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_col_per_row']
        real_row_id = scale_b_row_id + scale_a_row_id * self.scale_mapping['scale_b_row_num'] + self.scale_row_offset
        return real_row_id, real_col_id

    def copy(self):
        return TensorAddr(self.operand_info, self.tensor_size, self.tensor_mapping, self.row_offset)

# MARK: 基于循环顺序的 Mapping
class LoopfriendlyAddr():
    def __init__(self, operand_info, loop_info, tensor_mapping, is_weight = True, row_offset=0, set_k_block=None):
        self.operand_info = operand_info
        self.loop_info = loop_info
        self.tensor_mapping = tensor_mapping
        self.is_weight = is_weight
        self.row_offset = row_offset
        # self.scale_mix = scale_mix
        self.with_scale = operand_info['with_scale']
        self.set_k_block = set_k_block
        if is_weight:
            self.op_size = (loop_info.n_size, loop_info.k_size)
            # K0 - N - K1
            # assert loop_info.loop_order == ['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2']
            self.k_block = loop_info.block_sizes['k']['block1']
        else: # 非weight在reorder之后没有变化
            # 但是有可能存在 buffer b，从而导致激活成为需要改变 mapping 方式的情况
            # raise NotImplementedError("Not support non-weight operand with scale")
            self.op_size = (loop_info.m_size, loop_info.k_size)
            # K0 - M - K1
            assert loop_info.loop_order == ['n0', 'n1', 'n2', 'k0', 'k1', 'm0', 'm1', 'm2', 'k2']
            self.k_block = loop_info.block_sizes['k']['block1']
        self.data_in_col = SimConfig.co_w // operand_info['matrix_precision']
        self.scale_row_offset = self.get_scale_row_offset()
        if self.with_scale:
            self.end_row = self.scale_row_offset + tensor_mapping['scale_mapping']['scale_b_row_num'] * tensor_mapping['scale_mapping']['scale_a_row_num']
            # calculate the group info
            self.scale_mapping = tensor_mapping['scale_mapping']
            self.group_size = operand_info['scale_group'] # (outer, inner)
            self.scale_in_col = SimConfig.co_w // operand_info['scale_offset_precision']
            # mix data block info
            # self.db_col_num = self.scale_in_col * self.group_size[1] / self.data_in_col
            # block_sizes = loop_info.block_sizes
            # 我们只对不被 Buffer 的那侧做优化 
        else:
            self.end_row = self.scale_row_offset
        if set_k_block is not None:
            self.k_block = set_k_block
        
    def copy(self):
        return LoopfriendlyAddr(self.operand_info, self.loop_info, self.tensor_mapping, self.is_weight, self.row_offset, self.set_k_block)
    
    def get_meta(self):
        if self.k_block > 1:
            # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
            col_in_k_block = math.ceil(self.k_block / self.data_in_col)
            k_col_id = (self.op_size[1]-1) // self.data_in_col
            k_block_id = k_col_id // col_in_k_block
            k_inblock_id = k_col_id % col_in_k_block
            origin_col_id = k_inblock_id + (self.op_size[0]-1) * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
        return origin_col_id + 1,  origin_col_id // SimConfig.co + 1

    def get_scale_row_offset(self):
        if self.k_block > 1:
            # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
            col_in_k_block = math.ceil(self.k_block / self.data_in_col)
            k_col_id = (self.op_size[1]-1) // self.data_in_col
            k_block_id = k_col_id // col_in_k_block
            k_inblock_id = k_col_id % col_in_k_block
            origin_col_id = k_inblock_id + self.op_size[0] * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
            return int(self.row_offset + origin_col_id // SimConfig.co)+ 1
        else:
            raise NotImplementedError("Not support k_block == 1 for LoopfriendlyAddr")

    def get_end_row(self):
        return self.end_row

    def get_tensor_addr(self, a, b):
        assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
        # k_block -> DRAM col
        if self.k_block > 1:
            # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
            col_in_k_block = math.ceil(self.k_block / self.data_in_col)
            k_col_id = b // self.data_in_col
            k_block_id = k_col_id // col_in_k_block
            # assert k_block_id == 0, f"k_block_id: {k_block_id}, col_in_k_block: {col_in_k_block}, self.k_block: {self.k_block}, self.set_k_block: {self.set_k_block}"
            k_inblock_id = k_col_id % col_in_k_block
            origin_col_id = k_inblock_id + a * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
            return int(self.row_offset + origin_col_id // SimConfig.co), int(origin_col_id % SimConfig.co)
        else:
            raise NotImplementedError("Not support k_block == 1 for LoopfriendlyAddr")

    def get_scale_addr(self, a, b):
        assert self.with_scale, "No scale mapping"
        scale_group_a_id = a // self.operand_info['scale_group'][0]
        scale_group_b_id = b // self.operand_info['scale_group'][1]
        scale_b_col_num_id = scale_group_b_id // self.scale_mapping['scale_b_per_col']
        scale_a_col_num_id = scale_group_a_id // self.scale_mapping['scale_a_per_col']
        scale_b_row_id = scale_b_col_num_id // self.scale_mapping['scale_b_col_per_row']
        scale_a_row_id = scale_a_col_num_id // self.scale_mapping['scale_a_col_per_row']
        scale_b_col_id = scale_b_col_num_id % self.scale_mapping['scale_b_col_per_row']
        scale_a_col_id = scale_a_col_num_id % self.scale_mapping['scale_a_col_per_row']
        if scale_b_row_id == self.scale_mapping['scale_b_row_num'] - 1:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_corner_col_per_row']
        else:
            real_col_id = scale_b_col_id + scale_a_col_id * self.scale_mapping['scale_b_col_per_row']
        real_row_id = scale_b_row_id + scale_a_row_id * self.scale_mapping['scale_b_row_num'] + self.scale_row_offset
        return scale_group_a_id, scale_group_b_id, real_row_id, real_col_id

# MARK: 基于循环顺序的 Mapping
class ScaleMixAddr():
    def __init__(self, operand_info, loop_info, mx_gran=1, is_weight = True, row_offset=0):
        self.operand_info = operand_info
        self.loop_info = loop_info
        self.is_weight = is_weight
        self.row_offset = row_offset
        # self.scale_mix = scale_mix
        self.with_scale = operand_info['with_scale']
        if self.with_scale:
            # calculate the group info
            self.group_size = operand_info['scale_group'] # (outer, inner)
            self.scale_in_col = SimConfig.co_w // operand_info['scale_offset_precision']
            self.data_in_col = SimConfig.co_w // operand_info['matrix_precision']
            # block info
            self.mx_gran = mx_gran
            self.db_col_num = self.scale_in_col * self.group_size[1] / self.data_in_col * self.mx_gran
            # print(f"self.db_col_num: {self.db_col_num}, self.mx_gran: {self.mx_gran}")
            block_sizes = loop_info.block_sizes
            # 我们只对不被 Buffer 的那侧做优化
            if is_weight:
                self.op_size = (loop_info.n_size, loop_info.k_size)
                # K0 - N - K1
                assert loop_info.loop_order == ['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2']
                self.k_block = loop_info.block_sizes['k']['block1']
            else:
                raise NotImplementedError("Not support non-weight operand with scale")
                op_size = (loop_info.m_size, loop_info.k_size)
                # K0 - M - K1
                assert loop_info.loop_order == ['n0', 'n1', 'n2', 'k0', 'k1', 'm0', 'm1', 'm2', 'k2']
    
    def copy(self):
        return ScaleMixAddr(self.operand_info, self.loop_info, self.mx_gran, self.is_weight, self.row_offset)

    def get_end_row(self):
        if self.k_block > 1:
            # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
            col_in_k_block = math.ceil(self.k_block / self.data_in_col)
            k_col_id = self.op_size[1] // self.data_in_col
            k_block_id = k_col_id // col_in_k_block
            k_inblock_id = k_col_id % col_in_k_block
            origin_col_id = k_inblock_id + self.op_size[0] * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
            db_id = origin_col_id // self.db_col_num
            indb_id = origin_col_id % self.db_col_num
            real_col_id = db_id * (self.db_col_num + self.mx_gran) + self.mx_gran + indb_id - 1
            return int(self.row_offset + real_col_id // SimConfig.co + 1)
        else:
            raise NotImplementedError("Not support k_block == 1 for LoopfriendlyAddr")

    def get_tensor_addr(self, a, b):
        assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
        # k_block -> DRAM col
        if self.k_block > 1:
            # only test with w4a4s8 1,16
            # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
            group_in_block = math.ceil(self.k_block / self.group_size[1])
            k_group_id = b // self.group_size[1]
            k_in_group_id = b % self.group_size[1]
            
            k_group_block_id = k_group_id // group_in_block
            k_group_inblock_id = k_group_id % group_in_block
            
            group_id = k_group_inblock_id + a * group_in_block + k_group_block_id * self.op_size[0] * group_in_block
            # mystry number
            flat_id = 72 * group_id + 8 + k_in_group_id * 4
            real_col_id = flat_id // SimConfig.co_w
            # # group_id = a * 
            # col_in_k_block = math.ceil(self.k_block / self.data_in_col)
            # k_col_id = b // self.data_in_col
            # k_block_id = k_col_id // col_in_k_block
            # k_inblock_id = k_col_id % col_in_k_block
            # # 
            # origin_col_id = k_inblock_id + a * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
            
            # db_id = origin_col_id // self.db_col_num
            # indb_id = origin_col_id % self.db_col_num
            # real_col_id = db_id * (self.db_col_num + self.mx_gran) + self.mx_gran + indb_id
            return int(self.row_offset + real_col_id // SimConfig.co), int(real_col_id % SimConfig.co)
        else:
            raise NotImplementedError("Not support k_block == 1 for LoopfriendlyAddr")

    def get_scale_addr(self, a, b):
        assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
        scale_group_a_id = a // self.group_size[0]
        scale_group_b_id = b // self.group_size[1]
        # k_block -> DRAM col
        if self.k_block > 1:
            # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
            group_in_block = math.ceil(self.k_block / self.group_size[1])
            k_group_id = b // self.group_size[1]
            k_in_group_id = b % self.group_size[1]
            k_group_block_id = k_group_id // group_in_block
            k_group_inblock_id = k_group_id % group_in_block
            group_id = k_group_inblock_id + a * group_in_block + k_group_block_id * self.op_size[0] * group_in_block
            # mystry number
            flat_id = 72 * group_id + k_in_group_id // 16 * 8
            real_col_id = flat_id // SimConfig.co_w
            # col_in_k_block = math.ceil(self.k_block / self.data_in_col)
            # k_col_id = b // self.data_in_col
            # k_block_id = k_col_id // col_in_k_block
            # k_inblock_id = k_col_id % col_in_k_block
            # # 
            # origin_col_id = k_inblock_id + a * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
            
            # db_id = origin_col_id // self.db_col_num
            # indb_id = origin_col_id % self.db_col_num
            # scale_indb_id = indb_id // int(self.db_col_num/self.mx_gran)
            # assert scale_indb_id < self.mx_gran, f"scale_indb_id out of range, scale_indb_id: {scale_indb_id}, mx_gran: {self.mx_gran}"
            # real_col_id = db_id * (self.db_col_num + self.mx_gran) + scale_indb_id
            return int(scale_group_a_id), int(scale_group_b_id), int(self.row_offset + real_col_id // SimConfig.co), int(real_col_id % SimConfig.co)
        else:
            raise NotImplementedError("Not support k_block == 1 for LoopfriendlyAddr")

# 标准对齐，但可能会引起较多的
class ScaleMixAddr_v2():
    def __init__(self, operand_info, loop_info, mx_gran=1, is_weight = True, row_offset=0):
        self.operand_info = operand_info
        self.loop_info = loop_info
        self.is_weight = is_weight
        self.row_offset = row_offset
        # self.scale_mix = scale_mix
        self.with_scale = operand_info['with_scale']
        # self.op_size = operand_info['matrix_size']
        if self.with_scale:
            # calculate the group info
            self.group_size = operand_info['scale_group'] # (outer, inner)
            
            self.scale_in_col = SimConfig.co_w // operand_info['scale_offset_precision']
            self.data_in_col = SimConfig.co_w // operand_info['matrix_precision']
            # block info
            self.mx_gran = mx_gran
            # 填满 1 列 scale 需要多少列的数据
            self.db_scale_group_num = self.scale_in_col * self.mx_gran
            # print(f"self.db_col_num: {self.db_col_num}, self.mx_gran: {self.mx_gran}")
            block_sizes = loop_info.block_sizes
            # 我们只对不被 Buffer 的那侧做优化
            if is_weight:
                self.op_size = (loop_info.n_size, loop_info.k_size)
                # avoid corner
                self.group_size = (
                    min(self.op_size[0], self.group_size[0]),
                    min(self.op_size[1], self.group_size[1]),
                )
                self.db_col_num = self.scale_in_col * self.group_size[1] * self.group_size[0] / self.data_in_col * self.mx_gran
                # K0 - N - K1
                assert loop_info.loop_order == ['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2']
                self.k_block = loop_info.block_sizes['k']['block1']
                # 计算 k0 切块里包含多少个 scale group，这是 scale 在 k 维度的分块
                self.sg_k0 = min(math.ceil(self.k_block / self.group_size[1]),self.db_scale_group_num)
                # self.sg_k0 需要对齐到列
                if self.data_in_col > self.group_size[1]:
                    self.sg_in_col = int(self.data_in_col / self.group_size[1])
                    self.sg_k0 = math.ceil(self.sg_k0 / self.sg_in_col) * self.sg_in_col
                self.k_block_new = self.sg_k0 * self.group_size[1]
                
                # data block 对齐到分块
                self.actual_db_scale_group_num = self.sg_k0 * int(self.db_scale_group_num // self.sg_k0) # 虽然此处会造成存储的不紧密
                self.actual_db_col_num = self.actual_db_scale_group_num * self.group_size[0] * self.group_size[1] / self.data_in_col
                if False: # self.actual_db_col_num <= self.db_col_num - 1:
                    print(f"self.actual_db_col_num: {self.actual_db_col_num}, self.db_col_num: {self.db_col_num}")
                # 如果出现这种情况，说明 mx gran 选一个较小的值会更加合理（即 scale b prefetch size）
                if self.mx_gran - self.actual_db_scale_group_num / self.scale_in_col >= 1:
                    raise Exception(f"mx_gran: {self.mx_gran}, db_scale_group_num: {self.actual_db_scale_group_num}, scale_in_col: {self.scale_in_col}")
                assert self.mx_gran - self.actual_db_scale_group_num / self.scale_in_col < 1, f"mx_gran: {self.mx_gran}, db_scale_group_num: {self.actual_db_scale_group_num}, scale_in_col: {self.scale_in_col}"
                assert self.actual_db_scale_group_num % self.sg_k0 == 0, f"db_scale_group_num must be multiple of sg_k0, db_scale_group_num: {self.actual_db_scale_group_num}, sg_k0: {self.sg_k0}"
                self.col_per_block = self.actual_db_col_num + self.mx_gran
                # 此时是否意味着，scale值有可能不会存满？感觉好像也不太合理
                # 应该就按顺序存，然后 scale 个数大于 mx gran 容量，就切换区域？
                # 是不是不应该完全追求紧密存储，对一些 corner 情况在 mapping 中可以做 padding
            else:
                raise NotImplementedError("Not support non-weight operand with scale")
    
    def copy(self):
        return ScaleMixAddr_v2(self.operand_info, self.loop_info, self.mx_gran, self.is_weight, self.row_offset)

    def get_end_row(self):
        return self.get_tensor_addr(self.op_size[0] - 1, self.op_size[1] - 1)[0] + 1

    def get_meta(self):
        # # 1. scale < 1 col?
        # within_col = False
        # # 2. scale < 1 db?
        # within_db = False
        a = self.op_size[0] - 1
        b = self.op_size[1] - 1
        group_n_id = a // self.group_size[0]
        group_k_id = b // self.group_size[1]
        ingroup_n_id = a % self.group_size[0]
        ingroup_k_id = b % self.group_size[1]
        # 计算边界
        bound_group_n_id = (self.op_size[0] - 1) // self.group_size[0]
        bound_group_k_id = (self.op_size[1] - 1) // self.group_size[1] 
        # 计算对应的 scale group 在 k 维度上的位置
        group_k_block_id = group_k_id // self.sg_k0
        group_k_inblock_id = group_k_id % self.sg_k0
        # 对 scale group 的索引做 flatten
        # sg_k0 是否可以保证切块大小是整除的？应该是可以的！
        group_flat_id = group_k_inblock_id + self.sg_k0 * group_n_id + self.sg_k0 * (bound_group_n_id + 1) * group_k_block_id
        db_id = group_flat_id // self.actual_db_scale_group_num # * self.col_per_block
        indb_id = group_flat_id % self.actual_db_scale_group_num
        # 先索引 group id 在 k n 维度上的位置，然后再看组内位置，然后再展开为 k n 绝对坐标
        indb_n_id = indb_id // self.sg_k0 # scale group id 
        indb_k_id = indb_id % self.sg_k0
        indb_real_n_id = indb_n_id * self.group_size[0] + ingroup_n_id # data id
        indb_real_k_id = indb_k_id * self.group_size[1] + ingroup_k_id
        assert self.k_block_new % self.data_in_col == 0, "k_block_new must be multiple of data_in_col"
        indb_col_id = math.ceil(self.k_block_new / self.data_in_col) * indb_real_n_id + indb_real_k_id // self.data_in_col
        real_col_id = db_id * self.col_per_block + self.mx_gran + indb_col_id
        return real_col_id, bound_group_n_id, int(self.actual_db_scale_group_num / self.sg_k0), self.group_size, db_id, self.actual_db_col_num

    def get_tensor_addr(self, a, b):
        assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
        # k_block -> DRAM col
        # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
        # 先确定对应的 scale group 数量
        # 要求 k 分块一定要比 scale group 范围大
        """
        group = scale group
        block = k iteration inner loop block size
        db = mix data block

        n,k id -> group / in group id (scale group size)
            n group id -> unchanged
            k group id -> block / in block id (k block size / scale group size = self.sg_k0)
        
        flatten: group_flat_id = group_k_inblock_id 
                                + self.sg_k0 * group_n_id 
                                + self.sg_k0 * (bound_group_n_id + 1) * group_k_block_id

        """
        group_n_id = a // self.group_size[0]
        group_k_id = b // self.group_size[1]
        ingroup_n_id = a % self.group_size[0]
        ingroup_k_id = b % self.group_size[1]
        # 计算边界
        bound_group_n_id = (self.op_size[0] - 1) // self.group_size[0]
        bound_group_k_id = (self.op_size[1] - 1) // self.group_size[1]
        # 计算对应的 scale group 在 k 维度上的位置
        group_k_block_id = group_k_id // self.sg_k0
        group_k_inblock_id = group_k_id % self.sg_k0
        # 对 scale group 的索引做 flatten
        # sg_k0 是否可以保证切块大小是整除的？应该是可以的！
        group_flat_id = group_k_inblock_id + self.sg_k0 * group_n_id + self.sg_k0 * (bound_group_n_id + 1) * group_k_block_id
        db_id = group_flat_id // self.actual_db_scale_group_num # * self.col_per_block
        indb_id = group_flat_id % self.actual_db_scale_group_num
        # 先索引 group id 在 k n 维度上的位置，然后再看组内位置，然后再展开为 k n 绝对坐标
        indb_n_id = indb_id // self.sg_k0 # scale group id 
        indb_k_id = indb_id % self.sg_k0
        indb_real_n_id = indb_n_id * self.group_size[0] + ingroup_n_id # data id
        indb_real_k_id = indb_k_id * self.group_size[1] + ingroup_k_id
        assert self.k_block_new % self.data_in_col == 0, "k_block_new must be multiple of data_in_col"
        indb_col_id = math.ceil(self.k_block_new / self.data_in_col) * indb_real_n_id + indb_real_k_id // self.data_in_col
        real_col_id = db_id * self.col_per_block + self.mx_gran + indb_col_id

        # # 处理内部具体索引
        # # col_per_group = self.group_size[0] * self.group_size[1] / self.data_in_col # FIXME: 1 * 16 / 64 < 1，有可能出现这种情况
        # in_group_col = ingroup_k_id // self.data_in_col
        # # data_col_id = 

        # col_in_k_block = math.ceil(self.k_block_new / self.data_in_col)
        # k_col_id = b // self.data_in_col
        # k_block_id = k_col_id // col_in_k_block
        # k_inblock_id = k_col_id % col_in_k_block
        # # 
        # origin_col_id = k_inblock_id + a * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
        
        # db_id = origin_col_id // self.actual_db_col_num
        # indb_id = origin_col_id % self.actual_db_col_num
        # real_col_id = db_id * (self.actual_db_col_num + self.mx_gran) + self.mx_gran + indb_id

        return int(self.row_offset + real_col_id // SimConfig.co), int(real_col_id % SimConfig.co)

    def get_scale_addr(self, a, b):
        assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
        group_n_id = a // self.group_size[0]
        group_k_id = b // self.group_size[1]
        ingroup_n_id = a % self.group_size[0]
        ingroup_k_id = b % self.group_size[1]
        # 计算边界
        bound_group_n_id = (self.op_size[0] - 1) // self.group_size[0]
        bound_group_k_id = (self.op_size[1] - 1) // self.group_size[1] 
        # 计算对应的 scale group 在 k 维度上的位置
        group_k_block_id = group_k_id // self.sg_k0
        group_k_inblock_id = group_k_id % self.sg_k0
        # 对 scale group 的索引做 flatten
        # sg_k0 是否可以保证切块大小是整除的？应该是可以的！
        group_flat_id = group_k_inblock_id + self.sg_k0 * group_n_id + self.sg_k0 * (bound_group_n_id + 1) * group_k_block_id
        db_id = group_flat_id // self.actual_db_scale_group_num # * self.col_per_block
        indb_id = group_flat_id % self.actual_db_scale_group_num
        # # 先索引 group id 在 k n 维度上的位置，然后再看组内位置，然后再展开为 k n 绝对坐标
        # indb_n_id = indb_id // self.sg_k0 # scale group id 
        # indb_k_id = indb_id % self.sg_k0
        # indb_real_n_id = indb_n_id * self.group_size[0] + ingroup_n_id # data id
        # indb_real_k_id = indb_k_id * self.group_size[1] + ingroup_k_id
        assert self.k_block_new % self.data_in_col == 0, "k_block_new must be multiple of data_in_col"
        indb_col_id = indb_id // self.scale_in_col
        real_col_id = db_id * self.col_per_block + indb_col_id
        return int(group_n_id), int(group_k_id), int(self.row_offset + real_col_id // SimConfig.co), int(real_col_id % SimConfig.co)

# class ScaleMixAddr_v3():
#     def __init__(self, operand_info, loop_info, mx_gran=1, is_weight = True, row_offset=0):
#         self.operand_info = operand_info
#         self.loop_info = loop_info
#         self.is_weight = is_weight
#         self.row_offset = row_offset
#         # self.scale_mix = scale_mix
#         self.with_scale = operand_info['with_scale']
#         if self.with_scale:
#             # calculate the group info
#             self.group_size = operand_info['scale_group'] # (outer, inner)
#             self.scale_in_col = SimConfig.co_w // operand_info['scale_offset_precision']
#             self.data_in_col = SimConfig.co_w // operand_info['matrix_precision']
#             # block info
#             self.mx_gran = mx_gran
#             # 填满 1 列 scale 需要多少列的数据
#             self.db_scale_group_num = self.scale_in_col * self.mx_gran
#             self.db_col_num = self.scale_in_col * self.group_size[1] * self.group_size[0] / self.data_in_col * self.mx_gran
#             assert self.db_scale_group_num % self.sg_k0 == 0, f"db_scale_group_num must be multiple of sg_k0, db_scale_group_num: {self.db_scale_group_num}, sg_k0: {self.sg_k0}"
#             # print(f"self.db_col_num: {self.db_col_num}, self.mx_gran: {self.mx_gran}")
#             block_sizes = loop_info.block_sizes
#             # 我们只对不被 Buffer 的那侧做优化
#             if is_weight:
#                 self.op_size = (loop_info.n_size, loop_info.k_size)
#                 # K0 - N - K1
#                 assert loop_info.loop_order == ['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2']
#                 self.k_block = loop_info.block_sizes['k']['block1']
#                 self.sg_k0 = min(math.ceil(self.k_block / self.group_size[1]), self.db_scale_group_num)
#                 # 以 block 的粒度遍历就没问题？
#                 self.k_block_new = self.sg_k0 * self.group_size[1]
#                 # print(f"self.sg_k0: {self.sg_k0}, self.k_block_new: {self.k_block_new}")
#                 self.actual_db_col_num = int(self.db_scale_group_num // self.sg_k0 * self.k_block_new * self.group_size[0] / self.data_in_col)
#                 # 此时是否意味着，scale值有可能不会存满？感觉好像也不太合理
#                 # 应该就按顺序存，然后 scale 个数大于 mx gran 容量，就切换区域？
#                 # 是不是不应该完全追求紧密存储，对一些 corner 情况在 mapping 中可以做 padding
#             else:
#                 raise NotImplementedError("Not support non-weight operand with scale")
    
#     def copy(self):
#         return ScaleMixAddr_v3(self.operand_info, self.loop_info, self.mx_gran, self.is_weight, self.row_offset)

#     def get_end_row(self):
#         # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
#         col_in_k_block = math.ceil(self.k_block_new / self.data_in_col)
#         k_col_id = self.op_size[1] // self.data_in_col
#         k_block_id = k_col_id // col_in_k_block
#         k_inblock_id = k_col_id % col_in_k_block
#         origin_col_id = k_inblock_id + self.op_size[0] * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
#         db_id = origin_col_id // self.actual_db_col_num
#         indb_id = origin_col_id % self.actual_db_col_num
#         real_col_id = db_id * (self.actual_db_col_num + self.mx_gran) + self.mx_gran + indb_id - 1
#         return int(self.row_offset + real_col_id // SimConfig.co + 1)

#     def get_tensor_addr(self, a, b):
#         assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
#         # k_block -> DRAM col
#         # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
#         # 先确定对应的 scale group 数量
#         # 要求 k 分块一定要比 scale group 范围大
#         """
#         group = scale group
#         block = k iteration inner loop block size
#         db = mix data block

#         n,k id -> group / in group id (scale group size)
#             n group id -> unchanged
#             k group id -> block / in block id (k block size / scale group size = self.sg_k0)
        
#         flatten: group_flat_id = group_k_inblock_id 
#                                 + self.sg_k0 * group_n_id 
#                                 + self.sg_k0 * (bound_group_n_id + 1) * group_k_block_id

#         """
#         group_n_id = a // self.group_size[0]
#         group_k_id = b // self.group_size[1]
#         ingroup_n_id = a % self.group_size[0]
#         ingroup_k_id = b % self.group_size[1]
#         # 计算边界
#         bound_group_n_id = (self.op_size[0] - 1) // self.group_size[0]
#         bound_group_k_id = (self.op_size[1] - 1) // self.group_size[1] 
#         # 计算对应的 scale group 在 k 维度上的位置
#         group_k_block_id = group_k_id // self.sg_k0
#         group_k_inblock_id = group_k_id % self.sg_k0
#         # 对 scale group 的索引做 flatten
#         # sg_k0 是否可以保证切块大小是整除的？应该是可以的！
#         group_flat_id = group_k_inblock_id + self.sg_k0 * group_n_id + self.sg_k0 * (bound_group_n_id + 1) * group_k_block_id
#         db_id = group_flat_id // self.db_scale_group_num
#         indb_id = group_flat_id % self.db_scale_group_num
#         # 处理内部具体索引
#         # col_per_group = self.group_size[0] * self.group_size[1] / self.data_in_col # FIXME: 1 * 16 / 64 < 1，有可能出现这种情况
#         col_per_block = math.ceil(self.group_size[0] * self.group_size[1] * self.db_scale_group_num / self.data_in_col) + self.mx_gran
#         in_group_col = ingroup_k_id // self.data_in_col
#         # data_col_id = 

#         col_in_k_block = math.ceil(self.k_block_new / self.data_in_col)
#         k_col_id = b // self.data_in_col
#         k_block_id = k_col_id // col_in_k_block
#         k_inblock_id = k_col_id % col_in_k_block
#         # 
#         origin_col_id = k_inblock_id + a * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
        
#         db_id = origin_col_id // self.actual_db_col_num
#         indb_id = origin_col_id % self.actual_db_col_num
#         real_col_id = db_id * (self.actual_db_col_num + self.mx_gran) + self.mx_gran + indb_id
#         return int(self.row_offset + real_col_id // SimConfig.co), int(real_col_id % SimConfig.co)

#     def get_scale_addr(self, a, b):
#         assert a < self.op_size[0] and b < self.op_size[1], f"a or b out of range, a: {a}, b: {b}, loop_info: {self.op_size}"
#         scale_group_a_id = a // self.group_size[0]
#         scale_group_b_id = b // self.group_size[1]
#         # k_block -> DRAM col
#         # assert self.k_block % self.data_in_col == 0, "k_block must be multiple of data_in_col"
#         col_in_k_block = math.ceil(self.k_block_new / self.data_in_col)
#         k_col_id = b // self.data_in_col
#         k_block_id = k_col_id // col_in_k_block
#         k_inblock_id = k_col_id % col_in_k_block
#         # 
#         origin_col_id = k_inblock_id + a * col_in_k_block + k_block_id * self.op_size[0] * col_in_k_block
        
#         db_id = origin_col_id // self.actual_db_col_num
#         indb_id = origin_col_id % self.actual_db_col_num
#         # 此处采用了均分的想法，可能不太合适
#         scale_indb_id = int(indb_id // int(self.actual_db_col_num/self.mx_gran))
#         assert scale_indb_id < self.mx_gran, f"scale_indb_id out of range, scale_indb_id: {scale_indb_id}, mx_gran: {self.mx_gran}, actual_db_col_num: {self.actual_db_col_num}"
#         real_col_id = db_id * (self.actual_db_col_num + self.mx_gran) + scale_indb_id
#         return int(scale_group_a_id), int(scale_group_b_id), int(self.row_offset + real_col_id // SimConfig.co), int(real_col_id % SimConfig.co)
