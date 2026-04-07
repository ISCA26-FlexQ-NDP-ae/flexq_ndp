from functools import reduce
import numpy as np

"""
1. SimConfig: 
"""
import yaml
class SimConfig:
    verify = False
    bus_type = 8

    @classmethod
    def read_from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            for key, value in data.items():
                setattr(cls, key, value)
            if 'pu_lat' in data.keys():
                setattr(cls, 'pu_lat', int(max(data['pu_lat'], data['tCCDL'], data['BL']/cls.bus_type)))
            else:
                setattr(cls, 'pu_lat', int(max(data['tCCDL'], data['BL']/cls.bus_type)))
        # prepare other timing parameters
        cls.burst_apox = int(cls.BL/cls.bus_type)
        cls.col_change_apox = max(cls.tCCDL, cls.burst_apox)
        cls.read_row_change_apox = cls.tRCDRD + cls.tRP + cls.burst_apox + max(cls.tRTPL-cls.tCCDL, 0)
        cls.read_to_write_row_change_apox = cls.tRCDWR + cls.tRP + cls.burst_apox + max(cls.tRTPL-cls.tCCDL, 0)
        cls.write_row_change_apox = cls.tRCDWR + cls.tRP + cls.tWR + cls.WL + cls.burst_apox
        cls.write_to_read_row_change_apox = cls.tRCDRD + cls.tRP + cls.tWR + cls.WL + cls.burst_apox
        cls.read_to_write_apox = cls.RL + cls.burst_apox + cls.tRTRS - cls.WL - cls.col_change_apox
        cls.write_to_read_apox = cls.WL + cls.burst_apox + cls.tWTRL - cls.col_change_apox

"""
2. HW_info for compiler
"""
class HW_info:
    def __init__(self, require_power_of_2):
        self.row_num = SimConfig.ro
        self.col_num = SimConfig.co
        self.channel_num = SimConfig.ch
        self.rank_num = SimConfig.ra
        self.device_num = SimConfig.de
        self.bank_num = SimConfig.ba * SimConfig.bg
        # device_pu_num = SimConfig.de_pu
        self.simd = int(SimConfig.co_w / SimConfig.data_pr)
        self.de_pu_num_list = SimConfig.de_pu
        self.limit_div_to_power_of_two = require_power_of_2

    def powerof2(self, x):
        if self.limit_div_to_power_of_two:
            return x != 0 and ((x & (x - 1)) == 0)
        else:
            return True

    def get_div_id(self, index, div):
        assert len(index) == len(div)
        id = index[0]
        for i in range(len(index)-1):
            id = id * div[i+1] + index[i+1]
        return id
    
    def get_div_index(self, id, div):
        index = []
        for i in range(len(div)-1, -1, -1):
            index.append(id % div[i])
            id = id // div[i]
        return index[::-1]

"""
3. inst format
"""
from enum import Enum
# LEVEL, OPTYPE, ch_id, ra_id, de_id
class LEVEL(Enum):
    DE = 1
    RA = 2
    CH = 3
    SYS = 4
class OPTYPE(Enum):
    # level 1,2,3
    pu = 1 # ch_id, ra_id, de_id, pu:(num, mask), op1:(bank, row_id, col_offset), op2:(bank, row_id, col_offset)
    reg2buf = 2 # ch_id, ra_id, de_id, pu:(num, (mask), group), buffer_slot
    buf2reg = 3 # ch_id, ra_id, de_id, pu:(num, (mask), group), buffer_slot
    buf2bk = 4 # ch_id, ra_id, de_id, pu:(num, (mask), group), op1:(bank, row_id, col_offset), op2:(bank, row_id, col_offset)
    bk2buf = 5 # 
    bk2gb = 6
    gb2bk = 7
    # level 4
    host_read = 1
    host_write = 2
    host_write_device_buffer = 3
    host_read_device_buffer = 4
    host_write_pu_inbuf = 5
    host_read_mac_reg = 6
    host_write_mac_reg = 7
    host_read_rank_pu_reg = 8
    host_write_rank_pu_reg = 9

# Tools Functions
mul = lambda x: reduce(lambda x,y:x*y,x)

"""
4. Basic Resource
base class of bus / PU / Bank / Buffer
"""
class Resource:
    def __init__(self, numpy_object, index):
        # self.tick = 0
        self.occupy = False
        self.numpy_object = numpy_object
        self.index = index
    def check_state(self):
        # if self.occupy:
        #     return self.numpy_object[self.index]
        # else:
        #     return 0
        return self.numpy_object[self.index]
    def set_state(self, countdown, delay=0):
        # print(f"{self.numpy_object}, {self.index}")
        # assert not self.occupy
        assert delay >= self.numpy_object[self.index], "delay: %d, countdown: %d" % (delay, self.numpy_object[self.index])
        # self.occupy = True
        self.numpy_object[self.index] = countdown

"""
test code
"""
if __name__ == '__main__':
    print(OPTYPE.sys2pu.value)
    print(LEVEL.SYS)
    assert OPTYPE.host_read.value == 1
