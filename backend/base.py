import abc
from tools import *
import numpy as np

class BaseCodegen(HW_info):
    def __init__(self, require_power_of_2):
        super(BaseCodegen, self).__init__(require_power_of_2)
        self.de_pu_outbuf_col = SimConfig.de_pu_bf / SimConfig.co_w
        # print('de_pu_outbuf_col:', self.de_pu_outbuf_col)
        # FIXME: rank level computation should consider the outbuf of rank
        # FIXME: 输入写入的时候，写数应该是bank rank交替的，对aim8，其他架构cached的版本产生影响
        # FIXME: 合并的时候，读数也应该是bank间交替的
        self.gen_code = False
        self.last_buffer_col = 0
        self.last_buffer_row = 0
        self.buffered = []
        self.total_inst = 0
        self.inst_info = [
            'pu', 'pu_col', 'pu_row_change',
            'device_reg2buf', 'device_buf2reg',
            'device_buf2bk', 'device_buf2bk_col',
            'device_bk2buf', 'device_bk2buf_col',
            'device_bk2gb', 'device_bk2gb_col',
            'device_gb2bk', 'device_gb2bk_col',
            'host_read', 'host_read_col',
            'host_write', 'host_write_col',
            'host_write_device_buffer', 'host_write_device_buffer_col',
            'host_write_pu_inbuf', 'host_write_pu_inbuf_col',
            'host_read_mac_reg', 'host_write_mac_reg',
        ]
        self.inst_count = np.zeros(len(self.inst_info), dtype=np.int64)
        self.predictor = None # predictor should be defined in the child class

    def set_gen(self):
        self.gen_code = True

    def reset_inst_count(self):
        self.inst_count = np.zeros(len(self.inst_info), dtype=np.int64)

    # instruction generater
    def create_device_pu(self, 
                        ch_id: int, 
                        ra_id: int, 
                        de_id: int, 
                        pu_num: int, 
                        pu_mask: list[bool], 
                        op1: tuple[int, int, int], 
                        op2: tuple[int, int, int], 
                        col_num: int, 
                        auto_precharge: bool):
        """Create device-level processing unit MAC operation instruction.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)  
            de_id (int): Device ID (e.g., 0)
            pu_num (int): Number of processing units to use (e.g., 8, 16)
            pu_mask (list[bool]): Mask indicating which PUs are active (e.g., [True for _ in range(16)])
            op1 (tuple): Operand 1 bank location (bank_id, row_id, col_offset) (e.g., (0, 0, 0))
            op2 (tuple): Operand 2 bank location (bank_id, row_id, col_offset) (e.g., (1, 0, 0))
                        Note: If op1.bank == op2.bank and op2.row > 0, op2 comes from global buffer
                              If op1.bank == op2.bank and op2.row == 0, op2 comes from local input buffer
            col_num (int): Number of columns to process (e.g., 64)
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Device PU instruction tuple for simulation
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op1:(bank, row_id, col_offset), op2:,       col_num,auto_precharge    
        #return (LEVEL.DE,  OPTYPE.pu,      0,      0,      0,     (8, [True for _ in range(8)]),   (0, 0, 0),                      (1, 0, 0),  64,     False)
        # self.inst_count['pu'] += 1
        self.inst_count[0] += 1
        self.inst_count[1] += col_num
        if auto_precharge:
            # self.inst_count['pu_row_change'] += 1
            self.inst_count[2] += 1
        return (LEVEL.DE,  OPTYPE.pu, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, op2, col_num, auto_precharge)

    def create_device_reg2buf(self, 
                             ch_id: int, 
                             ra_id: int, 
                             de_id: int, 
                             pu_num: int, 
                             pu_mask: list[bool], 
                             buffer_addr: int, 
                             end_point=None):
        """Create instruction to transfer data from PU registers to local buffer.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_id (int): Device ID (e.g., 0)
            pu_num (int): Number of PUs involved (e.g., 16)
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(16)])
            buffer_addr (int): Buffer address to write to (e.g., 0)
            end_point (optional): Additional endpoint parameter
            
        Returns:
            tuple: Register to buffer transfer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
        #return (LEVEL.DE,  OPTYPE.reg2buf, 0,      0,      0,     (16, [True for _ in range(16)]), 0)
        # self.inst_count['device_reg2buf'] += 1
        self.inst_count[3] += 1
        if end_point is not None:
            return (LEVEL.DE,  OPTYPE.reg2buf, ch_id, ra_id, de_id, (pu_num, pu_mask), buffer_addr, end_point)
        else:
            return (LEVEL.DE,  OPTYPE.reg2buf, ch_id, ra_id, de_id, (pu_num, pu_mask), buffer_addr)

    def create_device_buf2reg(self, 
                             ch_id: int, 
                             ra_id: int, 
                             de_id: int, 
                             pu_num: int, 
                             pu_mask: list[bool], 
                             buffer_addr: int, 
                             end_point=None):
        """Create instruction to transfer data from local buffer to PU registers.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_id (int): Device ID (e.g., 0)
            pu_num (int): Number of PUs involved (e.g., 16)
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(16)])
            buffer_addr (int): Buffer address to read from (e.g., 1)
            end_point (optional): Additional endpoint parameter
            
        Returns:
            tuple: Buffer to register transfer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
        #return (LEVEL.DE,  OPTYPE.buf2reg, 0,      0,      0,     (16, [True for _ in range(16)]), 1)
        # self.inst_count['device_buf2reg'] += 1
        self.inst_count[4] += 1
        if end_point is not None:
            return (LEVEL.DE,  OPTYPE.buf2reg, ch_id, ra_id, de_id, (pu_num, pu_mask), buffer_addr, end_point)
        else:
            return (LEVEL.DE,  OPTYPE.buf2reg, ch_id, ra_id, de_id, (pu_num, pu_mask), buffer_addr)

    def create_device_buf2bk(self, 
                            ch_id: int, 
                            ra_id: int, 
                            de_id: int, 
                            pu_num: int, 
                            pu_mask: list[bool], 
                            op1: tuple[int, int, int], 
                            buf: tuple[bool, int, int], 
                            auto_precharge: bool):
        """Create instruction to transfer data from local buffer to bank.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_id (int): Device ID (e.g., 0)
            pu_num (int): Number of PUs involved (e.g., 8)
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(8)])
            op1 (tuple): Bank location (bank_id, row_id, col_offset) (e.g., (0, 1, 0))
            buf (tuple): Buffer info (is_input, buf_addr, col_len) (e.g., (False, 0, 0))
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Buffer to bank transfer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op:(bank, row_id, col_offset), auto_precharge
        #return (LEVEL.DE,  OPTYPE.buf2bk,  0,      0,      0,     (8, [True for _ in range(8)]),   (0, 1, 0),                      False)
        # self.inst_count['device_buf2bk'] += 1
        self.inst_count[5] += 1
        # self.inst_count['device_buf2bk_col'] += self.de_pu_outbuf_col
        self.inst_count[6] += self.de_pu_outbuf_col
        return (LEVEL.DE,  OPTYPE.buf2bk, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, buf, auto_precharge)

    def create_device_bk2buf(self, 
                            ch_id: int, 
                            ra_id: int, 
                            de_id: int, 
                            pu_num: int, 
                            pu_mask: list[bool], 
                            op1: tuple[int, int, int], 
                            buf: tuple[bool, int, int], 
                            auto_precharge: bool):
        """Create instruction to transfer data from bank to local buffer.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_id (int): Device ID (e.g., 0)
            pu_num (int): Number of PUs involved (e.g., 16)
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(16)])
            op1 (tuple): Bank location (bank_id, row_id, col_offset) (e.g., (0, 1, 32))
            buf (tuple): Buffer info (is_input, buf_addr, col_len)
                        For input buffer: (True, buf_addr, col_len) (e.g., (True, 0, 64))
                        For output buffer: (False, buf_addr, col_len) (e.g., (False, 0, 0))
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Bank to buffer transfer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op:(bank, row_id, col_offset), auto_precharge
        #return (LEVEL.DE,  OPTYPE.bk2buf,  0,      0,      0,     (16, [True for _ in range(16)]), (0, 1, 32),                     True)
        # self.inst_count['device_bk2buf'] += 1
        self.inst_count[7] += 1
        # self.inst_count['device_bk2buf_col'] += op1[2] if op1[0] else self.de_pu_outbuf_col
        self.inst_count[8] += buf[2] if buf[0] else self.de_pu_outbuf_col
        return (LEVEL.DE,  OPTYPE.bk2buf, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, buf, auto_precharge)

    def create_device_bk2gb(self, 
                           ch_id: int, 
                           ra_id: int, 
                           de_id: int, 
                           bank_id: int, 
                           op1: tuple[int, int], 
                           gb_col_offset: int, 
                           col_num: int, 
                           auto_precharge: bool):
        """Create instruction to transfer data from bank to global buffer.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_id (int): Device ID (e.g., 0)
            bank_id (int): Source bank ID (e.g., 0)
            op1 (tuple): Bank operation (row_id, col_offset) (e.g., (1, 0))
            gb_col_offset (int): Global buffer column offset (e.g., 0)
            col_num (int): Number of columns to transfer (e.g., 32)
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Bank to global buffer transfer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  bank_id,    op:(row_id, col_offset),    gb_col_offset   col_num, auto_precharge
        #return (LEVEL.DE,  OPTYPE.bk2gb,   0,      0,      0,      0,          (1, 0),                     0,              32,      True)
        # self.inst_count['device_bk2gb'] += 1
        self.inst_count[9] += 1
        # self.inst_count['device_bk2gb_col'] += col_num
        self.inst_count[10] += col_num
        return (LEVEL.DE,  OPTYPE.bk2gb, ch_id, ra_id, de_id, bank_id, op1, gb_col_offset, col_num, auto_precharge)

    def create_device_gb2bk(self, 
                           ch_id: int, 
                           ra_id: int, 
                           de_id: int, 
                           bank_mask: list[bool], 
                           op1: tuple[int, int], 
                           gb_col_offset: int, 
                           col_num: int, 
                           auto_precharge: bool):
        """Create instruction to transfer data from global buffer to bank(s).
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_id (int): Device ID (e.g., 0)
            bank_mask (list[bool]): Bank activation mask (e.g., [True for _ in range(16)])
            op1 (tuple): Bank operation (row_id, col_offset) (e.g., (2, 0))
            gb_col_offset (int): Global buffer column offset (e.g., 0)
            col_num (int): Number of columns to transfer (e.g., 32)
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Global buffer to bank transfer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  bank mask,        op:(row_id, col_offset),    gb_col_offset   col_num, auto_precharge
        #return (LEVEL.DE,  OPTYPE.gb2bk,   0,      0,      0,      0,                          (1, 0),                     0,              32,      True)
        # self.inst_count['device_gb2bk'] += 1
        self.inst_count[11] += 1
        # self.inst_count['device_gb2bk_col'] += col_num
        self.inst_count[12] += col_num
        return (LEVEL.DE,  OPTYPE.gb2bk, ch_id, ra_id, de_id, bank_mask, op1, gb_col_offset, col_num, auto_precharge)

    def create_rank_pu(self, 
                      ch_id: int, 
                      ra_id: int, 
                      pu_num: int, 
                      pu_mask: list[bool], 
                      op1: tuple[int, int, int, int], 
                      op2: tuple[int, int, int, int], 
                      col_num: int, 
                      auto_precharge: bool):
        """Create rank-level processing unit operation instruction.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            pu_num (int): Number of rank-level PUs (e.g., 4)
            pu_mask (list[bool]): Rank PU activation mask (e.g., [True for _ in range(4)])
            op1 (tuple): Operand 1 location (device, bank, row_id, col_offset) (e.g., (0, 0, 0, 0))
            op2 (tuple): Operand 2 location (device, bank, row_id, col_offset) (e.g., (1, 0, 0, 0))
            col_num (int): Number of columns to process (e.g., 15)
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Rank-level PU instruction tuple
        """
        # self.inst_count['pu'] += 1
        self.inst_count[0] += 1
        # self.inst_count['pu_col'] += col_num
        self.inst_count[1] += col_num
        if auto_precharge:
            # self.inst_count['pu_row_change'] += 1
            self.inst_count[2] += 1
        return (LEVEL.RA,  OPTYPE.pu, ch_id, ra_id, (pu_num, pu_mask), op1, op2, col_num, auto_precharge)

    def create_host_read(self, 
                        ch_id: int, 
                        ra_id: int, 
                        de_mask: list[bool], 
                        bank_id: int, 
                        row_id: int, 
                        col_offset: int, 
                        col_num: int, 
                        auto_precharge: bool):
        """Create host read operation from device bank.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_mask (list[bool]): Device activation mask (e.g., [True for _ in range(8)])
            bank_id (int): Target bank ID (e.g., 1)
            row_id (int): Row ID to read from (e.g., 0)
            col_offset (int): Column offset (e.g., 0)
            col_num (int): Number of columns to read (e.g., 64)
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Host read instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_list,  bank_id,    row_id, col_offset, col_num, auto_precharge
        #return (LEVEL.HOST,OPTYPE.read,    0,      0,      0,      0,          0,      0,          32)
        self.inst_count[13] += 1
        self.inst_count[14] += col_num
        return (LEVEL.SYS, OPTYPE.host_read, ch_id, ra_id, de_mask, bank_id, row_id, col_offset, col_num, auto_precharge)

    def create_host_write(self, 
                         ch_id: int, 
                         ra_id: int, 
                         de_mask: list[bool], 
                         bank_mask: list[bool], 
                         row_id: int, 
                         col_offset: int, 
                         col_num: int, 
                         auto_precharge: bool):
        """Create host write operation to device bank(s).
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            de_mask (list[bool]): Device activation mask (e.g., [True for _ in range(8)])
            bank_mask (list[bool]): Bank activation mask (e.g., [True for _ in range(16)])
            row_id (int): Row ID to write to (e.g., 0)
            col_offset (int): Column offset (e.g., 0)
            col_num (int): Number of columns to write (e.g., 64)
            auto_precharge (bool): Whether to auto-precharge after operation
            
        Returns:
            tuple: Host write instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  bank_mask,    row_id, col_offset, col_num, auto_precharge
        #return (LEVEL.HOST,OPTYPE.write,   0,      0,      0,      0,          0,      0,          32)
        self.inst_count[15] += 1
        self.inst_count[16] += col_num
        return (LEVEL.SYS, OPTYPE.host_write, ch_id, ra_id, de_mask, bank_mask, row_id, col_offset, col_num, auto_precharge)

    def create_host_write_device_buffer(self, 
                                       ch_id: int, 
                                       ra_id: int, 
                                       device_mask: list[bool], 
                                       buffer_addr: int, 
                                       col_num: int):
        """Create host write operation to device buffer.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            device_mask (list[bool]): Device activation mask (e.g., [True for _ in range(8)])
            buffer_addr (int): Buffer address to write to (e.g., 0)
            col_num (int): Number of columns to write (e.g., 64)
            
        Returns:
            tuple: Host write device buffer instruction
        """
    #   op-level    op-type                 ch_id,      ra_id,      device_mask-先有一个，实在不行给他全置为True即可
        self.inst_count[17] += 1
        self.inst_count[18] += col_num
        return (LEVEL.SYS, OPTYPE.host_write_device_buffer, ch_id, ra_id, device_mask, buffer_addr, col_num)

    def create_host_read_device_buffer(self, ch_id, ra_id, device_mask, buffer_addr, col_num):
        raise NotImplementedError

    def create_host_write_pu_inbuf(self, 
                                  ch_id: int, 
                                  ra_id: int, 
                                  device_mask: list[bool], 
                                  pu_mask: list[bool], 
                                  col_offset: int, 
                                  col_num: int):
        """Create host write operation to PU input buffer (supports broadcast).
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            device_mask (list[bool]): Device activation mask (e.g., [True for _ in range(8)])
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(16)])
            col_offset (int): Column offset (e.g., 0)
            col_num (int): Number of columns to write (e.g., 64)
            
        Returns:
            tuple: Host write PU input buffer instruction
        """
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
        #return (LEVEL.DE,  OPTYPE.reg2buf, 0,      0,      0,     (16, [True for _ in range(16)]), 0)
        self.inst_count[19] += 1
        self.inst_count[20] += col_num
        return (LEVEL.SYS, OPTYPE.host_write_pu_inbuf, ch_id, ra_id, device_mask, pu_mask, col_offset, col_num)

    def create_host_read_mac_reg(self, 
                                ch_id: int, 
                                ra_id: int, 
                                device_mask: list[bool], 
                                pu_mask: list[bool]):
        """Create host read operation from MAC register.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            device_mask (list[bool]): Device activation mask (e.g., [True for _ in range(8)])
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(16)])
            
        Returns:
            tuple: Host read MAC register instruction
        """
        self.inst_count[21] += 1
        return (LEVEL.SYS, OPTYPE.host_read_mac_reg, ch_id, ra_id, device_mask, pu_mask)

    def create_host_write_mac_reg(self, 
                                 ch_id: int, 
                                 ra_id: int, 
                                 device_mask: list[bool], 
                                 pu_mask: list[bool]):
        """Create host write operation to MAC register.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            device_mask (list[bool]): Device activation mask (e.g., [True for _ in range(8)])
            pu_mask (list[bool]): PU activation mask (e.g., [True for _ in range(16)])
            
        Returns:
            tuple: Host write MAC register instruction
        """
        self.inst_count[22] += 1
        return (LEVEL.SYS, OPTYPE.host_write_mac_reg, ch_id, ra_id, device_mask, pu_mask)
    
    def create_host_read_rank_pu_reg(self, 
                                    ch_id: int, 
                                    ra_id: int, 
                                    rank_pu_mask: list[bool]):
        """Create host read operation from rank-level PU register.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            rank_pu_mask (list[bool]): Rank PU activation mask (e.g., [True for _ in range(4)])
            
        Returns:
            tuple: Host read rank PU register instruction
        """
        self.inst_count[21] += 1
        return (LEVEL.SYS, OPTYPE.host_read_rank_pu_reg, ch_id, ra_id, rank_pu_mask)

    def create_host_write_rank_pu_reg(self, 
                                     ch_id: int, 
                                     ra_id: int, 
                                     rank_pu_mask: list[bool]):
        """Create host write operation to rank-level PU register.
        
        Args:
            ch_id (int): Channel ID (e.g., 0)
            ra_id (int): Rank ID (e.g., 0)
            rank_pu_mask (list[bool]): Rank PU activation mask (e.g., [True for _ in range(4)])
            
        Returns:
            tuple: Host write rank PU register instruction
        """
        self.inst_count[22] += 1
        return (LEVEL.SYS, OPTYPE.host_write_rank_pu_reg, ch_id, ra_id, rank_pu_mask)

    def output_buffer(self, 
                     result_col: int, 
                     result_row: int):
        """Manage output buffer state and determine if buffer flush is needed.
        
        Args:
            result_col (int): Current result column
            result_row (int): Current result row
            
        Returns:
            tuple: (need_flush, last_col, last_row)
                need_flush (bool): Whether buffer needs to be flushed
                last_col (int): Previous column position
                last_row (int): Previous row position
        """
        last_row = self.last_buffer_row
        last_col = self.last_buffer_col
        if result_row != self.last_buffer_row:
            # exit(3)
            self.last_buffer_row = result_row
            self.buffered = [result_col]
            # print('change row')
            return True, last_col, last_row
        else:
            if result_col not in self.buffered:
                self.buffered.append(result_col)
            if len(self.buffered) > self.de_pu_outbuf_col:
                # exit(3)
                self.buffered = [result_col]
                # print('change row')
                return True, last_col, last_row
        return False, last_col, last_row

    def reset_output_buffer(self):
        self.last_buffer_col = 0
        self.last_buffer_row = 0
        self.buffered = []

    @abc.abstractmethod
    def mm_micro(self, 
                mm_schedule: str, 
                base_group_id: int,
                channel_list: list, 
                rank_list: list, 
                device_list: list, 
                pu_num: int, 
                simd_l: int,
                input_bank: int, 
                input_row_offset: int, 
                weight_bank: int, 
                weight_row_offset: int, 
                output_bank: int, 
                output_row_offset: int,
                m_block: int, 
                k_block: int, 
                l_block: int, 
                b_block: int,
                m_row: int, 
                k_row: int, 
                l_row: int, 
                b_row: int, 
                m_block_corner: bool, 
                k_block_corner: bool, 
                l_block_corner: bool, 
                b_block_corner: bool,
                om_block: int, 
                ol_block: int, 
                ob_block: int,
                om_row: int, 
                ol_row: int, 
                ob_row: int,
                om_block_corner: bool, 
                ol_block_corner: bool, 
                ob_block_corner: bool,
                pu_m: int, 
                pu_k: int, 
                pu_l: int, 
                pu_b: int,
                pu_list: list, 
                performance_threshold: int):
        raise NotImplementedError

    @abc.abstractmethod
    def elewise_micro(self, 
                     mm_schedule: str, 
                     base_group_id: int,
                     channel_list: list, 
                     rank_list: list, 
                     device_list: list, 
                     pu_num: int, 
                     simd_l: int,
                     input_bank: int, 
                     input_row_offset: int, 
                     weight_bank: int, 
                     weight_row_offset: int, 
                     output_bank: int, 
                     output_row_offset: int,
                     m_block: int, 
                     k_block: int, 
                     l_block: int, 
                     b_block: int,
                     m_row: int, 
                     k_row: int, 
                     l_row: int, 
                     b_row: int, 
                     m_block_corner: bool, 
                     k_block_corner: bool, 
                     l_block_corner: bool, 
                     b_block_corner: bool,
                     om_block: int, 
                     ol_block: int, 
                     ob_block: int,
                     om_row: int, 
                     ol_row: int, 
                     ob_row: int,
                     om_block_corner: bool, 
                     ol_block_corner: bool, 
                     ob_block_corner: bool,
                     pu_m: int, 
                     pu_k: int, 
                     pu_l: int, 
                     pu_b: int,
                     pu_list: list, 
                     performance_threshold: int):
        raise NotImplementedError

    @abc.abstractmethod
    def softmax_micro(self, 
                     mm_schedule: str, 
                     base_group_id: int,
                     channel_list: list, 
                     rank_list: list, 
                     device_list: list, 
                     pu_num: int, 
                     simd_l: int,
                     input_bank: int, 
                     input_row_offset: int, 
                     weight_bank: int, 
                     weight_row_offset: int, 
                     output_bank: int, 
                     output_row_offset: int,
                     m_block: int, 
                     k_block: int, 
                     l_block: int, 
                     b_block: int,
                     m_row: int, 
                     k_row: int, 
                     l_row: int, 
                     b_row: int, 
                     m_block_corner: bool, 
                     k_block_corner: bool, 
                     l_block_corner: bool, 
                     b_block_corner: bool,
                     om_block: int, 
                     ol_block: int, 
                     ob_block: int,
                     om_row: int, 
                     ol_row: int, 
                     ob_row: int,
                     om_block_corner: bool, 
                     ol_block_corner: bool, 
                     ob_block_corner: bool,
                     pu_m: int, 
                     pu_k: int, 
                     pu_l: int, 
                     pu_b: int,
                     pu_list: list, 
                     performance_threshold: int):
        raise NotImplementedError

    @abc.abstractmethod
    def layernorm_micro(self, 
                       mm_schedule: str, 
                       base_group_id: int,
                       channel_list: list, 
                       rank_list: list, 
                       device_list: list, 
                       pu_num: int, 
                       simd_l: int,
                       input_bank: int, 
                       input_row_offset: int, 
                       weight_bank: int, 
                       weight_row_offset: int, 
                       output_bank: int, 
                       output_row_offset: int,
                       m_block: int, 
                       k_block: int, 
                       l_block: int, 
                       b_block: int,
                       m_row: int, 
                       k_row: int, 
                       l_row: int, 
                       b_row: int, 
                       m_block_corner: bool, 
                       k_block_corner: bool, 
                       l_block_corner: bool, 
                       b_block_corner: bool,
                       om_block: int, 
                       ol_block: int, 
                       ob_block: int,
                       om_row: int, 
                       ol_row: int, 
                       ob_row: int,
                       om_block_corner: bool, 
                       ol_block_corner: bool, 
                       ob_block_corner: bool,
                       pu_m: int, 
                       pu_k: int, 
                       pu_l: int, 
                       pu_b: int,
                       pu_list: list, 
                       performance_threshold: int):
        raise NotImplementedError

    def codegen(self, 
               kernel_name: str, 
               compute_level, 
               pu_num: int, 
               partition: tuple, 
               simd_k: int, 
               mkl_Input_to_row: tuple, 
               simd_l: int, 
               ml_Out_to_row: tuple, 
               hw_id_list: tuple, 
               mem_mapping: tuple, 
               mm_schedule: str = 'mkl', 
               cmd_threshold: int = 0):
        
        # get partition & mapping
        if SimConfig.pu_level == LEVEL.RA:
            pu_m, pu_k, pu_l, pu_b = partition[2]
        else:
            pu_m, pu_k, pu_l, pu_b = partition[3]

        input_bank, input_row_offset,\
            weight_bank, weight_row_offset,\
                output_bank, output_row_offset = mem_mapping

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
        # get assigned hw
        channel_list, rank_list, device_list, pu_list = hw_id_list

        # select the kernel for codegen
        if kernel_name == 'mm':
            kernel = self.mm_micro
        elif kernel_name == 'elewise':
            kernel = self.elewise_micro
        elif kernel_name == 'softmax':
            kernel = self.softmax_micro
        elif kernel_name in ['layernorm','batchnorm']:
            kernel = self.layernorm_micro
        else: raise NotImplementedError
        
        # generate code
        inst_groups, predict_result = \
            kernel(mm_schedule, 0, channel_list, rank_list, device_list, pu_num, simd_l,
                input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                m_block, k_block, l_block, b_block,
                m_row, k_row, l_row, b_row,
                m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                om_block, ol_block, ob_block,
                om_row, ol_row, ob_row,
                om_block_corner, ol_block_corner, ob_block_corner,
                pu_m, pu_k, pu_l, pu_b, pu_list, cmd_threshold)
        return inst_groups, self.inst_count.tolist(), predict_result

    def get_matrix(self):
        assert self.gen_code, "Please set_gen() before get_matrix()"
        # inst num
        # sum_list = [0, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19] micro inst
        sum_list = [1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22] # dram inst, 解释：Host端指令本身就是不区分Device的，所以只是在统计读写次数的时候用了
        inst_num = sum(self.inst_count[sum_list])
        inst_num += self.inst_count[2] * 2 # 增加换行次数的估计
        # DRAM access num
        sum_list = [1, 6, 8, 10, 12]
        pu_dram_num = sum(self.inst_count[sum_list])
        sum_list = [14, 16, 18, 20, 21, 22]
        host_dram_num = sum(self.inst_count[sum_list]) * self.device_num # 本质上是DRAM对总线的占用
        # row change num
        row_change_num = self.inst_count[2]
        return inst_num, pu_dram_num, host_dram_num, row_change_num