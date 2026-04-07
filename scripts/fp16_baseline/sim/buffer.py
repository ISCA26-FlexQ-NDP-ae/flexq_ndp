from tools import *

class Buffer(Resource):
    def __init__(self, buf_size, numpy_object, index):
        super(Buffer, self).__init__(numpy_object, index)
        self.size = buf_size
        self.nxt_op = 0

        if SimConfig.verify:
            self.data = [None] * self.size

    def check_inst(self, write=False):
        if write:
            return self.nxt_op
        else:
            return self.nxt_op

    def issue_inst(self, last_read_write, col_offset, col_len, write=False):
        assert col_len > 0
        assert col_offset >= 0
        assert col_offset + col_len <= self.size
        if write:
            self.nxt_op = last_read_write
        else:
            self.nxt_op = last_read_write
