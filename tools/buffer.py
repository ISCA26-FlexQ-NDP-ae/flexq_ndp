"""
Generic Buffer Implementation for PIM Simulation

This module provides a generic buffer class that implements trace-based prefetching
for optimal buffer management, as described in CLAUDE.md. The buffer maintains
an ideal situation where it collects trace information to ensure that each buffer
replacement loads data that will definitely be used in subsequent computations.
"""

from typing import List, Set, Tuple, Union, Any, Optional
import copy

class TraceBuffer:
    """
    Generic buffer class with trace-based prefetching for PIM simulation.
    
    This class implements the buffer management strategy described in CLAUDE.md:
    - Ideal buffering with trace information collection
    - Ensures buffer replacements load data that will be used in future computations
    - Supports both scale buffers and data buffers
    """
    
    def __init__(self, 
                 buffer_capacity: int,
                 trace_data: List[Tuple], 
                 buffer_name: str = "Buffer"):
        """
        Initialize the trace buffer.
        
        Args:
            buffer_capacity: Maximum number of items the buffer can hold
            trace_data: Complete trace of data accesses for optimal prefetching
            buffer_name: Name for debugging/logging purposes
        """
        self.buffer_capacity = buffer_capacity
        self.trace_data = copy.deepcopy(trace_data)  # Deep copy to avoid modifying original
        self.buffer_contents = []  # Current buffer contents
        self.buffer_name = buffer_name
        
    def is_data_available(self, data_key: Union[Tuple, Any]) -> bool:
        """
        Check if the requested data is currently in the buffer.
        
        Args:
            data_key: Key representing the data to check
            
        Returns:
            True if data is in buffer, False otherwise
        """
        return data_key in self.buffer_contents

    def is_list_available(self, data_key: List[Tuple]) -> bool:
        """
        Check if the requested data is currently in the buffer.
        
        Args:
            data_key: Key representing the data to check
            
        Returns:
            True if data is in buffer, False otherwise
        """
        for data in data_key:
            if data not in self.buffer_contents:
                return False
        return True
    
    def update_buffer(self, current_required: Set[Tuple]) -> Tuple[Set[Tuple], Set[Tuple]]:
        """
        Update buffer contents based on current required data.
        """
        data_kept = self.get_kept_data(current_required)
        data_required = self.get_required_new_data(current_required)
        # 接着，计算需要更新的数据量
        data_new_num = self.buffer_capacity - len(data_kept)
        # 根据需要更新的数据量，调用trace获取信息
        data_new = self.update_trace(data_new_num) - data_kept
        # if self.buffer_name == "B_DataBuffer":
        #     print(f"{current_required}\n{self.buffer_contents}\n{data_required}\n{data_new}\n {self.buffer_contents}\n {len(data_kept)}\n {self.trace_data}\n")
        assert len(data_required-data_new) == 0, f"{current_required}\n{data_required}\n {data_new}\n {self.buffer_contents}\n {len(data_kept)}\n {self.trace_data}"
        return data_new-set(self.buffer_contents), data_new | data_kept
    
    def get_required_new_data(self, current_required: Set[Tuple]) -> Set[Tuple]:
        """
        Determine what new data needs to be loaded into the buffer.
        
        Args:
            current_required: Set of data items currently needed
            
        Returns:
            Set of new data items that need to be loaded from DRAM
        """
        current_required = set(current_required)
        current_buffer_set = set(self.buffer_contents)
        return current_required - current_buffer_set
    
    def get_kept_data(self, current_required: Set[Tuple]) -> Set[Tuple]:
        """
        Get data that needs to be kept in the buffer.
        """
        current_required = set(current_required)
        current_buffer_set = set(self.buffer_contents)

        return current_buffer_set & current_required

    def update_trace(self, refresh_count: Optional[int] = None) -> Set[Tuple]:
        """
        Update buffer contents based on trace information.
        
        Args:
            refresh_count: Number of items to prefetch from trace (uses buffer_capacity if None)
            
        Returns:
            Tuple of (data_to_keep, new_data_to_load)
        """
        # 先清除当前 buffer 完全可以处理的部分
        current_buffer_set = set(self.buffer_contents)
        while len(self.trace_data) > 0:
            
            _len = min(self.buffer_capacity, len(self.trace_data))
            _next = set(self.trace_data[:_len])
            if _next - current_buffer_set == set():
                # 删除
                self.trace_data = self.trace_data[_len:]
            else:
                break
        # 如果是 scale buffer 类
        if refresh_count is None:
            refresh_count = self.buffer_capacity
            
        # Collect future requirements from trace
        future_required = set()
        while refresh_count > 0 and len(self.trace_data) > 0:
            trace_point = self.trace_data.pop(0)
            if trace_point not in future_required:
                future_required.add(trace_point)
                refresh_count -= 1
        
        return future_required

    def actual_update_buffer(self, new_contents: Set[Tuple]):
        """
        Actual update buffer contents.
        """
        self.buffer_contents = list(new_contents)
        if self.buffer_name == "-":
            print(f"update buffer {self.buffer_name} with {new_contents}")


class ScaleBuffer(TraceBuffer):
    """ 
    Specialized buffer for scale data in quantization operations.
    
    Used for storing quantization scale factors to reduce DRAM access.
    Supports both A_scale and B_scale buffering patterns from the original code.
    """
    
    def __init__(self, 
                 buffer_capacity: int,
                 scale_trace: List[Tuple[int, int]], 
                 buffer_name: str = "ScaleBuffer"):
        """
        Initialize scale buffer.
        
        Args:
            buffer_capacity: Maximum number of scale items to buffer
            scale_trace: Trace of (row, col) scale access patterns
            buffer_name: Name for debugging
        """
        super().__init__(buffer_capacity, scale_trace, buffer_name)
    
    def get_scale_row_dict(self, new_scales: Set[Tuple[int, int]]) -> dict:
        """
        Group scale coordinates by row for efficient DRAM access.
        
        Args:
            new_scales: Set of (row, col) scale coordinates to load
            
        Returns:
            Dictionary mapping row -> [list of columns]
        """
        scale_row_dict = {}
        for row, col in new_scales:
            if row not in scale_row_dict:
                scale_row_dict[row] = [col]
            else:
                scale_row_dict[row].append(col)
        return scale_row_dict


class DataBuffer(TraceBuffer):
    """
    Specialized buffer for dequantized weight data in weight-only quantization.
    
    Used for storing dequantized weight data (B_dequant) in weight-only scenarios.
    Includes scale dependency tracking as dequantization requires scale data.
    """
    
    def __init__(self, 
                 buffer_capacity: int,
                 data_trace: List[Tuple[int, int, int, int]], 
                 buffer_name: str = "DataBuffer"):
        """
        Initialize data buffer.
        
        Args:
            buffer_capacity: Maximum number of data items to buffer  
            data_trace: Trace of (data_row, data_col, scale_row, scale_col) access patterns
            buffer_name: Name for debugging
        """
        super().__init__(buffer_capacity, data_trace, buffer_name)
    
    def get_required_scales(self, new_data: Set[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        """
        Extract scale requirements from data requirements.
        
        Args:
            new_data: Set of (data_row, data_col, scale_row, scale_col) tuples
            
        Returns:
            List of (scale_row, scale_col) coordinates needed for dequantization
        """
        required_scales = []
        for data_row, data_col, scale_row, scale_col in new_data:
            required_scales.append((scale_row, scale_col))
        return required_scales
    
    def get_data_row_dict(self, new_data: Set[Tuple[int, int, int, int]]) -> dict:
        """
        Group data coordinates by row for efficient DRAM access.
        
        Args:
            new_data: Set of (data_row, data_col, scale_row, scale_col) tuples
            
        Returns:
            Dictionary mapping data_row -> [list of data_columns]
        """
        data_row_dict = {}
        for data_row, data_col, scale_row, scale_col in new_data:
            if data_row not in data_row_dict:
                data_row_dict[data_row] = [data_col]
            else:
                data_row_dict[data_row].append(data_col)
        return data_row_dict


class ResultBuffer:
    """
    Buffer for managing partial and final computation results.
    
    Handles the buffering of intermediate computation results before writing back to DRAM.
    Supports both partial results (during K accumulation) and final results.
    """
    
    def __init__(self, 
                 max_result_capacity: int,
                 buffer_name: str = "ResultBuffer"):
        """
        Initialize result buffer.
        
        Args:
            max_result_capacity: Maximum number of result items to buffer
            buffer_name: Name for debugging
        """
        self.max_result_capacity = max_result_capacity
        self.buffered_results = {}  # Maps (m, n) -> accumulation_count or list
        self.buffer_name = buffer_name
    
    def can_add_result(self, result_key: Tuple[int, int]) -> bool:
        """
        Check if a new result can be added to the buffer.
        
        Args:
            result_key: (m, n) coordinates of the result
            
        Returns:
            True if result can be added without exceeding capacity
        """
        if result_key in self.buffered_results:
            return True
        return len(self.buffered_results) < self.max_result_capacity
    
    def add_result(self, result_key: Tuple[int, int], value: Any = 1):
        """
        Add or update a result in the buffer.
        
        Args:
            result_key: (m, n) coordinates of the result
            value: Value to add (accumulation count or result data)
        """
        if result_key in self.buffered_results:
            if isinstance(self.buffered_results[result_key], int):
                self.buffered_results[result_key] += value
            else:
                self.buffered_results[result_key].append(value)
        else:
            self.buffered_results[result_key] = value
    
    def needs_writeback(self, next_result_key: Tuple[int, int]) -> bool:
        """
        Check if buffer needs to write back results to make space.
        
        Args:
            next_result_key: Next result that needs to be added
            
        Returns:
            True if writeback is needed
        """
        return (next_result_key not in self.buffered_results and 
                len(self.buffered_results) >= self.max_result_capacity)
    
    def get_writeback_data(self) -> dict:
        """
        Get current buffered results for writeback.
        
        Returns:
            Dictionary of current buffered results
        """
        return copy.deepcopy(self.buffered_results)
    
    def clear_buffer(self):
        """Clear all buffered results after writeback."""
        self.buffered_results.clear()
    
    def get_buffer_status(self) -> dict:
        """
        Get current buffer status for debugging.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'name': self.buffer_name,
            'current_size': len(self.buffered_results),
            'max_capacity': self.max_result_capacity,
            'utilization': len(self.buffered_results) / self.max_result_capacity,
            'contents': list(self.buffered_results.values())
        }