"""
命令地址辅助模块
基于配置驱动的动态属性映射，提供高可扩展性的地址访问接口
"""

from typing import List, Tuple, Optional, Any, Dict


class CmdAddrConfig:
    """命令地址格式配置 - 集中管理所有格式定义"""
    
    # 默认格式（test_baseline 和 test_baseline_w_only 当前使用的格式）
    DEFAULT_FORMAT = {
        # 基本地址 (0-5)
        'A_row': 0, 'A_col': 1,
        'B_row': 2, 'B_col': 3,
        'result_row': 4, 'result_col': 5,
        
        # 缩放组ID (6-9)
        'A_scale_group_a_id': 6, 'A_scale_group_b_id': 7,
        'B_scale_group_a_id': 8, 'B_scale_group_b_id': 9,
        
        # 缩放地址 (10-13)
        'A_scale_row': 10, 'A_scale_col': 11,
        'B_scale_row': 12, 'B_scale_col': 13,
        
        # 原始坐标 (14-15)
        'm': 14, 'n': 15
    }
    
    # W-only格式（目前与默认格式相同，但保留独立定义以便将来修改）
    W_ONLY_FORMAT = DEFAULT_FORMAT.copy()
    
    # A矩阵格式（用于A矩阵地址生成模式）
    A_MATRIX_FORMAT = {
        # A tensor addresses (0-1)
        'A_row': 0, 'A_col': 1,
        
        # Original A tensor addresses (2-3)
        'B_row': 2, 'B_col': 3,
        
        # A scale group IDs (4-5)
        'A_scale_group_a_id': 4, 'A_scale_group_b_id': 5,
        
        # A scale addresses (6-7)
        'A_scale_row': 6, 'A_scale_col': 7,
    }
    
    # 示例：未来可能的扩展格式
    EXTENDED_FORMAT = {
        'A_row': 0, 'A_col': 1,
        'B_row': 2, 'B_col': 3,
        'result_row': 4, 'result_col': 5,
        
        # 假设未来增加了时间戳
        'timestamp': 6,
        
        # 缩放组ID位置调整
        'A_scale_group_a_id': 7, 'A_scale_group_b_id': 8,
        'B_scale_group_a_id': 9, 'B_scale_group_b_id': 10,
        
        # 缩放地址
        'A_scale_row': 11, 'A_scale_col': 12,
        'B_scale_row': 13, 'B_scale_col': 14,
        
        # 原始坐标
        'm': 15, 'n': 16,
        
        # 新增字段
        'priority': 17,
        'metadata': 18
    }


class CmdAddr:
    """
    动态格式支持的命令地址类
    
    根据配置自动创建属性访问器，支持多种地址格式
    当格式变更时，只需修改配置，无需改动使用代码
    
    使用示例:
        # 使用默认格式
        cmd = CmdAddr(cmd_addr_list[i])
        print(f"A位置: ({cmd.A_row}, {cmd.A_col})")
        
        # 使用特定格式
        cmd = CmdAddr(cmd_addr_list[i], CmdAddrConfig.W_ONLY_FORMAT)
        
        # 组合属性访问
        if cmd.same_AB_col(next_cmd):
            print("列相同，可以累积")
    """
    
    # 类级别的属性缓存，避免重复创建
    _property_cache = {}
    
    def __init__(self, raw_addr: List[Any], format_config: Optional[Dict[str, int]] = None):
        """
        初始化命令地址
        
        Args:
            raw_addr: 原始地址数组
            format_config: 格式配置字典，默认使用 DEFAULT_FORMAT
        """
        self.raw = raw_addr
        self.format = format_config or CmdAddrConfig.DEFAULT_FORMAT
        
        # 为当前实例设置属性访问器
        self._setup_properties()
    
    def _setup_properties(self):
        """为当前实例设置属性访问器"""
        format_id = id(self.format)
        
        # 如果该格式的属性还未创建，则创建
        if format_id not in self._property_cache:
            self._property_cache[format_id] = {}
            for field_name, index in self.format.items():
                self._property_cache[format_id][field_name] = self._make_property(index)
        
        # 为当前实例设置属性
        for field_name, prop in self._property_cache[format_id].items():
            setattr(self, field_name, prop.fget(self))
    
    def _make_property(self, index: int):
        """
        创建属性访问器
        
        Args:
            index: 在raw数组中的索引位置
            
        Returns:
            property: 属性访问器
        """
        def getter(self):
            return self.raw[index] if index < len(self.raw) else None
        
        return property(getter)
    
    # === 组合属性 - 基于字段名而非索引，提供更高层次的抽象 ===
    
    @property
    def A_pos(self) -> Tuple[int, int]:
        """A矩阵位置 (row, col)"""
        return (self.A_row, self.A_col)
    
    @property
    def B_pos(self) -> Tuple[int, int]:
        """B矩阵位置 (row, col)"""
        return (self.B_row, self.B_col)

    @property
    def AB_pos(self) -> Tuple[int, int, int, int]:
        """A、B矩阵位置 (row, col)"""
        return (self.A_row, self.A_col, self.B_row, self.B_col)
    
    @property
    def result_pos(self) -> Tuple[int, int]:
        """结果矩阵位置 (row, col)"""
        return (self.result_row, self.result_col)
    
    @property
    def A_scale_pos(self) -> Tuple[Optional[int], Optional[int]]:
        """A缩放位置 (row, col)"""
        return (getattr(self, 'A_scale_row', None), getattr(self, 'A_scale_col', None))
    
    @property
    def B_scale_pos(self) -> Tuple[Optional[int], Optional[int]]:
        """B缩放位置 (row, col)"""
        return (getattr(self, 'B_scale_row', None), getattr(self, 'B_scale_col', None))
    
    @property
    def matrix_coord(self) -> Tuple[int, int]:
        """原始矩阵坐标 (m, n)"""
        return (self.m, self.n)
    
    @property
    def AB_col(self) -> Tuple[int, int]:
        """A、B矩阵列 (A_col, B_col) - 常用于列累积判断"""
        return (self.A_col, self.B_col)
    
    @property
    def AB_row(self) -> Tuple[int, int]:
        """A、B矩阵行 (A_row, B_row) - 常用于行比较"""
        return (self.A_row, self.B_row)
    
    @property
    def scale_positions(self) -> Tuple[Tuple[Optional[int], Optional[int]], Tuple[Optional[int], Optional[int]]]:
        """所有缩放位置 ((A_scale_row, A_scale_col), (B_scale_row, B_scale_col))"""
        return (self.A_scale_pos, self.B_scale_pos)
    
    # === 比较方法 - 基于字段名的比较，格式无关 ===
    
    def same_AB_col(self, other: Optional['CmdAddr']) -> bool:
        """检查A、B矩阵列是否相同"""
        if other is None:
            return False
        return self.AB_col == other.AB_col
    
    def same_AB_row(self, other: Optional['CmdAddr']) -> bool:
        """检查A、B矩阵行是否相同"""
        if other is None:
            return False
        return self.AB_row == other.AB_row

    def same_A_row(self, other: Optional['CmdAddr']) -> bool:
        """检查A矩阵行是否相同"""
        if other is None:
            return False
        return self.A_row == other.A_row

    def same_B_row(self, other: Optional['CmdAddr']) -> bool:
        """检查B矩阵行是否相同"""
        if other is None:
            return False
        return self.B_row == other.B_row
    
    def same_AB_pos(self, other: Optional['CmdAddr']) -> bool:
        """检查A、B矩阵位置是否完全相同"""
        if other is None:
            return False
        return self.A_pos == other.A_pos and self.B_pos == other.B_pos
    
    def same_result_coord(self, other: Optional['CmdAddr']) -> bool:
        """检查结果矩阵坐标是否相同"""
        if other is None:
            return False
        return self.matrix_coord == other.matrix_coord
    
    def same_scale_pos(self, other: Optional['CmdAddr']) -> bool:
        """检查缩放位置是否相同"""
        if other is None:
            return False
        return self.scale_positions == other.scale_positions
    
    def same_k_scale_group(self, other: Optional['CmdAddr']) -> bool:
        """检查K维度缩放组是否相同"""
        if other is None:
            return False
        return (getattr(self, 'A_scale_group_b_id', None) == getattr(other, 'A_scale_group_b_id', None) and 
                getattr(self, 'B_scale_group_b_id', None) == getattr(other, 'B_scale_group_b_id', None))
    
    # === 便捷方法 - 针对特定使用场景的组合操作 ===
    
    def get_current_scale_data(self) -> Tuple[Tuple[Optional[int], Optional[int]], Tuple[Optional[int], Optional[int]]]:
        """获取当前需要的缩放数据 - 用于缓冲区检查"""
        return self.A_scale_pos, self.B_scale_pos
    
    def get_current_b_data(self) -> Tuple[int, int, Optional[int], Optional[int]]:
        """获取当前B数据信息 - 用于W-only模式"""
        return (self.B_row, self.B_col, 
                getattr(self, 'B_scale_row', None), 
                getattr(self, 'B_scale_col', None))
    
    def get_current_a_data(self) -> Tuple[int, int, Optional[int], Optional[int]]:
        """获取当前A数据信息"""
        return (self.A_row, self.A_col, 
                getattr(self, 'A_scale_row', None), 
                getattr(self, 'A_scale_col', None))
    
    def needs_new_partial_result(self, other: Optional['CmdAddr'], 
                               max_partial_num: int, 
                               current_k_scale_count: int) -> bool:
        """
        检查是否需要新的部分结果处理
        
        Args:
            other: 下一个命令
            max_partial_num: 最大部分结果数量
            current_k_scale_count: 当前K缩放计数
            
        Returns:
            bool: 是否需要处理部分结果
        """
        if other is None:  # 最后一个命令
            return True
        
        # 结果位置改变或达到最大容量
        if (not self.same_result_coord(other) or 
            current_k_scale_count >= max_partial_num):
            return True
        
        return False
    
    # === 兼容性方法 - 保持与原始数组访问的兼容性 ===
    
    def __getitem__(self, index: int) -> Any:
        """保持与原始数组访问的兼容性: cmd[0] 等价于 cmd.raw[0]"""
        return self.raw[index]
    
    def __len__(self) -> int:
        """返回原始数组长度"""
        return len(self.raw)
    
    def __str__(self) -> str:
        """简洁的字符串表示"""
        return (f"CmdAddr(A{self.A_pos}, B{self.B_pos}, "
                f"Result{self.result_pos}, coord{self.matrix_coord})")
    
    def __repr__(self) -> str:
        """详细的字符串表示，用于调试"""
        return (f"CmdAddr(A{self.A_pos}, B{self.B_pos}, Result{self.result_pos}, "
                f"A_scale{self.A_scale_pos}, B_scale{self.B_scale_pos}, "
                f"coord{self.matrix_coord})")


class CmdAddrList:
    """
    命令地址列表包装器
    提供批量操作和迭代功能，支持不同的地址格式
    """
    
    def __init__(self, raw_cmd_addr_list: List[List[Any]], 
                 format_config: Optional[Dict[str, int]] = None):
        """
        初始化命令地址列表
        
        Args:
            raw_cmd_addr_list: 原始命令地址列表
            format_config: 格式配置，默认使用 DEFAULT_FORMAT
        """
        self.raw_list = raw_cmd_addr_list
        self.format = format_config or CmdAddrConfig.DEFAULT_FORMAT
        self._wrapped_list = [CmdAddr(addr, self.format) for addr in raw_cmd_addr_list]
    
    def __len__(self) -> int:
        return len(self._wrapped_list)
    
    def __getitem__(self, index: int) -> CmdAddr:
        return self._wrapped_list[index]
    
    def __iter__(self):
        return iter(self._wrapped_list)
    
    def get_pair(self, index: int) -> Tuple[CmdAddr, Optional[CmdAddr]]:
        """
        获取当前和下一个命令的配对
        
        Args:
            index: 当前索引
            
        Returns:
            Tuple: (当前命令, 下一个命令或None)
        """
        current = self._wrapped_list[index]
        next_cmd = self._wrapped_list[index + 1] if index + 1 < len(self._wrapped_list) else None
        return current, next_cmd
    
    # === 轨迹提取方法 ===
    
    def extract_scale_trace(self) -> List[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
        """提取完整缩放轨迹"""
        trace = []
        for cmd in self._wrapped_list:
            current_trace = (cmd.A_scale_pos[0], cmd.A_scale_pos[1], 
                           cmd.B_scale_pos[0], cmd.B_scale_pos[1])
            if not trace or current_trace != trace[-1]:
                trace.append(current_trace)
        return trace
    
    def extract_a_scale_trace(self) -> List[Tuple[Optional[int], Optional[int]]]:
        """提取A缩放轨迹"""
        trace = []
        for cmd in self._wrapped_list:
            if cmd.A_scale_pos[0] is not None and (not trace or cmd.A_scale_pos != trace[-1]):
                trace.append(cmd.A_scale_pos)
        return trace
    
    def extract_b_scale_trace(self) -> List[Tuple[Optional[int], Optional[int]]]:
        """提取B缩放轨迹"""
        trace = []
        for cmd in self._wrapped_list:
            if cmd.B_scale_pos[0] is not None and (not trace or cmd.B_scale_pos != trace[-1]):
                trace.append(cmd.B_scale_pos)
        return trace
    
    def extract_b_data_trace(self) -> List[Tuple[int, int, Optional[int], Optional[int]]]:
        """提取B数据轨迹 (用于W-only模式)"""
        trace = []
        for cmd in self._wrapped_list:
            current_trace = cmd.get_current_b_data()
            if not trace or current_trace != trace[-1]:
                trace.append(current_trace)
        return trace
    
    def extract_a_data_trace(self) -> List[Tuple[int, int, Optional[int], Optional[int]]]:
        """提取A数据轨迹"""
        trace = []
        for cmd in self._wrapped_list:
            current_trace = cmd.get_current_a_data()
            if not trace or current_trace != trace[-1]:
                trace.append(current_trace)
        return trace


# === 格式迁移辅助函数 ===

def migrate_cmd_addr_usage(old_code_snippet: str, 
                          field_mapping: Dict[int, str]) -> str:
    """
    辅助函数：帮助迁移旧的索引访问代码到新的属性访问
    
    Args:
        old_code_snippet: 旧代码片段
        field_mapping: 索引到字段名的映射
        
    Returns:
        str: 迁移后的代码片段
    """
    import re
    
    # 替换 cmd_addr[数字] 为 cmd.字段名
    def replace_index_access(match):
        index = int(match.group(1))
        if index in field_mapping:
            return f"cmd.{field_mapping[index]}"
        return match.group(0)  # 保持原样
    
    # 匹配 cmd_addr[数字] 模式
    pattern = r'cmd_addr\[(\d+)\]'
    return re.sub(pattern, replace_index_access, old_code_snippet)


# === 使用示例和测试 ===

def usage_example():
    """使用示例和测试"""
    print("=== CmdAddr 使用示例 ===")
    
    # 模拟原始cmd_addr_list数据
    raw_cmd_addr_list = [
        [10, 20, 30, 40, 50, 60, 1, 2, 3, 4, 100, 200, 300, 400, 5, 6],
        [11, 21, 31, 41, 51, 61, 1, 2, 3, 4, 101, 201, 301, 401, 5, 6],
        [12, 22, 32, 42, 52, 62, 1, 3, 3, 5, 102, 202, 302, 402, 6, 7],
    ]
    
    # 1. 基本使用 - 默认格式
    print("\n1. 基本属性访问:")
    cmd_list = CmdAddrList(raw_cmd_addr_list)
    
    for i, cmd in enumerate(cmd_list):
        print(f"  命令{i}: A{cmd.A_pos}, B{cmd.B_pos}, Result{cmd.result_pos}")
        print(f"    缩放: A_scale{cmd.A_scale_pos}, B_scale{cmd.B_scale_pos}")
        print(f"    原始坐标: {cmd.matrix_coord}")
    
    # 2. 命令比较
    print("\n2. 命令比较:")
    for i in range(len(cmd_list) - 1):
        current, next_cmd = cmd_list.get_pair(i)
        print(f"  命令{i} vs 命令{i+1}:")
        print(f"    列相同: {current.same_AB_col(next_cmd)}")
        print(f"    行相同: {current.same_AB_row(next_cmd)}")
        print(f"    结果坐标相同: {current.same_result_coord(next_cmd)}")
    
    # 3. 轨迹提取
    print("\n3. 轨迹提取:")
    print(f"  A缩放轨迹: {cmd_list.extract_a_scale_trace()}")
    print(f"  B缩放轨迹: {cmd_list.extract_b_scale_trace()}")
    print(f"  B数据轨迹: {cmd_list.extract_b_data_trace()}")
    
    # 4. 兼容性测试
    print("\n4. 兼容性测试:")
    cmd = cmd_list[0]
    print(f"  属性访问: cmd.A_row = {cmd.A_row}")
    print(f"  索引访问: cmd[0] = {cmd[0]}")
    print(f"  两者相等: {cmd.A_row == cmd[0]}")
    
    # 5. 格式扩展示例
    print("\n5. 格式扩展示例:")
    # 假设使用扩展格式（需要相应的数据）
    extended_data = [10, 20, 30, 40, 50, 60, 999, 1, 2, 3, 4, 100, 200, 300, 400, 5, 6, 10, "meta"]
    cmd_extended = CmdAddr(extended_data, CmdAddrConfig.EXTENDED_FORMAT)
    print(f"  扩展格式访问: A{cmd_extended.A_pos}, timestamp={getattr(cmd_extended, 'timestamp', 'N/A')}")


def test_format_migration():
    """测试格式迁移功能"""
    print("\n=== 格式迁移测试 ===")
    
    # 旧代码示例
    old_code = """
    current_a_scale = (cmd_addr[10], cmd_addr[11])
    if cmd_addr[0] != next_cmd_addr[0]:
        print(f"A_row changed: {cmd_addr[0]} -> {next_cmd_addr[0]}")
    """
    
    # 定义字段映射
    field_mapping = {
        0: 'A_row', 1: 'A_col', 2: 'B_row', 3: 'B_col',
        10: 'A_scale_row', 11: 'A_scale_col'
    }
    
    # 迁移代码
    new_code = migrate_cmd_addr_usage(old_code, field_mapping)
    print("迁移前:")
    print(old_code)
    print("迁移后:")
    print(new_code)


if __name__ == "__main__":
    usage_example()
    test_format_migration()