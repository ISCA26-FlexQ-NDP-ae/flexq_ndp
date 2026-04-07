#!/usr/bin/env python3

import math

class LoopGenerator:
    """重新设计的循环生成器，使用正确的block_sizes数据结构"""
    
    def __init__(self, m_size, n_size, k_size):
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        
        # 正确的数据结构：存储用户期望的实际块大小
        self.block_sizes = {
            'm': {'block0': 1, 'block1': 1},  # 默认：整个维度作为一个块，每个元素作为子块
            'n': {'block0': 1, 'block1': 1},
            'k': {'block0': 1, 'block1': 1}
        }
        
        # 默认循环顺序
        self.loop_order = ['m0', 'm1', 'm2', 'n0', 'n1', 'n2', 'k0', 'k1', 'k2']
    
    def set_block_sizes(self, **blocks):
        """设置块大小（用户友好的API）"""
        for key, size in blocks.items():
            if size <= 0:
                raise ValueError(f"Block size must be positive: {size}")
                
            if '_block0' in key:
                dim = key.split('_')[0]
                if dim not in ['m', 'n', 'k']:
                    raise ValueError(f"Invalid dimension: {dim}")
                
                total_size = getattr(self, f"{dim}_size")
                if size > total_size:
                    raise ValueError(f"{dim}_block0 ({size}) cannot be larger than {dim}_size ({total_size})")
                
                # 检查是否同时提供了block1
                block1_key = f"{dim}_block1"
                if block1_key in blocks:
                    block0_size = size
                    block1_size = blocks[block1_key]
                    
                    if block1_size > block0_size:
                        raise ValueError(f"{dim}_block1 ({block1_size}) cannot be larger than {dim}_block0 ({block0_size})")
                    
                    # 直接存储用户期望的块大小
                    self.block_sizes[dim]['block0'] = block0_size
                    self.block_sizes[dim]['block1'] = block1_size
                    
                    # print(f"[{dim}] 设置块大小: block0={block0_size}, block1={block1_size}")
                else:
                    # 只设置第一级
                    self.block_sizes[dim]['block0'] = size
                    self.block_sizes[dim]['block1'] = 1
                    # print(f"[{dim}] 单层块: block_size={size}")
                    
            elif '_block1' in key:
                # block1在处理block0时已经处理
                continue
            else:
                # 简化模式
                if key not in ['m', 'n', 'k']:
                    raise ValueError(f"Invalid dimension: {key}")
                self.block_sizes[key]['block0'] = size
                self.block_sizes[key]['block1'] = 1
                # print(f"[{key}] 简化设置: block_size={size}")
    
    def _calculate_loop_ranges(self):
        """根据block_sizes计算循环范围"""
        loop_ranges = {}
        
        for dim in ['m', 'n', 'k']:
            total_size = getattr(self, f"{dim}_size")
            block0 = self.block_sizes[dim]['block0'] 
            block1 = self.block_sizes[dim]['block1']
            
            # 计算三层循环的范围
            # 外层循环(dim0): 按block0大小分割总维度
            outer_range = math.ceil(total_size / block0)
            
            # 中层循环(dim1): 每个block0内部按block1分割  
            middle_range = math.ceil(block0 / block1)
            
            # 内层循环(dim2): 每个block1的大小
            inner_range = block1
            
            loop_ranges[f"{dim}0"] = outer_range
            loop_ranges[f"{dim}1"] = middle_range  
            loop_ranges[f"{dim}2"] = inner_range
            
            # print(f"[{dim}] 循环范围: {dim}0={outer_range}, {dim}1={middle_range}, {dim}2={inner_range}")
            # print(f"[{dim}] 切分方案: 总大小{total_size} -> {outer_range}个{block0}大小的块 -> 每块再分{middle_range}个{block1}大小的子块")
        
        return loop_ranges
    
    def _should_break_loop(self, dim_name, outer_idx, middle_idx, inner_idx):
        """检查是否应该跳出当前循环"""
        total_size = getattr(self, f"{dim_name}_size")
        block0 = self.block_sizes[dim_name]['block0']
        block1 = self.block_sizes[dim_name]['block1']
        
        # 检查1: 整体索引不能超出总大小
        id_total = block0 * outer_idx + block1 * middle_idx + inner_idx
        if id_total >= total_size:
            return True
            
        # 检查2: 次一层索引不能超出block0大小
        id_in_block0 = block1 * middle_idx + inner_idx
        if id_in_block0 >= block0:
            return True
            
        return False
    
    def _generate_loops(self):
        """按照loop_order顺序生成循环索引"""
        # loop_ranges = self._calculate_loop_ranges()
        
        # 使用递归方法按照loop_order顺序生成索引
        for indices in self._generate_nested_loops(0, {}):
            # 计算最终的m,n,k索引
            final_m = self._calculate_final_index('m', (
                indices.get('m0', 0), 
                indices.get('m1', 0), 
                indices.get('m2', 0)
            ))
            final_n = self._calculate_final_index('n', (
                indices.get('n0', 0), 
                indices.get('n1', 0), 
                indices.get('n2', 0)
            ))
            final_k = self._calculate_final_index('k', (
                indices.get('k0', 0), 
                indices.get('k1', 0), 
                indices.get('k2', 0)
            ))
            
            yield {'m': final_m, 'n': final_n, 'k': final_k}
    
    
    def _calculate_final_index(self, dim_name, indices):
        """计算给定维度的最终索引"""
        outer_idx, middle_idx, inner_idx = indices
        block0 = self.block_sizes[dim_name]['block0']
        block1 = self.block_sizes[dim_name]['block1']
        return outer_idx * block0 + middle_idx * block1 + inner_idx
    
    def generate(self):
        """生成有效的循环索引"""
        sizes = {'m': self.m_size, 'n': self.n_size, 'k': self.k_size}
        
        for indices in self._generate_loops():
            # 边界检查
            if all(indices[dim] < sizes[dim] for dim in ['m', 'n', 'k']):
                yield indices['m'], indices['n'], indices['k']
    
    def set_loop_order(self, order):
        """设置循环顺序"""
        expected_dims = {'m0', 'n0', 'k0', 'm1', 'n1', 'k1', 'm2', 'n2', 'k2'}
        if len(order) != 9 or set(order) != expected_dims:
            raise ValueError("Order must contain exactly 'm0', 'n0', 'k0', 'm1', 'n1', 'k1', 'm2', 'n2', 'k2'")
        
        # 检查同一维度的层次约束
        for dim in ['m', 'n', 'k']:
            pos0 = order.index(f'{dim}0')
            pos1 = order.index(f'{dim}1') 
            pos2 = order.index(f'{dim}2')
            
            if pos0 >= pos1:
                raise ValueError(f"Loop {dim}0 must come before {dim}1")
            if pos1 >= pos2:
                raise ValueError(f"Loop {dim}1 must come before {dim}2")
        
        self.loop_order = order
        # print(f"[o] 循环顺序: {self.loop_order}")

    def count_total_iterations(self):
        """计算总的有效迭代次数"""
        count = self.m_size * self.n_size * self.k_size
        return count
    
    
    def _generate_range_worker(self, worker_args):
        """
        工作进程函数：生成指定范围的循环索引
        
        Args:
            worker_args: (outermost_dim, start_idx, end_idx, loop_order, block_sizes, m_size, n_size, k_size)
        
        Returns:
            List of (m, n, k) tuples for the assigned range
        """
        outermost_dim, start_idx, end_idx, loop_order, block_sizes, m_size, n_size, k_size = worker_args
        
        # 重建对象状态
        temp_gen = LoopGenerator(m_size, n_size, k_size)
        temp_gen.block_sizes = block_sizes
        temp_gen.loop_order = loop_order
        
        results = []
        
        # 找到最外层维度在循环顺序中的位置
        outermost_pos = loop_order.index(outermost_dim)
        
        # 生成指定范围的索引
        for outer_idx in range(start_idx, end_idx):
            # 递归生成其他维度的所有组合
            for indices in temp_gen._generate_nested_loops(outermost_pos + 1, {outermost_dim: outer_idx}):
                # 计算最终的m,n,k索引
                final_m = temp_gen._calculate_final_index('m', (indices['m0'], indices['m1'], indices['m2']))
                final_n = temp_gen._calculate_final_index('n', (indices['n0'], indices['n1'], indices['n2']))
                final_k = temp_gen._calculate_final_index('k', (indices['k0'], indices['k1'], indices['k2']))
                
                # 边界检查
                if final_m < m_size and final_n < n_size and final_k < k_size:
                    results.append((final_m, final_n, final_k))
        
        return results
    
    def _generate_nested_loops(self, start_pos, fixed_indices):
        """
        从指定位置开始递归生成剩余维度的循环索引
        
        Args:
            start_pos: 从循环顺序的哪个位置开始
            fixed_indices: 已经固定的索引值
        
        Yields:
            完整的9维索引字典
        """
        if start_pos >= len(self.loop_order):
            yield fixed_indices
            return
            
        loop_dim = self.loop_order[start_pos]
        dim_name = loop_dim[0]  # 'm', 'n', 'k'
        level = int(loop_dim[1])  # 0, 1, 2
        
        loop_ranges = self._calculate_loop_ranges()
        
        for idx in range(loop_ranges[loop_dim]):
            # 检查边界条件
            current_indices = fixed_indices.copy()
            current_indices[loop_dim] = idx
            
            # 如果是最内层循环(level=2)，需要检查边界
            if level == 2:
                outer_idx = current_indices.get(f'{dim_name}0', 0)
                middle_idx = current_indices.get(f'{dim_name}1', 0)
                inner_idx = idx
                
                if self._should_break_loop(dim_name, outer_idx, middle_idx, inner_idx):
                    continue
            
            # 递归生成下一个维度
            yield from self._generate_nested_loops(start_pos + 1, current_indices)

    def generate_parallel(self, num_processes=None):
        """
        真正的并行版本的generate函数
        
        Args:
            num_processes: 使用的进程数。如果为None，使用min(cpu_count(), 8)
                          设置为1禁用多进程处理
        
        Returns:
            Generator yielding (m, n, k) tuples in the same order as generate()
        """
        import multiprocessing as mp
        from functools import partial
        import math
        
        if num_processes == 1:
            # 单进程回退
            yield from self.generate()
            return
            
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)
        
        # 计算最外层循环的范围
        loop_ranges = self._calculate_loop_ranges()
        outermost_dim = self.loop_order[0]
        total_outer_range = loop_ranges[outermost_dim]
        
        # 限制进程数不超过工作量
        actual_processes = min(num_processes, total_outer_range, mp.cpu_count())
        
        if actual_processes <= 1:
            yield from self.generate()
            return
        
        # 分配工作：将最外层循环的范围分配给各个进程
        ranges_per_process = math.ceil(total_outer_range / actual_processes)
        work_args = []
        
        for i in range(actual_processes):
            start_idx = i * ranges_per_process
            end_idx = min((i + 1) * ranges_per_process, total_outer_range)
            
            if start_idx < end_idx:
                work_args.append((
                    outermost_dim,
                    start_idx, 
                    end_idx,
                    self.loop_order.copy(),
                    self.block_sizes.copy(),
                    self.m_size,
                    self.n_size, 
                    self.k_size
                ))
        
        # 使用多进程池执行并行计算
        try:
            with mp.Pool(processes=actual_processes) as pool:
                # 创建绑定self的worker函数
                worker_func = partial(self._generate_range_worker_static, 
                                    loop_order=self.loop_order,
                                    block_sizes=self.block_sizes,
                                    m_size=self.m_size,
                                    n_size=self.n_size,
                                    k_size=self.k_size)
                
                # 并行计算各个工作块
                all_results = pool.map(worker_func, work_args)
                
                # 按顺序输出结果
                for result_batch in all_results:
                    for item in result_batch:
                        yield item
                        
        except Exception as e:
            # 如果并行化失败，回退到串行版本
            print(f"并行化失败，回退到串行版本: {e}")
            yield from self.generate()
    
    @staticmethod
    def _generate_range_worker_static(worker_args, loop_order, block_sizes, m_size, n_size, k_size):
        """
        静态工作进程函数（解决pickle问题）
        """
        outermost_dim, start_idx, end_idx = worker_args[:3]
        
        # 重建临时生成器对象
        temp_gen = LoopGenerator(m_size, n_size, k_size)
        temp_gen.block_sizes = block_sizes
        temp_gen.loop_order = loop_order
        
        results = []
        
        # 找到最外层维度在循环顺序中的位置
        outermost_pos = loop_order.index(outermost_dim)
        
        # 为指定的最外层范围生成所有组合
        for outer_idx in range(start_idx, end_idx):
            # 创建固定最外层索引的字典
            fixed_indices = {outermost_dim: outer_idx}
            
            # 生成剩余维度的所有有效组合
            try:
                for indices in temp_gen._generate_nested_loops(outermost_pos + 1, fixed_indices):
                    # 计算最终的m,n,k索引
                    final_m = temp_gen._calculate_final_index('m', (
                        indices.get('m0', 0), 
                        indices.get('m1', 0), 
                        indices.get('m2', 0)
                    ))
                    final_n = temp_gen._calculate_final_index('n', (
                        indices.get('n0', 0), 
                        indices.get('n1', 0), 
                        indices.get('n2', 0)
                    ))
                    final_k = temp_gen._calculate_final_index('k', (
                        indices.get('k0', 0), 
                        indices.get('k1', 0), 
                        indices.get('k2', 0)
                    ))
                    
                    # 边界检查
                    if final_m < m_size and final_n < n_size and final_k < k_size:
                        results.append((final_m, final_n, final_k))
                        
            except Exception as e:
                # 如果生成失败，记录错误但不中断整个过程
                print(f"Worker生成索引失败 outer_idx={outer_idx}: {e}")
                continue
        
        return results

def test_new_design():
    """测试新的设计"""
    print("=== 测试新的设计 ===\n")
    
    # 用户的具体场景
    print("用户场景: k=5120, block0=4096, block1=1703")
    gen = LoopGenerator(2, 3, 5120)
    gen.set_block_sizes(k_block0=4096, k_block1=1703)
    gen.set_loop_order(['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2'])
    
    print(f"\n块配置: {gen.block_sizes}")
    
    # 测试前20个迭代
    print(f"\n前20个(m,n,k):", end=" ")
    count = 0
    k_values = []
    for m, n, k in gen.generate():
        if count < 20:
            print(f"({m},{n},{k})", end=" ")
        k_values.append(k)
        count += 1
        if count % 10 == 0 and count <= 20:
            print()
        if count >= 100:  # 限制测试数量
            break
    
    print(f"\n前100个k值范围: [{min(k_values)}, {max(k_values)}]")
    
    # 检查总迭代数
    total = sum(1 for _ in gen.generate())
    expected = 2 * 3 * 5120
    print(f"总迭代数: {total}, 期望: {expected}, 正确: {total == expected}")

def test_generate_parallel():
    """测试generate_parallel函数"""
    print(f"\n=== 测试 generate_parallel 函数 ===\n")
    
    # 创建一个较小的测试场景
    gen = LoopGenerator(2, 2, 20)
    gen.set_block_sizes(k_block0=10, k_block1=5)
    gen.set_loop_order(['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2'])
    
    print(f"测试配置: m=2, n=2, k=20")
    print(f"k块配置: block0=10, block1=5")
    print(f"循环顺序: {' -> '.join(gen.loop_order)}")
    
    # 测试串行版本
    print(f"\n串行版本结果:")
    serial_results = list(gen.generate())
    print(f"总迭代数: {len(serial_results)}")
    print(f"前10个: {serial_results[:10]}")
    
    # 测试并行版本
    print(f"\n并行版本结果:")
    parallel_results = list(gen.generate_parallel(num_processes=2))
    print(f"总迭代数: {len(parallel_results)}")
    print(f"前10个: {parallel_results[:10]}")
    
    # 验证结果一致性
    print(f"\n一致性检查:")
    print(f"结果长度相同: {len(serial_results) == len(parallel_results)}")
    
    if len(serial_results) == len(parallel_results):
        # 检查结果是否相同（忽略顺序）
        serial_set = set(serial_results)
        parallel_set = set(parallel_results)
        print(f"结果内容相同: {serial_set == parallel_set}")
        
        if serial_set == parallel_set:
            print(f"✅ 并行函数工作正常")
        else:
            print(f"❌ 并行函数结果不一致")
            diff = serial_set.symmetric_difference(parallel_set)
            print(f"差异: {list(diff)[:10]}...")
    else:
        print(f"❌ 并行函数迭代数不匹配")

def test_user_scenario_with_parallel():
    """测试用户场景的并行版本"""
    print(f"\n=== 用户场景并行测试 ===\n")
    
    gen = LoopGenerator(2, 3, 5120)
    gen.set_block_sizes(k_block0=4096, k_block1=1703)
    gen.set_loop_order(['m0', 'm1', 'm2', 'k0', 'k1', 'n0', 'n1', 'n2', 'k2'])
    
    print(f"用户场景: M=2, N=3, K=5120")
    print(f"K配置: block0=4096, block1=1703")
    
    # 测试并行版本是否能正常工作（不验证具体结果，因为太多了）
    print(f"\n测试并行版本可用性:")
    
    try:
        parallel_gen = gen.generate_parallel(num_processes=2)
        first_10 = []
        for i, (m, n, k) in enumerate(parallel_gen):
            if i < 10:
                first_10.append((m, n, k))
            else:
                break
        
        print(f"✅ 并行版本可以正常生成")
        print(f"前10个结果: {first_10}")
        
        # 检查k值范围
        k_values = [k for _, _, k in first_10]
        print(f"k值范围: [{min(k_values)}, {max(k_values)}]")
        print(f"k值都在有效范围内: {all(0 <= k < 5120 for k in k_values)}")
        
    except Exception as e:
        print(f"❌ 并行版本出错: {e}")

if __name__ == "__main__":
    test_new_design()
    test_generate_parallel()
    test_user_scenario_with_parallel()