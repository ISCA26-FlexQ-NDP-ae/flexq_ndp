from tools import *

"""
Corner Case 的分析，可能会使能占用行数 & 换行数的减少（拼到一起后，通过 Snake 遍历）
Corner Case 重要的原因：容量问题 & 精度变低，同样的矩阵大小，产生 Corner Case 的可能性会增加
Corner Case 可能会影响 Scale Buffer 更换的时机，是否和换行保持一致会更好？
"""
def corner_case_analysis(baseline, operand_A_info, operand_B_info, result_precision, log_file=None):
    compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping = baseline
    if log_file is not None:
        log_print = lambda x: print(x, file=log_file)
    else:
        log_print = lambda x: print(x)
    log_print(f"compute_level: {compute_level}, pu_num: {pu_num}, partition: {partition}, A_mapping: {A_mapping}, B_mapping: {B_mapping}, result_mapping: {result_mapping}")
    # analyse corner case of baseline's A / B / Result Mapping
    log_print("\n=== Corner Case Analysis ===")
    
    # A Matrix Corner Case Analysis
    A_data_mapping = A_mapping['data_mapping']
    log_print(f"\n--- A Matrix Corner Case Analysis ---")
    log_print(f"A matrix size: {mm_size_per_pu[0]} x {mm_size_per_pu[1]}")
    
    # 获取 A 矩阵的映射参数
    a_size = mm_size_per_pu[0]
    a_k_size = mm_size_per_pu[1] 
    a_row_num = A_data_mapping['a_row_num']
    a_col_per_row = A_data_mapping['a_col_per_row'] 
    a_corner_col_per_row = A_data_mapping['a_corner_col_per_row']
    a_b_row_num = A_data_mapping['b_row_num']
    a_b_col_per_row = A_data_mapping['b_col_per_row']
    a_b_corner_col_per_row = A_data_mapping['b_corner_col_per_row']
    
    # 计算 A 矩阵的理论利用率
    a_theoretical_utilization = (a_size * a_k_size * operand_A_info['matrix_precision']) / (a_row_num * a_b_row_num * SimConfig.co * SimConfig.co_w)
    log_print(f"A theoretical_utilization: {a_theoretical_utilization:.4f} = ({a_size} * {a_k_size}) elements * {operand_A_info['matrix_precision']} bits / ( ({a_row_num} * {a_b_row_num}) rows * {SimConfig.co} cols * {SimConfig.co_w} bits)")
    
    # A 矩阵的 corner case 分析
    a_b_corner = (a_row_num-1) * (a_col_per_row * a_b_corner_col_per_row) / SimConfig.co
    a_a_corner = (a_b_row_num-1) * (a_b_col_per_row * a_corner_col_per_row) / SimConfig.co  
    a_ab_corner = 1 * (a_corner_col_per_row * a_b_corner_col_per_row) / SimConfig.co
    
    log_print(f"A b_corner: {a_row_num-1} * ({a_col_per_row} * {a_b_corner_col_per_row}) / {SimConfig.co} = {a_b_corner:.4f}")
    log_print(f"A a_corner: {a_b_row_num-1} * ({a_b_col_per_row} * {a_corner_col_per_row}) / {SimConfig.co} = {a_a_corner:.4f}")
    log_print(f"A a_b_corner: 1 * ({a_corner_col_per_row} * {a_b_corner_col_per_row}) / {SimConfig.co} = {a_ab_corner:.4f}")
    
    # B Matrix Corner Case Analysis
    B_data_mapping = B_mapping['data_mapping']
    log_print(f"\n--- B Matrix Corner Case Analysis ---")
    log_print(f"B matrix size: {mm_size_per_pu[2]} x {mm_size_per_pu[1]}")
    
    # 获取 B 矩阵的映射参数
    b_size = mm_size_per_pu[2]
    b_k_size = mm_size_per_pu[1]
    b_row_num = B_data_mapping['a_row_num']
    b_col_per_row = B_data_mapping['a_col_per_row']
    b_corner_col_per_row = B_data_mapping['a_corner_col_per_row']
    b_b_row_num = B_data_mapping['b_row_num']
    b_b_col_per_row = B_data_mapping['b_col_per_row']
    b_b_corner_col_per_row = B_data_mapping['b_corner_col_per_row']
    
    # 计算 B 矩阵的理论利用率
    b_theoretical_utilization = (b_size * b_k_size * operand_B_info['matrix_precision']) / (b_row_num * b_b_row_num * SimConfig.co * SimConfig.co_w)
    log_print(f"B theoretical_utilization: {b_theoretical_utilization:.4f} = ({b_size} * {b_k_size}) elements * {operand_B_info['matrix_precision']} bits / ( ({b_row_num} * {b_b_row_num}) rows * {SimConfig.co} cols * {SimConfig.co_w} bits)")
    
    # B 矩阵的 corner case 分析
    b_b_corner = (b_row_num-1) * (b_col_per_row * b_b_corner_col_per_row) / SimConfig.co
    b_a_corner = (b_b_row_num-1) * (b_b_col_per_row * b_corner_col_per_row) / SimConfig.co
    b_ab_corner = 1 * (b_corner_col_per_row * b_b_corner_col_per_row) / SimConfig.co
    
    log_print(f"B b_corner: {b_row_num-1} * ({b_col_per_row} * {b_b_corner_col_per_row}) / {SimConfig.co} = {b_b_corner:.4f}")
    log_print(f"B a_corner: {b_b_row_num-1} * ({b_b_col_per_row} * {b_corner_col_per_row}) / {SimConfig.co} = {b_a_corner:.4f}")
    log_print(f"B a_b_corner: 1 * ({b_corner_col_per_row} * {b_b_corner_col_per_row}) / {SimConfig.co} = {b_ab_corner:.4f}")
    
    # Result Matrix Corner Case Analysis
    result_data_mapping = result_mapping['data_mapping']
    log_print(f"\n--- Result Matrix Corner Case Analysis ---")
    log_print(f"Result matrix size: {mm_size_per_pu[0]} x {mm_size_per_pu[2]}")
    
    # 获取 Result 矩阵的映射参数
    result_m_size = mm_size_per_pu[0]  
    result_n_size = mm_size_per_pu[2]
    result_row_num = result_data_mapping['a_row_num']
    result_col_per_row = result_data_mapping['a_col_per_row']
    result_corner_col_per_row = result_data_mapping['a_corner_col_per_row']
    result_b_row_num = result_data_mapping['b_row_num']
    result_b_col_per_row = result_data_mapping['b_col_per_row']
    result_b_corner_col_per_row = result_data_mapping['b_corner_col_per_row']
    
    # 计算 Result 矩阵的理论利用率
    result_theoretical_utilization = (result_m_size * result_n_size * result_precision) / (result_row_num * result_b_row_num * SimConfig.co * SimConfig.co_w)
    log_print(f"Result theoretical_utilization: {result_theoretical_utilization:.4f} = ({result_m_size} * {result_n_size}) elements * {result_precision} bits / ( ({result_row_num} * {result_b_row_num}) rows * {SimConfig.co} cols * {SimConfig.co_w} bits)")
    
    # Result 矩阵的 corner case 分析
    result_b_corner = (result_row_num-1) * (result_col_per_row * result_b_corner_col_per_row) / SimConfig.co
    result_a_corner = (result_b_row_num-1) * (result_b_col_per_row * result_corner_col_per_row) / SimConfig.co
    result_ab_corner = 1 * (result_corner_col_per_row * result_b_corner_col_per_row) / SimConfig.co
    
    log_print(f"Result b_corner: {result_row_num-1} * ({result_col_per_row} * {result_b_corner_col_per_row}) / {SimConfig.co} = {result_b_corner:.4f}")
    log_print(f"Result a_corner: {result_b_row_num-1} * ({result_b_col_per_row} * {result_corner_col_per_row}) / {SimConfig.co} = {result_a_corner:.4f}")
    log_print(f"Result a_b_corner: 1 * ({result_corner_col_per_row} * {result_b_corner_col_per_row}) / {SimConfig.co} = {result_ab_corner:.4f}")
    
    log_print(f"=== End Corner Case Analysis ===\n")