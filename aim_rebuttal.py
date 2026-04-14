"""
aim baseline 考虑 scale 的实现
"""
from backend import *
from midend import *
from sim import sim
from tools import *
import tools.logger
import argparse
import os
import multiprocessing as mp
import yaml
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# 允许 safe_load 解析由 yaml.dump 写出的 !!python/tuple（仅反序列化为 tuple，安全）
def _yaml_safe_tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', _yaml_safe_tuple_constructor, yaml.SafeLoader)

# tqdm.tqdm = tqdm_replacement
"""
MARK: Tool functions
"""
def profile_test(test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info, buffer_specify=None, showalldetail=False):
    if buffer_specify is not None:
        print(f"Buffer Specify: {buffer_specify}")
    null_log = open("/dev/null", "w")
    base_latency = test_func(batch_size=batch_size, result_precision=result_precision, accumulation_precision=accumulation_precision,
                    operand_A_info=operand_A_info, operand_B_info=operand_B_info, log_file = null_log, silent=True, buffer_specify=buffer_specify) # log_file=None
    # print("="*10, "[+ DATA READ]")
    data_read_latency = test_func(batch_size=batch_size, result_precision=result_precision, accumulation_precision=accumulation_precision,
                    operand_A_info=operand_A_info, operand_B_info=operand_B_info,
                    consider_data_read=True, log_file = null_log, silent=True, buffer_specify=buffer_specify) - base_latency
    # print("="*10, "[+ DATA READ + SCALE READ]")
    scale_read_latency = test_func(batch_size=batch_size, result_precision=result_precision, accumulation_precision=accumulation_precision,
                    operand_A_info=operand_A_info, operand_B_info=operand_B_info,
                    consider_data_read=True, consider_scale_buffer=True, log_file = null_log, silent=True, buffer_specify=buffer_specify) - base_latency - data_read_latency
    # print("="*10, "[+ DATA READ + SCALE READ + RESULT]")
    result_latency = test_func(batch_size=batch_size, result_precision=result_precision, accumulation_precision=accumulation_precision,
                    operand_A_info=operand_A_info, operand_B_info=operand_B_info,
                    consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, breakdown_analysis=True, silent=True, buffer_specify=buffer_specify) - base_latency - data_read_latency - scale_read_latency
    print(f"Total Latency: {base_latency + data_read_latency + scale_read_latency + result_latency}")
    print(f"Base Latency: {base_latency} ({base_latency / (base_latency + data_read_latency + scale_read_latency + result_latency) * 100:.2f}%)")
    print(f"Data Read Latency: {data_read_latency} ({data_read_latency / (base_latency + data_read_latency + scale_read_latency + result_latency) * 100:.2f}%)")
    print(f"Scale Read Latency: {scale_read_latency} ({scale_read_latency / (base_latency + data_read_latency + scale_read_latency + result_latency) * 100:.2f}%)")
    print(f"Result Latency: {result_latency} ({result_latency / (base_latency + data_read_latency + scale_read_latency + result_latency) * 100:.2f}%)")

def test_buffer_worker(args):
    buffer_specify, test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info = args
    null_log = open("/dev/null", "w")
    latency = test_func(batch_size=batch_size, result_precision=result_precision, accumulation_precision=accumulation_precision,
            operand_A_info=operand_A_info, operand_B_info=operand_B_info,
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, log_file=null_log, silent=True, buffer_specify=buffer_specify)
    # print(f"Buffer Specify: {buffer_specify}, Latency: {latency}")
    null_log.close()
    return buffer_specify, latency


def _run_baseline_worker(args):
    """子进程：只跑 baseline，返回延迟。args = (test_func, operand_A_info, operand_B_info, buffer_specify, scale_factor)"""
    test_func, operand_A_info, operand_B_info, buffer_specify, scale_factor = args
    test_baseline = partial(test_func, loop_friendly_mapping=True)
    lat = test_baseline(
        batch_size=1, result_precision=16, accumulation_precision=32,
        operand_A_info=operand_A_info, operand_B_info=operand_B_info,
        consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True,
        buffer_specify=buffer_specify, silent=True,
    )
    return lat * scale_factor


def _run_best_worker(args):
    """子进程：只跑最优策略，返回延迟。args = (test_ours_func, operand_A_info, operand_B_info, best_buffer_specify, best_partition_specify)"""
    test_ours_func, operand_A_info, operand_B_info, best_buffer_specify, best_partition_specify, scale_factor = args
    lat = test_ours_func(
        batch_size=1, result_precision=16, accumulation_precision=32,
        operand_A_info=operand_A_info, operand_B_info=operand_B_info,
        consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True,
        buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True,
    )
    return lat * scale_factor

def search_buffer_size(test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info, total_buffer_size, force_b_scale = 0):
    """
    Search for the best buffer size for the given test function.
    """
    if True:
        buffer_specify_list = []
        if force_b_scale > 0:
            if isinstance(total_buffer_size, int):
                for data_buf_size in range(1, total_buffer_size - force_b_scale):
                    scale_a_buf_size = total_buffer_size - data_buf_size - force_b_scale
                    buffer_specify_list.append((data_buf_size, scale_a_buf_size, force_b_scale))
            elif isinstance(total_buffer_size, list) and len(total_buffer_size) == 3:
                for data_buf_size in range(1, total_buffer_size[0]):
                    for scale_a_buf_size in range(1, total_buffer_size[1]):
                        buffer_specify_list.append((data_buf_size, scale_a_buf_size, force_b_scale))
        else:
            if isinstance(total_buffer_size, int):
                for data_buf_size in range(1, total_buffer_size):
                    for scale_a_buf_size in range(1, total_buffer_size - data_buf_size):
                        scale_b_buf_size = total_buffer_size - data_buf_size - scale_a_buf_size
                        buffer_specify_list.append((data_buf_size, scale_a_buf_size, scale_b_buf_size))
            elif isinstance(total_buffer_size, list) and len(total_buffer_size) == 3:
                for data_buf_size in range(1, total_buffer_size[0]):
                    for scale_a_buf_size in range(1, total_buffer_size[1]):
                        for scale_b_buf_size in range(1, total_buffer_size[2]):
                            buffer_specify_list.append((data_buf_size, scale_a_buf_size, scale_b_buf_size))
        
        args_list = [(buffer_specify, test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info) 
                     for buffer_specify in buffer_specify_list]
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            results = list(executor.map(test_buffer_worker, args_list))
        best_lat = float('inf')
        best_buffer_specify = None
        for buffer_specify, latency in results:
            if latency < best_lat:
                best_lat = latency
                best_buffer_specify = buffer_specify
        print(f"Best Latency: {best_lat}, Best Buffer Specify: {best_buffer_specify}")
        return best_lat, best_buffer_specify

def search_buffer_size_with_dequant_buf(test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info, total_buffer_size, force_b_scale = 0, slient=False, answer_buffer_specify=None):
    """
    Search for the best buffer size for the given test function.
    """
    buffer_specify_list = []
    assert isinstance(total_buffer_size, int)
    for data_buf_size in range(1, total_buffer_size):
        for scale_a_buf_size in range(1, total_buffer_size - data_buf_size):
            for scale_b_buf_size in range(1, total_buffer_size - data_buf_size - scale_a_buf_size + 1): # 确保能取到使底下两个都是 0
                # print(total_buffer_size - data_buf_size - scale_a_buf_size - scale_b_buf_size + 1)
                for quant_group_result_buf_size in range(0, total_buffer_size - data_buf_size - scale_a_buf_size - scale_b_buf_size + 1):
                    extra_scale_b_buf_size = total_buffer_size - data_buf_size - scale_a_buf_size - scale_b_buf_size - quant_group_result_buf_size
                    if force_b_scale == 0 or scale_b_buf_size == force_b_scale:
                        buffer_specify_list.append((data_buf_size, scale_a_buf_size, scale_b_buf_size, quant_group_result_buf_size, extra_scale_b_buf_size))
    if not operand_A_info['with_scale']:
        # 合并后两项加起来一样的 buffer size
        tmp_set = set()
        for buffer_specify in buffer_specify_list:
            tmp_set.add((buffer_specify[0], buffer_specify[1], buffer_specify[2], buffer_specify[3]+buffer_specify[4], 0))
        buffer_specify_list = list(tmp_set)
    # print(buffer_specify_list)
    args_list = [(buffer_specify, test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info) 
                    for buffer_specify in buffer_specify_list]
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        results = list(executor.map(test_buffer_worker, args_list))
    best_lat = float('inf')
    best_buffer_specify = None
    before_prune_count = len(results)
    after_prune_count = 0
    for buffer_specify, latency in results:
        if latency < math.inf:
            after_prune_count += 1
            if buffer_specify == answer_buffer_specify and not slient:
                print(f"found answer_buffer_specify")
        else:
            assert buffer_specify is not answer_buffer_specify, "buffer_specify is not best_buffer_specify"
        if latency < best_lat:
            best_lat = latency
            best_buffer_specify = buffer_specify
    if not slient:
        print(f"Best Latency: {best_lat}, Best Buffer Specify: {best_buffer_specify}")
        print(f"Before Prune Count: {before_prune_count}, After Prune Count: {after_prune_count}")
    return best_lat, best_buffer_specify

def search_partition_space_and_buffer(mm_size, test_func, batch_size, result_precision, accumulation_precision, operand_A_info, operand_B_info, total_buffer_size, force_b_scale = 0, slient=False, answer_partition = None, answer_buffer_specify = None):
    # partition_space = get_partition_space([m, k, n, 1])
    # print(f"answer_partition: {answer_partition}, answer_buffer_specify: {answer_buffer_specify}")
    partition_space = get_partition_space(mm_size)
    best_lat = float('inf')
    best_partition_specify = None
    best_buffer_specify = None
    for partition in partition_space:
        compute_level, pu_num, baseline_partition = partition
        test_tmp = partial(
            test_func,
            partition_specify=baseline_partition
        )
        if not slient:
            print(f"DIV: {baseline_partition}", end=", ")
        inner_answer_buffer = None
        if answer_partition == partition[2]: 
            inner_answer_buffer = answer_buffer_specify
        lat, buffer_specify = search_buffer_size_with_dequant_buf(
            test_func=test_tmp,
            batch_size=batch_size,
            result_precision=result_precision,
            accumulation_precision=accumulation_precision,
            operand_A_info=operand_A_info,
            operand_B_info=operand_B_info,
            total_buffer_size=total_buffer_size,
            force_b_scale=force_b_scale,
            slient=slient,
            answer_buffer_specify=inner_answer_buffer
        )
        if lat < best_lat:
            best_lat = lat
            best_partition_specify = baseline_partition
            best_buffer_specify = buffer_specify
    return best_lat, best_partition_specify, best_buffer_specify



"""
MARK: experiment for intro
"""
def latency_profile(**kwargs):
    """
    latency profile ( motivation experiment 3 )
    """
    #
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    for k_group in [16, 32, 64, 128]:
        for n_group in [1, 4, 8, 32, 128]:
            print("="*10, f"mkn={m},{k},{n}, k_group={k_group}, n_group={n_group}, total_buffer_size={total_buffer_size}")
            print("="*10, "test w8a8")
            w8a8_best_lat, w8a8_best_buffer_specify = search_buffer_size_with_dequant_buf(
                test_func=test_w_and_a,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info={'matrix_size': (m,k), 'scale_group': (1, k_group), 'matrix_precision': 8, 'scale_offset_precision': 32, 'with_scale': True, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 8, 'scale_offset_precision': 32, 'with_scale': True, 'with_offset': False},
                total_buffer_size=total_buffer_size,
            )
            profile_test(
                test_func=test_w_and_a,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info={'matrix_size': (m,k), 'scale_group': (1, k_group), 'matrix_precision': 8, 'scale_offset_precision': 32, 'with_scale': True, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 8, 'scale_offset_precision': 32, 'with_scale': True, 'with_offset': False},
                buffer_specify=w8a8_best_buffer_specify
            )
            print("="*10, "test w4a4")
            w4a4_best_lat, w4a4_best_buffer_specify = search_buffer_size_with_dequant_buf(
                test_func=test_w_and_a,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info={'matrix_size': (m,k), 'scale_group': (1, k_group), 'matrix_precision': 4, 'scale_offset_precision': 8, 'with_scale': True, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 4, 'scale_offset_precision': 8, 'with_scale': True, 'with_offset': False},
                total_buffer_size=total_buffer_size,
            )
            profile_test(
                test_func=test_w_and_a,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info={'matrix_size': (m,k), 'scale_group': (1, k_group), 'matrix_precision': 4, 'scale_offset_precision': 8, 'with_scale': True, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 4, 'scale_offset_precision': 8, 'with_scale': True, 'with_offset': False},
                buffer_specify=w4a4_best_buffer_specify
            )
            print("="*10, "test w4a16")
            w4a16_best_lat, w4a16_best_buffer_specify = search_buffer_size_with_dequant_buf(
                test_func=test_w_only,
                batch_size=1,
                result_precision=16,
                accumulation_precision=16,
                operand_A_info={'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 4, 'scale_offset_precision': 16, 'with_scale': True, 'with_offset': False},
                total_buffer_size=total_buffer_size,
            )
            profile_test(
                test_func=test_w_only,
                batch_size=1,
                result_precision=16,
                accumulation_precision=16,
                operand_A_info={'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 4, 'scale_offset_precision': 16, 'with_scale': True, 'with_offset': False},
                buffer_specify=w4a16_best_buffer_specify
            )
            print("="*10, "test w4a16 with buffer_b=True")
            test_w_only_bufb = partial(
                test_w_only,
                buffer_b = True
            )
            w4a16_best_lat, w4a16_best_buffer_specify = search_buffer_size_with_dequant_buf(
                test_func=test_w_only_bufb,
                batch_size=1,
                result_precision=16,
                accumulation_precision=16,
                operand_A_info={'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 4, 'scale_offset_precision': 16, 'with_scale': True, 'with_offset': False},
                total_buffer_size=total_buffer_size,
            )
            profile_test(
                test_func=test_w_only_bufb,
                batch_size=1,
                result_precision=16,
                accumulation_precision=16,
                operand_A_info={'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}, 
                operand_B_info={'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': 4, 'scale_offset_precision': 16, 'with_scale': True, 'with_offset': False},
                buffer_specify=w4a16_best_buffer_specify
            )
            print(" ")

"""
MARK: final experiment
"""

def predictor_validation(**kwargs):
    """
    predictor validation
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    predict_w_a_func, predict_w_only_func = parse_predict_args(kwargs)
    # INSERT_YOUR_CODE
    yaml_file = '/home/xietongxin/nfs/Quant-PIM/quantpim/scripts/final/2_predictor_validation/w_a.yaml'
    with open(yaml_file, 'r') as f:
        predictor_data = yaml.load(f, Loader=yaml.FullLoader)
    real_func = partial(
        test_w_and_a,
        mix_scale_mapping=True,
        reorder=True,
    )
    for data in predictor_data:
        m = data['m']
        if m == 1:
            continue
        k = data['k']
        n = data['n']
        data_pre = data['data_pre']
        scale_pre = data['scale_pre']
        n_group = data['n_group']
        k_group = data['k_group']
        best_partition_specify = data['best_partition_specify']
        best_buffer_specify = data['best_buffer_specify']
        best_lat = data['best_lat']
        operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, k_group), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
        operand_B_info = {'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
        # search best compiliation strategy
        print("="*10 + f"({m},{k},{n}) D{data_pre}S{scale_pre} group_k={k_group} group_n={n_group}" + "="*10)
        answer_predicted_lat = predict_w_a_func(batch_size=1, result_precision=16, accumulation_precision=32, 
        operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)
        my_predicted_lat, my_predicted_partition_specify, my_predicted_buffer_specify = search_partition_space_and_buffer(
            mm_size=[m,k,n,1],
            test_func=predict_w_a_func,
            batch_size=1,
            result_precision=16,
            accumulation_precision=32,
            operand_A_info=operand_A_info,
            operand_B_info=operand_B_info,
            total_buffer_size=total_buffer_size,
            slient=True,
            answer_partition=tuple(tuple(x) for x in best_partition_specify),
            answer_buffer_specify=tuple(best_buffer_specify)
        )
        # print("==> finish search")
        print(f"==> Answer Partition = {best_partition_specify}, Buffer = {best_buffer_specify}")
        print(f"Answer Predicted Latency: {answer_predicted_lat}, Answer Real Latency: {best_lat}")
        # test my real latency
        my_real_latency = real_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=my_predicted_buffer_specify, partition_specify=my_predicted_partition_specify, silent=True)
        # best_latency = real_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        # consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)

        print(f"==> My Partition = {my_predicted_partition_specify}, Buffer = {my_predicted_buffer_specify}")
        print(f"My Predicted Latency: {my_predicted_lat}, My Real Latency: {my_real_latency}")
        print(f"reach {best_lat/my_real_latency*100:.2f}% of answer performance")
        
def predictor_validation_w_only(**kwargs):
    """
    predictor validation
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    predict_w_a_func, predict_w_only_func = parse_predict_args_rebuttal(kwargs)
    # INSERT_YOUR_CODE
    yaml_file = '/home/xietongxin/nfs/Quant-PIM/quantpim/scripts/final/2_predictor_validation/w_only.yaml'
    with open(yaml_file, 'r') as f:
        predictor_data = yaml.load(f, Loader=yaml.FullLoader)
    real_func = partial(
        test_w_only, 
        loop_friendly_mapping=True,
        buffer_b = True,
    )
    for data in predictor_data:
        m = data['m']
        # if m in [1,2]:
        #     continue
        k = data['k']
        n = data['n']
        data_pre = data['data_pre']
        scale_pre = data['scale_pre']
        n_group = data['n_group']
        k_group = data['k_group']
        best_partition_specify = data['best_partition_specify']
        best_buffer_specify = data['best_buffer_specify']
        best_lat = data['best_lat']
        operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
        operand_B_info = {'matrix_size': (n,k), 'scale_group': (n_group, k_group), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
        # search best compiliation strategy
        print("="*10 + f"({m},{k},{n}) D{data_pre}S{scale_pre} group_k={k_group} group_n={n_group}" + "="*10)
        answer_predicted_lat = predict_w_only_func(batch_size=1, result_precision=16, accumulation_precision=32, 
        operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)
        my_predicted_lat, my_predicted_partition_specify, my_predicted_buffer_specify = search_partition_space_and_buffer(
            mm_size=[m,k,n,1],
            test_func=predict_w_only_func,
            batch_size=1,
            result_precision=16,
            accumulation_precision=32,
            operand_A_info=operand_A_info,
            operand_B_info=operand_B_info,
            total_buffer_size=total_buffer_size,
            slient=False,
            answer_partition=tuple(tuple(x) for x in best_partition_specify),
            answer_buffer_specify=tuple(best_buffer_specify)
        )
        # print("==> finish search")
        print(f"==> Answer Partition = {best_partition_specify}, Buffer = {best_buffer_specify}")
        print(f"Answer Predicted Latency: {answer_predicted_lat}, Answer Real Latency: {best_lat}")
        # test my real latency
        my_real_latency = real_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=my_predicted_buffer_specify, partition_specify=my_predicted_partition_specify, silent=True)
        # best_latency = real_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        # consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)

        print(f"==> My Partition = {my_predicted_partition_specify}, Buffer = {my_predicted_buffer_specify}")
        print(f"My Predicted Latency: {my_predicted_lat}, My Real Latency: {my_real_latency}")
        print(f"reach {best_lat/my_real_latency*100:.2f}% of answer performance")

def get_ideal_latency(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    predict_w_a_func, predict_w_only_func = parse_predict_args(kwargs)
    if 'qconfig' in kwargs:
        quant_config_path = kwargs['qconfig']
    else:
        quant_config_path = "./scripts/final/3_single_op_with_predictor/workload/quant.yaml"
    # quant_config_path = "./scripts/final/3_single_op_with_predictor/workload/quant.yaml"
    precision_list = []
    quant_group_list = []
    type_list = []
    
    with open(quant_config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        for precision in yaml_config["precision"]:
            precision_list.append(precision)
        for quant_group in yaml_config["group"]:
            quant_group_list.append(quant_group)
        for type in yaml_config["type"]:
            type_list.append(type)
    
    for type in type_list:
        if type == 'w_only': 
            optimal_func = get_optimal_latency_w_only
            addtion_precision_list = [[4,16]]
        else:
            optimal_func = get_optimal_latency_w_a
            addtion_precision_list = []
        
        for precision in precision_list + addtion_precision_list:
            for quant_group in quant_group_list:
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                # baseline 0
                # print("==> baseline 0")
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                partition_space = get_partition_space([m,k,n,1])
                best_lat = float('inf')
                for partition in partition_space:
                    compute_level, pu_num, baseline_partition = partition
                    optimal_lat = optimal_func(args = None, batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                    consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, partition_specify=baseline_partition, silent=True)
                    if optimal_lat < best_lat:
                        best_lat = optimal_lat
                        best_partition_specify = baseline_partition
                print(f"==> Optimal Latency: {best_lat}, Optimal Partition Specify: {best_partition_specify}")

def change_buffer_size(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    predict_w_a_func, _ = parse_predict_args(kwargs)
    quant_group_list = [[1, 16]]
    precision_list = [[4,8]]
    for precision in precision_list:
        for quant_group in quant_group_list:
            for buffer_size in range(4,30,4):
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                print("="*10, f"D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n} buffer_size={buffer_size}", "="*10)
                operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                
                baseline_test_func = partial(test_w_and_a, loop_friendly_mapping=True)
                baseline_latency, baseline_buffer_specify = search_buffer_size_with_dequant_buf(
                    test_func=baseline_test_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=buffer_size,
                    slient=True,
                )
                print(f"Baseline Latency: {baseline_latency}, Baseline Buffer Specify: {baseline_buffer_specify}")                
                with_mx_test_func = partial(test_w_and_a, mix_scale_mapping=True)
                with_mx_best_lat, with_mx_best_partition_specify, with_mx_best_buffer_specify = search_partition_space_and_buffer(
                    mm_size=[m,k,n,1],
                    test_func=with_mx_test_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=buffer_size,
                    slient=True,
                )
                print(f"With MX Best Latency: {with_mx_best_lat}, With MX Best Partition Specify: {with_mx_best_partition_specify}, With MX Best Buffer Specify: {with_mx_best_buffer_specify}")
                
                with_reorder_test_func = partial(test_w_and_a, mix_scale_mapping=True, reorder=True)
                _, with_reorder_best_partition_specify, with_reorder_best_buffer_specify = search_partition_space_and_buffer(
                    mm_size=[m,k,n,1],
                    test_func=with_reorder_test_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=buffer_size,
                    slient=True,
                )
                with_reorder_best_lat = with_reorder_test_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=with_reorder_best_buffer_specify, partition_specify=with_reorder_best_partition_specify, silent=True)

                print(f"With Reorder Best Latency: {with_reorder_best_lat}, With Reorder Best Partition Specify: {with_reorder_best_partition_specify}, With Reorder Best Buffer Specify: {with_reorder_best_buffer_specify}")
                print(f"Speedup: {baseline_latency / with_reorder_best_lat:.2f}")

def ablation_study(**kwargs):
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    predict_w_a_func, predict_w_only_func = parse_predict_args_rebuttal(kwargs)
    # read workload from yaml
    precision_list = [[4,8]]
    quant_group_list = [[1, 16]]
    # test w_a
    for precision in precision_list:
        # print("="*10, f"precision: {precision}, group: {quant_group_list[0]}", "="*10)
        data_pre, scale_pre = precision
        sg_n, sg_k = quant_group_list[0]
        # print("="*10, "test w_a", "="*10)
        test_func = test_w_and_a
        operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
        operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
        baseline_0_buffer_specify = [16, 1, 2, 1, 0]
        test_baseline_func = partial(test_func, loop_friendly_mapping=True)
        test_mapping = partial(
            test_func, 
            mix_scale_mapping=True,
            reorder=False,
        )
        test_mapping_reorder = partial(
            test_func, 
            mix_scale_mapping=True,
            reorder=True,
        )
        baseline_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=False)
        print(f"[Baseline] Latency: {baseline_lat}, Partition: baseline, Buffer: baseline")
        # 2. + mapping
        mapping_lat = test_mapping(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
        consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=False)
        print(f"[Mapping] Latency: {mapping_lat}, Partition: baseline, Buffer: baseline")
        # 6. + mapping + reorder
        mapping_reorder_best_lat, mapping_reorder_best_buffer_specify = search_buffer_size_with_dequant_buf(
            test_func=test_mapping_reorder,
            batch_size=1,
            result_precision=16,
            accumulation_precision=32,
            operand_A_info=operand_A_info,
            operand_B_info=operand_B_info,
            total_buffer_size=total_buffer_size,
            slient=True,
        )
        print(f"[Mapping + Reorder] Latency: {mapping_reorder_best_lat}, Partition: baseline, Buffer: {mapping_reorder_best_buffer_specify}")
        dse_lat, dse_partition_specify, dse_buffer_specify = search_partition_space_and_buffer(
            mm_size=[m,k,n,1],
            test_func=predict_w_a_func,
            batch_size=1,
            result_precision=16,
            accumulation_precision=32,
            operand_A_info=operand_A_info,
            operand_B_info=operand_B_info,
            total_buffer_size=total_buffer_size,
            slient=True,
        )
        dse_lat = test_mapping_reorder(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=dse_buffer_specify, partition_specify=dse_partition_specify, silent=True)
        print(f"[DSE + Partition + Reorder] Latency: {dse_lat}, Partition: {dse_partition_specify}, Buffer: {dse_buffer_specify}")
    

"""
NOTE: Rebuttal
"""

# def new_single_op(**kwargs):
#     """
#     new single op experiment
#     """
#     m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
#     if 'qconfig' in kwargs:
#         quant_config_path = kwargs['qconfig']
#     else:
#         quant_config_path = "./scripts/final/1_single_op/workload/quant_w_only_64.yaml"
#     # read workload from yaml
#     precision_list = []
#     quant_group_list = []
#     type_list = []
#     with open(quant_config_path, "r") as f:
#         yaml_config = yaml.load(f, Loader=yaml.FullLoader)
#         for precision in yaml_config["precision"]:
#             precision_list.append(precision)
#         for quant_group in yaml_config["group"]:
#             quant_group_list.append(quant_group)
#         for type in yaml_config["type"]:
#             type_list.append(type)
#     for type in type_list:

#         if type == 'w_only': 
#             test_func = test_w_only
#             baseline_0_buffer_specify = [16, 0, 4, 0, 0]
#             if m > 1:
#                 test_ours_func = partial(
#                     test_func, 
#                     loop_friendly_mapping=True,
#                     buffer_b = True,
#                     # buffer_b_old = True
#                 )
#             else:
#                 test_ours_func = partial(
#                     test_func, 
#                     loop_friendly_mapping=True,
#                     # buffer_b_old = True
#                 )
#         else:
#             test_func = test_w_and_a
#             add_precision_list = []
#             baseline_0_buffer_specify = [16, 1, 2, 1, 0]
#             test_ours_func = partial(
#                 test_func, 
#                 mix_scale_mapping=True,
#                 reorder=True,
#             )
#         # logfile_name = f"/home/xietongxin/nfs/Quant-PIM/quantpim/scripts/final/1_single_op/rbt/7B_MVM_{m}_{k}_{n}.log"
#         # log_dir = os.path.dirname(logfile_name)
#         # if not os.path.exists(log_dir):
#         #     os.makedirs(log_dir, exist_ok=True)
#         # logfile = open(logfile_name, 'w')
#         for precision in precision_list:
#             for quant_group in quant_group_list:
#                 test_baseline_func = partial(test_func, loop_friendly_mapping=True)
#                 data_pre, scale_pre = precision 
#                 sg_n, sg_k = quant_group
#                 print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
#                 # baseline 0
#                 # print("==> baseline 0")
#                 if type == 'w_only': 
#                     operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
#                 else:
#                     operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
#                 operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
#                 baseline_0_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
#                 consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True)
#                 print(f"==> Baseline 0 Latency: {baseline_0_lat}")
#                 # ours
#                 #c print("==> ours")
#                 best_lat, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
#                     mm_size=[m,k,n,1],
#                     test_func=test_ours_func,
#                     batch_size=1,
#                     result_precision=16,
#                     accumulation_precision=32,
#                     operand_A_info=operand_A_info,
#                     operand_B_info=operand_B_info,
#                     total_buffer_size=total_buffer_size,
#                     slient=True,
#                 )
#                 # 再运行一次，留下 log
#                 test_ours_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
#                 consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)
#                 print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
#                 print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

# def single_op(**kwargs):
#     """
#     single op experiment
#     """
#     m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
#     if 'qconfig' in kwargs:
#         quant_config_path = kwargs['qconfig']
#     else:
#         quant_config_path = "./scripts/final/1_single_op/workload/quant_w_only_64.yaml"
#     # read workload from yaml
#     precision_list = []
#     quant_group_list = []
#     type_list = []
    
#     with open(quant_config_path, "r") as f:
#         yaml_config = yaml.load(f, Loader=yaml.FullLoader)
#         for precision in yaml_config["precision"]:
#             precision_list.append(precision)
#         for quant_group in yaml_config["group"]:
#             quant_group_list.append(quant_group)
#         for type in yaml_config["type"]:
#             type_list.append(type)
#     for type in type_list:
#         if type == 'w_only': 
#             test_func = test_w_only
#             add_precision_list = [[4,16]]
#             baseline_0_buffer_specify = [16, 0, 4, 0, 0]
#             test_ours_func = partial(
#                 test_func, 
#                 loop_friendly_mapping=True,
#                 buffer_b = True,
#                 # buffer_b_old = True
#             )
#         else:
#             test_func = test_w_and_a
#             add_precision_list = []
#             baseline_0_buffer_specify = [16, 1, 2, 1, 0]
#             test_ours_func = partial(
#                 test_func, 
#                 mix_scale_mapping=True,
#                 reorder=True,
#             )
#         for precision in precision_list + add_precision_list:
#             for quant_group in quant_group_list:
#                 # if (m,k,n) == (1, 4096, 4096) and type == "w_a":
#                 #     continue
#                 # if (m,k,n) == (1, 6656, 6656):
#                 #     continue
#                 # if (m,k,n) == (16, 4096, 4096) and type == "w_a" and quant_group[0]==1:
#                 #     continue
#                 # if m in [2,4,8] and type == "w_a":
#                 #     continue
#                 test_baseline_func = partial(test_func, loop_friendly_mapping=True)
#                 data_pre, scale_pre = precision 
#                 sg_n, sg_k = quant_group
#                 print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
#                 # baseline 0
#                 # print("==> baseline 0")
#                 if type == 'w_only': 
#                     operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
#                 else:
#                     operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
#                 operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
#                 baseline_0_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
#                 consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True)
#                 print(f"==> Baseline 0 Latency: {baseline_0_lat}")
#                 # ours
#                 #c print("==> ours")
#                 best_lat, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
#                     mm_size=[m,k,n,1],
#                     test_func=test_ours_func,
#                     batch_size=1,
#                     result_precision=16,
#                     accumulation_precision=32,
#                     operand_A_info=operand_A_info,
#                     operand_B_info=operand_B_info,
#                     total_buffer_size=total_buffer_size,
#                     slient=True,
#                 )
#                 # 再运行一次，留下 log
#                 test_ours_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
#                 consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)
#                 print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
#                 print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

def test_cnn(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    type_list = ['w_a']
    precision_list = [(4,8)]
    if (m,k,n) == (1, 1600, 192):
        quant_group_list = [(1, 25)]
    else:
        quant_group_list = [(1, 9)]
    for type in type_list:
        if type == 'w_only': 
            test_func = test_w_only
            add_precision_list = []
            baseline_0_buffer_specify = [16, 0, 4, 0, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                buffer_b = True,
                # buffer_b_old = True
            )
        else:
            test_func = test_w_and_a
            add_precision_list = []
            baseline_0_buffer_specify = [16, 1, 2, 1, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                reorder=True,
            )
        for precision in precision_list + add_precision_list:
            for quant_group in quant_group_list:
                test_baseline_func = partial(test_func, loop_friendly_mapping=True)
                data_pre, scale_pre = precision 
                sg_n, sg_k = quant_group
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                # baseline 0
                # print("==> baseline 0")
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                baseline_0_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=True)
                print(f"==> Baseline 0 Latency: {baseline_0_lat}")
                # ours
                #c print("==> ours")
                best_lat, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
                    mm_size=[m,k,n,1],
                    test_func=test_ours_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=total_buffer_size,
                    slient=True,
                )
                # 再运行一次，留下 log
                test_ours_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True, breakdown_analysis=True)
                print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
                print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

def case_test(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    if 'qconfig' in kwargs:
        quant_config_path = kwargs['qconfig']
    else:
        quant_config_path = "./scripts/final/1_single_op/workload/quant_w_only_64.yaml"
    # read workload from yaml
    precision_list = [(4,8)]
    quant_group_list = [(1,16)]
    type_list = ['w_a']
    
    # with open(quant_config_path, "r") as f:
    #     yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    #     for precision in yaml_config["precision"]:
    #         precision_list.append(precision)
    #     for quant_group in yaml_config["group"]:
    #         quant_group_list.append(quant_group)
    #     for type in yaml_config["type"]:
    #         type_list.append(type)
    for type in type_list:
        if type == 'w_only': 
            test_func = test_w_only
            add_precision_list = []
            baseline_0_buffer_specify = [16, 0, 4, 0, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                buffer_b = True,
                # buffer_b_old = True
            )
        else:
            test_func = test_w_and_a
            add_precision_list = []
            baseline_0_buffer_specify = [16, 1, 2, 1, 0]
            test_ours_func = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=True,
            )
        for precision in precision_list + add_precision_list:
            for quant_group in quant_group_list:
                # if (m,k,n) == (1, 4096, 4096) and type == "w_a":
                #     continue
                # if (m,k,n) == (1, 6656, 6656):
                #     continue
                # if (m,k,n) == (16, 4096, 4096) and type == "w_a" and quant_group[0]==1:
                #     continue
                # if m in [2,4,8] and type == "w_a":
                #     continue
                test_baseline_func = partial(test_func, loop_friendly_mapping=True)
                data_pre, scale_pre = precision 
                sg_n, sg_k = quant_group
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                # baseline 0
                # print("==> baseline 0")
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                baseline_0_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=True)
                print(f"==> Baseline 0 Latency: {baseline_0_lat}")
                
                # best_partition_specify = ((1, 64, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))
                # best_buffer_specify =  (14, 4, 2, 0, 0)

                best_partition_specify = ((1, 8, 8, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))
                best_buffer_specify = (1, 1, 1, 1, 0)
                # 再运行一次，留下 log
                best_lat = test_ours_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                    consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True, breakdown_analysis=True)
                print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
                print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

def mapping_ablation(**kwargs):
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    quant_config_path = "./scripts/final/1_single_op/workload/quant.yaml"
    precision_list = [[4,8]]
    # quant_group_list = [[16, 16]]
    quant_group_list = [[1, 16],[1, 32],[16, 16],[32, 32],[64, 64],[128, 128]]
    # test w_a
    for precision in precision_list:
        for quant_group in quant_group_list:
            print("="*10, f"precision: {precision}, group: {quant_group}", "="*10)
            if True:
                type = 'w_a'
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                print("="*10, "test w_a", "="*10)
                test_func = test_w_and_a
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                baseline_0_buffer_specify = [16, 1, 2, 1, 0]
                test_baseline_func = partial(test_func, loop_friendly_mapping=True)
                test_mapping = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=False,
            )
            test_mapping_old = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=False,
                old_scale_mix=True,
            )
            test_reorder = partial(
                test_func, 
                mix_scale_mapping=False,
                reorder=True,
            )
            test_mapping_reorder = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=True,
            )
            baseline_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=True)
            print(f"[Baseline] Latency: {baseline_lat}, Partition: baseline, Buffer: baseline")
                        # 3. + old scale mix
            # old_scale_mix_lat, old_scale_mix_buffer_specify = search_buffer_size_with_dequant_buf(
            #     test_func=test_mapping_old,
            #     batch_size=1,
            #     result_precision=16,
            #     accumulation_precision=32,
            #     operand_A_info=operand_A_info,
            #     operand_B_info=operand_B_info,
            #     total_buffer_size=total_buffer_size,
            #     slient=True,
            # )
            old_scale_mix_lat = test_mapping_old(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=True)
            print(f"[Old Scale Mix] Latency: {old_scale_mix_lat}, Partition: baseline, Buffer: baseline`")
            
            mix_scale_lat = test_mapping(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True, breakdown_analysis=True)
            print(f"[Mix Scale] Latency: {mix_scale_lat}, Partition: baseline, Buffer: baseline")
            # force b scale
            force_b_lat, force_b_buffer_specify = search_buffer_size_with_dequant_buf(
                test_func=test_mapping,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info=operand_A_info,
                operand_B_info=operand_B_info,
                total_buffer_size=total_buffer_size,
                slient=True,
                force_b_scale=1,
            )
            force_b_lat = test_mapping(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=force_b_buffer_specify, silent=True, breakdown_analysis=True)
            print(f"[Force B] Latency: {force_b_lat}, Partition: baseline, Buffer: {force_b_buffer_specify}")
            # # force b scale with dse
            # _, force_b_dse_partition_specify, force_b_dse_buffer_specify = search_partition_space_and_buffer(
            #     mm_size=[m,k,n,1],
            #     test_func=test_mapping,
            #     batch_size=1,
            #     result_precision=16,
            #     accumulation_precision=32,
            #     operand_A_info=operand_A_info,
            #     operand_B_info=operand_B_info,
            #     total_buffer_size=total_buffer_size,
            #     force_b_scale=1,
            #     slient=True,
            # )
            # force_b_dse_lat = test_mapping(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            # consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=force_b_dse_buffer_specify, partition_specify=force_b_dse_partition_specify, silent=True, breakdown_analysis=True)
            # print(f"[Force B + Partition DSE] Latency: {force_b_dse_lat}, Partition: {force_b_dse_partition_specify}, Buffer: {force_b_dse_buffer_specify}")

            # # # 1. + dse
            # # dse_lat, dse_partition_specify, dse_buffer_specify = search_partition_space_and_buffer(
            # #     mm_size=[m,k,n,1],
            # #     test_func=test_baseline_func,
            # #     batch_size=1,
            # #     result_precision=16,
            # #     accumulation_precision=32,
            # #     operand_A_info=operand_A_info,
            # #     operand_B_info=operand_B_info,
            # #     total_buffer_size=total_buffer_size,
            # #     slient=True,
            # # )
            # # print(f"[DSE] Latency: {dse_lat}, Partition: {dse_partition_specify}, Buffer: {dse_buffer_specify}")
            # 2. + mapping
            mapping_lat, mapping_buffer_specify = search_buffer_size_with_dequant_buf(
                test_func=test_mapping,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info=operand_A_info,
                operand_B_info=operand_B_info,
                total_buffer_size=total_buffer_size,
                slient=True,
            )
            mapping_lat = test_mapping(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=mapping_buffer_specify, silent=True, breakdown_analysis=True)
            print(f"[Mapping] Latency: {mapping_lat}, Partition: baseline, Buffer: {mapping_buffer_specify}")

            # mapping_dse_lat, mapping_dse_partition_specify, mapping_dse_buffer_specify = search_partition_space_and_buffer(
            #     mm_size=[m,k,n,1],
            #     test_func=test_mapping,
            #     batch_size=1,
            #     result_precision=16,
            #     accumulation_precision=32,
            #     operand_A_info=operand_A_info,
            #     operand_B_info=operand_B_info,
            #     total_buffer_size=total_buffer_size,
            #     slient=True,
            # )
            # mapping_dse_lat = test_mapping(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            # consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=mapping_dse_buffer_specify, partition_specify=mapping_dse_partition_specify, silent=True, breakdown_analysis=True)
            # print(f"[Mapping + Partition DSE] Latency: {mapping_dse_lat}, Partition: {mapping_dse_partition_specify}, Buffer: {mapping_dse_buffer_specify}")
        else:
            # test w_only
            type = 'w_only'
            data_pre, scale_pre = precision_list[0]
            sg_n, sg_k = quant_group_list[0]
            print("="*10, "test w_only", "="*10)
            test_func = test_w_only
            if type == 'w_only': 
                operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
            else:
                operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
            operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
            test_func = test_w_only
            baseline_0_buffer_specify = [12, 4, 4, 0, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                buffer_b = True,
                # buffer_b_old = True
            )
            test_baseline_func = partial(test_func, loop_friendly_mapping=True)
            # 1. + dse
            dse_lat, dse_partition_specify, dse_buffer_specify = search_partition_space_and_buffer(
                mm_size=[m,k,n,1],
                test_func=test_baseline_func,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info=operand_A_info,
                operand_B_info=operand_B_info,
                total_buffer_size=total_buffer_size,
                slient=True,
            )
            print(f"[DSE] Latency: {dse_lat}, Partition: {dse_partition_specify}, Buffer: {dse_buffer_specify}")
            
            # 2. + buffer b
            buffer_b_lat = test_ours_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True)
            print(f"[Buffer B] Latency: {buffer_b_lat}, Partition: baseline, Buffer: baseline")

def _yaml_convert_tuples_to_lists(obj):
    """递归把 tuple 转为 list，便于用 safe_load 可读的 YAML 写出。"""
    if isinstance(obj, tuple):
        return [_yaml_convert_tuples_to_lists(x) for x in obj]
    if isinstance(obj, list):
        return [_yaml_convert_tuples_to_lists(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _yaml_convert_tuples_to_lists(v) for k, v in obj.items()}
    return obj

tmp_strategy_file = "./scripts/final/3_single_op_with_predictor/tmp_strategy/tmp_strategy.yaml"
def single_op_with_predictor_part1(**kwargs):
    """
    search for optimal
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    predict_w_a_func, predict_w_only_func = parse_predict_args_rebuttal(kwargs)
    if 'qconfig' in kwargs:
        quant_config_path = kwargs['qconfig']
    else:
        quant_config_path = "./scripts/final/3_single_op_with_predictor/workload/quant.yaml"
    precision_list = []
    quant_group_list = []
    type_list = []
    
    with open(quant_config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        for precision in yaml_config["precision"]:
            precision_list.append(precision)
        for quant_group in yaml_config["group"]:
            quant_group_list.append(quant_group)
        for type in yaml_config["type"]:
            type_list.append(type)

    tmp_strategy = {}
    if os.path.exists(tmp_strategy_file):
        try:
            with open(tmp_strategy_file, "r") as f:
                tmp_strategy = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load existing strategy file {tmp_strategy_file}: {e}")
            tmp_strategy = {}
        
    for type in type_list:
        if type == 'w_only': 
            if m == 1:
                predict_func = partial(
                    predict_w_only_func,
                    buffer_b = False
                )
            else: 
                predict_func = predict_w_only_func
        else:
            predict_func = predict_w_a_func
        for precision in precision_list:
            for quant_group in quant_group_list:
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                _, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
                    mm_size=[m,k,n,1],
                    test_func=predict_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=total_buffer_size,
                    slient=True,
                )
                tmp_strategy[f"{type} M{m} K{k} N{n} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}"] = [
                    best_partition_specify,
                    best_buffer_specify,
                ]
    with open(tmp_strategy_file, "w") as f:
        yaml.dump(_yaml_convert_tuples_to_lists(tmp_strategy), f)

def single_op_with_predictor_part2(**kwargs):
    """
    simulation
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    test_mode = kwargs.get('test_mode', 0)
    # read workload from yaml
    if 'qconfig' in kwargs:
        quant_config_path = kwargs['qconfig']
    else:
        quant_config_path = "./scripts/final/3_single_op_with_predictor/workload/quant.yaml"
    precision_list = []
    quant_group_list = []
    type_list = []

    with open(tmp_strategy_file, "r") as f:
        buffered_strategy = yaml.safe_load(f) or {}
    
    with open(quant_config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        for precision in yaml_config["precision"]:
            precision_list.append(precision)
        for quant_group in yaml_config["group"]:
            quant_group_list.append(quant_group)
        for type in yaml_config["type"]:
            type_list.append(type)
        
    for type in type_list:
        if type == 'w_only': 
            test_func = test_w_only
            add_precision_list = []
            baseline_0_buffer_specify = [16, 0, 4, 0, 0]
            if m == 1:
                test_ours_func = partial(
                    test_func, 
                    loop_friendly_mapping=True,
                )
            else:
                test_ours_func = partial(
                    test_func, 
                    loop_friendly_mapping=True,
                    buffer_b = True,
                    # buffer_b_old = True
                )
        else:
            test_func = test_w_and_a
            add_precision_list = []
            baseline_0_buffer_specify = [16, 1, 2, 1, 0]
            test_ours_func = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=True,
            )
        for precision in precision_list + add_precision_list:
            for quant_group in quant_group_list:
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                key = f"{type} M{m} K{k} N{n} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}"
                if not key in buffered_strategy:
                    continue
                best_partition_specify, best_buffer_specify = buffered_strategy[key]
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                if m == 4096:
                    if type == 'w_only':
                        operand_A_baseline = {'matrix_size': (8,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                        operand_A_best = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                        operand_B_baseline = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                        operand_B_best = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                        baseline_args = (test_func, operand_A_baseline, operand_B_baseline, baseline_0_buffer_specify, 4096 / 8)
                        best_args = (test_ours_func, operand_A_best, operand_B_best, best_buffer_specify, best_partition_specify, 1)
                    else:
                        operand_A_baseline = {'matrix_size': (8,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                        operand_A_best = {'matrix_size': (256,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                        operand_B_baseline = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                        operand_B_best = operand_B_baseline
                        baseline_args = (test_func, operand_A_baseline, operand_B_baseline, baseline_0_buffer_specify, 4096 / 8)
                        best_args = (test_ours_func, operand_A_best, operand_B_best, best_buffer_specify, best_partition_specify, 4096 / 256)
                else:
                    if type == 'w_only':
                        operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                    else:
                        operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                    operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                    baseline_args = (test_func, operand_A_info, operand_B_info, baseline_0_buffer_specify, 1.0)
                    best_args = (test_ours_func, operand_A_info, operand_B_info, best_buffer_specify, best_partition_specify, 1)
                # 主进程内顺序执行，避免 ProcessPool 导致子进程崩溃（test_func 可能依赖 CUDA/全局状态，fork 后不可用）
                
                if test_mode in [0, 1]:
                    baseline_0_lat = _run_baseline_worker(baseline_args)
                    print(f"==> Baseline 0 Latency: {baseline_0_lat}")
                if test_mode in [0, 2]:
                    best_lat = _run_best_worker(best_args)
                    print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
                if test_mode in [0]:
                    print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

def test_energy(**kwargs):
    """
    simulation
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    predict_w_a_func, predict_w_only_func = parse_predict_args_rebuttal(kwargs)
    if 'qconfig' in kwargs:
        quant_config_path = kwargs['qconfig']
    else:
        quant_config_path = "./scripts/final/8_energy/workload/quant.yaml"
    precision_list = []
    quant_group_list = []
    type_list = []

    # with open("./scripts/final/8_energy/tmp_strategy/strategy_rebuttal.yaml", "r") as f:
    #     buffered_strategy = yaml.safe_load(f) or {}
    
    with open(quant_config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        for precision in yaml_config["precision"]:
            precision_list.append(precision)
        for quant_group in yaml_config["group"]:
            quant_group_list.append(quant_group)
        for type in yaml_config["type"]:
            type_list.append(type)
        
    for type in type_list:
        if type == 'w_only': 
            test_func = test_w_only
            add_precision_list = []
            baseline_0_buffer_specify = [16, 0, 4, 0, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                buffer_b = True,
                # buffer_b_old = True
            )
            predict_func = predict_w_only_func
        else:
            test_func = partial(
                test_w_and_a,
                report_energy = True,
            )
            add_precision_list = []
            baseline_0_buffer_specify = [16, 1, 2, 1, 0]
            test_ours_func = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=True,
                report_energy = True,
            )
            predict_func = predict_w_a_func
            # if (n==6656 and k==6656):
            #     quant_group_list = [[1, 32]]
            #     precision_list = [[4,8]]
        for precision in precision_list + add_precision_list:
            for quant_group in quant_group_list:
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                # key = f"{type} M{m} K{k} N{n} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}"
                # if not key in buffered_strategy:
                #     continue
                # best_partition_specify, best_buffer_specify = buffered_strategy[key]
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                if type == 'w_only':
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                baseline_args = (test_func, operand_A_info, operand_B_info, baseline_0_buffer_specify, 1.0)
                _, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
                    mm_size=[m,k,n,1],
                    test_func=predict_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=total_buffer_size,
                    slient=True,
                )
                best_args = (test_ours_func, operand_A_info, operand_B_info, best_buffer_specify, best_partition_specify, 1.0)
                # 主进程内顺序执行，避免 ProcessPool 导致子进程崩溃（test_func 可能依赖 CUDA/全局状态，fork 后不可用）
                print(f"==> Baseline Energy:")
                baseline_0_lat = _run_baseline_worker(baseline_args)
                print(f"==> Best Energy:")
                best_lat = _run_best_worker(best_args)
                print(f"==> Baseline Latency: {baseline_0_lat}")
                print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
                print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

def test_mix_precision_pu(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args_mix_precision_pu(kwargs)
    if 'qconfig' in kwargs:
        quant_config_path = kwargs['qconfig']
    else:
        quant_config_path = "./scripts/final/9_mix_precision_pu/workload/quant_gemm_left.yaml"
    # read workload from yaml
    precision_list = []
    quant_group_list = []

    with open(quant_config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        for precision in yaml_config["precision"]:
            precision_list.append(precision)
        for quant_group in yaml_config["group"]:
            quant_group_list.append(quant_group)

    test_func = test_w_and_a
    baseline_0_buffer_specify = [16, 0, 3, 1, 0]
    test_ours_func = partial(
        test_func, 
        mix_scale_mapping=True,
        reorder=True,
        sequential_gen = True,
    )
    for precision in precision_list:
        for quant_group in quant_group_list:
            test_baseline_func = partial(test_func, loop_friendly_mapping=True)
            data_pre, scale_pre = precision
            sg_n, sg_k = quant_group
            print("="*10, f"w_a D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
            # baseline 0
            # print("==> baseline 0")
            operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
            operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
            baseline_0_lat = test_baseline_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=baseline_0_buffer_specify, silent=True)
            print(f"==> Baseline 0 Latency: {baseline_0_lat}")
            # ours
            #c print("==> ours")
            best_lat, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
                mm_size=[m,k,n,1],
                test_func=test_ours_func,
                batch_size=1,
                result_precision=16,
                accumulation_precision=32,
                operand_A_info=operand_A_info,
                operand_B_info=operand_B_info,
                total_buffer_size=total_buffer_size,
                slient=True,
            )
            # 再运行一次，留下 log
            test_ours_func(batch_size=1, result_precision=16, accumulation_precision=32, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
            consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=best_buffer_specify, partition_specify=best_partition_specify, silent=True)
            print(f"==> Best Latency: {best_lat}, Best Partition Specify: {best_partition_specify}, Best Buffer Specify: {best_buffer_specify}")
            print(f"==> Speedup: {baseline_0_lat / best_lat:.2f}")

def test_dse(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    predict_w_a_func, predict_w_only_func = parse_predict_args(kwargs)
    quant_config_path = "./scripts/final/3_single_op_with_predictor/workload/quant.yaml"
    precision_list = [[4,8]]
    quant_group_list = [[1, 16]]
    type_list = ['w_a']
        
    for type in type_list:
        if type == 'w_only': 
            test_func = test_w_only
            add_precision_list = [[4,16]]
            baseline_0_buffer_specify = [16, 0, 4, 0, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                buffer_b = True,
                # buffer_b_old = True
            )
            predict_func = predict_w_only_func
        else:
            test_func = test_w_and_a
            add_precision_list = []
            baseline_0_buffer_specify = [16, 1, 2, 1, 0]
            test_ours_func = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=True,
            )
            predict_func = predict_w_a_func
            # if (n==6656 and k==6656):
            #     quant_group_list = [[1, 32]]
            #     precision_list = [[4,8]]
        for precision in precision_list + add_precision_list:
            for quant_group in quant_group_list:
                test_baseline_func = partial(test_func, loop_friendly_mapping=True)
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                # baseline 0
                # print("==> baseline 0")
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                # run a original dse test, without predictor
                print("==> run a original dse test, without predictor")
                import time
                start_time = time.time()
                # _, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
                #     mm_size=[m,k,n,1],
                #     test_func=test_ours_func,
                #     batch_size=1,
                #     result_precision=16,
                #     accumulation_precision=32,
                #     operand_A_info=operand_A_info,
                #     operand_B_info=operand_B_info,
                #     total_buffer_size=total_buffer_size,
                #     slient=True,
                # )
                end_time = time.time()
                print(f"==> Time consumed: {end_time - start_time:.2f} seconds")
                # run a dse test with predictor
                print("==> run a dse test with predictor")
                start_time = time.time()
                _, best_partition_specify, best_buffer_specify = search_partition_space_and_buffer(
                    mm_size=[m,k,n,1],
                    test_func=predict_func,
                    batch_size=1,
                    result_precision=16,
                    accumulation_precision=32,
                    operand_A_info=operand_A_info,
                    operand_B_info=operand_B_info,
                    total_buffer_size=total_buffer_size,
                    slient=True,
                )
                end_time = time.time()
                print(f"==> Time consumed: {end_time - start_time:.2f} seconds")
                # compare the time consumed
                print(f"==> Speedup: {start_time / end_time:.2f}")

def test_other_pu_rebuttal(**kwargs):
    """
    single op experiment
    """
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    # read workload from yaml
    predict_w_a_func, predict_w_only_func = parse_predict_args(kwargs)
    quant_config_path = "./scripts/final/3_single_op_with_predictor/workload/quant.yaml"
    precision_list = [[4,8]]
    quant_group_list = [[1, 16]]
    type_list = ['w_a']
        
    for type in type_list:
        if type == 'w_only': 
            test_func = test_w_only
            add_precision_list = []
            baseline_0_buffer_specify = [16, 0, 4, 0, 0]
            test_ours_func = partial(
                test_func, 
                loop_friendly_mapping=True,
                buffer_b = True,
                # buffer_b_old = True
                sequential_gen=True,
            )
            predict_func = predict_w_only_func
        else:
            test_func = test_w_and_a
            add_precision_list = []
            baseline_0_buffer_specify = [16, 1, 2, 1, 0]
            test_ours_func = partial(
                test_func, 
                mix_scale_mapping=True,
                reorder=True,
                sequential_gen=True,
            )
            predict_func = predict_w_a_func
            # if (n==6656 and k==6656):
            #     quant_group_list = [[1, 32]]
            #     precision_list = [[4,8]]
        for precision in precision_list + add_precision_list:
            for quant_group in quant_group_list:
                test_baseline_func = partial(test_func, loop_friendly_mapping=True)
                data_pre, scale_pre = precision
                sg_n, sg_k = quant_group
                print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={sg_k} group_n={sg_n}", "="*10)
                # baseline 0
                # print("==> baseline 0")
                if type == 'w_only': 
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': None, 'matrix_precision': 16, 'scale_offset_precision': None, 'with_scale': False, 'with_offset': False}
                else:
                    operand_A_info = {'matrix_size': (m,k), 'scale_group': (1, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                operand_B_info = {'matrix_size': (n,k), 'scale_group': (sg_n, sg_k), 'matrix_precision': data_pre, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}
                fp32_lat_change_list = [2, 4]
                # run a original dse test, without predictor
                for fp32_lat_change in fp32_lat_change_list:
                    throughput_tag = 1/fp32_lat_change
                    print(f"==> run pu with {throughput_tag}x fp32 throughput")
                    accumulation_precision = int(32 * fp32_lat_change)
                    baseline_best_lat, baseline_buffer_specify = search_buffer_size_with_dequant_buf(
                        test_func=test_baseline_func,
                        batch_size=1,
                        result_precision=16,
                        accumulation_precision=accumulation_precision,
                        operand_A_info=operand_A_info,
                        operand_B_info=operand_B_info,
                        total_buffer_size=total_buffer_size,
                        slient=True,
                    )
                    print(f"Baseline Buffer Search Best Latency: {baseline_best_lat}, Baseline Partition Specify: baseline, Baseline Buffer Specify: {baseline_buffer_specify}")
                    ours_best_lat, ours_partition_specify, ours_buffer_specify = search_partition_space_and_buffer(
                        mm_size=[m,k,n,1],
                        test_func=test_ours_func,
                        batch_size=1,
                        result_precision=16,
                        accumulation_precision=accumulation_precision,
                        operand_A_info=operand_A_info,
                        operand_B_info=operand_B_info,
                        total_buffer_size=total_buffer_size,
                        slient=False,
                    )
                    ours_best_lat = test_ours_func(batch_size=1, result_precision=16, accumulation_precision=accumulation_precision, operand_A_info=operand_A_info, operand_B_info=operand_B_info, 
                    consider_data_read=True, consider_scale_buffer=True, consider_result_buffer=True, buffer_specify=ours_buffer_specify, partition_specify=ours_partition_specify, silent=True)
                    print(f"Ours Best Latency: {ours_best_lat}, Ours Partition Specify: {ours_partition_specify}, Ours Buffer Specify: {ours_buffer_specify}")
                    print(f"Speedup: {baseline_best_lat / ours_best_lat:.2f}")

def test_quant_latency(**kwargs):
    m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only = parse_test_args(kwargs)
    SS = kwargs.get('SS')
    
    def parse_log_line(line):
        """
        Parse a log line from quant_latency/log/13B_MVM1_B4_4_5120_5120.log, which has 4 config strings:
        "((4, 8, 2, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))", "((4, 4, 4, 1), ...)", ...
        Return a list of 4 partition strategies, each as list-of-lists or list-of-tuples.
        """
        import re
        if not line or not isinstance(line, str):
            return None
        configs = [seg.strip() for seg in line.strip().split('","')]
        configs[0] = configs[0].lstrip('"')
        if len(configs) > 1:
            configs[-1] = configs[-1].rstrip('"')
        results = []
        for conf in configs:
            # Remove leading and trailing quotes if any
            conf = conf.strip('"')
            # Find all groups inside ( )
            groups = re.findall(r'\(([^)]*)\)', conf)
            this_spec = []
            for group in groups:
                nums = [int(x) for x in re.findall(r'-?\d+', group)]
                this_spec.append(tuple(nums))
            results.append(this_spec)
        return results if results else None
    # Example usage with one line from @file_context_0:
    # log_line = '"((4, 8, 2, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))","((4, 4, 4, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))","((1, 1, 64, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))","((1, 1, 64, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 16, 1))"'
    # parsed = parse_log_line(log_line)
    # print(parsed)  # Should output a list of 4 elements, each as list of 4 tuples
    s1, s2, s3, s4 = parse_log_line(SS)
    config_list = [
        ('w_a', 4, 8, 1, 16, s1),
        ('w_a', 4, 8, 1, 32, s2),
        ('w_only', 4, 8, 1, 16, s3),
        ('w_only', 4, 8, 1, 32, s4),
    ]
    # get_baseline_partition
    baseline_partition=None
    partition_tool = Partition(require_power_of_2 = False)
    partition_space = partition_tool.get_partition_space_mm([m,k,n,1])
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    partition_space = filtered_partition_space    
    for compute_level, pu_num, partition in partition_space:
        # m,k,l,b: only partition on l, and k = 8
        if partition[3][0] * partition[3][1] * partition[3][3] * \
            partition[2][0] * partition[2][1] * partition[2][3] * \
            partition[1][0] * partition[1][1] * partition[1][3] * \
            partition[0][0] * partition[0][1] * partition[0][3] == 1:
            # baseline = compute_level, pu_num, partition, mm_size_per_pu, A_mapping, B_mapping, result_mapping
            baseline_partition = partition
            break
    if baseline_partition == None: # corner case
        baseline_partition = partition_space[0][2]

    for type, data_pre, scale_pre, group_n, group_k,  S in config_list:
        # INSERT_YOUR_CODE
        # The input S is in the format "((1  8  8  1)  (1  1  1  1)  (1  1  1  1)  (1  1  16  1))"
        # We need to parse it into Python list-of-lists (partition strategy format)
        print("="*10, f"{type} D{data_pre}S{scale_pre} group_k={group_k} group_n={group_n}", "="*10)
        partition_tool = Partition(require_power_of_2 = False)
        # baseline_partition=[[1,1,64,1],[1,1,1,1],[1,1,1,1],[1,1,16,1]]
        mm_size_per_pu = partition_tool.mem_partition_mm([m,k,n,1], S, (1, group_k, group_n))
        mm_size_per_pu_baseline = partition_tool.mem_partition_mm([m,k,n,1], baseline_partition, (1, group_k, group_n))
        if mm_size_per_pu_baseline[0] >= 16:
            lat1, lat2 = test_baseline_quant(
                args,
                1,
                operand_A_info={'matrix_size': (16,mm_size_per_pu_baseline[1]), 
                'scale_group': (1, group_k), 'matrix_precision': data_pre, 'original_matrix_precision': 16, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}, 
            )
            lat1 = int(lat1 * mm_size_per_pu_baseline[0] / 16)
            lat2 = int(lat2 * mm_size_per_pu_baseline[0] / 16)
        else:
            lat1, lat2 = test_baseline_quant(
                args,
                1,
                operand_A_info={'matrix_size': (mm_size_per_pu_baseline[0],mm_size_per_pu_baseline[1]), 
                'scale_group': (1, group_k), 'matrix_precision': data_pre, 'original_matrix_precision': 16, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}, 
            )
        print(f"baseline lat all: {lat1}, baseline lat original: {lat2}")
        if mm_size_per_pu[0] >= 16:
            lat1, lat2 = test_baseline_quant(
                args,
                1,
                operand_A_info={'matrix_size': (16,mm_size_per_pu[1]), 
                'scale_group': (1, group_k), 'matrix_precision': data_pre, 'original_matrix_precision': 16, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}, 
            )
            lat1 = int(lat1 * mm_size_per_pu[0] / 16)
            lat2 = int(lat2 * mm_size_per_pu[0] / 16)
        else:
            lat1, lat2 = test_baseline_quant(
                args,
                1,
                operand_A_info={'matrix_size': (mm_size_per_pu[0],mm_size_per_pu[1]), 
                'scale_group': (1, group_k), 'matrix_precision': data_pre, 'original_matrix_precision': 16, 'scale_offset_precision': scale_pre, 'with_scale': True, 'with_offset': False}, 
            )
        # partition_tool = PartitionTool(m, k, n, data_pre, scale_pre, group_k, group_n)
        print(f"lat all: {lat1}, lat original: {lat2}")

def test_nvfp4(**kwargs):
    m = kwargs.get('m', 1)
    k = kwargs.get('k', 5120)
    n = kwargs.get('n', 5120)
    buffer_specify = kwargs.get('buffer_specify', (56, 4, 4))
    test_func = kwargs.get('test_func', test_baseline_with_buffer)
    profile_w4a4 = partial(
        profile_test,
        test_func=test_func,
        batch_size=1,
        result_precision=16,
        accumulation_precision=16,
        operand_A_info={'matrix_size': (m,k), 'scale_group': (1, 16), 'matrix_precision': 4, 'scale_offset_precision': 8, 'with_scale': True, 'with_offset': False}, 
        operand_B_info={'matrix_size': (n,k), 'scale_group': (1, 16), 'matrix_precision': 4, 'scale_offset_precision': 8, 'with_scale': True, 'with_offset': False},
        buffer_specify=buffer_specify,
        buffer_b = False
    )
    profile_w4a4()

def parse_test_args(args):
    m = args.get('m', 1)
    k = args.get('k', 5120)
    n = args.get('n', 5120)
    buffer_specify = args.get('buffer_specify',[1,1,1])
    k_group = args.get('k_group', 16)
    n_group = args.get('n_group', 1)
    k_tile = args.get('kt', 0)
    reorder = args.get('reorder', False)
    loop_friendly_mapping = args.get('lf_mapping', False)
    mix_scale_mapping = args.get('mx_mapping', False)
    new_partition = args.get('new_partition', False)
    total_buffer_size = args.get('total_buffer_size', 32)
    aargs = args.get("args")
    test_w_and_a = partial(
        test_baseline_with_buffer,
        loop_friendly_mapping = loop_friendly_mapping,
        mix_scale_mapping = mix_scale_mapping,
        new_partition = new_partition,
        args=aargs,
        reorder = reorder,
        sequential_gen = True
    )
    test_w_only = partial(
        test_baseline_w_only_with_buffer,
        loop_friendly_mapping = loop_friendly_mapping,
        mix_scale_mapping = mix_scale_mapping,
        new_partition = new_partition,
        args=aargs
    )
    return m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only

def parse_test_args_mix_precision_pu(args):
    m = args.get('m', 1)
    k = args.get('k', 5120)
    n = args.get('n', 5120)
    buffer_specify = args.get('buffer_specify',[1,1,1])
    k_group = args.get('k_group', 16)
    n_group = args.get('n_group', 1)
    k_tile = args.get('kt', 0)
    reorder = args.get('reorder', False)
    loop_friendly_mapping = args.get('lf_mapping', False)
    mix_scale_mapping = args.get('mx_mapping', False)
    new_partition = args.get('new_partition', False)
    total_buffer_size = args.get('total_buffer_size', 32)
    aargs = args.get("args")
    test_w_and_a = partial(
        test_baseline_with_buffer_mixprepu,
        loop_friendly_mapping = loop_friendly_mapping,
        mix_scale_mapping = mix_scale_mapping,
        new_partition = new_partition,
        args=aargs,
        reorder = reorder
    )
    test_w_only = partial(
        test_baseline_w_only_with_buffer,
        loop_friendly_mapping = loop_friendly_mapping,
        mix_scale_mapping = mix_scale_mapping,
        new_partition = new_partition,
        args=aargs
    )
    return m, k, n, buffer_specify, k_group, n_group, k_tile, loop_friendly_mapping, mix_scale_mapping, new_partition, total_buffer_size, test_w_and_a, test_w_only


def parse_predict_args(args):
    m = args.get('m', 1)
    k = args.get('k', 5120)
    n = args.get('n', 5120)
    buffer_specify = args.get('buffer_specify',[1,1,1])
    k_group = args.get('k_group', 16)
    n_group = args.get('n_group', 1)
    k_tile = args.get('kt', 0)
    reorder = args.get('reorder', False)
    loop_friendly_mapping = args.get('lf_mapping', False)
    mix_scale_mapping = args.get('mx_mapping', False)
    new_partition = args.get('new_partition', False)
    total_buffer_size = args.get('total_buffer_size', 32)
    aargs = args.get("args")

    predict_w_a_func = partial(
        predict_w_a,
        mix_scale_mapping=True,
        reorder=True,
        args = aargs
    )
    predict_w_only_func = partial(
        predict_w_only,
        loop_friendly_mapping = True,
        buffer_b = True,
        args=aargs
    )
    return predict_w_a_func, predict_w_only_func

def parse_predict_args_rebuttal(args):
    m = args.get('m', 1)
    k = args.get('k', 5120)
    n = args.get('n', 5120)
    buffer_specify = args.get('buffer_specify',[1,1,1])
    k_group = args.get('k_group', 16)
    n_group = args.get('n_group', 1)
    k_tile = args.get('kt', 0)
    reorder = args.get('reorder', False)
    loop_friendly_mapping = args.get('lf_mapping', False)
    mix_scale_mapping = args.get('mx_mapping', False)
    new_partition = args.get('new_partition', False)
    total_buffer_size = args.get('total_buffer_size', 32)
    aargs = args.get("args")

    predict_w_a_func = partial(
        predict_w_a,
        mix_scale_mapping=True,
        reorder=True,
        args = aargs
    )
    predict_w_only_func = partial(
        predict_w_only_rebuttal,
        loop_friendly_mapping = True,
        buffer_b = True,
        args=aargs
    )
    return predict_w_a_func, predict_w_only_func

# # MARK: main test
# def main():
# 解析未知参数为键值对
argparser = argparse.ArgumentParser()
argparser.add_argument('--workloadsize', '-S', nargs='+', type=int, default=[5000,5000])
argparser.add_argument('--po2', '-P', action='store_true')
argparser.add_argument('--allow_under_ultize', '-UU', action='store_true')
argparser.add_argument('--scale-buffer', action='store_true', help='Enable scale buffer management')
argparser.add_argument('--data-read', action='store_true', help='Enable data read buffer management')
argparser.add_argument('--result-buffer', action='store_true', help='Enable result buffer management')
argparser.add_argument('--use-global-buffer', '-UGB', action='store_true', help='Use global buffer')
argparser.add_argument('--silent', action='store_true', help='Silent mode')
argparser.add_argument('--num-processes', type=int, default=mp.cpu_count(), help='Number of processes for parallel address generation')
argparser.add_argument('--test', '-t', type=str, default='test3', help='Select which test function to run')
args, unknown_args = argparser.parse_known_args()
# set up global output directory
output_dir_name = "rebuttal" # NOTE: you can change the output dir name
output_dir_name = "result/" + output_dir_name
tools.logger.output_dir_name = output_dir_name
os.makedirs(f"{output_dir_name}/csv", exist_ok=True)
os.makedirs(f"{output_dir_name}/log", exist_ok=True)
# Global variable to track the current top-level test function name
# workload size: [M, K, N, B]
assert len(args.workloadsize) == 2 , f"Invalid workload size: {args.workloadsize}"
# hw config
SimConfig.read_from_yaml('./config/gddr6-aim.yaml')
# set pu level
SimConfig.pu_level = LEVEL.DE
test_kwargs = {}
i = 0
# print(unknown_args)
while i < len(unknown_args):
    if unknown_args[i].startswith('--') or unknown_args[i].startswith('-'):
        key = unknown_args[i][2:] if unknown_args[i].startswith('--') else unknown_args[i][1:]  # 移除 '--' 前缀
        if key == 'buffer_specify':
            # Special handling for buffer_specify which needs multiple values
            values = []
            j = i + 1
            while j < len(unknown_args) and not unknown_args[j].startswith('--') and not unknown_args[j].startswith('-'):
                try:
                    values.append(int(unknown_args[j]))
                except ValueError:
                    values.append(unknown_args[j])
                j += 1
            test_kwargs[key] = tuple(values) if len(values) > 1 else values[0] if values else True
            i = j
        elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--') and not unknown_args[i + 1].startswith('-'):
            # 下一个参数是值
            value = unknown_args[i + 1]
            # 尝试转换为数字
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # 保持字符串
            test_kwargs[key] = value
            i += 2
        else:
            # 布尔标志
            test_kwargs[key] = True
            i += 1
    else:
        i += 1
test_kwargs['args'] = args
# 添加一些通用参数的映射
k_tile = test_kwargs.get('kt', 0)
# 动态调用测试函数
test_func_name = args.test
# Set global variable to track current test function for directory naming
tools.logger.current_test_function_name = test_func_name
test_func_inner = partial(
    test_baseline_with_buffer,
    args=args,
    k_tile = k_tile,
    loop_friendly_mapping = test_kwargs.get('lf_mapping', False),
    mix_scale_mapping = test_kwargs.get('mx_mapping', False),
    new_partition = test_kwargs.get('new_partition', False),
)
test_kwargs['test_func'] = test_func_inner
if test_func_name in globals() and callable(globals()[test_func_name]):
    test_func = globals()[test_func_name]
    test_func(**test_kwargs)
else:
    print(f"Unknown test function: {args.test}")
    exit()
