import csv
import os
import inspect

global output_dir_name
global current_test_function_name
def create_function_log_csv(function_name=None, workloadsize=None, data_buf_size=None, scale_a_buf_size=None, scale_b_buf_size=None, quant_group_result_buf_size=None, extra_scale_b_buf_size=None, quanted_buf_size=None):
    """Create function-specific log and CSV files with buffer size information
    
    Args:
        function_name: Optional function name. If None, automatically detects the calling function name.
        workloadsize: Workload size string
        data_buf_size: Size of data buffer
        scale_a_buf_size: Size of scale A buffer  
        scale_b_buf_size: Size of scale B buffer
        quant_group_result_buf_size: Size of quant group result buffer
        extra_scale_b_buf_size: Size of extra scale B buffer
    """
    # Auto-detect function name if not provided
    if function_name is None:
        # Use global test function name if available (for nested calls from profile_test)
        if current_test_function_name is not None:
            function_name = current_test_function_name
        else:
            # Fallback to caller detection
            caller_frame = inspect.currentframe().f_back
            caller_function_name = caller_frame.f_code.co_name
            function_name = caller_function_name
    
    # Create test function specific directory
    test_dir = f"{output_dir_name}/{function_name}"
    os.makedirs(f"{test_dir}/log", exist_ok=True)
    os.makedirs(f"{test_dir}/csv", exist_ok=True)
    
    # Create buffer size suffix for filename
    buffer_suffix = ""
    if data_buf_size is not None or scale_a_buf_size is not None or scale_b_buf_size is not None or quant_group_result_buf_size is not None or extra_scale_b_buf_size is not None:
        buffer_parts = []
        if data_buf_size is not None:
            buffer_parts.append(f"data{data_buf_size}")
        if quanted_buf_size is not None:
            buffer_parts.append(f"quantedData{quanted_buf_size}")
        if scale_a_buf_size is not None:
            buffer_parts.append(f"scaleA{scale_a_buf_size}")
        if scale_b_buf_size is not None:
            buffer_parts.append(f"scaleB{scale_b_buf_size}")
        if quant_group_result_buf_size is not None:
            buffer_parts.append(f"quantGroupResult{quant_group_result_buf_size}")
        if extra_scale_b_buf_size is not None:
            buffer_parts.append(f"extraScaleB{extra_scale_b_buf_size}")
        buffer_suffix = "_" + "_".join(buffer_parts)
    
    log_name = f"./{test_dir}/log/{workloadsize}{buffer_suffix}.log"
    outcsv_name = f"./{test_dir}/csv/{workloadsize}{buffer_suffix}.csv"
    
    log_file = open(log_name, 'w+')
    csvfile = open(outcsv_name, 'w', newline='')
    writer = csv.writer(csvfile)
    
    return log_file, csvfile, writer