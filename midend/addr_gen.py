import tqdm
import multiprocessing as mp

def process_address_chunk_with_dedup(chunk_data):
    """Process a chunk of (m, n, k) coordinates and generate addresses with local deduplication."""
    coords_chunk, A_addr, B_addr, result_addr = chunk_data
    
    # Use copy methods to create new instances for this worker
    A_addr = A_addr.copy()
    B_addr = B_addr.copy()
    result_addr = result_addr.copy()
    
    chunk_results = []
    for m, n, k in coords_chunk:
        A_row, A_col = A_addr.get_tensor_addr(m, k)
        B_row, B_col = B_addr.get_tensor_addr(n, k)
        result_row, result_col = result_addr.get_tensor_addr(m, n)
        
        if A_addr.with_scale:
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = A_addr.get_scale_addr(m, k)
        else:
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = None, None, None, None
            
        if B_addr.with_scale:
            B_scale_group_a_id, B_scale_group_b_id, B_scale_row, B_scale_col = B_addr.get_scale_addr(n, k)
        else:
            B_scale_group_a_id, B_scale_group_b_id, B_scale_row, B_scale_col = None, None, None, None
            
        addr_entry = [A_row, A_col, B_row, B_col, result_row, result_col, 
                     A_scale_group_a_id, A_scale_group_b_id, B_scale_group_a_id, B_scale_group_b_id, 
                     A_scale_row, A_scale_col, B_scale_row, B_scale_col, m, n]
        
        # Local deduplication within chunk
        if not chunk_results or addr_entry != chunk_results[-1]:
            chunk_results.append(addr_entry)
    
    return chunk_results

def generate_cmd_addr_list_parallel(gen, A_addr, B_addr, result_addr, num_processes=None, silent = False):
    """Generate command address list using multiprocessing with proper deduplication."""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)
    
    # Generate all coordinates first
    all_coords = list(gen.generate_parallel(num_processes=num_processes))
    total_coords = len(all_coords)
    
    if total_coords < 1000 or num_processes == 1:
        # For small workloads, use sequential processing
        return generate_cmd_addr_list_sequential(gen, A_addr, B_addr, result_addr)
    
    # Split coordinates into chunks
    chunk_size = max(1, total_coords // num_processes)
    chunks = []
    for i in range(0, total_coords, chunk_size):
        chunk = all_coords[i:i + chunk_size]
        chunks.append((chunk, A_addr, B_addr, result_addr))
    
    # Process chunks in parallel with progress bar
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = []
        if silent:
            for result in pool.imap(process_address_chunk_with_dedup, chunks):
                chunk_results.append(result)
        else:
            with tqdm.tqdm(total=len(chunks), desc=f"Processing {len(chunks)} chunks with {num_processes} processes") as pbar:
                for result in pool.imap(process_address_chunk_with_dedup, chunks):
                    chunk_results.append(result)
                    pbar.update(1)
        
    
    # Merge results with global deduplication
    cmd_addr_list = []
    total_entries = sum(len(chunk_result) for chunk_result in chunk_results)
    if silent:
        for chunk_result in chunk_results:
            for addr_entry in chunk_result:
                # Global deduplication across chunks
                if not cmd_addr_list or addr_entry != cmd_addr_list[-1]:
                    cmd_addr_list.append(addr_entry)
    else:
        with tqdm.tqdm(total=total_entries, desc="Merging and deduplicating results") as pbar:
            for chunk_result in chunk_results:
                for addr_entry in chunk_result:
                    # Global deduplication across chunks
                    if not cmd_addr_list or addr_entry != cmd_addr_list[-1]:
                        cmd_addr_list.append(addr_entry)
                    pbar.update(1)
    return cmd_addr_list

def generate_cmd_addr_list_from_parallel_gen(gen, A_addr, B_addr, result_addr, num_processes=None, silent=False):
    """Generate command address list from parallel loop generator with address calculation parallelization."""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)
    
    # Collect all coordinates from parallel generator
    if not silent:
        print("Collecting coordinates from parallel loop generator...")
    all_coords = list(gen.generate_parallel(num_processes))
    total_coords = len(all_coords)
    
    if total_coords < 1000 or num_processes == 1:
        # For small workloads, use sequential processing
        return generate_cmd_addr_list_from_coords_sequential(all_coords, A_addr, B_addr, result_addr, silent=silent)
    
    # Prepare serializable data for TensorAddr objects
    A_addr_data = {
        'info': A_addr.operand_info,
        'size': A_addr.tensor_size, 
        'mapping': A_addr.tensor_mapping,
        'row_offset': A_addr.row_offset
    }
    B_addr_data = {
        'info': B_addr.operand_info,
        'size': B_addr.tensor_size,
        'mapping': B_addr.tensor_mapping, 
        'row_offset': B_addr.row_offset
    }
    result_addr_data = {
        'info': result_addr.operand_info,
        'size': result_addr.tensor_size,
        'mapping': result_addr.tensor_mapping,
        'row_offset': result_addr.row_offset
    }
    
    # Split coordinates into chunks
    chunk_size = max(1, total_coords // num_processes)
    chunks = []
    for i in range(0, total_coords, chunk_size):
        chunk = all_coords[i:i + chunk_size]
        chunks.append((chunk, A_addr_data, B_addr_data, result_addr_data))
    
    # Process chunks in parallel with optional progress bar
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = []
        if silent:
            for result in pool.imap(process_address_chunk_with_dedup, chunks):
                chunk_results.append(result)
        else:
            with tqdm.tqdm(total=len(chunks), desc=f"Processing {len(chunks)} address chunks with {num_processes} processes") as pbar:
                for result in pool.imap(process_address_chunk_with_dedup, chunks):
                    chunk_results.append(result)
                    pbar.update(1)
    
    # Merge results with global deduplication
    cmd_addr_list = []
    total_entries = sum(len(chunk_result) for chunk_result in chunk_results)
    if silent:
        for chunk_result in chunk_results:
            for addr_entry in chunk_result:
                # Global deduplication across chunks
                if not cmd_addr_list or addr_entry != cmd_addr_list[-1]:
                    cmd_addr_list.append(addr_entry)
    else:
        with tqdm.tqdm(total=total_entries, desc="Merging and deduplicating address results") as pbar:
            for chunk_result in chunk_results:
                for addr_entry in chunk_result:
                    # Global deduplication across chunks
                    if not cmd_addr_list or addr_entry != cmd_addr_list[-1]:
                        cmd_addr_list.append(addr_entry)
                    pbar.update(1)
    
    return cmd_addr_list

def generate_cmd_addr_list_from_coords_sequential(coords_list, A_addr, B_addr, result_addr, silent=False):
    """Generate command address list sequentially from pre-generated coordinates."""
    cmd_addr_list = []
    
    if silent:
        coords_iter = coords_list
    else:
        coords_iter = tqdm.tqdm(coords_list, desc="Processing coordinates sequentially")
    
    for m, n, k in coords_iter:
        A_row, A_col = A_addr.get_tensor_addr(m, k)
        B_row, B_col = B_addr.get_tensor_addr(n, k)
        result_row, result_col = result_addr.get_tensor_addr(m, n)
        if A_addr.with_scale:
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = A_addr.get_scale_addr(m, k)
        else:
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = None, None, None, None
        if B_addr.with_scale:
            B_scale_group_a_id, B_scale_group_b_id, B_scale_row, B_scale_col = B_addr.get_scale_addr(n, k)
        else:
            B_scale_group_a_id, B_scale_group_b_id, B_scale_row, B_scale_col = None, None, None, None
        
        addr_entry = [A_row, A_col, B_row, B_col, result_row, result_col, A_scale_group_a_id, A_scale_group_b_id, B_scale_group_a_id, B_scale_group_b_id, A_scale_row, A_scale_col, B_scale_row, B_scale_col, m, n]
        
        if cmd_addr_list == []:
            cmd_addr_list.append(addr_entry)
        else:
            if addr_entry != cmd_addr_list[-1]:
                cmd_addr_list.append(addr_entry)
    
    return cmd_addr_list

def generate_cmd_addr_list_sequential(gen, A_addr, B_addr, result_addr):
    """Generate command address list sequentially (fallback implementation)."""
    cmd_addr_list = []
    total_iterations = gen.count_total_iterations()
    
    for m, n, k in tqdm.tqdm(gen.generate(), total=total_iterations, desc="Processing loops"):
        A_row, A_col = A_addr.get_tensor_addr(m, k)
        B_row, B_col = B_addr.get_tensor_addr(n, k)
        result_row, result_col = result_addr.get_tensor_addr(m, n)
        if A_addr.with_scale:
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = A_addr.get_scale_addr(m, k)
        else:
            A_scale_group_a_id, A_scale_group_b_id, A_scale_row, A_scale_col = None, None, None, None
        if B_addr.with_scale:
            B_scale_group_a_id, B_scale_group_b_id, B_scale_row, B_scale_col = B_addr.get_scale_addr(n, k)
        else:
            B_scale_group_a_id, B_scale_group_b_id, B_scale_row, B_scale_col = None, None, None, None
        if cmd_addr_list == []:
            cmd_addr_list.append([A_row, A_col, B_row, B_col, result_row, result_col, A_scale_group_a_id, A_scale_group_b_id, B_scale_group_a_id, B_scale_group_b_id, A_scale_row, A_scale_col, B_scale_row, B_scale_col, m, n])
        else:
            if [A_row, A_col, B_row, B_col, result_row, result_col, A_scale_group_a_id, A_scale_group_b_id, B_scale_group_a_id, B_scale_group_b_id, A_scale_row, A_scale_col, B_scale_row, B_scale_col, m, n] != cmd_addr_list[-1]:
                cmd_addr_list.append([A_row, A_col, B_row, B_col, result_row, result_col, A_scale_group_a_id, A_scale_group_b_id, B_scale_group_a_id, B_scale_group_b_id, A_scale_row, A_scale_col, B_scale_row, B_scale_col, m, n])
    
    return cmd_addr_list
