# Documention for MidEnd

The MidEnd section primarily focuses on obtaining the design space for operator partitioning and mapping on hardware.

- For an operator Op, the input includes the number of dimensions and the size of each dimension. These dimensions must be divisible.

- For a hardware configuration, the input includes the parallelism that can be allocated on the hardware, typically including channel, rank, device, and pu for DRAM hierarchy.

## 1. Partitioning on Hardware Hierarchy

### 1.1. get_partition_space_mm

- Function input:

    - Sizes of each dimension of the operator (the length represents the number of dimensions that can be allocated). There are mainly three types of dimensions:
        
        - Dimensions unique to operator A / Dimensions unique to operator B
        
        - Dimensions shared by operator A and operator B
        
            - Dimensions that need to be reduced: input dimensions of MVM
            
            - Dimensions that do not need to be reduced: Multi-head, batch, etc.
        
        - Example 1: For MM, the dimension unique to operator A is M, the dimension unique to operator B is N, and the shared dimension is K, which needs to be reduced.
        
        - Example 2: For Multi-head/Batched MM, the dimension unique to operator A is M, the dimension unique to operator B is N, and the shared dimensions are K (needs to be reduced) and B (does not need to be reduced).

    - Precision bit width of the model (default in the current version of the code)

- The function output is the design space of partition schemes, formatted as follows:
    ```python
    [ # possible divide schemes
        (
            level, # LEVEL.CH/RA/DE
            device_pu_num, # in case multiple computation modes are supported, e.g. Hynix AiM
            (
                (divide_for_the_first_dim, divide_for_the_second_dim, ...), # divide on channel
                (rank_divide),
                (device_divide),
                (pu_divide)
            )
        )
    ]
    ```

### 1.2 choose_from_partition_space

- Used to filter out partition schemes that utilize full parallelism.

## 2. Mapping of Partitioned Operators on DRAM

### 2.1 mem_partition_mm

- Used to calculate the block size each computation unit needs to process after partitioning and calls mem_mapping_mm. The return value is:

    ```Python
    (
        block size in k dimension in columns,
        [ # input data mapping on DRAM
            (m_block_in_row, k_block_in_row, n_block_in_row,
            m_row_num, k_row_num, n_row_num,
            m_block_corner, k_block_corner, n_block_corner)
        ],
        block size in n dimension in output,
        [ # output data mapping on DRAM
            (m_block_in_row, n_block_in_row,
            m_row_num, n_row_num,
            m_block_corner, n_block_corner)
        ]
    )
    ```

### 2.2 mem_mapping_mm

- Used to map the blocks each computation unit needs to process to the rows and columns of DRAM.

### 2.3 choose_from_mem_space
