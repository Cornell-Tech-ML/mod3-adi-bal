# MiniTorch Module 3

## Task 3.1

Output of `python project/parallel_check.py`

```
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (180)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (180) 
--------------------------------------------------------------------------|loop #ID
    def _map(                                                             | 
        out: Storage,                                                     | 
        out_shape: Shape,                                                 | 
        out_strides: Strides,                                             | 
        in_storage: Storage,                                              | 
        in_shape: Shape,                                                  | 
        in_strides: Strides,                                              | 
    ) -> None:                                                            | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(    | 
            out_shape, in_shape                                           | 
        ):                                                                | 
            for out_index in prange(len(out)):----------------------------| #2
                out[out_index] = fn(in_storage[out_index])                | 
        else:                                                             | 
            for out_index in prange(len(out)):----------------------------| #3
                out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)-------| #0
                to_index(out_index, out_shape, out_idx)                   | 
                                                                          | 
                in_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #1
                broadcast_index(out_idx, out_shape, in_shape, in_idx)     | 
                                                                          | 
                out_value = index_to_position(out_idx, out_strides)       | 
                in_pos = index_to_position(in_idx, in_strides)            | 
                                                                          | 
                out[out_value] = fn(in_storage[in_pos])                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (195) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (198) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (232)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (232) 
-------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                        | 
        out: Storage,                                                                | 
        out_shape: Shape,                                                            | 
        out_strides: Strides,                                                        | 
        a_storage: Storage,                                                          | 
        a_shape: Shape,                                                              | 
        a_strides: Strides,                                                          | 
        b_storage: Storage,                                                          | 
        b_shape: Shape,                                                              | 
        b_strides: Strides,                                                          | 
    ) -> None:                                                                       | 
        if (                                                                         | 
            np.array_equal(out_strides, a_strides)                                   | 
            and np.array_equal(out_strides, b_strides)                               | 
            and np.array_equal(out_shape, a_shape)                                   | 
            and np.array_equal(out_shape, b_shape)                                   | 
        ):                                                                           | 
            for index in prange(len(out)):-------------------------------------------| #7
                out[index] = fn(a_storage[index], b_storage[index])                  | 
        else:                                                                        | 
            for out_index in prange(len(out)):---------------------------------------| #8
                out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)------------------| #4
                to_index(out_index, out_shape, out_idx)                              | 
                                                                                     | 
                a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------------------| #5
                broadcast_index(out_idx, out_shape, a_shape, a_index)                | 
                                                                                     | 
                b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------------------| #6
                broadcast_index(out_idx, out_shape, b_shape, b_index)                | 
                                                                                     | 
                out_value = index_to_position(out_idx, out_strides)                  | 
                a_position = index_to_position(a_index, a_strides)                   | 
                b_position = index_to_position(b_index, b_strides)                   | 
                                                                                     | 
                out[out_value] = fn(a_storage[a_position], b_storage[b_position])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (256) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (253) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (259) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (292)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (292) 
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        a_storage: Storage,                                              | 
        a_shape: Shape,                                                  | 
        a_strides: Strides,                                              | 
        reduce_dim: int,                                                 | 
    ) -> None:                                                           | 
        for out_ordinal_idx in prange(len(out)):-------------------------| #10
            out_idx = np.zeros(len(out_shape), dtype=np.int32)-----------| #9
            to_index(out_ordinal_idx, out_shape, out_idx)                | 
                                                                         | 
            out_pos = index_to_position(out_idx, out_strides)            | 
                                                                         | 
            reduce_size = a_shape[reduce_dim]                            | 
                                                                         | 
            for s in range(reduce_size):                                 | 
                out_idx[reduce_dim] = s                                  | 
                j = index_to_position(out_idx, a_strides)                | 
                out[out_ordinal_idx] = fn(out[out_pos], a_storage[j])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (302) is 
hoisted out of the parallel loop labelled #10 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (317)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/adityaballaki/cornell/MLE/mod3-adi-bal/minitorch/fast_ops.py (317) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    size_batch = out_shape[0]  # if len(out_shape) > 1 else 0                               | 
    rows_out, cols_out = out_shape[1], out_shape[2]                                         | 
    dim_common = a_shape[-1]                                                                | 
                                                                                            | 
    for batch in prange(size_batch):--------------------------------------------------------| #11
        for row in range(rows_out):                                                         | 
            for col in range(cols_out):                                                     | 
                total = 0.0                                                                 | 
                index_out: Index = (                                                        | 
                    batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]    | 
                )                                                                           | 
                                                                                            | 
                for k in range(dim_common):                                                 | 
                    index_a = (                                                             | 
                        batch * a_batch_stride + row * a_strides[1] + k * a_strides[2]      | 
                    )                                                                       | 
                                                                                            | 
                    index_b = (                                                             | 
                        batch * b_batch_stride + k * b_strides[1] + col * b_strides[2]      | 
                    )                                                                       | 
                    total += a_storage[index_a] * b_storage[index_b]                        | 
                                                                                            | 
                out[index_out] = total                                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No all
```

## CPU vs GPU optimisation 

![alt text](CPU-GPU.png)


| Size | CPU Optimization (seconds) | GPU Optimization (seconds) |
|------|----------------------------|----------------------------|
| 64   | 0.004                      | 0.008                      |
| 128  | 0.018                      | 0.017                      |
| 256  | 0.105                      | 0.058                      |
| 512  | 1.300                      | 0.310                      |
| 1024 | 8.000                      | 1.000                      |


## Training logs for task 3.5

## Split Dataset