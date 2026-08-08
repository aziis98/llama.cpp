# Research Log: SYCL Implementation of TQ1 and TQ2 Quantization

This document records durable findings and rejected approaches discovered while implementing and debugging the SYCL backend support for TQ1_0 and TQ2_0 quantization types, based on recent commits (`ccaaa95` and `4b5a903`).

## Durable Findings

1. **Mathematical Simplification for `dp4a` (Distributive Property)**
   - When evaluating dot products for 2-bit quantization formats (like TQ2_0), the weights need to be unpacked and shifted by an offset (typically `w - 1`).
   - Using the `dp4a` instruction (which computes the dot product of four 8-bit integers packed in a 32-bit word) is highly efficient, but manually packing `(w - 1)` into a 32-bit word byte-by-byte is slow and error-prone.
   - **Finding:** You can apply the distributive property to avoid manual byte packing: `sum((w - 1) * a) = sum(w * a) - sum(1 * a)`.
   - **Implementation:** 
     - Extract four 2-bit weights directly using a single shift and mask: `int sym_pack = (w_pack >> (2 * lane)) & 0x03030303;`.
     - Compute the weighted sum: `sumi = ggml_sycl_dp4a(sym_pack, a_pack, sumi);`.
     - Compute the offset sum using a packed word of ones: `suma = ggml_sycl_dp4a(0x01010101, a_pack, suma);`.
     - Final adjustment outside the loop: `sumi -= suma;`.
   - This approach is significantly faster, uses fewer instructions, and fixed the issue where the model generated gibberish.

2. **Work-Per-Thread Tuning in Vector Dot Products (VDR/QI)**
   - The threading block size and data processed per thread must be carefully tuned. For TQ2_0, moving from processing 1 chunk per thread (`VDR_TQ2_0_Q8_1_MMVQ = 1`, with `QI = 32`) to 4 chunks per thread (`VDR = 4`, with `QI = 8`) in `mmvq.cpp` and `vecdotq.hpp` drastically improved correctness and memory access patterns. 
   - Processing 4 iterations in the unrolled loop (`v = 0..3`) allows the thread to amortize the cost of reading the scale factor `d` and coordinates better with the warp layout for reductions.

## Rejected Approaches

1. **Manual Bitwise Packing for Offset Values**
   - **Approach:** Iterating over bytes (`j = 0..3`), extracting the 2-bit value, subtracting 1, masking with `0xFF`, and shifting it back into a new 32-bit word: `sym_pack |= ((val - 1) & 0xFF) << (8 * j);`.
   - **Reason for Rejection:** This was mathematically buggy (leading to the "gibberish state" output from the LLM) and computationally expensive due to the high number of dependent bitwise operations inside the innermost unrolled loop.

2. **Dedicated `mul_mat_tq2_0` Matrix-Matrix Kernel**
   - **Approach:** Implementing a complex, custom `mul_mat_tq2_0` kernel in SYCL for full matrix-matrix multiplication.
   - **Reason for Rejection:** The custom matrix kernel was entirely removed in favor of relying on the Matrix-Vector (MMVQ) implementations (`mul_mat_vec_tq2_0_q8_1_sycl`). The MMVQ routines are more robust, easier to debug, and map well to the existing SYCL warp primitives without adding unnecessary maintenance overhead for a dedicated matmul kernel that was causing correctness issues.
