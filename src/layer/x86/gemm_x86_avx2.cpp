// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

#include "gemm_int8.h"

void pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_int8(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_int8(B, BT, j, max_jj, k, max_kk);
}

void pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void transpose_pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void pack_B_tile_fp32_to_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void transpose_pack_B_tile_fp32_to_int8_avx2(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void unpack_output_tile_int32_to_fp32_avx2(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
    unpack_output_tile_int32_to_fp32(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta, output_transpose);
}

void gemm_transB_packed_tile_int8_avx2(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}

} // namespace ncnn
