---
title: DP
type: docs
description: Data Parallelism 并行详解
weight: 5
---

DP 主要是在 attention 阶段，如果开开启 dp attention，就是跑一个 TP8 的 attention。

**那么 dp attention 何时有收益呢？**

首先，这个需要看你是否使用的是正常


### Example

#### TP4

| Step | Kernel Name | Avg(us) | Min(us) | Max(us) | Count | MFU | MBU | Percentage | Annotation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | nvjet_tst_176x128_64x8_1x2_2cta_h_bz_TNN | 607.7 | 606.8 | 609.4 | 3 | 72.6% | 13.6% | 2.81% | q_a_and_kv_a |
| 2 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 61.6 | 61.4 | 61.9 | 3 | N/A | N/A | 0.28% |  |
| 3 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 181.3 | 180.8 | 181.8 | 3 | 75.8% | 22.3% | 0.84% | q_b |
| 4 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 38.7 | 38.5 | 38.9 | 3 | N/A | N/A | 0.18% |  |
| 5 | void flashinfer::BatchQKApplyRotaryPosIdsCosSinCache... | 33.1 | 32.8 | 33.5 | 3 | N/A | N/A | 0.15% | fused_rope |
| 6 | set_mla_kv_buffer_kernel | 89.9 | 89.8 | 89.9 | 3 | N/A | N/A | 0.42% |  |
| 7 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 89.7 | 89.6 | 90.0 | 3 | 68.1% | 44.3% | 0.41% | kv_b |
| 8 | concat_and_cast_mha_k_kernel | 53.8 | 53.7 | 53.9 | 3 | N/A | N/A | 0.25% |  |
| 9 | void at::native::vectorized_elementwise_kernel<8, at... | 37.1 | 36.9 | 37.2 | 3 | N/A | N/A | 0.17% |  |
| 10 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk192HV128Sepa... | 1032.6 | 1032.3 | 1032.8 | 3 | 60.0% | 21.1% | 4.77% | attn mha |
| 11 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 538.6 | 538.6 | 538.6 | 3 | 79.4% | 8.9% | 2.49% | o_proj |
| 12 | void at::native::vectorized_elementwise_kernel<8, at... | 207.9 | 207.6 | 208.2 | 3 | N/A | N/A | 0.96% |  |
| 13 | void at::native::vectorized_elementwise_kernel<8, at... | 245.1 | 239.9 | 247.7 | 3 | N/A | N/A | 1.13% |  |
| 14 | memcpy_triton_kernel | 159.0 | 155.9 | 160.8 | 3 | N/A | N/A | 0.74% |  |
| 15 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKern... | 2511.1 | 2490.8 | 2525.2 | 3 | N/A | N/A | 11.61% | all reduce |
| 16 | void at::native::vectorized_elementwise_kernel<8, at... | 123.0 | 119.2 | 124.9 | 3 | N/A | N/A | 0.57% |  |
| 17 | memcpy_triton_kernel | 141.5 | 140.4 | 142.1 | 3 | N/A | N/A | 0.65% |  |
| 18 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 615.0 | 602.3 | 621.4 | 3 | N/A | N/A | 2.84% |  |
| 19 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 300.8 | 295.7 | 304.0 | 3 | 71.1% | 10.5% | 1.39% | shared w13 |
| 20 | void flashinfer::activation::act_and_mul_kernel<__nv... | 72.1 | 70.1 | 73.2 | 3 | N/A | N/A | 0.33% |  |
| 21 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 241.9 | 236.5 | 244.8 | 3 | 44.2% | 9.7% | 1.12% | shared w2 |
| 22 | nvjet_tst_192x128_64x6_2x2_2cta_h_bz_TNT | 268.2 | 267.3 | 268.8 | 3 | N/A | N/A | 1.24% |  |
| 23 | void at::native::vectorized_elementwise_kernel<8, at... | 2.2 | 2.0 | 2.4 | 3 | N/A | N/A | 0.01% |  |
| 24 | void at::native::unrolled_elementwise_kernel<at::nat... | 70.7 | 70.1 | 71.0 | 3 | N/A | N/A | 0.33% |  |
| 25 | void moe::dev::routing::routingDeepSeek::routingMain... | 357.1 | 348.3 | 361.7 | 3 | 15.0% | 17.8% | 1.65% | moe router |
| 26 | void moe::dev::routing::routingDeepSeek::routingIndi... | 10.8 | 10.6 | 11.0 | 3 | N/A | N/A | 0.05% |  |
| 27 | bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_t128x6... | 5213.1 | 5122.3 | 5320.4 | 3 | 16.4% | 1.8% | 24.10% | fused_moe_w13 |
| 28 | bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_t128x6... | 2949.7 | 2926.5 | 2994.8 | 3 | 14.5% | 1.6% | 13.64% | fused_moe_w2 |
| 29 | void moe::dev::finalize::finalizeKernelVecLoad<moe::... | 1224.4 | 1218.4 | 1230.0 | 3 | N/A | N/A | 5.66% |  |
| 30 | void at::native::vectorized_elementwise_kernel<8, at... | 404.5 | 402.7 | 407.2 | 3 | N/A | N/A | 1.87% |  |
| 31 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKern... | 2887.7 | 2645.7 | 3136.7 | 3 | N/A | N/A | 13.35% | all reduce |
| 32 | void at::native::vectorized_elementwise_kernel<8, at... | 123.2 | 121.6 | 125.0 | 3 | N/A | N/A | 0.57% |  |
| 33 | memcpy_triton_kernel | 142.0 | 141.6 | 142.3 | 3 | N/A | N/A | 0.66% |  |
| 34 | void flashinfer::norm::FusedAddRMSNormKernel<8u, __n... | 591.9 | 586.8 | 598.8 | 3 | 0.0% | 10.3% | 2.74% | fused_add_norm |

total time: 21627.18633333333 us

