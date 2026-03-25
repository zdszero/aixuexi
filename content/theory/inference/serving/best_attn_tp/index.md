---
title: Attn TP & DP
type: docs
description: Prefill/Decode 是否应该使用 dp attention 以及最好的 dp size 设置
weight: 5
---

### Prefill

如果不开启 dp attention，就是跑一个 TP8 的 attention。

#### Normal MoE

**那么 dp attention 何时有收益呢？**

首先，这个需要看你是否使用的是正常 MoE，如果是正常的 MoE，损耗会比较大，因为在 TP4, TP8 的 attention 之后需要增加一个 ReduceScatter，然后再运行 dispatch/combine 的 EP 模式的 MoE，这一步是额外的损耗。

如果使用 FusedMoE，那个差别其实不那么大，因为 TP1, TP4, TP8 在 attention 过后，fused moe 之前，都需要做一个 AllReduce。

> 注意在 sglang 的 TP8 DP8 的时候是通过 AllReduce 操作来模拟 AllGather，这样可以统一 DP1, DP2, DP4, DP8 的代码逻辑。

#### Fused MoE

这一节来讨论 dp attention 对于 prefill 阶段极限吞吐的影响。

首先控制变量，保证单卡 attention / moe  计算量一致的情况下 TP1 / TP4 进行对比。

后文给出三种 TP 情况下 Kimi k25 一层的时间（使用 Fused MoE + 不同 tp size 的 attention）进行性能对比。

- TP1: chunk size = 8k
- TP4: chunk size = 32k
- TP8: chunk size = 64k

三种情况下都是用 8k 的请求打满，保证每张卡的 attention 和 moe 计算量一致。

在该场景下，TP1 极限吞吐相比 TP4 提升约 7%，相比 TP8 提升约 12%

---

观察数据分析可知：

* attention MHA 和 moe 的 w13/w2 算子时间基本一致
* 通信时间基本一致
    * 由于使用 fused moe，TP1 的 attention 之后也要做 all gather，TP4/TP8 的 attention 之后直接做 8 卡的 all reduce，这个步骤两者通信量一致。当然 sglang 的实现逻辑，在 dp_gather 中使用 all_reduce 模拟了 all_gather。
* 差异在于小算子，TP4 的一些小算子需要额外计算：
    * q_a_and_kv_a_proj：在 MHA 的 qkv head 切分之前，每个 rank 重复计算，4 倍差距
    * norm
        * q_norm和 k_norm：4 倍差距
        * attention 之后的 fused_add_norm：8 倍差距。
        * MoE 之后的 fused_add_norm：4 倍差距。
    * 一些其他小算子也存在额外计算
        * set_mla_kv_buffer
        * dp_scatter 中的 trtion_mem_cpy 和 local_tokens.fill_(0)

{{< details "TP1 kimi k25 layer time" >}}
| Step | Kernel Name | Avg(us) | Min(us) | Max(us) | Count | MFU | MBU | Percentage | Annotation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 166.8 | 159.9 | 177.3 | 3 | 66.1% | 14.2% | 0.85% | q_a_and_kv_a |
| 2 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 15.8 | 15.5 | 16.4 | 3 | N/A | N/A | 0.08% |  |
| 3 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 183.2 | 178.0 | 192.8 | 3 | 75.0% | 18.7% | 0.93% | q_b |
| 4 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 11.9 | 11.6 | 12.2 | 3 | N/A | N/A | 0.06% |  |
| 5 | void flashinfer::BatchQKApplyRotaryPosIdsCosSinCache... | 30.7 | 30.0 | 31.7 | 3 | N/A | N/A | 0.16% | fused_rope |
| 6 | set_mla_kv_buffer_kernel | 23.5 | 22.6 | 24.8 | 3 | N/A | N/A | 0.12% |  |
| 7 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 94.7 | 91.7 | 99.0 | 3 | 64.5% | 40.3% | 0.48% | kv_b |
| 8 | concat_and_cast_mha_k_kernel | 54.3 | 53.9 | 54.7 | 3 | N/A | N/A | 0.27% |  |
| 9 | void at::native::vectorized_elementwise_kernel<8, at... | 36.4 | 35.0 | 38.5 | 3 | N/A | N/A | 0.18% |  |
| 10 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk192HV128Sepa... | 1042.3 | 1025.2 | 1074.2 | 3 | 59.4% | 8.4% | 5.28% | attn mha |
| 11 | nvjet_tst_128x256_64x6_2x2_2cta_h_bz_TNT | 592.4 | 588.9 | 597.1 | 3 | 72.2% | 8.1% | 3.00% | o_proj |
| 12 | void at::native::vectorized_elementwise_kernel<8, at... | 50.6 | 50.4 | 50.8 | 3 | N/A | N/A | 0.26% |  |
| 13 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 81.0 | 79.9 | 83.1 | 3 | N/A | N/A | 0.41% |  |
| 14 | void at::native::vectorized_elementwise_kernel<8, at... | 244.0 | 235.2 | 257.9 | 3 | N/A | N/A | 1.24% |  |
| 15 | memcpy_triton_kernel | 39.2 | 37.9 | 41.6 | 3 | N/A | N/A | 0.20% |  |
| 16 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKern... | 2603.4 | 2587.1 | 2634.0 | 3 | N/A | N/A | 13.19% | all reduce |
| 17 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 301.4 | 295.2 | 313.0 | 3 | 70.9% | 10.5% | 1.53% | shared w13 |
| 18 | void flashinfer::activation::act_and_mul_kernel<__nv... | 71.9 | 69.5 | 76.0 | 3 | N/A | N/A | 0.36% |  |
| 19 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 245.0 | 236.2 | 260.6 | 3 | 43.6% | 9.6% | 1.24% | shared w2 |
| 20 | nvjet_tst_192x128_64x6_2x2_2cta_h_bz_TNT | 269.2 | 266.2 | 273.2 | 3 | N/A | N/A | 1.36% |  |
| 21 | void at::native::vectorized_elementwise_kernel<8, at... | 2.6 | 2.5 | 2.8 | 3 | N/A | N/A | 0.01% |  |
| 22 | void at::native::unrolled_elementwise_kernel<at::nat... | 70.6 | 69.8 | 72.2 | 3 | N/A | N/A | 0.36% |  |
| 23 | void moe::dev::routing::routingDeepSeek::routingMain... | 355.6 | 343.6 | 377.5 | 3 | 3.8% | 4.6% | 1.80% | moe router |
| 24 | void moe::dev::routing::routingDeepSeek::routingIndi... | 10.6 | 10.3 | 10.8 | 3 | N/A | N/A | 0.05% |  |
| 25 | bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_t128x6... | 5219.4 | 5026.8 | 5594.8 | 3 | 4.1% | 1.8% | 26.45% | fused_moe_w13 |
| 26 | bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_t128x6... | 2935.0 | 2825.4 | 3145.1 | 3 | 3.6% | 1.6% | 14.87% | fused_moe_w2 |
| 27 | void moe::dev::finalize::finalizeKernelVecLoad<moe::... | 1220.5 | 1218.1 | 1223.6 | 3 | N/A | N/A | 6.18% |  |
| 28 | void at::native::vectorized_elementwise_kernel<8, at... | 405.9 | 400.1 | 417.2 | 3 | N/A | N/A | 2.06% |  |
| 29 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKern... | 3137.6 | 2734.6 | 3391.8 | 3 | N/A | N/A | 15.90% | all reduce |
| 30 | void at::native::vectorized_elementwise_kernel<8, at... | 31.9 | 30.8 | 33.9 | 3 | N/A | N/A | 0.16% |  |
| 31 | memcpy_triton_kernel | 38.0 | 37.5 | 38.4 | 3 | N/A | N/A | 0.19% |  |
| 32 | void flashinfer::norm::FusedAddRMSNormKernel<8u, __n... | 149.8 | 145.4 | 157.6 | 3 | 0.1% | 40.7% | 0.76% | fused_add_norm |

total time: 19735.244666666666 us
{{< /details >}}


{{< details "TP4 kimi k25 layer time" >}}
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
{{< /details >}}

{{< details "TP8 kimi k25 layer time" >}}
| Step | Kernel Name | Avg(us) | Min(us) | Max(us) | Count | MFU | MBU | Percentage | Annotation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | nvjet_tst_176x128_64x8_1x2_2cta_h_bz_TNN | 1220.1 | 1211.6 | 1229.5 | 7 | 72.3% | 13.3% | 5.38% | q_a_and_kv_a |
| 2 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 125.4 | 124.3 | 126.2 | 7 | N/A | N/A | 0.55% |  |
| 3 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 177.3 | 176.4 | 177.8 | 7 | 77.5% | 29.8% | 0.78% | q_b |
| 4 | void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat... | 74.3 | 73.5 | 75.7 | 7 | N/A | N/A | 0.33% |  |
| 5 | void flashinfer::BatchQKApplyRotaryPosIdsCosSinCache... | 30.8 | 30.7 | 31.2 | 7 | N/A | N/A | 0.14% | fused_rope |
| 6 | set_mla_kv_buffer_kernel | 179.1 | 177.5 | 182.7 | 7 | N/A | N/A | 0.79% |  |
| 7 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 89.4 | 88.8 | 91.0 | 7 | 68.3% | 49.0% | 0.39% | kv_b |
| 8 | concat_and_cast_mha_k_kernel | 60.1 | 59.8 | 60.3 | 7 | N/A | N/A | 0.26% |  |
| 9 | void at::native::vectorized_elementwise_kernel<8, at... | 37.3 | 36.9 | 38.1 | 7 | N/A | N/A | 0.16% |  |
| 10 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk192HV128Sepa... | 1029.5 | 1026.9 | 1033.7 | 7 | 60.2% | 38.1% | 4.54% | attn mha |
| 11 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 556.7 | 555.1 | 559.5 | 7 | 76.8% | 8.6% | 2.45% | o_proj |
| 12 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKern... | 2558.2 | 2475.5 | 2653.6 | 7 | N/A | N/A | 11.28% | all reduce |
| 13 | void flashinfer::norm::FusedAddRMSNormKernel<8u, __n... | 1188.6 | 1167.6 | 1210.8 | 7 | N/A | N/A | 5.24% |  |
| 14 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 304.6 | 300.3 | 307.7 | 7 | 70.2% | 10.4% | 1.34% | shared w13 |
| 15 | void flashinfer::activation::act_and_mul_kernel<__nv... | 73.5 | 71.9 | 75.3 | 7 | N/A | N/A | 0.32% |  |
| 16 | nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 246.4 | 238.9 | 252.1 | 7 | 43.4% | 9.5% | 1.09% | shared w2 |
| 17 | nvjet_tst_192x128_64x6_2x2_2cta_h_bz_TNT | 269.6 | 267.1 | 273.6 | 7 | N/A | N/A | 1.19% |  |
| 18 | void at::native::vectorized_elementwise_kernel<8, at... | 2.3 | 2.0 | 2.7 | 7 | N/A | N/A | 0.01% |  |
| 19 | void at::native::unrolled_elementwise_kernel<at::nat... | 70.9 | 70.0 | 71.9 | 7 | N/A | N/A | 0.31% |  |
| 20 | void moe::dev::routing::routingDeepSeek::routingMain... | 364.3 | 354.9 | 374.4 | 7 | 29.3% | 34.8% | 1.61% | moe router |
| 21 | void moe::dev::routing::routingDeepSeek::routingIndi... | 11.1 | 10.9 | 11.3 | 7 | N/A | N/A | 0.05% |  |
| 22 | bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_t128x6... | 5339.1 | 5190.2 | 5504.8 | 7 | 32.0% | 1.7% | 23.54% | fused_moe_w13 |
| 23 | bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_t128x6... | 3012.1 | 2925.3 | 3103.1 | 7 | 28.4% | 1.5% | 13.28% | fused_moe_w2 |
| 24 | void moe::dev::finalize::finalizeKernelVecLoad<moe::... | 1225.4 | 1220.9 | 1233.1 | 7 | N/A | N/A | 5.40% |  |
| 25 | void at::native::vectorized_elementwise_kernel<8, at... | 410.8 | 402.1 | 421.8 | 7 | N/A | N/A | 1.81% |  |
| 26 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKern... | 2826.4 | 2527.8 | 3094.1 | 7 | N/A | N/A | 12.46% | all reduce |
| 27 | void flashinfer::norm::FusedAddRMSNormKernel<8u, __n... | 1196.8 | 1164.6 | 1232.7 | 7 | 0.0% | 5.1% | 5.28% | fused_add_norm |

total time: 22679.95042857143 us
{{< /details >}}

### Decode

在 decode 阶段 dp attention 需要根据场景分析，如果不开启 dp attention，那么你不同卡的显存可能没有办法充分利用，导致 bs 打不上去，从而导致吞吐不高。

#### MLA & MQA

首先，MLA 在 decode 阶段使用 mqa，kv heads 数量为 1，在只有 1 个 dp 的情况下，相当于每张卡都要保存相同的 kv cache。

所以显存实际利用率不高，你的实际 bs 打不上去。

#### GQA

所以如果想要提升 decode 阶段的吞吐，需要保证不同卡能存储不同 kv head 的 cache，这样不重复存储相同的 kv cache，显存利用率才高。

考虑到现在一般大模型的 kv heads 数量都为 2 或 4，所以在这种情况下设置 decode attn 阶段的 dp_size = 8 // kv_heads，吞吐相比于 dp8 的模式不会明显下滑。

#### 其他损耗

其实损耗就是跟 prefill 阶段分析一致了：

- 小算子是否有重复计算
- 是否会额外 ReduceScatter 的通信（后面是正常 MoE 还是 FusedMoE）
