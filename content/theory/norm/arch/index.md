---
title: 架构
type: docs
description: Normalization
weight: 10
---

### Norm 位置

根据 Layer Normalization 放置的不同位置，可以分为 Pre-Norm 和 Post-Norm 两种常见形式。

{{< svg "/images/pre_post_ln.svg" "90%" >}}

```
# pre norm
y = x + F(LayerNorm(x))
z = y + F(LayerNorm(y))

# post nrom
y = LayerNorm(x + F(x))
z = LayerNorm(y + F(y))

```


现在模型已经全面专项 pre norm：因为训练更稳定，能堆更深

以 deepseek 为例：进入第一层之前，需要做一个 Norm

```
35    void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat...        3.1        3.0        3.4       30      N/A      N/A     0.38%    →   
total time: 807.6974 us

==========================================================================================================================================================================
                                                                           Kimi K25 Layer 1                                                                           
==========================================================================================================================================================================
Step  Kernel Name (Most Common)                                  Avg(us)    Min(us)    Max(us)    Count      MFU      MBU Percentage
```

在每一层的最后，需要做一个 fused add norm，直接准备下一个 Module 的输入：

```
27    bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16_t128x64x512_s4...      106.4      103.9      111.0       30    12.6%    86.4%    21.28%    →   fused_moe_w13
28    bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_t128x64x512_s3_...      105.1      103.5      106.9       30     6.4%    43.8%    21.03%    →   fused_moe_w2
29    void moe::dev::finalize::finalizeKernelVecLoad<moe::...       12.4       12.0       13.1       30      N/A      N/A     2.49%    →   
30    void at::native::vectorized_elementwise_kernel<8, at...        4.0        3.8        4.2       30      N/A      N/A     0.79%    →   
31    ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDev...       43.6       42.4       49.1       30      N/A      N/A     8.72%    →   reduce scatter
32    void flashinfer::norm::FusedAddRMSNormKernel<8u, __n...        6.9        6.6        7.2       30     0.0%     7.0%     1.38%    →   fused_add_norm
total time: 499.9073 us  
```

### QK-Norm

在计算 attention score 之前，对 Q 和 K 做一个 Norm：

```python
class Attention:
    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x) 
        keys = self.W_key(x)
        values = self.W_value(x) 

        # ...

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
```
