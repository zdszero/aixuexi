---
title: FP8
type: docs
description: scale in quantization
weight: 20
---

```
量化:   bf16_tensor → (fp8_tensor, scale)

反量化: real_value = fp8_tensor × scale

GEMM:   A_real @ B_real = (A_fp8 @ B_fp8) × A_scale × B_scale
                          ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                          tensor core 做     需要额外乘两个 scale
```


### FP8 量化返回

```python
q_fp8, q_scale = act_quant(query, block_size, scale_fmt)  # bf16 → fp8
k_fp8, k_scale = act_quant(key, block_size, scale_fmt)    # bf16 → fp8
```

量化过程（对应 kernel trace 中的 `per_token_group_quant_8bit_kernel`）：

```
对每个 block (128 elements):
    amax = max(|x_i|)
    scale = amax / fp8_max         # fp8_max = 448 for e4m3
    x_fp8 = round(x / scale)       # 量化
```

返回的 `scale` 是 **per-block** 的 fp32 标量，用于后续反量化。

#### Ex

假设我们有一个 `query` 张量，其中一部分元素（一个 block）为 128 个 bf16 数值，例如：
`x = [0.12, -0.45, 1.23, 0.87, -2.10, 0.01, ..., 0.34]`（共 128 个）

1. **计算 amax**：  
   在这个 block 中，取绝对值最大的数，假设是 `| -2.10 | = 2.10`，所以 `amax = 2.10`。

2. **计算 scale**：  
   使用 e4m3 格式的 fp8，其最大可表示的正数（fp8_max）为 448。  
   `scale = amax / fp8_max = 2.10 / 448 ≈ 0.0046875`（存储为 fp32）。

3. **量化**：  
   对 block 中每个元素计算 `round(x_i / scale)`，例如：
   - 对于 `-2.10`：`-2.10 / 0.0046875 ≈ -448`，在 e4m3 范围内，直接取整为 `-448`（fp8 表示）。
   - 对于 `0.12`：`0.12 / 0.0046875 ≈ 25.6`，取整为 `26`（fp8 表示）。
   最终得到 128 个 fp8 数值，组成 `q_fp8`。

4. **返回**：  
   `q_fp8` 是量化后的 fp8 数据块，`q_scale` 是上面计算的 `0.0046875`（fp32），用于后续反量化时恢复数值：`x_dequant ≈ q_fp8 * q_scale`。

#### fp8 × fp8

数学上：

\[
\begin{aligned}
\text{原始值:} \quad Q_{\text{real}} &= Q_{\text{fp8}} \times q_{\text{scale}} \\
K_{\text{real}} &= K_{\text{fp8}} \times k_{\text{scale}} \\[6pt]
Q_{\text{real}} @ K_{\text{real}}^T &= (Q_{\text{fp8}} \times q_{\text{scale}}) @ (K_{\text{fp8}} \times k_{\text{scale}})^T \\
&= (Q_{\text{fp8}} @ K_{\text{fp8}}^T) \times q_{\text{scale}} \times k_{\text{scale}}
\end{aligned}
\]

所以 fp8 GEMM 的结果需要乘上**两个 scale** 才能得到正确的数值。

#### fp8 workflow

FP8 GEMM 的累加精度：FP32，不会溢出

**Tensor Core 内部的精度**

FP8 × FP8 的 tensor core 指令（不管是 Hopper WGMMA 还是 Blackwell UMMA），**累加器（accumulator）都是 FP32**：

```
GMMA::MMA_64x64x32_F32E4M3E4M3_SS
//                  ^^^           FP32 accumulator
//                      ^^^^^^^ FP8 × FP8 输入
```


##### 硬件级操作流程

1.  **乘法**：Tensor Core 的专用电路直接处理 FP8 输入，执行 `fp8 * fp8` 的乘法。这个乘法操作是“原生”的，硬件有专门为 FP8 设计的乘法器单元，效率极高。
2.  **结果扩展与累加**：乘法产生的原始乘积结果（其位宽大于 FP8）会被**立即转换/扩展到更高精度**（例如 FP16 或直接到 FP32 的中间表示），然后送入 FP32 累加器阵列。
3.  **累加器**：所有扩展后的部分积在 FP32 累加器中进行求和。正如你引用的指令 `GMMA::MMA_64x64x32_F32E4M3E4M3_SS` 所示，`F32` 明确指明了累加器精度为 FP32。

**为什么不会溢出**

FP32 的范围是 ±3.4×10³⁸，而 FP8 e4m3 的最大值是 448。即使 GEMM 的 K 维度很大（比如 K=7168），最坏情况的累加和：

\[
448 \times 448 \times 7168 \approx 1.44 \times 10^9
\]

远远小于 FP32 的上限（3.4×10³⁸），所以**不存在溢出风险**。

**数学正确性**

因为从**数值效果**和**编程模型**上看，整个过程**等价于**：

\[
\text{output\_f32}[i,j] = \sum_k \left( \text{float}(A\_fp8[i,k]) \times \text{float}(B\_fp8[k,j]) \right)
\]

也就是说，最终结果的数值精度与“先将每个 FP8 数转换为浮点数（FP32），再进行乘法和累加”所得到的结果是**一致**的。这是 Tensor Core 设计所要保证的数学正确性。

##### 计算复杂度

(M, K) @ (K, N) 的 FP8 GEMM:

- 矩阵乘法: \( 2 \times M \times N \times K \) 次 FP8 乘加
    - \( O(M \cdot N \cdot K) \)，tensor core 执行  
- 乘 scale: \( M \times N \) 次 FP32 乘法
    - \( O(M \cdot N) \)，elementwise

scale 的计算量比矩阵乘法少了一个 K 的量级（K 通常是几千到上万），占比可以忽略。

所以衡量 FP8 GEMM 的 MFU 时，只算 2MNK 的 FP8 FLOPS，用 FP8 峰值算力做分母，scale 的 elementwise 开销不计入。

**最终输出 dtype**

deep_gemm 线性层的输出一般会被**转回 bf16**（用于后续计算），但中间 GEMM + scale 的过程全部在 FP32 下完成。具体的转换时机取决于 kernel 实现：

| 场景 | GEMM 累加 | × scale | 输出 dtype |
|---|---|---|---|
| deep_gemm 线性层 | FP32 | FP32 | **bf16**（kernel 内转） |
| fp8_mqa_logits | FP32 | FP32 (weights+k_scale) | **FP32**（给 topk 用） |
| FlashMLA fp8 decode | FP32 (bf16 MMA 后) | N/A（已反量化） | **bf16** |

所以整个流程的精度保障是：**FP8 只用于存储和 tensor core 输入，所有累加和中间计算都在 FP32 下进行**。FP8 牺牲的是单个元素的精度（3-4 位有效数字），但不会引入溢出问题。

### 不同 kernel 处理 scale 的方式

#### fp8 deep gemm

deep_gemm 的 `fp8_gemm` 接受两个 scale 输入，**kernel 内部自动处理**反量化：

\[
\begin{aligned}
\text{output} &= \text{fp8\_gemm}(A_{\text{fp8}}, B_{\text{fp8}}, A_{\text{scale}}, B_{\text{scale}}) \\
              &= (A_{\text{fp8}} \ @ \ B_{\text{fp8}}) \times A_{\text{scale}} \times B_{\text{scale}} \quad \text{// kernel 内部完成}
\end{aligned}
\]

注意这里 **权重的 scale 是预计算好的**（模型加载时 checkpoint 已有），activation 的 scale 是 `per_token_group_quant` 动态计算的。

#### indexer mqa logits

在 indexer 中，`q_scale` 被预乘到 `weights` 里了，kernel 内部只需要乘 `k_scale`：

\[
\begin{aligned}
&\text{weights} = \frac{\text{gate}}{\sqrt{H}} \times \text{q\_scale} \times \frac{1}{\sqrt{d}} \quad \text{(q\_scale 已包含在内)} \\
\\
&\text{logits}[q,k] = \sum_{h} \Bigl( \underbrace{\text{weights}[q,h] \times \text{q\_fp8}[q,h]}_{\text{q\_scale 已在此处}} \cdot \text{k\_fp8}[k] \Bigr) \times \underbrace{\text{k\_scale}[k]}_{\text{k\_scale 在 kernel 内乘}}
\end{aligned}
\]

这是一种**优化技巧**：把 q_scale 提前乘进 weights，减少 kernel 内的计算量。展开后等价于：

\[
\begin{aligned}
&= \sum_{h} \frac{\text{gate}[q,h]}{\sqrt{H}} \times \frac{(q_{\text{fp8}} \times q_{\text{scale}}) \cdot (k_{\text{fp8}} \times k_{\text{scale}})}{\sqrt{d}} \\
&= \sum_{h} \frac{\text{gate}[q,h]}{\sqrt{H}} \times \frac{q_{\text{real}} \cdot k_{\text{real}}}{\sqrt{d}}
\end{aligned}
\]

两个 scale 都乘上了，只是时机不同。

#### flashmla sparse attention

DSA 的 FlashMLA FP8 路径中，KV cache 存储的是 `[fp8_data, fp32_scale]`，kernel 内部读取 fp8 data 和 scale，**在 shared memory 中反量化为 bf16**：

```cuda
// dequant.h
bf16_val = fp8_to_float(fp8_val) * scale;  // 在 producer warpgroup 中完成
```

然后做 bf16 × bf16 的 MMA。这里 k_scale 在反量化时就消耗掉了，Q 本身是 bf16 不需要 scale。

两个 scale **一定都要乘回去**，否则数值就错了。区别只是在哪里乘、怎么融合：
- **deep_gemm 线性层**：kernel 内部自动乘两个 scale
- **MQA logits**：q_scale 预吸收到 weights，k_scale 在 kernel 内乘
- **FlashMLA fp8**：k_scale 在 shared memory 反量化时消耗，Q 本身是 bf16 无需 scale
