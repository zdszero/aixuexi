---
title: Roofline
type: docs
weight: 120
---

### 瓶颈分析

在深度学习中，模型的计算本质上可以看作是大量矩阵乘法的组合，而矩阵乘法又由浮点乘法和加法等基本运算（FLOPs）构成。模型在计算阶段所花费的时间，取决于所需的计算量以及加速器本身的算力，其估计公式为：

\[
\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}
\]

除了计算之外，数据在系统中的流动同样会带来时间开销。无论是芯片内部的数据访问，还是跨芯片、跨节点的通信，这部分成本通常用带宽（Bytes/s）来衡量，对应的通信时间可以估算为：

\[
\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}
\]

在大多数（但并非所有）情况下，计算与通信是可以部分甚至高度重叠的，例如在执行计算的同时预取数据，或在计算下一层时传输上一层的中间结果。基于这一点，我们通常可以用**计算时间和通信时间中的较大者**来作为训练或推理耗时的一个**下界估计**；而在最保守的情况下，则可以用两者之和作为**上界估计**。

\[
\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}
\]

\[
\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}
\]

如果以 \(\max(T_\text{math}, T_\text{comms})\) 为优化目标，那么上下界之间的差距最多只有 2 倍，因为始终有：

\[
T_{\text{math}} + T_{\text{comms}} \leq 2 \times \max(T_{\text{math}}, T_{\text{comms}})
\]

这种近似在工程实践中非常常见：一方面数学形式更简单，另一方面通过精心设计的流水线和调度机制，往往可以接近这一理想下界。进一步的精度提升，则需要引入对“重叠区间”和系统额外开销的建模，这通常依赖于对具体模型和硬件平台的 profiling 结果。

---

如果假设计算与通信能够做到**完全重叠**，那么系统的瓶颈就由两者中更慢的一方决定：

* 当 \( T_{\text{math}} > T_{\text{comms}} \) 时，整体性能受限于计算能力，此时加速器可以被充分利用，我们称这种情况为**计算受限（compute-bound）**。
* 当 \( T_{\text{comms}} > T_{\text{math}} \) 时，系统主要在等待数据传输，部分算力被闲置，这种情况称为**通信受限（communication-bound）**。

为了判断某个算子或算法更可能属于哪一类，我们通常引入一个关键指标——**算术强度（Arithmetic Intensity，也称 Operational Intensity）**。

### 计算访存比

计算访存比（Arithmetic Intensity，也叫做 Compute-to-Memory Ratio）的 **定义** 如下：算法执行过程中，每传输 1 字节数据所能完成的浮点运算数量，也就是：

\[
\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}
\]

也就是说，算术强度刻画了 “**每字节数据能榨出多少 FLOPs**”。从一阶近似的角度看：

* 算术强度高 → 计算占主导，\( T_{\text{math}} \gg T_{\text{comms}} \)，算力利用率高；
* 算术强度低 → 通信占主导，系统花大量时间在搬数据上，FLOPs 难以充分发挥。

对于一套具体硬件而言，存在一个“分界点”，即硬件的**峰值算术强度**，它由计算峰值与带宽峰值之比决定：

\[
\text{Peak Intensity} = \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}}
\]

当算法的算术强度高于这一值时，系统趋向于计算受限；反之则更容易落入通信受限状态。这一点可以通过以下等价关系清晰地表示出来：

\[
\begin{aligned}
T_{\text{math}} > T_{\text{comms}}
&\quad\Leftrightarrow\quad 
\frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} 
\;>\; 
\frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} \\[0.5em]
&\quad\Leftrightarrow\quad 
\frac{\text{Computation FLOPs}}{\text{Communication Bytes}} 
\;>\; 
\frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \\[0.5em]
&\quad\Leftrightarrow\quad 
\text{Intensity}_{\text{Computation}} \;>\; \text{Intensity}_{\text{Accelerator}}
\end{aligned}
\]

这一视角也是 **Roofline Model** 等性能分析方法的核心基础，用来指导我们在模型结构、算子实现和并行策略上的优化方向。

以 Nvidia A100 80GB 为例来直观说明计算与显存带宽之间的关系。在计算能力方面，A100 在 FP16 / BF16 Tensor Core 精度下的理论峰值约为 **312 TFLOPs**（不考虑 2:4 稀疏带来的额外加速，这也是 Roofline 一般使用的标准）。在内存系统方面，其 **HBM2e 显存带宽约为 1.55 TB/s**。

将二者放在同一量纲下，可以得到该 GPU 的 **计算访存比**：

\[
\text{Intensity} = \frac{312 \times 10^{12}\ \text{FLOPs/s}}{1.55 \times 10^{12}\ \text{Bytes/s}}
\approx 200\ \text{FLOPs/Byte}
\]

这意味着：只有当算子的算术强度（Arithmetic Intensity）达到或超过约 200 FLOPs/Byte 时，才能真正进入计算受限（compute-bound）区间，从而充分发挥 A100 的峰值算力；否则，性能将主要受限于显存带宽（memory-bound）。

### 矩阵乘法

来看一个即将成为我们最常用的算法：矩阵乘法（简称 matmul）。  
记作 \( X * Y \rightarrow Z \)，其中：  
- \( X \) 的形状为 \( \text{bf16}[B, D] \)  
- \( Y \) 的形状为 \( \text{bf16}[D, F] \)  
- \( Z \) 的形状为 \( \text{bf16}[B, F] \)  

执行该矩阵乘法时，需要：  
- 加载数据量：\( 2DF + 2BD \) 字节  
- 浮点运算次数：\( 2BDF \) 次  
- 写回结果量：\( 2BF \) 字节  

因此：

\[
\begin{equation} \text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF} \end{equation}
\]

这里将问题简化一下，我们假设 \(B\) 相比 \(D\) 和 \(F\) 是一个比较小的值，那么我们可以这样估计 gemm 的计算访存比：

\[
\begin{equation} \frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B \end{equation}
\]

\[
\begin{equation} \text{Intensity}(\text{matmul}) > \text{Intensity}(\text{A100}) \implies B > 200 \end{equation}
\]

需要注意的是，这里的 \(B\) 并不是 sequence 维度的 batch size，
