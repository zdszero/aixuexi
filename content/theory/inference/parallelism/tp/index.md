---
title: TP
type: docs
description: Tensor Parallelism 并行详解
weight: 10
---

在分布式模型训练中，张量并行（Tensor Parallelism）是一种将运算拆分到多个 GPU 上的关键技术。其核心思想是将输入矩阵或权重矩阵沿特定维度进行分割，使每个设备仅计算部分结果，最后通过集合通信操作合成最终结果。

### 通信原语

#### Reduce

#### AllReduce

{{< svg "/images/reduce_op.drawio.svg" >}}

- **Reduce 语义：** 所有进程的输入值进行归约（如求和），结果只保存在指定的一个进程（通常是 Rank 0）中。
- **AllReduce 语义：** 所有进程的输入值进行归约（如求和），结果被广播给所有进程。

| 操作       | 输出位置           | 数学表达式                            |
|------------|--------------------|----------------------------------------|
| Reduce     | 只在 Rank 0        | \( \text{out}_0 = \sum_{i=0}^{N-1} \text{in}_i \) |
| AllReduce  | 所有 rank 都有     | \( \forall i, \text{out}_i = \sum_{j=0}^{N-1} \text{in}_j \) |

{{< region note >}}
这里的操作是“求和”作为示例，实际中也可以是 `max`, `min`, `prod` 等归约操作。
{{< /region >}}

#### Gather 

#### AllGather

{{< svg "/images/gather_op.drawio.svg" >}}

- **Gather 语义：** 将多个进程（通常每个进程位于不同的设备或节点上）持有的不同数据块，收集到一个指定的目标进程（根进程）中。目标进程的输出张量是所有输入张量的拼接。其他非根进程通常需要提供输入，但不会收到完整的输出结果。
- **AllGather 语义：** 将多个进程持有的不同数据块，收集到**所有**进程中。执行后，每个进程都拥有一个相同的、完整的输出张量，该张量是所有进程输入张量的拼接。

| 操作 | 输出位置 | 数学表达式 |
| :--- | :--- | :--- |
| **Gather** | 仅根进程 (Rank \( r \)) | \( \text{Output}_r = \text{concat}(\text{Input}_0, \text{Input}_1, ..., \text{Input}_{n-1}) \) <br> \( \text{Output}_i = \text{None} \quad (\text{对于所有 } i \ne r) \) |
| **AllGather** | 所有进程 | \( \text{Output}_i = \text{concat}(\text{Input}_0, \text{Input}_1, ..., \text{Input}_{n-1}) \quad (\text{对于所有 Rank } i) \) |

#### Scatter 

#### ReduceScatter

{{< svg "/images/scatter_op.drawio.svg" >}}

- **Scatter 语义：** 将根进程（通常为 rank 0）拥有的一个完整数据缓冲区，按照固定的顺序，分割成若干块，并分发到通信组中的所有进程（包括根进程自身）。每个进程收到且只收到数据的一个子块。
- **ReduceScatter 语义：** 通信组中的每个进程都拥有一个完整的缓冲区。首先，在所有进程之间按元素进行规约操作（如求和、求最大值等），得到一个全局规约结果。然后，将这个全局结果缓冲区进行分割，并按照固定的顺序，将不同的数据块分发（Scatter）给不同的进程。每个进程最终只得到全局规约结果的一个子块。

| 操作 | 输出位置 | 数学表达式 |
| :--- | :--- | :--- |
| **Scatter** | 所有进程接收不同的数据块。 | 假设根进程（rank 0）有数据 \([x_0, x_1, ..., x_{n-1}]\)，进程 \(i\) 接收到的数据为 \(x_i\)。 |
| **ReduceScatter** | 所有进程接收规约后结果的不同数据块。 | 假设进程 \(i\) 拥有数据 \([d_i^{(0)}, d_i^{(1)}, ..., d_i^{(n-1)}]\)。首先进行逐元素全局规约（以求和为例），得到 \([\sum_i d_i^{(0)}, \sum_i d_i^{(1)}, ..., \sum_i d_i^{(n-1)}]\)，然后进程 \(i\) 得到第 \(i\) 块数据 \(\sum_j d_j^{(i)}\)。 |

#### All2All

{{< svg "/images/all2all_op.drawio.svg" >}}

- **All2All 语义：** All2All 操作允许通信组中的每个进程将自己的一部分数据发送给组内的其他所有进程。具体来说，假设有一个由 \(P\) 个进程组成的通信组，每个进程都有一个大小为 \(N\) 的缓冲区。在执行 All2All 操作后，每个进程都会从其他每个进程中接收一部分数据，并且这些接收到的数据会被组织成一个新的缓冲区，该缓冲区的总大小同样为 \(N\)。这样，最终每个进程都拥有了一份来自其他所有进程的数据副本集合。这个过程确保了信息在整个通信组中均匀分布，适用于需要跨多个节点共享或同步数据的应用场景。

| 操作 | 输出位置 | 数学表达式 |
| :--- | :--- | :--- |
| **All2All** | 每个进程接收到来自其他所有进程的数据部分。 | 假设共有 \(P\) 个进程，每个进程初始时持有数据 \([d_0, d_1, ..., d_{N-1}]\)。对于任意两个进程 \(i\) 和 \(j\)，\(i\) 将其数据的第 \(j/P\) 部分发送给 \(j\)；完成交换后，进程 \(i\) 将会拥有一个新数组 \([d_{i,0}, d_{i,1}, ..., d_{i,N-1}]\)，其中 \(d_{i,k}\) 表示来自某个特定进程的数据片段。|

### 矩阵乘法切分策略

考虑矩阵 \(A B\) 相乘，我们可以考虑三种切分方式：

1. 将 \(B\) 按照列切（列拆分）
2. 将 \(A\) 按照行切（行拆分）
3. 将 \(A\) 按照列切，将 \(B\) 按照行切分（行列拆分）

#### 列拆分

{{< svg "/images/tp_col_split.svg" "120%" >}}

**适用场景**：当权重矩阵 \(B\) 的列数（即输出特征维度）非常大时，适合沿列维度进行拆分。

**拆分方式**：

- 将权重矩阵 \(B\) 沿**列**维度均匀拆分为 \(N\) 块，其中 \(N\) 为并行设备数。
- 输入矩阵 \(A\) **保持完整**地分发到所有设备。

\[AB = A [B_1, B_2, \cdots, B_{N-1}] = [C_1, C_2, \cdots, C_{N-1}] = C\]

矩阵 \(B\) 的每个子矩阵可以分别与矩阵 \(A\) 相乘，在不同的设备上进行计算，计算完之后之后做一个 all gather 进行结果聚合：

\[
\begin{aligned}
B &= [B_0, B_1, \dots, B_{N-1}] \\
&\Downarrow \quad \text{每个设备 } k \text{ 计算} \\
C_k &= A \times B_k \\
&\Downarrow \quad \text{All-Gather 聚合} \\
C &= [C_0, C_1, \dots, C_{N-1}]
\end{aligned}
\]

其中 \(A \in \mathbb{R}^{L \times K}\)， \(B \in \mathbb{R}^{K \times M}\)， \(B_k \in \mathbb{R}^{K \times \frac{M}{N}}\)，\(C_k \in \mathbb{R}^{L \times \frac{M}{N}}\)，\(C \in \mathbb{R}^{L \times M}\)。

#### 行列拆分

{{< svg "/images/tp_row_col_split.svg" "120%" >}}

**适用场景**：当权重矩阵 \(B\) 的行数（即输入特征维度）非常大，或当输入 \(A\) 的批次/序列维度很大时，适合沿行维度进行拆分。

**拆分方式**：
- 将权重矩阵 \(B\) 沿**行**维度均匀拆分为 \(N\) 块。
- 同时，将输入矩阵 \(A\) 沿**列**维度进行对应的拆分，以匹配 \(B\) 的行分割。

用数学公式表示为：
\[
A = [A_0, A_1, ..., A_{N-1}], \quad B = \begin{bmatrix} B_0 \\ B_1 \\ \vdots \\ B_{N-1} \end{bmatrix}
\]
其中，\(A_k\) 的形状为 \((L, \frac{K}{N})\)，\(B_k\) 的形状为 \((\frac{K}{N}, M)\)。

**计算过程**：
每个设备 \(k\) 计算局部结果：
\[
C_k = A_k \times B_k
\]
这里，每个 \(C_k\) 的形状与最终输出 \(C\) 相同，均为 \((L, M)\)，但每个 \(C_k\) 只是基于局部输入和权重计算的**部分和**。

**通信与聚合**：
为了得到最终结果，需要对所有设备的局部结果 \(C_k\) 进行求和。这通过 **All-Reduce**（或更优化的 **Reduce-Scatter**）操作实现。
\[
C = \sum_{k=0}^{N-1} C_k \quad \text{（在每个设备上通过 All-Reduce 后得到）}
\]

### 通信量分析

#### 估算模型

在分布式训练或推理中，通信时间通常可以通过以下公式进行估算：

\[
T_{\text{comm}} =\frac{\text{Bytes per GPU}}{\text{Effective bandwidth per GPU}}
\]

这个公式的核心思想是：**从“单个 GPU 的视角”出发，在通信进入稳定状态后，估算其完成全部通信所需的时间。**

其中 **分子** 的含义为每个 GPU 的通信量，**分母** 的含义为 每个 GPU 的有效带宽。下面通过几个问题来回答估算过程中的常见疑问：

{{< details "为什么要用 per GPU 的通信量？" >}}
在 AllReduce、AllGather、ReduceScatter 等集体通信中：
* 通信是 **并行发生的**
* 所有 GPU **同时在发送和接收数据**
* 整体通信时间由 **最慢的一张 GPU** 决定

因此，在时间建模时，**不关心系统总通信量，而关心“单张 GPU 需要完成多少通信”。**

以 **Ring AllGather** 为例，假设总数据大小为 \( S \)，GPU 数为 \( N \)。
Ring AllGather 的通信过程是：
* 数据被划分为 \( N \) 份
* 每一轮中，每个 GPU 向右邻居发送 \( \frac{S}{N} \)，同时从左邻居接收 \( \frac{S}{N} \)
* 共进行 \( N - 1 \) 轮

因此，每个 GPU 的通信量为：
\[
\text{Bytes per GPU} = (N-1) \times \frac{S}{N} = \frac{N-1}{N} S
\]
{{< /details >}}

{{< details "在涉及到 ring 的实现中，send 和 recv 为什么只算“一份”？" >}}
以 Ring AllGather 为例：

* 每个 GPU 在每一轮中 **同时进行 send 和 recv**
* 这两个操作在硬件和通信库（如 NCCL）中是 **并行执行的**
* 在稳定阶段，send 和 recv 会被 **流水化（pipelined）**，并不会形成时间上的串行依赖

因此，在时间估算时，**只需要计算“单向 payload 的总量”，而不是 send 和 recv 的简单相加。** 通信时间由 **最慢的一侧链路吞吐量** 决定，而不是由“发送量 + 接收量”决定。
{{< /details >}}

{{< details "为什么不用 NVLink 的标称带宽？" >}}
以 H100 为例：
* 单 GPU 的 NVLink 标称聚合带宽约为 **900 GB/s（双向）**

但在真实的集体通信中：
* 不可能同时、持续地用满所有 NVLink
* 通信算法只暴露有限数量的并发数据流
* 存在协议、调度、流水线、仲裁等开销

因此，**直接使用 NVLink 的理论带宽会严重高估性能。**
{{< /details >}}

{{< details "什么是有效带宽？" >}}
**Effective bandwidth per GPU** 定义为：在给定通信算法和硬件拓扑下，**单个 GPU 在通信稳定阶段能够持续达到的有效 payload 吞吐率**。

它综合反映了：
* 通信算法的并行度（如 Ring / Tree）
* 拓扑结构（NVSwitch / 直连 NVLink）
* 通信库实现（如 NCCL 的 pipeline 和 chunk 策略）
* 实际硬件限制（DMA 引擎、仲裁机制等）

一个经验性的范围（以 H100 为例）

在 **Ring、AllGather、AllReduce** 等典型操作中：
* NVLink 标称聚合带宽：\( 900\ \text{GB/s} \)
* 实际可持续达到的有效带宽通常为：
\[
\text{Effective bandwidth per GPU} \approx 450 \sim 600\ \text{GB/s}
\]
也就是大约 **50%–70% 的理论峰值**。

在性能建模或教程中，常用：
* **保守估计**：450–500 GB/s
* **乐观估计**：550–600 GB/s
{{< /details >}}

#### 对比

\[
\begin{array}{|c|c|}
\hline
\textbf{通信类型} & \textbf{计算公式} \\[4pt]
\hline
\textbf{all\_reduce (ring)} & 2 \times (P-1) \times \text{local\_bytes} \\[8pt]
\hline
\textbf{all\_reduce (tree)} & 2.0 \times P \times \text{local\_bytes} \\[8pt]
\hline
\textbf{all\_gather (ring)} & (P-1) \times \text{local\_bytes} \\[8pt]
\hline
\textbf{reduce\_scatter} & (P-1) \times \text{local\_bytes} \\[8pt]
\hline
\textbf{all\_to\_all} & (P-1) \times \text{local\_bytes} \\[8pt]
\hline
\textbf{broadcast} & \text{local\_bytes} \\[8pt]
\hline
\textbf{reduce} & \text{local\_bytes} \\[8pt]
\hline
\textbf{gather} & \frac{P-1}{P} \times \text{total\_bytes} \\[8pt]
\hline
\textbf{scatter} & \frac{P-1}{P} \times \text{total\_bytes} \\[8pt]
\hline
\textbf{dispatch (MoE)} & \text{tokens} \times (\text{moe\_top\_k}-1) \times \text{dim} \times \text{dbytes} \\[8pt]
\hline
\textbf{combine (MoE)} & \text{tokens} \times (\text{moe\_top\_k}-1) \times \text{dim} \times \text{dbytes} \\[8pt]
\hline
\end{array}
\]

