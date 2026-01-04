---
title: Transformer 模型结构
type: docs
weight: 50
---

### 符号定义

首先给出一系列 dimension 的符号表示：

\[
\begin{array}{|c|c|}
\hline
\text{symbol} & \text{dimension} \\
\hline
B & \text{batch size} \\
L & \text{number of layers} \\
T & \text{sequence length (query)} \\
S & \text{sequence length (key value)} \\
V & \text{vocab} \\
D & \text{embedding dimension} \\
F & \text{MLP intermediate dimension} \\
H & \text{attention head dimension} \\
N & \text{number of query heads} \\
K & \text{number of key/value heads} \\
G & \text{q heads per kv head} = N // K \\
\hline
\end{array}
\]

### Transformer Block

下图给出了一个当今大模型实现中的一个典型 transformer block 实现，它具备以下特点：

- attention 阶段支持 MHA/MQA/GQA 三种情况

{{< svg "/images/transformer_block.drawio.svg" "120%" >}}

### 估算法则

如果上下文比较短，然后忽略 attention 阶段的 self dot product 的计算量的话，那么计算量可以近似为：

\[
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 * BT * (3DF + 2D(N+K)H)L \\
    = 6 * \text{num tokens} * \text{parameter count}
\end{align*}
\]

这引出了一个著名的经验法则，用于估算密集型 Transformer 的浮点运算数（FLOP），同时忽略了注意力机制的浮点运算。 （去嵌入（Unembedding）是另一个简单的矩阵乘法，有 \(6BSDV\) 浮点运算和 \(DV\) 参数，也遵循相同的经验法则。）
