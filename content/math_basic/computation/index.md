---
title: 数理基础
type: docs
description: 本节分析 transformer 模型在 attention 和 MLP 的各阶段的计算量。
weight: 1
---

### 矩阵乘法

本节重点阐述矩阵乘法的计算量分析，这是后续 Transformer 模型的计算量分析的基础。

#### 向量和二维矩阵

假设有两个向量 $x$、$y$ 和两个矩阵 $A$、$B$ 有如下的形状：

| 向量/矩阵 | 形状 |
|---|---|
| $x$ | $[K]$ |
| $y$ | $[K]$ |
| $A$ | $[M, K]$ |
| $B$ | $[K, N]$ |

那么 向量/矩阵 之间的浮点计算量如下：

- 向量点积 $x \cdot y$ 需要执行 $K$ 次乘法和 $K-1$ 次加法，总共需要约 $2K$ 次浮点运算。
- 矩阵向量乘法 $Ax$ 等同于 $K$ 次向量点积，计算量为 $2MK$。
- 矩阵乘法 $AB$ 等同于对于矩阵 $B$ 的每一列都与矩阵 $A$ 进行一次矩阵向量乘法，总共计算量为 $2MKN$。

#### 高维矩阵

高维矩阵的计算量分析更加复杂一些，因为其中的维度分为三种情况：

- 收缩维度（$\textcolor{red}{\text{contracting dimensions}}$）
    - 这是两个张量在相乘时需要 __求和消去的维度__。
    - 它们在两个输入张量中同时出现，但不会出现在结果中。
- 批处理维度（$\textcolor{blue}{\text{batching dimensions}}$）
    - 在乘法过程中，这些维度会被保留，并且 __在批次上并行执行乘法__。
    - 它们在两个张量中同时出现，并且会保留到输出结果里。
- 自由维度（$\textcolor{green}{\text{free dimensions}}$）
    - 那些 __只在一个输入张量中出现__ 的维度。
    - 在乘法过程中不会被求和，而是直接保留到输出结果里。

为了更好地理解这三种维度，我们首先来看一个简单的例子：

假设我们有一个张量 $A$，其形状为 $(\textcolor{blue}{B}, \textcolor{green}{M}, \textcolor{red}{K})$，以及另一个张量 $B$，其形状为 $(\textcolor{blue}{B}, \textcolor{red}{K}, \textcolor{green}{N})$。我们希望计算 $C = A \times B$，在这种情况下：

- $\textcolor{blue}{B}$ 是批处理维度：它在 $A$ 和 $B$ 中都存在，并且会保留到结果 $C$ 中。
- $\textcolor{red}{K}$ 是收缩维度：它在 $A$ 和 $B$ 中都存在，但在计算过程中会被求和，从结果 $C$ 中消失。
- $\textcolor{green}{M}$ 和 $\textcolor{green}{N}$ 是自由维度：$\textcolor{green}{M}$ 只存在于 $A$ 中，$\textcolor{green}{N}$ 只存在于 $B$ 中，它们都会保留到结果 $C$ 中。

结果张量 $C$ 的形状为 $(\textcolor{blue}{B}, \textcolor{green}{M}, \textcolor{green}{N})$。可以观察到，收缩维度被消去，批处理维度只保留一份，自由维度都被保留。矩阵乘法 $A \times B$ 的计算量为 $2\textcolor{blue}{B}\textcolor{green}{MN}\textcolor{red}{K}$。

### 推理过程的计算量分析

#### 符号表示

在推导大模型推理阶段的计算量之前，首先需要引入一系列符号用于表示推理过程中的关键概念：

| 符号 | 含义             |
|-----|------------------|
| $B$ | batch size       |
| $L$ | number of layers |
| $T$ | sequence length  |
