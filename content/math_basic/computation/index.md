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
| $T$ | sequence length (query) |
| $S$ | sequence length (key/value) |
| $V$ | vocab |
| $D$ | dimension of model (embedding size) |
| $F$ | MLP hidden dimension |
| $H$ | attention head dimension |
| $N$ | number of query heads |
| $K$ | number of key/value heads |
| $G$ | q heads per kv head |

在 MHA 中，query 的多头数量和 key/value 一致，都设置为 $H$。但是在 MQA 和 GQA 中，key/value 的头数量比 query 更少，上表中的 $K$ 和 $G$ 参数的引入也是为了方便对于这两种 attention 计算情况的论证：

- 对于 MQA，$K=1$，$G=N$。
- 对于 GQA，$K = N/G$，其中 $K>1$。

$G$ 的含义是一个 key/value 的头被几个 query 的头共用，所以 $K \times G = N$。

#### Embedding 阶段

Embedding 本质是一个查表操作（look-up），不是 gemm，计算量相对小。

- 输入：token id $[B,T]$
- 查表：$E[\text{Vocab}, D]$，取出对应行
- 输出：$X[B,T,D]$

该阶段计算量很小，几乎可以忽略不计。

#### Attention 阶段

Attention 阶段核心包含以下几个数学公式：

- $Q = X W_{Q}$
- $K = X W_{K}$
- $V = X W_{V}$
- $Y = \text{Attention}(Q, K, V) = \text{Softmax}(\frac{Q K^{T}}{\sqrt{d_k}})V$
- $Z = YW_{O}$
- $\text{Output} = \text{LayerNorm}(X+Z)$

Attention 计算可以分为三大部分：

- 线性投影/矩阵乘法（linear projection/gemm）：$X$ 和 $W_{Q}, W_{K}, W_{V}, W_{O}$ 相乘
- 注意力得分计算（attention score）
- 其他运算：layernorm

其中 __gemm__ 的计算量和访存量如下表所示：

| operation | inference FLOPs | params | output shape |
| :-: | :-: | :-: | :-: |
| $A[B,T,\textcolor{red}{D}] \cdot W_Q[\textcolor{red}{D},N,H]$ | $2BTDNH$ | $DNH$ | $[B,T,D,H]$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_K[\textcolor{red}{D},K,H]$ | $2BTDKH$ | $DKH$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_V[\textcolor{red}{D},K,H]$ | $2BTDKH$ | $DKH$ |
| $A[B,T,\textcolor{red}{N,H}] \cdot W_O[N,\textcolor{red}{H,D}]$ | $2BTDNH$ | $DNH$ |

其中 __attention score__ 的计算量如下表所示：

| operation | inference FLOPs |
|-----------|-----------------|
| $Q[\textcolor{blue}{B},T,\textcolor{blue}{K},G,\textcolor{red}{H}] \cdot K[\textcolor{blue}{B},S,\textcolor{blue}{K},\textcolor{red}{H}]$ | $2BTSKGH=2BTSNH$ |
| $\text{softmax}_{S}\ L[B,T,S,K,G]$ | $O(BTSKG)=O(BTSN)$ |
| $S[\textcolor{blue}{B},T,\textcolor{red}{S},\textcolor{blue}{K},G] \cdot V[\textcolor{blue}{B},\textcolor{red}{S},\textcolor{blue}{K},H]$ | $2BTSKGH=2BTSNH$ |

attention score 的计算量其实取决于 $T$（q length）

#### MLP/MOE 阶段

首先说一说 MLP，MLP 在当前 transformer 的模型中有两种常见实现方式，一种是 up/down，另一种是 in1/in2/out。

第一种 up/down 就是经典的 transformer 论文中提到的两层线性层，包含三个数学公式：

- $H_{\text{up}} = \sigma(XW_{\text{up}} + b_{\text{up}})$
- $H_{\text{down}} = H_{\text{up}}W_{\text{down}} + b_{\text{down}}$
- $\text{Output} = \text{LayerNorm}(X + H_{\text{down}})$

第二种方式是 in1/in2/out，两个 in 是并行的线性映射，一个负责主通道（值），一个负责门控（控制开关）。比传统 up/down 更灵活，计算量略多，但性能通常更好。

- $U = XW_{\text{in1}} + b_{\text{in1}}$
- $G = XW_{\text{in1}} + b_{\text{in2}}$
- $H_{\text{gated}} = \sigma(G) \odot U$
- $H_{\text{out}} = H_{\text{gated}} W_{\text{out}} + b_{\text{out}}$
- $\text{Output} = \text{LayerNorm}(X + H_{\text{out}})$

现在 transformer 架构通常使用第二种方式，几个核心的操作都是 gemm 操作，其计算量如下：

| operation | inference FLOPs | params |
|---|---|---|
| $A[B,T,\textcolor{red}{D}] \cdot W_{in1}[\textcolor{red}{D},F]$ | $2BTDF$ | $DF$ |
| $A[B,T,\textcolor{red}{D}] \cdot W_{in2}[\textcolor{red}{D},F]$ | $2BTDF$ | $DF$ |
| $\sigma(A_{in1})[B,T,F] * A_{in2}[B,T,F]$ | $O(BTF)$ | |
| $A[B,T,\textcolor{red}{F}] \cdot W_{out}[\textcolor{red}{F},D]$ | $2BTDF$ | $DF$ |

---

如果是使用 MOE 的模型，主要包含以下几个数学公式：

- 路由器：$g= W_{gate}x$
- 选择 k 专家：$S = TopK(g,k)$
- 归一化权重：$w_{i} = \frac{exp(g_i)}{\sum_{j \in S} exp(g_i)}, i \in S$
- 专家计算：$f_{i}(x) = \sigma(x W_{i,1})W_{i,2}$
- 组合：$y = \sum_{i=1}^{N} w_i f_i(x)$

#### Unembedding 阶段
