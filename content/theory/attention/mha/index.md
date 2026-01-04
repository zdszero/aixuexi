---
title: MHA
type: docs
weight: 13
---

### 单头注意力

多头注意力是单头注意力（SHA，single head attention）的升级版本，从理论层面来说，就是增加了 head 的并行个数，所以首先说明单头的理论基础。

假设我们正在处理一个序列，当前要计算第 \( t \) 个 token 的输出。输入是历史所有 token 的隐藏状态（包括当前）：
\[
\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}
\]

#### 线性投影

对每个位置 \( j \in \{1, 2, ..., t\} \)，通过可学习权重矩阵 \( W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_h} \) 投影：

\[
\begin{aligned}
\mathbf{q}_t &= W^Q \mathbf{h}_t \quad &\in \mathbb{R}^{d_h} \\
\mathbf{k}_j &= W^K \mathbf{h}_j \quad &\in \mathbb{R}^{d_h} \\
\mathbf{v}_j &= W^V \mathbf{h}_j \quad &\in \mathbb{R}^{d_h}
\end{aligned}
\]

{{< svg "/images/w_qkv.drawio.svg" >}}

#### 注意力分数计算

计算 Q 和 K 的点积，这里假设 Q 有 s 个 q 向量，K 有 t 个 k 向量，那么注意力分数的计算过程如下图所示：

{{< svg "/images/sha_attention_score.drawio.svg" "80%" >}}

图中每一个线条交错点在实际运算过程中就是一个注意力分数：

\[
\text{score}_{t,j} = \mathbf{q}_t^\top \mathbf{k}_j \quad \in \mathbb{R}
\]

这衡量了当前位置 \( i \) 与历史位置 \( j \) 的相关性。

#### 缩放

为防止点积过大导致 softmax 梯度消失，除以 \( \sqrt{d_h} \)：

\[
\text{scaled\_score}_{t,j} = \frac{\mathbf{q}_t^\top \mathbf{k}_j}{\sqrt{d_h}}
\]

#### Softmax 归一化

对所有 \( j = 1 \) 到 \( t \) 的 scaled scores 做 softmax，得到权重：

\[
\alpha_{t,j} = \text{Softmax}_j\left( \frac{\mathbf{q}_t^\top \mathbf{k}_j}{\sqrt{d_h}} \right) = 
\frac{\exp\left( \frac{\mathbf{q}_t^\top \mathbf{k}_j}{\sqrt{d_h}} \right)}{\sum_{l=1}^{t} \exp\left( \frac{\mathbf{q}_t^\top \mathbf{k}_l}{\sqrt{d_h}} \right)}
\]

这些 \( \alpha_{t,j} \) 构成一个概率分布，表示“在生成第 \( t \) 个 token 时，应关注前面哪些位置”。

#### 加权求和得到输出

用注意力权重对 value 加权求和：

\[
\mathbf{o}_t = \sum_{j=1}^{t} \alpha_{t,j} \cdot \mathbf{v}_j
\quad \in \mathbb{R}^{d_h}
\]

这就是单头注意力在位置 \( t \) 的输出。

### 多头注意力

现在引入 **\( n_h \) 个注意力头（heads）**，每个头独立学习不同的子空间表示。

#### 并行投影

将 \( \mathbf{h}_t \) 同时投影到所有头的 Q/K/V 空间（通常通过大矩阵一次完成）：

\[
\begin{aligned}
\mathbf{q}_t &= W^Q \mathbf{h}_t = [\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; \dots; \mathbf{q}_{t,n_h}] \quad &\in \mathbb{R}^{n_h d_h} \\
\mathbf{k}_j &= W^K \mathbf{h}_j = [\mathbf{k}_{j,1}; \mathbf{k}_{j,2}; \dots; \mathbf{k}_{j,n_h}] \quad &\in \mathbb{R}^{n_h d_h} \\
\mathbf{v}_j &= W^V \mathbf{h}_j = [\mathbf{v}_{j,1}; \mathbf{v}_{j,2}; \dots; \mathbf{v}_{j,n_h}] \quad &\in \mathbb{R}^{n_h d_h}
\end{aligned}
\]

其中每个 head 的向量维度为 \( d_h \)，总维度 \( d_{\text{model}} = n_h \cdot d_h \)。

> 实际实现中，\( W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \)，然后 reshape 成 \( (n_h, d_h) \)。

#### 注意力计算

对每个头 \( i = 1, \dots, n_h \)，独立执行单头注意力：

\[
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left( \frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h}} \right) \mathbf{v}_{j,i}
\quad \in \mathbb{R}^{d_h}
\]

注意：每个头有自己的 \( \mathbf{q}_{t,i}, \mathbf{k}_{j,i}, \mathbf{v}_{j,i} \)，且 softmax 仅在该头内部进行。

#### 输出拼接

将所有头的输出拼接起来：

\[
[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}] \quad \in \mathbb{R}^{n_h d_h}
\]

#### 输出投影

通过输出投影矩阵 \( W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \) 得到最终输出：

\[
\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}] \quad \in \mathbb{R}^{d_{\text{model}}}
\]

这就是多头注意力在位置 \( t \) 的最终输出。

---

下表总结了单头和多头注意力的不同：

| 步骤 | 单头注意力 | 多头注意力 |
|------|-----------|-----------|
| Q/K/V 投影 | \( \mathbf{q}_t = W^Q \mathbf{h}_t \) | \( \mathbf{q}_t = W^Q \mathbf{h}_t \)，然后 split 成 \( n_h \) 个 head |
| Attention 计算 | 1 个 attention map | \( n_h \) 个独立 attention maps |
| 输出 | \( \mathbf{o}_t \in \mathbb{R}^{d_h} \) | 拼接后 \( \in \mathbb{R}^{n_h d_h} \)，再经 \( W^O \) 变换 |
| 直觉 | 学习一种“关注模式” | 并行学习多种关注模式（如语法、语义、位置等） |
