---
title: Linear Attention
type: docs
weight: 27
---

### Softmax Attention

\[
\begin{aligned}
\boldsymbol{q}_i, \boldsymbol{k}_i, \boldsymbol{v}_i, \boldsymbol{o}_i &\in \mathbb{R}^{d \times 1} \\
\boldsymbol{Q} &= [\boldsymbol{q}_1, \boldsymbol{q}_2, \cdots, \boldsymbol{q}_n]^\top \in \mathbb{R}^{n \times d} \\
\boldsymbol{K} &= [\boldsymbol{k}_1, \boldsymbol{k}_2, \cdots, \boldsymbol{k}_n]^\top \in \mathbb{R}^{n \times d} \\
\boldsymbol{V} &= [\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n]^\top \in \mathbb{R}^{n \times d} \\
\boldsymbol{O} &= [\boldsymbol{o}_1, \boldsymbol{o}_2, \cdots, \boldsymbol{o}_n]^\top \in \mathbb{R}^{n \times d} \\[6pt]
\text{注意力分数矩阵：} & \quad \boldsymbol{S} = \frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d}} \in \mathbb{R}^{n \times n} \\[2pt]
\text{注意力权重矩阵：} & \quad \boldsymbol{A} = \mathrm{softmax}(\boldsymbol{S}) \in \mathbb{R}^{n \times n} \\[2pt]
\text{输出矩阵：} & \quad \boldsymbol{O} = \boldsymbol{A} \boldsymbol{V} \in \mathbb{R}^{n \times d}
\end{aligned}
\]

计算量为 \(BT^{2}HD\)，主要在于计算注意力得分矩阵的过程中 \(QK\) 时 q_len == kv_len = T，计算结果的 shape 为 \([B, H, T, T]\)

### Linear Attention

#### insight

根据矩阵乘法结合率，如果去掉 Softmax 操作的话，可以使用 \(QK^{T}V = Q(K^{T}V)\)，这样就降低了计算复杂度。

考虑完整的 MHA，\(Q, K, V\) 的 shape 都是 \([B, T, H, D]\)，当我们计算 \(K^{T} V\) 时：

- \(B, H\) 是 batching dimension
- \(D\) 是 free dimension
- \(T\) 是 contracting dimension

\(K^{T} V\) 计算结果的 shape 为 \([B, H, D, D]\)，计算量为 \(BTHD^2\)

所以如果使用这种方案的话，计算复杂度从 \(T^2\) 变成了 \(D^2\)

#### recursion

\[o_{t}=\sum_{j=1}^{t} v_{j}\left(k_{j}^{\top} q_{t}\right)=\sum_{j=1}^{t}\left(v_{j} k_{j}^{\top}\right) q_{t}=\left(\sum_{j=1}^{t} v_{j} k_{j}^{\top}\right) q_{t}\]

如果我们记括号部分为 \(S_t\)，那么有

\[o_{t}=S_{t} q_{t}, \quad S_{t}=S_{t-1}+v_{t} k_{t}^{\top}\]

使用这种方式后，Causal 格式的 Attention 体现为一个以 \(S_{t}\) 为 State 的 RNN。

但是这种最简单的递归方式本质上就是一个 cumsum，所有历史信息都等权地叠加，不难想象当叠加的 token 足够多时，每个 token 的信息占比都会变得极小，无法重建信息。

所以问题在于如何在递归的过程可以动态地赋予 \(v_{j} k_{j}^{\top}\)

#### evolution

| 模型/方法             | 公式                                                                 |
|-----------------------|----------------------------------------------------------------------|
| Softmax Attention     | \((\exp(QK^\top) \odot M)V\)                                          |
| 最早的线性Attention   | \((QK^\top \odot M)V\)                                                |
| 加入遗忘门后          | \((QK^\top \odot \Gamma)V\)                                           |
| DeltaNet              | \((QK^\top \odot M)(I + KK^\top \odot M^-)^{-1}V\)                    |
| Gated DeltaNet        | \(((QK^\top \odot M)(I + KK^\top \odot M^-)^{-1} \odot \Gamma)V= (QK^\top \odot \Gamma)(I + KK^\top \odot \Gamma^-)^{-1}V\) |

其中

\[
\Gamma_{i,j} = 
\begin{cases}
\displaystyle \prod_{\tau=j+1}^{i} \gamma_\tau, & i > j \\
1, & i = j \\
0, & i < j
\end{cases}
\]
