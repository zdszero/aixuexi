---
title: 架构
type: docs
description: Normalization
weight: 10
---

### Norm 位置

根据 Layer Normalization 放置的不同位置，可以分为 Pre-Norm 和 Post-Norm 两种常见形式。

{{< svg "/images/pre_post_ln.svg" "90%" >}}

#### Pre-Norm

在 Pre-Norm 结构中，Layer Normalization 被放置在残差连接之前。这意味着对于每一个输入向量\(\mathbf{u}_t^l\)（其中\(t\) 表示时间步或序列中的位置，\(l\) 表示层号），首先通过 Layer Normalization 进行标准化，然后将结果传递给前馈神经网络 (FFN)。最后，原始输入与经过 FFN 处理后的输出相加得到最终的隐藏状态\(\mathbf{h}_t^l\)。具体公式如下：

\[
\mathbf{h}_t^l = \mathbf{u}_t^l + \mathrm{FFN}(\mathrm{LayerNorm}(\mathbf{u}_t^l))
\]

使用 Pre Norm 的完整 transformer block 服从以下公式：

\[ x^{1}_{l,i} = \text{LayerNorm}(x_{l,i}) \]

\[ x^{2}_{l,i} = \text{MultiHeadAtt}(x^{1}_{l,i}, [x^{1}_{l,1}, \cdots, x^{1}_{l,n}]) \]

\[ x^{3}_{l,i} = x_{l,i} + x^{2}_{l,i} \]

\[ x^{4}_{l,i} = \text{LayerNorm}(x^{3}_{l,i}) \]

\[ x^{5}_{l,i} = \text{ReLU}(x^{4}_{l,i} W^{1,l} + b^{1,l}) W^{2,l} + b^{2,l} \]

\[ x_{l+1,i} = x^{3}_{l,i} + x^{5}_{l,i} \]

#### Post-Norm

与 Pre-Norm 相反，在 Post-Norm 结构里，Layer Normalization 被置于残差连接之后。也就是说，先将输入\(\mathbf{u}_t^l\) 直接送入前馈网络 (FFN)，再将其与 FFN 的结果相加，最后对这个总和执行 Layer Normalization 操作以获得\(\mathbf{h}_t^l\)。其数学表达式为：

\[
\mathbf{h}_t^l = \mathrm{LayerNorm}\left( \mathbf{u}_t^l + \mathrm{FFN}(\mathbf{u}_t^l) \right)
\]

使用 Post Norm 的完整 transformer block 服从以下公式：

\[ x^{1}_{l,i} = \text{MultiHeadAtt}(x_{l,i}, [x_{l,1}, \cdots, x_{l,n}]) \]

\[ x^{2}_{l,i} = x_{l,i} + x^{1}_{l,i} \]

\[ x^{3}_{l,i} = \text{LayerNorm}(x^{2}_{l,i}) \]

\[ x^{4}_{l,i} = \text{ReLU}(x^{3}_{l,i} W^{1,l} + b^{1,l}) W^{2,l} + b^{2,l} \]

\[ x^{5}_{l,i} = x^{3}_{l,i} + x^{4}_{l,i} \]

\[ x_{l+1,i} = \text{LayerNorm}(x^{5}_{l,i}) \]
