---
title: 方案
type: docs
description: Norm 可以采用的一些常见方法介绍。
weight: 20
---

### 数据处理层面

Normalization 的英文本身含义就是让数据变得 Normal，但是根据使用场景常常拥有不同的语义。

一个是在数据处理层面，另外一个是处理模型内部的数值。

#### Min-Max

(最小-最大归一化)

*   **公式**：\(x' = \frac{x - \min(x)}{\max(x) - \min(x)}\)
*   **结果**：数据被压缩到 **\([0, 1]\)** 区间。
*   **用途**：图像处理（像素值 0-255 转 0-1）、某些对范围敏感的算法。

#### Standardization

*   **公式**：\(x' = \frac{x - \mu}{\sigma}\) （减去均值，除以标准差）
*   **结果**：数据变为**均值为 0，方差为 1**的标准正态分布。**注意：数值范围不再是 0-1，可能是 -3 到 +3 甚至更大。**
*   **用途**：大多数深度学习模型的首选预处理方式（如 ImageNet 的均值减法）。

{{< details "Standardization 和 LayerNorm 的语义区别" >}}
Standardization 是：

> 在数据进入模型之前做的

LayerNorm 是：

> 在每一层之间做的

- 前者是数据工程
- 后者是数值工程
{{< /details >}}

### 模型层面

#### LayerNorm

\[y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta\]

- \(y\) 是经过 Layer Normalization 处理后的输出。
- \(x\) 表示输入张量中的元素。
- \(\mathbb{E}[x]\) 代表输入张量 \(x\) 的所有元素的平均值（期望），即：
  \[\mathbb{E}[x] = \frac{1}{N} \sum_{i=1}^{N} x_i\]
  这里 \(N\) 是指 \(x\) 中元素的数量。
- \(\mathrm{Var}[x]\) 是 \(x\) 的方差，用来衡量数据分散程度的一个统计量，计算方式如下：
  \[\mathrm{Var}[x] = \mathbb{E}[(x - \mathbb{E}[x])^2] = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mathbb{E}[x])^2\]
- \(\sqrt{\mathrm{Var}[x]}\) 则是标准差，表示数据相对于均值的离散程度。
- \(\epsilon\) 是一个很小的正数（如 \(1e-5\) 或者 \(1e-6\)），加在分母上是为了防止当方差接近于零时发生除以零的情况。
- \(\gamma\) 和 \(\beta\) 分别是缩放和平移因子，它们都是可学习的参数，允许模型调整归一化后数据的分布。

以上公式已经高度凝练过，也可以通过以下四个公式进行理解：

\[
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
\]
\[
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
\]
\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]
\[
\text{LayerNorm}(\mathbf{x})_i = y_i = g_i \cdot \hat{x}_i + b_i
\]


Layer Normalization 的主要目的是使网络各层的输入具有更加稳定的分布，从而加速训练过程并可能提高模型性能。通过上述步骤，每个样本内部特征之间的相对差异被保留下来，同时整体分布变得更为稳定。这种方法特别适用于循环神经网络 (RNNs) 和变长序列处理任务中，因为它是在每个样本内部对特征进行归一化，而不是跨样本。

#### RMSNorm

RMSNorm 是 LayerNorm 的一个简化变体。它认为 LayerNorm 的成功主要来自于其重新缩放不变性，而减去均值的操作不是必须的。因此，它只对输入的均方根进行归一化。

\[y = \frac{x}{\sqrt{\underbrace{\frac{1}{n} \sum_{i=1}^n x_i^2}_{\text{均方值}} + \epsilon}} \cdot \gamma\]

- \(x\)：输入张量，表示模型中的某一层的输出或激活。
- \(x_i\)：\(x\) 中的第 \(i\) 个元素。
- \(n\)：\(x\) 的元素数量（即维度大小）。
- \(\sum_{i=1}^n x_i^2\)：对 \(x\) 中所有元素的平方求和。
- \(\frac{1}{n} \sum_{i=1}^n x_i^2\)：计算 \(x\) 中所有元素平方后的平均值，这被称为均方值。
- \(\epsilon\)：一个小的正数（通常设置为 \(10^{-5}\) 或 \(10^{-6}\)），用于防止分母为零的情况，确保数值稳定性。
- \(\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 + \epsilon}\)：计算均方根（Root Mean Square, RMS），这是归一化因子。
- \(\gamma\)：缩放因子，是学习得到的参数，允许模型调整归一化后数据的尺度。

通过这种方式，RMSNorm 对输入张量 \(x\) 进行了归一化处理，而不需要像 LayerNorm 那样先减去均值，从而简化了运算过程并减少了需要训练的参数数量（因为它没有偏置项 \(\beta\)）。这种设计使得 RMSNorm 在某些情况下比 LayerNorm 更高效。


### 归一语义

很好，我们把三种“归一”的**数学约束条件**直接写出来，对比它们到底在“归一”什么。

我只写最核心的“归一结果满足什么条件”。

---

1️⃣ **Softmax —— 概率归一**

Softmax function

定义：

\[
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
\]

它满足两个数学约束：

\[
\sum_i y_i = 1
\]

\[
y_i > 0
\]

**语义**

* 把任意实数向量映射到 **概率单纯形**
* 归一对象：**L1 范数（总和）**
* 作用空间：概率空间

👉 它控制的是“总质量”

---

2️⃣ **LayerNorm —— 方差归一**

Layer Normalization

定义（忽略 γ、β）：

\[
y_i = \frac{x_i - \mu}{\sigma}
\]

其中：

\[
\mu = \frac{1}{d}\sum_i x_i
\]

\[
\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
\]

它满足：

\[
\frac{1}{d}\sum_i y_i = 0
\]

\[
\frac{1}{d}\sum_i y_i^2 = 1
\]

也就是说：

\[
\mathrm{Var}(y) = 1
\]

**语义**

* 均值归零
* 方差归一
* 控制分布形状

👉 它控制的是“中心 + 波动幅度”

---

3️⃣ **RMSNorm —— 二阶矩归一**

RMSNorm

定义（忽略 γ）：

\[
y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2}}
\]

它满足：

\[
\frac{1}{d}\sum_i y_i^2 = 1
\]

注意：

\[
\frac{1}{d}\sum_i y_i \neq 0
\]

**语义**

* 不减均值
* 只控制平均平方值
* 控制“能量规模”

👉 它控制的是“二阶矩”

---

🔥 三者统一对比

| 方法        | 归一对象    | 数学约束            | 几何意义      |
| --------- | ------- | --------------- | --------- |
| Softmax   | L1 和    | \( \sum y_i =1 \) | 投影到概率单纯形  |
| LayerNorm | 均值 + 方差 | 均值=0，Var=1      | 投影到零均值单位球 |
| RMSNorm   | 二阶矩     | \( E[y^2]=1 \)    | 投影到固定半径球面 |

---

#### 统一表达

可以写成：

\[
\text{Normalize}(x) = \frac{x}{|x|_p}
\]

不同方法只是：

* Softmax：本质类似 L1 归一（但带指数）
* RMSNorm：L2 归一
* LayerNorm：中心化后 L2 归一

---

Softmax：

> 归一“概率结构”

LayerNorm：

> 归一“统计分布”

RMSNorm：

> 归一“能量尺度”

---

* Softmax 让向量“像概率”
* LayerNorm 让向量“像标准正态”
* RMSNorm 让向量“长度固定”
