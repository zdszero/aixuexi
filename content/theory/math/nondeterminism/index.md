---
title: Nondeterminism
type: docs
weight: 140
---

### floating-point non-associativity

在真实的硬件计算中，浮点计算不像数学上一样严格地遵循结合率：

\[(a + b) + c \ne a + (b + c)\]

```python
(0.1 + 1e20) - 1e20
>>> 0
0.1 + (1e20 - 1e20)
>>> 0.1
```

本质原因在于浮点数在计算机中是 **有限精度 + 舍入（rounding）**

计算机里的数是：

\[
\text{float} = sign \times mantissa \times 2^{exponent}
\]

特点：

* 有限位数（比如 fp32 只有 23 位 mantissa）
* 每一步运算都要 **round**

### batch invariance

矩阵乘法满足：

\[
\begin{bmatrix}
A \\
B
\end{bmatrix}
C = \begin{bmatrix}
AC \\
BC
\end{bmatrix}
\]

这就是所谓的：

👉 **对“batch 维度”是线性的 / 可分解的**

> **无论你怎么切 batch（甚至 batch size = 1），最终结果不变**

换句话说：

```python
# 一次性算
Y = X @ W

# 分 batch 算
Y1 = X[:k] @ W
Y2 = X[k:] @ W
Y = concat(Y1, Y2)
```

👉 两者完全一样（数值上）


### MSE

> 对每个 token 的 logprob 差值平方，再取平均

数学形式：

\[\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2\]

其中：

* \(x_i\)：单条推理的 logprob
* \(y_i\)：batch 推理的 logprob

