---
title: Quant Math
type: docs
description: scale in quantization
weight: 10
---


### 通用量化模型

对于任意张量 (X)，量化可以抽象为：

\[
X \approx \hat{X} = s_X \cdot Q_X
\]

其中：

* \(Q_X\)：低精度表示（int8 / fp8 / nf4 等）
* \(s_X\)：scale（可以是：
  * per-tensor
  * per-channel
  * per-group）
* \(\hat{X}\)：反量化后的近似值


### 核心计算目标

**GEMM 的通用形式**

\[
Y = XW
\]

量化后：

\[
X \approx s_X Q_X,\quad W \approx s_W Q_W
\]

代入：

\[
Y \approx (Q_X Q_W) \cdot (s_X s_W)
\]


> **在低精度空间完成大部分计算，同时最小化恢复到高精度的代价**

也就是：

\[
\text{目标：} \quad XW \approx (Q_X \cdot Q_W) \cdot (s_X \cdot s_W)
\]

### 理解视角

把 GEMM 拆成两部分：

#### 低精度核心计算

\[
Z = Q_X Q_W
\]

特点：

* 完全在低精度执行（FP8 / INT8）
* 吞吐最大
* 是**主要算力消耗**

####  高精度缩放恢复

\[
Y = Z \cdot (s_X s_W)
\]

特点：

* 标量 / 向量乘
* 可以：
  * fuse 到 epilogue
  * 或延迟到后续算子

#### 进一步抽象

我们可以写成一个**统一计算框架**：

\[
Y = f(Q_X, Q_W, s_X, s_W)
\]

其中：

\[
f = \text{LowPrecisionMatmul}(Q_X, Q_W) \cdot g(s_X, s_W)
\]

### 优化目标

大模型推理中，真正优化的是：

#### 计算效率

最大化：

\[
\text{Throughput}(Q_X Q_W)
\]

也就是：

* Tensor Core 利用率
* memory bandwidth 利用率

---

#### 精度误差最小化

最小化：

\[
|XW - (Q_X Q_W) s_X s_W|
\]

这决定：

* FP8 vs INT8 vs NF4
* per-channel vs per-tensor
* scale 动态/静态

###  scale 的计算/存储成本

你那句其实点到了关键：

```
需要额外乘两个 scale
```

工程上目标是：

👉 **让 scale 成本“消失”**

方法：

* fuse scale 到：
  * bias
  * activation
  * 下一层 weight
* 或：
  * 提前合并：\(s_X s_W → s_{fused}\)

---

其实可以把整个量化推理写成：

\[
\boxed{
\min_{Q_X, Q_W, s_X, s_W}
|XW - s_X s_W (Q_X Q_W)|
}
\]

约束：

* \(Q_X, Q_W\) ∈ 低精度集合（FP8 / INT8）
* scale 受限（硬件友好）

