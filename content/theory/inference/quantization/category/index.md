---
title: Category
type: docs
description: Categories of quantization methods
weight: 15
---

### 量化粒度

{{< svg "/images/quantization_granularity_v2.svg" >}}

粒度从粗到细：

| 粒度 | 含义 | scale 数量 |
|------|------|-----------|
| **Per-tensor** | 整个张量共享一个 scale | 1 |
| **Per-channel / Per-token** | 权重按输出通道、激活按 token 各一个 scale | N 或 M |
| **Per-group** | 输入通道每 group_size 个共享一个 scale | N × (K/group_size) |
| **Per-block** | 二维分块，如 128×128 的 tile 一个 scale（DeepSeek FP8 用的就是这种） | 按 block 数 |

Per-group 内还可以细分：
- **顺序分组**：通道 [0..127] 一组、[128..255] 一组（常见，Kimi K2.5 就是这种）
- **act_order 分组**：先按激活幅度排序再分组（GPTQ desc_act=True）

### 量化对象

量化对象有两个：weight（权重）和 activation（激活）

> 默会前提：现代大模型量化，默认原始精度 ≈ BF16
>
> 如果 W4，就是代表 weight 用 4bit，4 < 16，就是用了量化
> 如果 A8，就是代表 activation 用 8bit，8 < 16，用了量化


这是一个重要维度，命名惯例如 `WxAy` 表示权重 x-bit、激活 y-bit：

- **Weight-only（W4A16, W8A16）**：只量化权重，激活保持 FP16/BF16。推理时 on-the-fly 反量化权重再与 FP16 激活相乘。Kimi K2.5 就是 W4A16。
- **Weight + Activation（W8A8, W4A8）**：权重和激活都量化，计算在低精度域完成（如 INT8 matmul 或 FP8 matmul），吞吐更高。
- **KV Cache 量化**：单独对 attention 的 KV cache 量化（如 FP8 KV），节省显存。

### 数据类型

两大阵营：

#### 整数

- **INT8**：`x_q = round(x / scale) + zero_point`，反量化 `x = (x_q - zp) * scale`
- **INT4 / UINT4**：同上，但 4-bit。又分：
  - **对称量化**（symmetric）：zero_point = 0，`x_q = round(x / scale)`。GPTQ 的 uint4b8 就是偏移 8 的对称量化
  - **非对称量化**（asymmetric）：有 zero_point，需要额外存储 `qzeros`。AWQ 用这种

#### 浮点

- **FP8 (E4M3 / E5M2)**：IEEE 风格的 8-bit 浮点。E4M3 精度更高（推理常用），E5M2 动态范围更大（训练常用）
- **FP4 (NF4)**：QLoRA 提出的 NormalFloat4，假设权重近似正态分布，4-bit 非均匀量化
- **MXFP（Microscaling）**：block-wise 的微缩放浮点，如 MXFP4/MXFP8

关键区别：整数量化是**均匀映射**（量化值等间距），浮点量化是**非均匀映射**（小值处密、大值处疏），更符合权重的实际分布。

先说明几个单词：

> Sign：符号位
> Exponent（E）：指数位
> Mantissa（M）：小数位

| 浮点格式  | 符号位 | 指数位 | 小数位 | 量化 group size（典型值）  |
|-----------|--------|--------|--------|----------------------------|
| bfloat16  | 1      | 8      | 7      | 通常不分组或按层量化       |
| fp16      | 1      | 5      | 10     | 通常不分组或按层量化       |
| fp8(e4m3) | 1      | 4      | 3      | 128（常见于权重分组）      |
| fp8(e5m2) | 1      | 5      | 2      | 128（常见于权重分组）      |
| fp8(e8m0) | 1      | 8      | 0      | 128（blackwell 默认使用）  |
| fp4(e2m1) | 1      | 2      | 1      | 32 或 64（常用于权重）     |
| mxfp8     | 1      | 4      | 3      | 128（类似 fp8(e4m3)）      |
| mxfp4     | 1      | 2      | 1      | 32 或 64（类似 fp4(e2m1)） |

### 量化算法

上面说的是"格式"，还有一个维度是"怎么选 scale"：

- **RTN（Round-to-Nearest）**：直接取 min/max 算 scale，最简单
- **GPTQ**：基于 Hessian 逆矩阵逐列量化 + 误差补偿，数学上是二阶优化
- **AWQ（Activation-Aware）**：根据激活幅度给重要通道乘一个保护系数再量化，等效于调整 per-channel scale
- **SmoothQuant**：把激活的离群值"平滑"到权重上（`diag(s) × W × diag(1/s)`），使两者都更好量化
- **AQLM / QuIP**：向量量化 / lattice 量化，多个权重联合编码，压缩率更高但 kernel 复杂

### Example

```
量化方案 = 粒度 × 对象 × 数据类型 × 算法

Kimi K2.5 的位置:
  粒度:   per-group (group_size = 32)
  对象:   weight-only (W4A16)
  类型:   INT4 对称 (uint4b8, 无 qzeros)

DeepSeek V3/R1 的位置:
  粒度:   Per-block 128×128
  对象:   weight+activation (W8A8)
  类型:   FP8 (e4m3)
```


> uint4b8 就是 [0, 15] - 8，范围是和 int4 一样的，那为什么要用 uint4
> 
> 区别纯粹是存储编码层面的：unsigned 4-bit 在 bit packing 到 int32 时不需要处理符号位扩展，实现上更简单。GPU kernel 解包时直接 (v >> shift) & 0xF 就行，不用额外做符号位处理。
