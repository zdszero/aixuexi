---
title: 语法糖
type: docs
description: Torch 经典语法糖
weight: 20
---

### 形状变换

1️⃣  **view / reshape / flatten** → 逻辑重新解释维度

```python
x.view(b, s, h, d)
x.reshape(-1, hidden)
x.flatten(0, 1)
```

| API     | 是否要求连续        | 是否可能复制 |
| ------- | ------------- | ------ |
| view    | 必须 contiguous | 不复制    |
| reshape | 不要求           | 可能复制   |
| flatten | 本质 reshape    | 可能复制   |

2️⃣  **permute / transpose** → 转置

```python
x.permute(0, 2, 1, 3)
x.transpose(1, 2)
```

注意：

- permute 不会复制
- 但 permute 后通常 不连续
- 很多 kernel 前要 `.contiguous()`

3️⃣  **unsqueeze / squeeze**

插入/维度一个维度（1）

```python
x.unsqueeze(-1)
```

### 广播

从 **右往左对齐维度，逐维比较**：

规则：

- 两个维度相等 ✅
- 其中一个是 1 ✅（可以扩展）
- 否则 ❌ 报错

### stride

在 PyTorch（以及 NumPy）中，`stride` 是一个**元组**，表示为了在内存中沿着某个维度移动到下一个元素，需要跨过多少个“存储单元”（通常是一个元素的大小，比如 4 字节的 float）。

```python
import torch

x = torch.arange(12).reshape(3, 4)  # 形状 (3, 4)
print(x.stride())  # 输出 (4, 1)
```
- 形状 `(3, 4)` 表示有 3 行、4 列。
- `stride = (4, 1)` 的含义：
  - 沿着**第 0 维**（行）移动一行，需要在内存中跳过 4 个元素（因为每行有 4 列）。
  - 沿着**第 1 维**（列）移动一列，只需要跳过 1 个元素。

内存布局（一维数组索引）：
```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
行0: 0 1 2 3
行1: 4 5 6 7  ← 行1第一个元素在内存索引 4，比行0第一个元素索引大 4
行2: 8 9 10 11
```

### 广播语法糖

1️⃣  **None / Ellipsis**

```
x[:, None, :, :]
x[..., None]
```

功能和 unsqueeze 类似，`...` 表示位置前后的所有维度。

2️⃣  **自动广播**

```
x + bias

x: (bs, seq, hidden)
bias: (hidden,)
```

会自动广播为：

```
(bs, seq, hidden)
```

### 索引

### 矩阵

```python
# 矩阵乘法
y = x @ w
torch.matmul(x, w)

# 3D 矩阵乘法
torch.bmm(x, y)

# 灵活矩阵乘法
torch.einsum("bshd,bthd->bhst", q, k)

```
