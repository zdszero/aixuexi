---
title: 推理范式
type: docs
description: 推理范式的演变
weight: 10
---

### 范式演变

```
第一阶段：静态图 vs 动态图之争（2016）
第二阶段：动态图 + 编译器融合（2020）
第三阶段：动态图 + 编译器 + CUDA Graph（大模型时代）
```

时间线大概是：

**第一代**

静态图框架（TensorFlow 1.x）

**第二代**

ONNX + TensorRT Engine 编译

**第三代（现在）**

动态框架 + CUDA Graph + 手写 kernel

### ONNX

ONXX 本质上是

> 一种中间表示（IR，计算图格式）

```
PyTorch
   ↓ export
ONNX 文件（.onnx）
   ↓
TensorRT 编译
   ↓
Engine 文件（.plan）
```


tensorrt 基于 onnx 计算图构建 runtime，做了以下优化：

1. 完全静态图
2. kernel 融合
3. 内存布局重排
4. tactic 搜索
5. 几乎无 Python 参与

runtime 是一种专用编译产物，非常快，但是编译速度慢、产物巨大、只能在某种特定硬件运行（比如 GPU 中的 A100）、灵活性很差、十分麻烦。

### 权衡

如果完全是

> 单 kernel 超大融合

静态图 + 编译器就够了。

但 LLM 结构太复杂：

- Attention
- KV cache
- MoE
- AllReduce

不可能完全 fuse。

所以现在进入：

> “静态图不够，还要静态执行”

也就是 CUDA Graph。
