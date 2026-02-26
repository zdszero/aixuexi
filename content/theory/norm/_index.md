---
title: Norm
type: docs
description: Normalization
weight: 30
---

首先澄清 “归一化” 这个翻译，很多中文翻译都是十分不精确，导致概念混乱，建议直接记住英语，不用关系其中文。

Softmax 在数学上是概率归一化，但是在大模型中一般说到 Norm 指的是 LayerNorm 和 RMSNorm 这种名字有带有 Norm 的，代表的是 Norm 层，中文又把 Norm 翻译做归一化，就会引起混乱。

| 英文                        | 精确含义       |
| ------------------------- | ---------- |
| normalization             | 变到某种“标准尺度” |
| standardization           | 零均值单位方差    |
| scaling                   | 缩放         |
| probability normalization | 归一为概率      |

中文全都翻成：归一化，这就炸了。

| 方法        | 数学操作       | 更准确中文 |
| --------- | ---------- | ----- |
| LayerNorm | 减均值 / 除标准差 | 标准化   |
| RMSNorm   | 除 RMS      | 尺度归一  |
| Softmax   | 除以总和       | 概率归一  |
| MinMax    | 映射到[0,1]   | 线性归一  |
| 乘 γ       | scaling    | 缩放    |
