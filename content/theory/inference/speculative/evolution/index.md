---
title: 演变史
type: docs
description: 技术演变历史
weight: 10
---

| 阶段                   | 核心思想                    |
| -------------------- | ----------------------- |
| Speculative sampling | 小模型 draft               |
| Lookahead            | target self draft       |
| Medusa               | multi-head draft        |
| Eagle                | hidden state prediction |
| Eagle2               | tree draft              |
| Eagle3               | trajectory learning     |

### Transformer Block Sematic

很多 probing 研究发现不同层次有不同语义：

| 层   | 主要作用                          |
| --- | ----------------------------- |
| 前几层 | 语法结构                          |
| 中间层 | 语义关系                          |
| 后几层 | token distribution sharpening |

如果在中间层加一个 LM Head，预测的 __top-1 token__ 往往是一致的，不同点在于 logit margin。比如在中间可能是 P(token1) = 0.55, P(token2) = 0.30，但是跑完所有层可能是 P(token1) = 0.85，P(token2) = 0.05。

所以 Transformer 有一个 __渐进确定性__：

- layer 0 → 不确定
- layer 10 → 大致确定
- layer 30 → 非常确定
- layer 40 → sharpening

这种现象也被叫做 representation crystallization（语义结晶）。

### Eagle

__hidden state prediction__

传统 speculative decoding，draft model 直接预测 token。eagle 提出应该支持预测 hidden state：

\[h_t → h_{(t+1)} → h_{(t+2)} → h_{(t+3)}\]

因为 token 空间大，假设 vocab_size=100k，那么预测 token 相当于 100k 分类问题。但是 hidden state 是连续向量空间，预测 hidden state 相当于 vector regression。

__使用模型的前若干层进行预测__

思路：前几层已经包含大部分语义信息，只差概率 refinement。

```
context
   │
只跑前 8~12 层
   │
hidden state
   │
Eagle predictor
   │
draft tokens
```
