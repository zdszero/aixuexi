---
title: NoPE
type: docs
description: NoPE
weight: 5
---

In transformer-based LLMs, positional encoding is typically necessary because self-attention treats tokens independently of order. Absolute position embeddings solve this by adding an additional embedding layer that adds information to the token embeddings.

> attention 计算某个 qk 之间的相关性之时不会考虑两者的位置关联，所以需要在编码中考虑这一部分信息

{{< img "/images/abs-encode.jpg" >}}

the [NoPE](https://arxiv.org/abs/2305.19466) paper not only found that no positional information injection is necessary, but it also found that NoPE has better length generalization, which means that LLM answering performance deteriorates less with increased sequence length
