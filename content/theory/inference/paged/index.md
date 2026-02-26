---
title: 分页机制
type: docs
description: Paged Attention
weight: 10
---

### 两层 Pool

在大模型推理中（尤其是 serving 场景）：

- 每个 request 长度不同
- 每个 request 动态增长（decode 逐 token）
- KV cache 体积巨大
- 不能频繁 cudaMalloc / free
- 不能因为某个 request 结束就整体搬移内存

所以必须做：

- **逻辑 token 管理**：`req_to_token_pool`
- **物理 KV 存储管理**：`token_to_kv_pool`


你这个问题问得非常好，本质已经触到 **vLLM / PagedAttention 内存管理模型的核心设计**了。

我直接用推理系统视角给你讲清楚三层语义：

### req_to_token

`req_to_token_pool` 是：

> request 视角的逻辑 token 分配表

语义：

```
request_id  ->  token_index list
```

你可以理解为：

```python
req_to_token_pool[req_id] = [t0, t1, t2, ...]
```

这些 token_index 是：

> 全局 token id（逻辑 token id）

不是物理地址。

**它解决什么问题？**

* request 有多少 token
* decode 新 token 时往哪里 append
* request 结束时释放哪些 token

它只管理：

> token 的逻辑生命周期

不关心 KV 在哪。

### token_to_kv

它是：

> token_index -> KV 物理存储位置

可以理解为：

```python
token_to_kv_pool[token_id] = physical_block_id + offset
```

也就是说：

* 每个 token
* 对应一段 KV cache
* KV cache 存在 GPU 上的 paged block 里

### Paged Attn

PagedAttention 的核心思想是：

> KV cache 不连续存储，而是分 block（类似操作系统分页）

PagedAttention 解决的是：

1️⃣ 避免内存碎片

因为 request 长度不同，如果连续存：

```
req1: 100
req2: 1000
req3: 5
```

删除 req2 后会出现巨大空洞。

Paged 把 KV 切成固定 block，比如 16 tokens/block：

```
[block1][block2][block3]...
```

释放时只回收 block。

---

2️⃣ 支持共享

prefix cache / speculative decode 里：

多个 request 可能共享前缀 token

只需要让：

```
req_to_token_pool
```

指向相同 token_index

而 token_to_kv_pool 指向同一 block

无需复制 KV。

---

3️⃣ 支持动态扩展

decode 时：

* 新 token
* 从 free block list 拿一个 block
* 绑定到 token_id

完全 O(1) 扩展。

### Example

假设：

block_size = 4 token

req1 输入 6 token：

```
req_to_token_pool[1] = [0,1,2,3,4,5]
```

token_to_kv_pool:

```
0 -> block0 offset0
1 -> block0 offset1
2 -> block0 offset2
3 -> block0 offset3
4 -> block1 offset0
5 -> block1 offset1
```

block1 剩下两个空位可以给别的 request 用。

这就是 paged 的意义。
