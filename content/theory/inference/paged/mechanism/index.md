---
title: 分页机制
type: docs
description: Paged Attention
weight: 10
---

### 逻辑 token 和物理 KV

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

也就是说：

* 每个 token
* 对应一段 KV cache
* KV cache 存在 GPU 上的 paged block 里

### page

在 decode 阶段，每个 request 有一条 **逻辑上的 KV 序列**：

```
req i:
K = [k0, k1, k2, ..., k_{L-1}]
V = [v0, v1, v2, ..., v_{L-1}]
```

但实际存储在 `kv_cache` 里不是连续的，而是：

```
kv_cache:
[page0, page1, page2, ...]
```

#### page size

每个 page 存 page_size 个 token 的 kv cache：

```
Q: [page_size, num_kv_heads, qk_head_dim]
K: [page_size, num_kv_heads, v_head_dim]
```


如果是 MLA，kv 共享一个 latent，只需要存储：

```
latent: [page_size, kv_lora_rank]
```

#### page table

page table（block table）记录了每个 req 中所有 token 使用的 physical pages。

```
block_tables: [batch_size, num_pages_per_seq]
```

##### Find Token KV

如何查找到某个 token 的 kv 的物理地址：

**step 1**. 定位到该 req 需要使用到哪些 pages：`block_tables[req_id]`

对第 i 个请求：

```
block_tables[i] = [p3, p7, p1, ...]
```

表示：

👉 第 i 个序列的第 0 页在 page3
👉 第 1 页在 page7
👉 第 2 页在 page1

也就是：逻辑 page index  ->  物理 page index

**step 2**. 把“token index”映射到 page

对于 KV 的某个位置 `t`：

```
t ∈ [0, L)
```

计算：

```
logical_page_id  = t // page_size
offset   = t % page_size
```

**step 3**. 定位到物理地址

物理地址 = physical page address + offset

```
physical_page_id = block_tables[i, logical_page_id]
K = kv_cache[physical_page_id, 0, :, offset, :]
V = kv_cache[physical_page_id, 1, :, offset, :]
```

__整合逻辑__

```
for each query token:
    for t in [0, seq_len):
        page_id  = t // page_size
        offset   = t % page_size
        phy_page = block_tables[req_id, page_id]

        k = kv_cache[phy_page, ..., offset, :]
        v = kv_cache[phy_page, ..., offset, :]
```
