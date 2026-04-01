---
title: memory pool
type: docs
description: 内存池的设计
weight: 20
---

### 核心问题：如何为 Attention 算子组织 KV Cache

在深入实现细节之前，我们需要先理解一个根本问题：**为什么需要 Memory Pool？它的存在解决的是什么问题？**

#### 问题起源

Transformer 模型在推理时，每生成一个新 token，都需要访问之前所有 token 的 Key 和 Value（KV Cache）。这带来三个核心挑战：

1. **动态长度**：不同请求的序列长度不同，且每个请求的长度都在持续增长
2. **内存碎片**：频繁的分配和释放会产生内存碎片，降低内存利用率
3. **共享需求**：多个请求可能有相同的 prefix，共享 KV Cache 可以节省计算和内存

传统的连续内存分配方式难以同时解决这三个问题。这引出了 **Page 机制** 的设计思想。

---

### 核心抽象：三层映射关系

SGLang 的 Memory Pool 本质上建立了一个**三层映射架构**，让逻辑上的 token 序列能够找到物理上的 KV 数据：

```
┌─────────────────────────────────────────────────────────────────┐
│                     逻辑视图 (Logical View)                     │
│                                                                 │
│   Request A:  [Token_0] [Token_1] [Token_2] [Token_3] ...       │
│   Request B:  [Token_0] [Token_1] [Token_2] ...                 │
│   Request C:  [Token_0] [Token_1] ...                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              第一层映射：ReqToTokenPool                         │
│                                                                 │
│   将「请求 + token位置」映射到「KV Cache槽位索引」              │
│                                                                 │
│   req_to_token[req_pool_idx, token_pos] = kv_cache_index        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              第二层映射：TokenToKVPoolAllocator                 │
│                                                                 │
│   管理 kv_cache_index 的分配与释放（物理槽位分配器）            │
│                                                                 │
│   Page Size = 1: 连续分配                                       │
│   Page Size > 1: 页式分配（PagedAttention）                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              第三层映射：KVCache                                │
│                                                                 │
│   实际存储 KV 数据的物理内存                                    │
│                                                                 │
│   kv_buffer[layer_id][kv_cache_index] = (K, V)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

这个三层架构的设计蕴含了深刻的工程智慧：

- **ReqToTokenPool** 解决「请求维度」的寻址问题
- **TokenToKVPoolAllocator** 解决「物理内存」的分配问题
- **KVCache** 解决「数据存储」的实际问题

三者分离，各司其职，又通过索引紧密关联。

---

### Page 机制的深层含义

#### 为什么是 Page？

Page 的概念来自操作系统虚拟内存管理。在 KV Cache 场景中，Page 的引入解决了两个关键问题：

**问题一：内存碎片**

假设不使用 Page，每次为一个 token 分配一个槽位。当请求 A 需要扩展，但紧邻的槽位已被请求 B 占用时，只能分配到其他位置。这导致：
- 请求 A 的 KV 数据分散在各处
- 需要额外的索引结构记录每个 token 的位置
- 访问模式变得不规则，影响性能

**Page 的解决方案**：以 Page（如 64 个 token）为单位分配。同一请求的数据倾向于聚集在相同的 Page 中，减少了索引开销。

**问题二：共享效率**

多个请求共享 prefix 时，如果不使用 Page，需要逐个 token 复制或引用。Page 级别的共享只需引用 Page 编号，效率更高。

#### Page Size 的权衡

```
Page Size 小（如 1）：
  优点：精确分配，无浪费
  缺点：索引开销大，碎片化严重

Page Size 大（如 64）：
  优点：索引开销小，便于共享
  缺点：内部碎片，分配延迟
```

SGLang 支持可配置的 Page Size，允许根据场景选择：

- **Page Size = 1**：连续分配模式，适合短序列或内存紧张场景
- **Page Size > 1**：页式分配模式，适合长序列和 prefix 共享场景

---

### req_to_token 与 token_to_kv 的深层关联

#### 数据结构对比

| 特性 | req_to_token | token_to_kv（由 Allocator 管理） |
|------|-------------|--------------------------------|
| 维度 | `[max_batch, max_context_len]` | 一维索引池 |
| 含义 | 每个请求每个位置的 KV 槽位 | 物理 KV Cache 槽位 |
| 分配单位 | 按「请求」分配一行 | 按 Page 分配槽位 |
| 生命周期 | 与请求绑定 | 与 KV 数据绑定 |

#### 索引流转过程

当 Attention 算子需要访问 KV Cache 时，发生了一次**索引解引用链**：

```
步骤 1: 获取请求的 pool index
        req_pool_idx = request.req_pool_idx

步骤 2: 找到该请求所有 token 的 KV 索引
        kv_indices = req_to_token[req_pool_idx, :seq_len]

步骤 3: 通过 KV 索引访问实际数据
        K = kv_cache.k_buffer[layer_id][kv_indices]
        V = kv_cache.v_buffer[layer_id][kv_indices]
```

这个设计的关键洞察是：**分离「寻址」和「存储」**。

- `req_to_token` 负责「寻址」：知道数据在哪里
- `kv_cache` 负责「存储」：知道数据是什么
- `allocator` 负责「管理」：知道哪里有空位

#### 为什么需要两层索引？

一个直观的疑问是：为什么不直接让 `req_to_token` 指向物理地址？

答案在于**灵活性**：

1. **内存重分配**：当需要整理内存碎片时，只需修改 `kv_cache` 的物理布局，无需遍历所有请求更新 `req_to_token`

2. **Prefix 共享**：多个请求的 `req_to_token` 可以指向相同的 `kv_cache_index`，实现零拷贝共享

3. **CPU Offload**：当 KV Cache 需要换入换出时，`req_to_token` 保持不变，只移动 `kv_cache` 的数据

---

### Attention 算子与元数据的结合

#### Attention 需要哪些元数据？

Attention 算子执行时，需要回答以下问题：

```
Q1: Query 来自哪些位置？        → qo_indptr
Q2: Key/Value 来自哪些位置？    → kv_indptr, kv_indices
Q3: 每个请求有多少 KV？         → seq_lens
Q4: 最后一个 Page 有多少有效？  → kv_last_page_len
```

这些问题的答案，都需要从 Memory Pool 的数据结构中推导出来。

#### 元数据构建流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     ScheduleBatch                               │
│                                                                 │
│   requests: [Req1, Req2, ...]                                   │
│   req_pool_indices: [req_pool_idx_1, req_pool_idx_2, ...]       │
│   seq_lens: [seq_len_1, seq_len_2, ...]                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 构建元数据
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Attention Backend Metadata                      │
│                                                                 │
│   kv_indptr:    [0, seq_len_1, seq_len_1+seq_len_2, ...]        │
│                 CSR 格式的指针数组，标记每个请求的 KV 范围      │
│                                                                 │
│   kv_indices:   [所有请求的 kv_cache_index 展平]                │
│                 通过 create_flashinfer_kv_indices_triton 生成   │
│                                                                 │
│   kv_last_page_len: [每个请求最后一页的有效长度]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 传入 Attention Kernel
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FlashInfer / Triton Backend                   │
│                                                                 │
│   for each request i:                                           │
│       kv_start = kv_indptr[i]                                   │
│       kv_end = kv_indptr[i+1]                                   │
│       my_kv_indices = kv_indices[kv_start:kv_end]               │
│       K = gather(kv_cache.k_buffer, my_kv_indices)              │
│       V = gather(kv_cache.v_buffer, my_kv_indices)              │
│       Attention(Q[i], K, V)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 关键转换：req_to_token → kv_indices

`create_flashinfer_kv_indices_triton` 是连接 Memory Pool 和 Attention Backend 的桥梁：

```python
## 输入：
##   req_to_token: [max_batch, max_context_len] - 所有请求的 KV 索引映射
##   req_pool_indices: [batch_size] - 当前批次请求的 pool index
##   page_kernel_lens: [batch_size] - 每个请求的序列长度

## 输出：
##   kv_indices: [total_kv_tokens] - 展平的 KV 索引数组

## 逻辑（简化版）：
for i in range(batch_size):
    req_pool_idx = req_pool_indices[i]
    seq_len = page_kernel_lens[i]
    kv_indices[kv_indptr[i]:kv_indptr[i+1]] = req_to_token[req_pool_idx, :seq_len]
```

这个 kernel 的高效性至关重要，因为它在每个 forward pass 都会执行。

---

### 代码实现的关键节点

#### 核心类关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                      ReqToTokenPool                             │
│                                                                 │
│   属性：                                                        │
│     - req_to_token: Tensor [size, max_context_len]             │
│     - free_slots: List[int]                                    │
│                                                                 │
│   方法：                                                        │
│     - alloc(reqs) → req_pool_indices                           │
│     - free(req)                                                │
│     - write(indices, values)                                   │
│                                                                 │
│   文件：python/sglang/srt/mem_cache/memory_pool.py:126         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 为请求分配索引槽位
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TokenToKVPoolAllocator (基类)                      │
│                                                                 │
│   派生类：                                                      │
│     ├─ TokenToKVPoolAllocator      (page_size=1, 连续分配)      │
│     ├─ PagedTokenToKVPoolAllocator (page_size>1, 页式分配)      │
│     └─ SWATokenToKVPoolAllocator   (滑动窗口注意力)             │
│                                                                 │
│   核心方法：                                                    │
│     - alloc(need_size) → kv_cache_indices                       │
│     - alloc_extend(prefix_lens, seq_lens, last_loc)             │
│     - alloc_decode()                                            │
│     - free(indices)                                             │
│                                                                 │
│   文件：python/sglang/srt/mem_cache/allocator.py                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 管理 KV 槽位
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        KVCache (基类)                           │
│                                                                 │
│   派生类：                                                      │
│     ├─ MHATokenToKVPool    (标准多头注意力)                     │
│     ├─ MLATokenToKVPool    (DeepSeek MLA)                       │
│     ├─ NSATokenToKVPool    (Native Sparse Attention)            │
│     └─ DoubleSparseTokenToKVPool                                │
│                                                                 │
│   核心属性：                                                    │
│     - k_buffer: List[Tensor] per layer                          │
│     - v_buffer: List[Tensor] per layer                          │
│                                                                 │
│   核心方法：                                                    │
│     - get_key_buffer(layer_id)                                  │
│     - get_value_buffer(layer_id)                                │
│     - set_kv_buffer(layer, loc, k, v)                           │
│                                                                 │
│   文件：python/sglang/srt/mem_cache/memory_pool.py:601          │
└─────────────────────────────────────────────────────────────────┘
```

#### 内存分配的两个关键路径

**路径一：Prefill（Extend）阶段**

```
场景：处理新的 prompt，需要为大量 token 分配 KV Cache

1. ReqToTokenPool.alloc(reqs)
   → 为每个请求分配一行 req_to_token

2. PagedTokenToKVPoolAllocator.alloc_extend(
      prefix_lens,    # 已缓存的 prefix 长度
      seq_lens,       # 扩展后的总长度
      last_loc        # 上次分配的位置
   )
   → 智能复用未满的 Page
   → 分配新的完整 Page
   → 分配部分 Page（如果需要）

3. write_cache_indices(req_pool_indices, new_kv_indices)
   → 将新分配的 KV 索引写入 req_to_token
   → 使用 Triton kernel 高效执行
```

**路径二：Decode 阶段**

```
场景：每次生成一个新 token，需要最小化分配

1. PagedTokenToKVPoolAllocator.alloc_decode()
   → 如果当前 Page 未满，复用
   → 如果当前 Page 已满，分配新 Page
   → 返回单个 token 的 kv_cache_index

2. 更新 req_to_token[req_pool_idx, seq_len] = new_index
```

#### Prefix Cache：Radix Tree 的角色

Radix Tree（基数树）位于 Memory Pool 之上，实现了 **Prefix 共享**：

```
┌─────────────────────────────────────────────────────────────────┐
│                       RadixCache                               │
│                                                                 │
│   TreeNode:                                                     │
│     - key: token_ids (这个节点代表的 token 序列)               │
│     - value: kv_cache_indices (这些 token 的 KV 索引)          │
│     - children: dict[token_id -> TreeNode]                     │
│     - lock_ref: 引用计数（有多少请求在使用）                   │
│                                                                 │
│   核心操作：                                                    │
│     - match_prefix(tokens) → matched_node, matched_len        │
│       查找最长匹配的 prefix                                    │
│                                                                 │
│     - insert(key, value)                                       │
│       插入新的 prefix 节点                                     │
│                                                                 │
│   文件：python/sglang/srt/mem_cache/radix_cache.py             │
└─────────────────────────────────────────────────────────────────┘
```

当新请求到达时：

```
1. 在 Radix Tree 中 match_prefix(request_tokens)
   → 找到可复用的 prefix

2. 复用 prefix 对应的 kv_cache_indices
   → 无需重新计算这些 token 的 KV

3. 只需为剩余的 tokens 分配新的 KV Cache
```

这是 Memory Pool 设计中「共享需求」的具体实现。

---

### 总结：架构的核心价值

回顾整个 Memory Pool 架构，其核心价值体现在三个方面：

#### 解耦带来的灵活性

三层映射关系将「逻辑请求」「索引管理」「物理存储」解耦：

- 调整分配策略（连续 vs 页式）不影响存储层
- 实现共享（Radix Cache）不需要修改底层存储
- 支持 Offload 只需扩展存储层的操作

#### 抽象带来的可扩展性

KVCache 的多种实现（MHA、MLA、NSA）共享相同的索引机制：

```
MHATokenToKVPool:  k_buffer[layer], v_buffer[layer]  ← 标准 KV 分离
MLATokenToKVPool:  kv_buffer[layer]                  ← KV 融合
NSATokenToKVPool:  + index_k_buffer                  ← 额外的稀疏索引
```

这些差异被封装在 `get_key_buffer`、`get_value_buffer` 等接口背后，上层代码无需关心。

#### 元数据流的清晰性

从 Request → req_to_token → kv_indices → Attention Kernel，数据流转路径清晰：

```
Request 级别信息 (req_pool_idx, seq_len)
         ↓
    req_to_token 解引用
         ↓
    Token 级别索引 (kv_cache_index)
         ↓
    展平为 kv_indices
         ↓
    Attention Kernel 使用
```

每一步的输入输出明确，便于调试和优化。
