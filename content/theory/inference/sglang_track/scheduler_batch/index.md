---
title: Schduler Batch
type: docs
description: 通信组件实现和模型
weight: 20
---

### 从理论出发：LLM 推理到底在调度什么

#### 两阶段推理的本质

LLM 推理不是一次性完成的，它天然分为两个阶段：

- **Prefill（预填充）**：把用户输入的全部 token 一次性送入模型，计算出每一层的 KV Cache。这是一个**计算密集**操作，token 数量多、GPU 算力利用率高。
- **Decode（解码）**：逐个生成新 token，每次只算一个 token，但需要读取之前所有的 KV Cache。这是一个**访存密集**操作，GPU 利用率较低。

调度器的核心挑战是：**如何在有限的 GPU 显存（KV Cache 池）和计算带宽下，将不断涌入的请求高效地组织成 batch，交替执行 prefill 和 decode，使整体吞吐最大化、延迟可控。**

#### 调度器需要管理的三个核心资源

要理解调度器的设计，先要看清它管理着哪三种资源：

```
┌──────────────────────────────────────────────────────────┐
│                     GPU 显存                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  KV Cache Pool：所有请求共享的 KV 缓存池             │ │
│  │  - 每个 token 在每一层都需要一个 KV slot             │ │
│  │  - 总容量固定，是调度的硬约束                        │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Prefix Cache（RadixAttention Tree）：              │ │
│  │  - 已完成请求的 KV 可以被缓存复用                    │ │
│  │  - 新请求如果和已有请求共享前缀，可以跳过计算         │ │
│  │  - 但这些缓存可以被驱逐，是"弹性"资源               │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  计算带宽：                                         │ │
│  │  - 每次 forward 的 token 数有上限（max_prefill_tokens）│
│  │  - Prefill 和 Decode 争夺同一个 GPU                  │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

#### 请求的生命周期（逻辑视角）

一个请求在系统中经历以下状态流转：

```
用户请求到达
    │
    ▼
┌─────────┐     prefill 完成      ┌─────────┐     生成结束      ┌──────────┐
│ Waiting  │ ──────────────────► │ Decoding │ ──────────────► │ Finished │
│ Queue    │                      │ (Running)│                  │          │
└─────────┘                      └─────────┘                  └──────────┘
    ▲                                  │
    │          显存不足时 retract        │
    └──────────────────────────────────┘
```

关键观察：
- **请求不会在 prefill 和 decode 之间反复切换**——prefill 只做一次（或分 chunk 做几次），然后进入 decode 循环。
- **Retract（回退）是安全阀**：当 KV Cache 不够时，decode 中的请求会被驱逐回 waiting queue，释放显存。被驱逐的请求下次需要重新 prefill。

#### 每个请求需要保存什么？

在推理过程中，调度器必须为每个"活跃"请求维护以下信息：

| 类别 | 内容 | 为什么需要 |
|------|------|-----------|
| **身份** | 请求 ID、原始输入 token | 标识请求、支持 retract 后重做 |
| **KV 缓存指针** | req_pool_idx → token 位置映射 | 知道这个请求的 KV 数据存在哪里 |
| **进度** | 已生成的 output_ids、当前 seq_len | 知道生成到哪了，还需要多少 |
| **KV 长度追踪** | kv_committed_len, kv_allocated_len | 区分"已经确认写入"和"预分配但可能要回收"的 KV |
| **前缀缓存** | prefix_indices, last_node | 记录命中了多少 prefix cache |
| **采样参数** | temperature, top_p, stop 条件 | 控制生成行为 |
| **Chunk 状态** | is_chunked, extend_input_len | 支持长输入分段 prefill |

---

### 数据流全景：从请求到输出

#### 主循环：调度器的心跳

调度器运行一个无限循环，每个 tick 做五件事：

```
┌──────────────────────────────────────────────────────────────────┐
│                     Scheduler Event Loop                         │
│                                                                  │
│  while True:                                                     │
│    ① recv_requests()         ← 从 Tokenizer 接收新请求           │
│    ② process_input_requests() ← 解析请求，放入 waiting_queue      │
│    ③ get_next_batch_to_run() ← 【核心决策】组装下一个 batch       │
│    ④ run_batch()             ← 提交给 GPU 执行 forward           │
│    ⑤ process_batch_result()  ← 处理输出，判断完成，流式返回       │
│                                                                  │
│  其中 ③ 是灵魂——它决定"这一步到底做什么"                          │
└──────────────────────────────────────────────────────────────────┘
```

#### 核心决策：get_next_batch_to_run()

这是调度器最重要的函数。它的逻辑可以用一张决策图表达：

```
                    get_next_batch_to_run()
                            │
                            ▼
              ┌──── 上一轮是 prefill 吗？────┐
              │ 是                           │ 否
              ▼                              │
    把上轮 prefill 完成的请求                  │
    合并进 running_batch                     │
    （它们要进入 decode 循环了）                │
              │                              │
              ▼                              ▼
         尝试从 waiting_queue 构建新的 prefill batch
                            │
                    ┌───────┴───────┐
                    │ 有新 prefill   │ 没有
                    ▼               ▼
             返回 prefill batch   running_batch 非空？
            （prefill 优先！）      │
                              ┌───┴───┐
                              │ 是     │ 否
                              ▼       ▼
                     准备 decode     返回 None
                     batch          （服务器空闲）
                     并返回
```

**关键设计：Prefill 优先于 Decode。** 只要 waiting queue 中有请求可以 prefill，就先做 prefill。这保证了新请求的首 token 延迟（TTFT）不会因 decode 阻塞而过长。

#### Batch 的三级变换

请求被组装成 batch 后，经历三次数据结构变换，逐步从"调度语义"下沉到"GPU 执行语义"：

```
ScheduleBatch ──────► ModelWorkerBatch ──────► ForwardBatch
 (scheduler.py)        (tp_worker.py)          (model_runner.py)

 包含：                 包含：                   包含：
 - 请求列表 (Req[])    - 纯 GPU 张量            - 底层注意力张量
 - 调度元信息          - 采样参数                - FlashAttention 输入
 - 内存池引用          - forward_mode            - 分页 KV 索引
 - 前缀缓存指针        - 无请求生命周期方法       - CUDA kernel 参数
 - 合并/过滤/回退方法
```

这三级抽象的目的是**解耦**：调度器不需要知道 GPU kernel 怎么跑，GPU worker 不需要知道请求排队逻辑。

---

### 关键机制深入

#### 显存预算管理：如何决定 batch 能装多少请求？

构建 prefill batch 时，调度器使用 `PrefillAdder` 跟踪三个预算：

```
可用显存 = KV Cache 空闲 slots + 可驱逐的 prefix cache
                      │
                      ▼
              减去：running_batch 中每个请求未来
              可能产生的 token 数（乘以 new_token_ratio）
                      │
                      ▼
              得到：rem_total_tokens（真正可分配的 token 数）

同时受限于：
  - rem_input_tokens：单次 prefill 的 token 上限（max_prefill_tokens）
  - rem_chunk_tokens：chunked prefill 的分片大小
  - max_running_requests：最大并发请求数
```

`new_token_ratio` 是一个**动态参数**：正常运行时逐渐衰减（更激进地填充 batch），一旦发生 retract（OOM 回退），立即跳回高水位（更保守）。这是一个**自适应反馈机制**。

#### Prefix Caching 如何影响调度顺序？

SGLang 的标志性优化是 RadixAttention——通过 Radix Tree 管理 KV Cache 的前缀共享。这深刻影响调度策略：

```
                  Radix Tree (KV Cache)
                       [root]
                      /      \
                 [system      [system
                  prompt A]    prompt B]
                 /    \            \
            [user1]  [user2]     [user3]
```

- **LPM 策略**（Longest Prefix Match）：优先调度与已缓存前缀匹配最长的请求——命中缓存最多，节省计算。
- **DFS-Weight 策略**：按 Radix Tree 的深度优先权重排序，让共享前缀的请求聚集在一起。
- **In-batch 去重**：如果多个等待请求共享大量前缀，先调度其中一个（填充缓存），然后后续请求就能直接命中。

#### Chunked Prefill 与 Mixed Batch

当输入很长时，一次 prefill 会占用过多计算资源，阻塞 decode。解决方案：

```
长输入请求 (10000 tokens)
    │
    ▼
chunk 1: 前 4096 tokens → prefill → 中间状态保存
chunk 2: 接下来 4096 tokens → prefill → 中间状态保存
chunk 3: 最后 1808 tokens → prefill → 进入 decode 循环

在 chunk 之间，decode 请求可以穿插执行（Mixed Batch）：
┌──────────────┬───────────────┐
│ chunk 2 的    │ running_batch  │  ← 一个 batch 同时做
│ prefill tokens│ 的 decode     │     prefill + decode
└──────────────┴───────────────┘
```

#### Retract（回退）机制

当 KV Cache 不足以为 decode batch 中所有请求分配下一个 token 的存储时：

```
check_decode_mem() 失败
        │
        ▼
按 (output_len 降序, input_len 升序) 排序
选出"代价最小"的请求逐个回退
        │
        ▼
释放它们的 KV Cache → 放回 waiting_queue
（设置 is_retracted = True）
        │
        ▼
提高 new_token_ratio（下次预留更多空间）
```

回退策略倾向于驱逐"已生成最多 output 的请求"——因为它们占用显存最多，释放后收益最大。但被驱逐的请求下次需要重新 prefill，所以 retract 是有代价的。

---

### 代码中的实现方案

#### 核心文件地图

```
python/sglang/srt/managers/
├── scheduler.py          ← 调度器主体：事件循环、batch 组装、执行
├── schedule_batch.py     ← 数据结构：Req, ScheduleBatch, ModelWorkerBatch
├── schedule_policy.py    ← 调度策略：SchedulePolicy, PrefillAdder
└── tp_worker.py          ← GPU 执行器：接收 ModelWorkerBatch 做 forward

python/sglang/srt/mem_cache/
├── memory_pool.py        ← KV Cache 物理存储：ReqToTokenPool, KVCache
├── allocator.py          ← token slot 分配器：BaseTokenToKVPoolAllocator
└── radix_cache.py        ← RadixAttention 前缀缓存树
```

#### 关键类与职责

**`Req`**（`schedule_batch.py:512`）：单个请求的全生命周期状态。核心字段：
- `origin_input_ids` / `output_ids` / `fill_ids` — 输入输出 token
- `req_pool_idx` — 在 ReqToTokenPool 中的行索引
- `kv_committed_len` / `kv_allocated_len` — KV 缓存的已确认/已分配长度
- `prefix_indices` / `last_node` — prefix cache 命中信息
- `finished_reason` / `is_retracted` / `is_chunked` — 生命周期状态标记

**`ScheduleBatch`**（`schedule_batch.py:1202`）：调度器侧的 batch。核心能力：
- `prepare_for_extend()` — 为 prefill 分配 KV、构建 GPU 张量
- `prepare_for_decode()` — 为 decode 分配单 token KV
- `filter_batch()` — 移除已完成的请求
- `merge_batch()` — 将 prefill 完成的请求合并进 decode batch
- `retract_decode()` — OOM 时回退请求
- `get_model_worker_batch()` — 导出为 GPU 执行所需的精简数据

**`PrefillAdder`**（`schedule_policy.py:372`）：构建 prefill batch 的预算控制器。逐个尝试将 waiting queue 中的请求加入 batch，直到任一预算耗尽。

#### 事件循环的两种模式

**Normal 模式**（`scheduler.py:1109`）— 串行执行：
```python
while True:
    recv_reqs = self.recv_requests()
    self.process_input_requests(recv_reqs)
    batch = self.get_next_batch_to_run()     # CPU 调度
    if batch:
        result = self.run_batch(batch)        # GPU 执行（同步等待）
        self.process_batch_result(batch, result)
```

**Overlap 模式**（`scheduler.py:1136`）— CPU/GPU 流水线并行：
```
时间轴 →
GPU: ─── [forward batch N] ──────── [forward batch N+1] ────
CPU: ──────── [process result N-1] [schedule N+1] ──────────
                                    ↑
                        GPU 执行 N 的同时，CPU 处理 N-1 的结果并调度 N+1
```

Overlap 模式通过双 CUDA stream 实现，将 CPU 调度延迟隐藏在 GPU 计算之后，是生产环境的默认模式。

#### 三层内存池架构

```
ReqToTokenPool                TokenToKVPoolAllocator         KVCache
┌─────────────────┐           ┌──────────────────┐          ┌──────────┐
│ req_to_token:   │           │ free_pages: [...]  │          │ 物理 GPU  │
│ [max_batch ×    │ ──索引──► │ alloc_extend()    │ ──索引──►│ K/V 张量  │
│  max_ctx_len]   │           │ alloc_decode()    │          │ per layer │
│                 │           │ free()            │          │          │
│ 作用：请求 → 位置映射│       │ 作用：slot 分配/回收 │          │          │
└─────────────────┘           └──────────────────┘          └──────────┘
```

`prepare_for_extend()` 和 `prepare_for_decode()` 的核心工作就是在这三层之间做分配和写入。
