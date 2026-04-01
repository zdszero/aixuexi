---
title: disaggregation
type: docs
description: PD 分离式实现
weight: 30
---

**Disaggregation 的核心思路**：把 Prefill 和 Decode 拆到不同的 GPU 集群上，各自独立调度。Prefill 集群专注吞吐，Decode 集群专注延迟。两者之间通过高速网络（RDMA）传输 KV Cache。

这个架构的收益：
1. **消除干扰**：Prefill 不会阻塞 Decode，两个阶段独立 SLO
2. **异构部署**：Prefill 用计算型 GPU，Decode 用带宽型 GPU，资源利用率更高
3. **独立扩缩容**：可以根据流量特征分别调整 Prefill 和 Decode 的实例数

但它引入一个新问题：**Prefill 计算出的 KV Cache 必须高效传输到 Decode 节点**。这就是整个实现方案要解决的核心问题。

---

### 全局视角：一个请求的完整生命周期

下图展示了一个请求从进入到完成的完整路径。注意 Prefill 和 Decode 是**两个独立的 Scheduler 进程**，通过 Bootstrap Server 协调、通过 RDMA 传数据。

```
                         ┌─────────────────────┐
                         │   Bootstrap Server   │  ← HTTP 注册/发现服务
                         │  (aiohttp, 轻量级)   │
                         └────┬────────────┬────┘
                              │            │
                    HTTP PUT  │            │  HTTP GET
                    注册地址   │            │  查询地址
                              │            │
              ┌───────────────▼──┐    ┌───▼───────────────┐
              │  Prefill Server  │    │   Decode Server    │
              │                  │    │                    │
              │ ┌──────────────┐ │    │ ┌───────────────┐  │
         ①──►│ │ Bootstrap Q  │ │    │ │ Prealloc Q    │◄──── ①
              │ │ (握手等待)    │ │    │ │ (握手+预分配)  │  │
              │ └──────┬───────┘ │    │ └──────┬────────┘  │
              │        │ 握手完成  │    │        │ 分配KV内存  │
              │ ┌──────▼───────┐ │    │ ┌──────▼────────┐  │
              │ │ Waiting Q    │ │    │ │ Transfer Q    │  │
              │ │ (等待调度)    │ │    │ │ (等待KV到达)   │  │
              │ └──────┬───────┘ │    │ └──────┬────────┘  │
              │        │ Prefill  │    │        │ KV到达     │
              │ ┌──────▼───────┐ │    │ ┌──────▼────────┐  │
         ④◄──│ │ Inflight Q   │─┼─ RDMA ─►│ Waiting Q  │  │
              │ │ (KV传输中)   │ │    │ │ (构造Prebuilt) │  │
              │ └──────────────┘ │    │ └──────┬────────┘  │
              │                  │    │        │           │
              └──────────────────┘    │ ┌──────▼────────┐  │
                                      │ │ Running Batch │──── ⑤
                                      │ │ (自回归Decode) │  │
                                      │ └───────────────┘  │
                                      └────────────────────┘

① 请求同时发往 Prefill 和 Decode（携带相同的 bootstrap_room 作为关联 ID）
④ Prefill 完成后释放 KV，返回首 token
⑤ Decode 持续生成直到结束
```

关键要理解的是：**请求是同时发到两侧的**。两侧各自维护一套队列系统，通过 `bootstrap_room`（一个唯一 ID）在 Bootstrap Server 上"会合"。

---

### 协调机制：两端如何找到对方

Prefill 和 Decode 运行在不同机器上，它们需要解决两个问题：
1. **怎么找到对方**（服务发现）
2. **怎么把 GPU 显存地址告诉对方**（RDMA 地址交换）

#### Bootstrap Server：轻量级注册中心

Bootstrap Server 是一个 aiohttp HTTP 服务，角色类似服务发现。它只有一个核心接口 `/route`：

- **Prefill 启动时**：HTTP PUT 注册自己的 IP、ZMQ 端口、TP/DP/PP rank
- **Decode 收到请求时**：HTTP GET 查询目标 Prefill worker 的地址

Bootstrap Server 不参与数据传输，只是一个"电话簿"。

#### 握手流程

```
  Decode (Receiver)                    Bootstrap Server              Prefill (Sender)
       │                                     │                            │
       │  ─── GET /route (查询拓扑) ────────►│                            │
       │  ◄── {tp_size, dp_size, pp_size} ───│                            │
       │                                     │                            │
       │  ─── GET /route (查询地址) ────────►│                            │
       │  ◄── {rank_ip, rank_port} ──────────│                            │
       │                                     │                            │
       │  ═══ ZMQ: 注册GPU内存地址 ═══════════════════════════════════════►│
       │      (kv_data_ptrs, session_id)     │      (一次性, 首次连接时)    │
       │                                     │                            │
       │  ═══ ZMQ: 本次请求的目标地址 ════════════════════════════════════►│
       │      (bootstrap_room,               │                            │
       │       dst_kv_indices,               │       状态: Bootstrapping   │
       │       dst_aux_index)                │         → WaitingForInput   │
       │                                     │                            │
       │         此时 Decode 侧等待数据传输                                │
       │         Prefill 侧正常执行 forward                               │
       │                                     │                            │
       │  ◄══════════ RDMA 写入 KV Cache ═════════════════════════════════│
       │                                     │         状态: Transferring  │
       │                                     │                            │
       │  ◄══ ZMQ: 传输完成通知 ══════════════════════════════════════════│
       │                                     │         状态: → Success     │
```

注意两个层面的通信：
- **HTTP**（Bootstrap Server）：用于服务发现，低频
- **ZMQ**（点对点）：用于地址交换和状态通知，每请求
- **RDMA**（Mooncake 引擎）：用于 KV Cache 数据传输，高吞吐

这三层协议各司其职，HTTP 解决"去哪找"，ZMQ 解决"写到哪"，RDMA 解决"怎么快速写"。

---

### Prefill 侧：三阶段流水线

Prefill Scheduler 为请求维护三个队列，对应请求的三个生命阶段：

#### 阶段 1：Bootstrap Queue —— 等待握手

请求进入时，Scheduler 创建一个 `KVSender`（由 Mooncake/Nixl 等后端实现），并把请求放入 Bootstrap Queue。每个 event loop 迭代都会 poll 这些 sender 的状态：

- `KVPoll.Bootstrapping`：继续等待（Decode 侧还没发来目标地址）
- `KVPoll.WaitingForInput`：握手完成，分配 metadata buffer 槽位，调用 `sender.init()`，移入 Waiting Queue
- `KVPoll.Failed`：超时或对端故障，abort 请求

**关键设计**：poll 结果通过 `poll_and_all_reduce()` 在所有 TP rank 之间做 Gloo AllReduce（取 MIN），保证所有 rank 看到一致的状态。任何一个 rank 失败，所有 rank 都视为失败。

#### 阶段 2：Waiting Queue —— 正常 Prefill 调度

握手完成的请求进入 Waiting Queue，和普通请求一样参与 batch 调度。Scheduler 的 `get_next_disagg_prefill_batch_to_run()` 从中取出请求，组 batch 执行 Prefill forward。

**唯一的区别**：这些请求的 `max_new_tokens` 被设为 1——因为 Prefill 只需要计算出 KV Cache 和第一个输出 token，不需要做自回归。

#### 阶段 3：Inflight Queue —— KV 传输中

Prefill forward 完成后，调用 `send_kv_chunk()` 发起 KV 传输：

```python
## 核心逻辑（简化）
def send_kv_chunk(req, last_chunk):
    # 1. 从 req_to_token_pool 读取 KV Cache 的物理位置
    kv_indices = req_to_token_pool[req.req_pool_idx, start:end]

    # 2. token 级索引 → page 级索引（KV Cache 按页管理）
    page_indices = kv_to_page_indices(kv_indices, page_size)

    # 3. 最后一个 chunk 时，把元数据（首 token、logprobs）写入 metadata buffer
    if last_chunk:
        metadata_buffers.set_buf(req)

    # 4. 发起 RDMA 传输
    req.disagg_kv_sender.send(page_indices, state_indices)
```

请求移入 Inflight Queue，每轮 event loop poll 传输状态。传输完成后：
- 释放本地 KV Cache（因为数据已经到了 Decode 侧）
- 返回首 token 给用户（作为流式输出的第一个 token）

---

### Decode 侧：四阶段流水线

Decode 侧更复杂，因为它需要**先分配内存**才能告诉 Prefill "写到哪里"。

#### 阶段 1：Prealloc Queue —— 握手 + 内存预分配

请求进入时创建 `KVReceiver`，先完成握手。握手完成后，进入**内存预分配**：

```python
def pop_preallocated():
    for req in waiting_for_input_reqs:
        # 1. 检查是否有足够的 KV Cache 空间
        available = _allocatable_tokens()
        if available < req.input_length:
            break  # 等下一轮

        # 2. 分配 req pool 槽位 + KV Cache 页
        _pre_alloc(req)

        # 3. 读取分配到的 KV 页索引
        kv_indices = req_to_token_pool[req.req_pool_idx, :fill_len]
        page_indices = kv_to_page_indices(kv_indices, page_size)

        # 4. 告诉 Prefill 侧写到哪里
        receiver.init(page_indices, metadata_buffer_index, state_indices)
```

**关键设计**：`DecodeReqToTokenPool` 有一个独立的 `pre_alloc_size` 池，使得预分配的内存不会挤占正在运行的 Decode 请求的 KV 空间。这是因为预分配的内存可能要等较长时间才会被使用（取决于 Prefill 何时完成），如果和 running batch 共享池，会导致 Decode 吞吐下降。

#### 阶段 2：Transfer Queue —— 等待 KV 数据

预分配完成后请求进入 Transfer Queue。这个阶段 Decode 侧只需要 poll——数据由 Prefill 侧主动推送（RDMA 单边写）。

传输完成后，从 metadata buffer 中读取 Prefill 侧写入的元数据：

```python
def _commit_transfer_to_req(decode_req):
    idx = decode_req.metadata_buffer_index
    output_id, cached_tokens, logprobs, ... = metadata_buffers.get_buf(idx)

    # 校验 bootstrap_room 防止 buffer 冲突
    assert stored_room == req.bootstrap_room

    req.output_ids.append(output_id)
    req.cached_tokens = cached_tokens
    ...
```

#### 阶段 3：Waiting Queue —— 构造 Prebuilt Batch

KV 到达的请求进入 Waiting Queue，但**不需要再跑 Prefill forward**——KV Cache 已经在 GPU 上了。

Scheduler 构造一个特殊的 **Prebuilt Batch**（`ForwardMode.PREBUILT`）：

```python
def prepare_for_prebuilt():
    # 和 prepare_for_extend() 类似，但跳过 forward 计算
    # KV Cache 位置直接从 req_to_token_pool 读取（已被 RDMA 写入）
    forward_mode = ForwardMode.PREBUILT
    out_cache_loc = req_to_token_pool[req.req_pool_idx, :]
    ...

def process_prebuilt():
    # 用从 metadata buffer 拿到的首 token 作为"forward 结果"
    req.output_ids.append(transferred_output_id)
    # 如果用了 EAGLE 投机解码，还要恢复 draft 模型的输入
    ...
```

这是整个设计中最精妙的一步：**Prebuilt Batch 让 Decode 侧的调度器以为刚做完一次 Prefill，而实际上 KV Cache 是远程传来的**。后续的 Decode 逻辑完全复用已有代码路径。

#### 阶段 4：Running Batch —— 正常 Decode

Prebuilt Batch 与 Running Batch 合并，后续就是标准的自回归 Decode 流程，不再有 disaggregation 的特殊逻辑。

---

### KV Cache 传输：传了什么、怎么传

#### 传输的内容

传输分三类数据，走同一条 RDMA 通道但逻辑上独立：

| 类别 | 内容 | 用途 |
|------|------|------|
| **KV Data** | 每层 Attention 的 Key/Value Cache 页 | Decode 推理的核心输入 |
| **Metadata** | 首 output token、cached_tokens 数、logprobs、EAGLE top-k 概率/索引/hidden states | 让 Decode 侧构造 Prebuilt Batch |
| **State**（可选） | Mamba SSM state、SWA 滑动窗口 KV、NSA 索引 | 混合架构模型（Mamba+Attention）需要 |

#### RDMA 传输机制（Mooncake 后端为例）

传输引擎的核心是 **单边 RDMA 写**（one-sided write）：Prefill 侧直接把数据写入 Decode 侧的 GPU 显存，Decode 侧的 CPU 不需要参与。

```
Prefill GPU                                    Decode GPU
┌──────────────┐                          ┌──────────────┐
│ KV Page [3]  │ ── RDMA Write ──────────►│ KV Page [7]  │
│ KV Page [5]  │ ── RDMA Write ──────────►│ KV Page [12] │
│ KV Page [8]  │ ── RDMA Write ──────────►│ KV Page [20] │
│ Metadata[i]  │ ── RDMA Write ──────────►│ Metadata[j]  │
└──────────────┘                          └──────────────┘

源地址 = local_ptr + src_page_idx × item_len
目标地址 = remote_ptr + dst_page_idx × item_len
（remote_ptr 通过 ZMQ 握手获得）
```

实现上：
1. **启动时**：`MooncakeKVManager` 向 RDMA 引擎注册所有 GPU buffer（`register_buffer_to_engine()`）
2. **握手时**：Decode 通过 ZMQ 发送 `kv_data_ptrs`（GPU 原始指针），Prefill 存储为 `KVArgsRegisterInfo`
3. **传输时**：Prefill 的 `transfer_worker` 线程池从队列中取任务，对每一层执行 `engine.batch_transfer_sync()`

传输是**异步流水线化**的——多个请求的传输可以并行，由多线程 worker 处理。

#### Page 索引转换

KV Cache 在 SGLang 中按页（page）管理，一页包含 `page_size` 个 token 的 KV。传输时需要把 token 级索引转换为 page 级索引：

```python
def kv_to_page_indices(kv_indices, page_size):
    # 每 page_size 个 token 取一个，得到 page 编号
    return kv_indices[::page_size] // page_size
```

---

### 与 Scheduler 的关系

Disaggregation 通过 **Mixin** 模式注入 Scheduler，而不是修改 Scheduler 核心逻辑。

#### 初始化

`Scheduler.init_disaggregation()` 根据 `disaggregation_mode` 决定创建哪些组件：

```
DisaggregationMode.PREFILL:
    → MetadataBuffers + ReqToMetadataIdxAllocator
    → PrefillBootstrapQueue
    → disagg_prefill_inflight_queue (list)

DisaggregationMode.DECODE:
    → MetadataBuffers + ReqToMetadataIdxAllocator
    → DecodePreallocQueue + DecodeTransferQueue
    → disagg_decode_waiting_queue (deque)
```

#### 请求路由

`Scheduler._add_request_to_queue()` 根据模式分流：

```
NULL   → waiting_queue           （标准路径）
PREFILL → disagg_prefill_bootstrap_queue.add()
DECODE  → disagg_decode_prealloc_queue.add()
```

#### 事件循环

最核心的差异在 event loop。Scheduler 启动时根据模式选择不同的事件循环函数：

| 模式 | 事件循环 | 核心差异 |
|------|----------|----------|
| NULL | `event_loop_normal` | 标准 prefill + decode |
| PREFILL | `event_loop_normal_disagg_prefill` | 多了 bootstrap poll + inflight poll，forward 后发起 KV 传输 |
| DECODE | `event_loop_normal_disagg_decode` | 多了 prealloc + transfer poll，用 Prebuilt Batch 替代 prefill forward |

Mixin 的好处是：**disaggregation 的所有逻辑都在独立文件中**（`prefill.py`、`decode.py`、`decode_schedule_batch_mixin.py`），不污染 Scheduler 主循环。这也意味着可以方便地支持新的传输后端（Mooncake、Nixl、Mori、Ascend），只需实现 `BaseKVSender/Receiver/Manager` 接口。

---

### 代码地图

```
python/sglang/srt/disaggregation/
├── base/
│   └── conn.py              # 抽象接口：KVArgs, KVPoll, BaseKVManager/Sender/Receiver
├── common/
│   └── conn.py              # 通用实现：HTTP Bootstrap Server, ZMQ 控制面, 连接池
├── mooncake/
│   └── conn.py              # Mooncake RDMA 后端：传输引擎、worker 线程池
├── nixl/ mori/ ascend/      # 其他传输后端
├── fake/
│   └── conn.py              # 测试用假后端
├── prefill.py               # Prefill 侧：PrefillBootstrapQueue + SchedulerPrefillMixin
├── decode.py                # Decode 侧：DecodePreallocQueue + DecodeTransferQueue + SchedulerDecodeMixin
├── decode_schedule_batch_mixin.py  # Prebuilt Batch 构造逻辑（混入 ScheduleBatch）
├── decode_kvcache_offload_manager.py  # Decode 侧 KV Cache 卸载到 CPU（内存不足时）
├── utils.py                 # 工具：MetadataBuffers, page 索引转换, poll_and_all_reduce
└── kv_events.py             # 可观测性：传输事件追踪

managers/
└── scheduler.py             # init_disaggregation(), 请求路由, 事件循环分发
```

#### 关键抽象关系

```
BaseKVManager (每个 worker 一个)
 ├── 持有 KVArgs（所有 GPU buffer 地址、拓扑信息）
 ├── 创建 Sender/Receiver（每个请求一个）
 └── 管理传输 worker 线程池

BaseKVSender (Prefill 侧, 每请求)
 ├── init(num_pages, aux_index) → 设置传输参数
 ├── send(page_indices, state_indices) → 发起 RDMA 写
 └── poll() → 查询传输状态

BaseKVReceiver (Decode 侧, 每请求)
 ├── init(dst_page_indices, aux_index, state_indices) → 告知目标地址
 ├── poll() → 查询传输状态
 └── clear() / abort() → 清理资源

MetadataBuffers (两侧各一份)
 ├── set_buf(req) → Prefill 侧写入首 token、logprobs 等
 └── get_buf(idx) → Decode 侧读取
```

