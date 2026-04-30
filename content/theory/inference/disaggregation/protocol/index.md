---
title: 协议
type: docs
description: PD disaggregation protocol
weight: 10
---

### bootstrap

**Bootstrap time 不是"传输前的会话"这么简单**，它是 PD 分离（Prefill-Decode 分离）架构中，Prefill 节点和 Decode 节点**建立连接并完成元数据交换**的完整握手阶段耗时。

#### Bootstrap 阶段具体做了什么

| 步骤 | 操作 | 执行方 |
|---|---|---|
| 1. 拓扑注册 | Prefill 启动时向 Bootstrap HTTP Server 注册自身的 TP/CP/PP rank 和 ZMQ 端口 | Prefill |
| 2. 拓扑查询 | Decode 向 Bootstrap Server 查询目标 Prefill 节点的完整拓扑（IP、Port、rank 信息） | Decode |
| 3. DP rank 解析 | 确定当前请求由哪个 Prefill DP worker 处理 | Decode |
| 4. 连接建立 | Decode 为每个目标 Prefill rank 建立 ZMQ PUSH socket 连接 | Decode |
| 5. 握手完成 | KVReceiver 状态变为 `WaitingForInput`，记录 `bootstrap_done_time` | Decode |

**prefill 请求类型**

```
进入 bootstrap 队列
    ↓
（与 Decode 节点握手，建立 KV 传输通道）← prealloc-req 就是这里的请求
    ↓
bootstrap 完成 → 移入 waiting_queue（等待真正做 prefill）
    ↓
prefill 计算完成 → 移入 inflight 队列（传输 KV）← #inflight-req
    ↓
KV 传输完成，请求结束
```

#### 时间边界

```
bootstrap_ms = bootstrap_done_time - prefill_bootstrap_queue_entry_time
             = 请求进入 bootstrap 队列 → 握手成功（ZMQ 连接就绪）
```

Bootstrap 完成后才会进入下一步：分配 KV 内存槽（`alloc_ms`），然后才是实际的 KV cache 传输。

#### 状态机

```
Bootstrapping (1) → WaitingForInput (2) → Transferring (3) → Success (4)
     ↑                    ↑
  进入队列时          bootstrap_done_time 记录点
```

#### Prometheus 指标

- `sglang:kv_transfer_bootstrap_ms` — bootstrap 握手耗时直方图
- `sglang:kv_transfer_alloc_ms` — bootstrap 后 KV 内存分配等待耗时
- `sglang:num_bootstrap_failed_reqs_total` — bootstrap 失败请求计数

**总结：** Bootstrap time 是"传输前准备"阶段的耗时，核心是 Decode 节点找到并连接到正确的 Prefill 节点（通过 Bootstrap HTTP Server 做服务发现 + ZMQ 建连），而不只是一次简单会话，它涵盖了整个服务发现、拓扑解析、连接建立的过程。
