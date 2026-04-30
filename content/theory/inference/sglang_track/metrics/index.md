---
title: metrics
type: docs
description: sglang metrics
weight: 60
---

### Stage

该指标带有 `stage` 标签，区分请求流转的不同阶段，当前定义的阶段（`req_time_stats.py:91`）包括：

| stage | 说明 |
|---|---|
| `request_process` | 请求处理阶段（调度器） |
| `prefill_forward` | Prefill 前向计算 |
| `chunked_prefill` | Chunked Prefill 前向计算 |
| `prefill_bootstrap` | 分离推理：Prefill Bootstrap |
| `prefill_transfer_kv_cache` | 分离推理：KV Cache 传输 |
| `decode_prepare` | 分离推理：Decode 准备 |
| `decode_bootstrap` | 分离推理：Decode Bootstrap |
| `decode_waiting` | 分离推理：Decode 等待 KV |
| `decode_transferred` | 分离推理：Decode KV 已传输完成 |
| `fake_output` | 分离推理：Decode 伪输出 |
| `quick_finish` | 分离推理：快速完成 |


- prefill_bootstrap：Prefill 侧从"请求进队"到"握手 + 内存分配完成、可以真正做 prefill"的等待时间
- decode_bootstrap：Decode 侧从"收到请求"到"与 Prefill 握手成功、等待 KV 传输"的等待时间

两者度量的是同一条链路中不同端点的视角，理想情况下应该大致对称。如果 decode_bootstrap 显著大于 prefill_bootstrap，说明 Decode 节点在等 Prefill 侧主动发起连接；反之则相反。

### 使用场景

用于定位请求延迟的瓶颈在哪个阶段，例如是卡在 prefill 阶段、KV Cache 传输、还是 decode 等待。典型 PromQL 查询：

```promql
# 各阶段 P99 延迟
histogram_quantile(0.99, sum by (stage, le) (rate(sglang:per_stage_req_latency_seconds_bucket[1m])))
```
