---
title: 校验
type: docs
description: sglang 中的 kv cache 传输如何保证正确性？
weight: 20
---

SGLang 的 PD 分离**没有对 KV tensor 数据内容本身做 checksum/CRC 校验**，但存在以下几类验证机制：

### bootstrap_room 元数据碰撞检测

**`decode.py:1006~1038`**

Prefill 端将自己的 `bootstrap_room` 写入元数据 buffer 随 KV 一起传输，Decode 端读取后与期望值比对：
- `actual_room == 0`：元数据尚未就绪，等待重试
- `actual_room != expected_room`：检测到 metadata buffer index 碰撞，调用 `prepare_abort` 中止请求并记录 error

---

### bootstrap 握手阶段配置一致性校验

保证 kv cache dbyte 和 page_size 一致：

Decode 端连接 Prefill 端时验证：
- `page_size` 不一致 → 抛出 `RuntimeError`
- `kv_cache_dtype` 不一致 → 抛出 `RuntimeError`

---

### 协议消息格式校验

- **Mori** (`mori/conn.py:307~314`)：ZMQ 消息首帧必须为 `b"MoriMsgGuard"`，否则丢弃并 warning
- **NIXL** (`nixl/conn.py:~858`)：同样有 GUARD magic bytes 断言

###  KV page hash（非传输校验）

**`decode_kvcache_offload_manager.py:288~295`**

对每个 page 的 token 序列计算链式 hash，但用途是 HiCache 的**存储索引**，不是对传输后 KV tensor 数值做精度验证。

**总结**

| 验证类型 | 说明 |
|---|---|
| bootstrap_room 碰撞检测 | 验证元数据路由是否正确，防止 request 读到别人的元数据 |
| page_size/dtype 握手校验 | 确保 P/D 两端配置一致 |
| GUARD magic bytes | 防止恶意/乱序消息污染 bootstrap 通道 |
| KV tensor 内容校验 | **不存在**，传输后不验证 tensor 数值正确性 |
