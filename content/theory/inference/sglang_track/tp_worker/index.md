---
title: TP Worker
type: docs
description: 整体架构
weight: 30
---

### TP rank with data

```
                    Tokenizer / HTTP Frontend
                              |
                              | ZMQ IPC (仅连接 rank 0)
                              v
                      Scheduler (rank 0)
                              |
                              | torch.distributed.broadcast (Gloo CPU)
                              | 广播的是 Python 请求对象（pickle 序列化）
                              v
  Scheduler(rank0)    Scheduler(rank1)    Scheduler(rank2)    Scheduler(rank3)
       |                   |                   |                   |
       | 本地构造          | 本地构造          | 本地构造          | 本地构造
       v                   v                   v                   v
  ForwardBatch        ForwardBatch        ForwardBatch        ForwardBatch
  (含 input_ids      (含 input_ids       (含 input_ids       (含 input_ids
   等 GPU tensor)     等 GPU tensor)      等 GPU tensor)      等 GPU tensor)
       |                   |                   |                   |
       v                   v                   v                   v
  ModelRunner         ModelRunner         ModelRunner         ModelRunner
  (TP shard 0)        (TP shard 1)        (TP shard 2)        (TP shard 3)
```


**1. 每个 TP rank 启动一个独立的 Scheduler 进程**

在 `engine.py` 中，引擎为**每个 TP rank** 各启一个进程：

```python
# engine.py:911-991
for tp_rank in tp_rank_range:
    proc = mp.Process(
        target=run_scheduler_process,
        args=(server_args, port_args, gpu_id, tp_rank, ...),
    )
```

每个进程内创建自己的 `Scheduler`，Scheduler 又在**同一进程内**创建自己的 `TpModelWorker` 和 `ModelRunner`（`scheduler.py:520`）。Worker 和 Scheduler 之间是**直接的 Python 方法调用**，没有任何 IPC。

**2. 只有 rank 0 通过 ZMQ 接收请求**

```python
# scheduler.py:421-461
def init_ipc_channels(self, port_args):
    if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
        # 只有 rank 0 建立 ZMQ socket
        self.recv_from_tokenizer = get_zmq_socket(context, zmq.PULL, ...)
    else:
        self.recv_from_tokenizer = None  # 其他 rank 没有 ZMQ
```

**3. 请求通过 torch.distributed.broadcast 广播到所有 rank**

```python
# scheduler.py:1218-1307
def recv_requests(self):
    if self.attn_tp_rank == 0:
        # rank 0 通过 ZMQ 收请求
        recv_reqs = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
    else:
        recv_reqs = None

    # 广播给所有 TP rank（使用 Gloo CPU group）
    if self.tp_size != 1:
        recv_reqs = broadcast_pyobj(recv_reqs, self.tp_group.rank,
                                     self.tp_cpu_group,
                                     src=self.tp_group.ranks[0])
```

`broadcast_pyobj`（`utils/common.py:1268`）的实现是：

1. **pickle 序列化**请求对象为 bytes
2. 通过 `torch.distributed.broadcast()` 在 **Gloo CPU group** 上广播
3. 其他 rank 反序列化得到相同的请求

```python
def broadcast_pyobj(data, rank, dist_group, src=0):
    if rank == src:
        serialized_data = pickle.dumps(data)
        tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, ...))
        dist.broadcast(tensor_size, src=src, group=dist_group)  # 先广播大小
        dist.broadcast(tensor_data, src=src, group=dist_group)  # 再广播数据
    else:
        dist.broadcast(tensor_size, src=src, group=dist_group)  # 接收大小
        dist.broadcast(tensor_data, src=src, group=dist_group)  # 接收数据
        data = pickle.loads(bytes(tensor_data.numpy()))
```

**4. 各 rank 独立、确定性地构造相同的 batch**

所有 rank 收到相同的请求后，各自独立执行**完全相同的调度逻辑**：

```python
# 每个 rank 的 scheduler 独立执行，但因为输入相同，结果也相同
batch = self.get_next_batch_to_run()          # 确定性调度，各 rank 结果一致
model_worker_batch = batch.get_model_worker_batch()  # 本地构造
forward_batch = ForwardBatch.init_new(model_worker_batch)  # 本地构造 GPU tensor
self.model_runner.forward(forward_batch)      # 直接 Python 调用，无 IPC
```

**`input_ids` 等 tensor 是每个 rank 在本地 GPU 上独立创建的**，不存在从一个 rank 传输 tensor 到另一个 rank 的过程。

## DP Attention 下的 token 分布

当启用 DP Attention（例如 TP=4, DP=2）时，情况不同。此时 `attn_tp_size = tp_size / dp_size = 2`：

```
TP Attention Group 0: Rank 0, Rank 1  →  处理序列 a, b
TP Attention Group 1: Rank 2, Rank 3  →  处理序列 c, d
```

token 分布（`ScatterMode` 定义在 `communicator.py:104`）：

```
TP_ATTN_FULL 模式:  Rank0=[ab], Rank1=[ab], Rank2=[cd], Rank3=[cd]
                    同一 TP attention 组内共享相同 token

FULL 模式:          Rank0=[abcd], Rank1=[abcd], Rank2=[abcd], Rank3=[abcd]
                    所有 rank 持有全部 token

SCATTERED 模式:     Rank0=[a], Rank1=[b], Rank2=[c], Rank3=[d]
                    每个 rank 只有自己的分片
```

**此时需要 token 维度的通信：**

| 阶段 | 模式转换 | 通信操作 | 为什么需要 |
|------|---------|---------|-----------|
| `prepare_mlp` | TP_ATTN_FULL → FULL | `dp_gather_partial` (AllGather) | MLP 权重是 TP=4 切分的，需要全部 token |
| `postprocess_layer` | FULL → TP_ATTN_FULL | `dp_scatter` | 把 token 散回各自的 attention 组 |
