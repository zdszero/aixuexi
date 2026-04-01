---
title: Communictator
type: docs
description: 通信组件实现和模型
weight: 20
---

### SGLang LayerCommunicator 深度解析

#### 从一个问题出发

一个 Transformer 层由 Attention 和 MLP 两个子模块组成。在张量并行（TP）下，这两个子模块内部各自通过列并行/行并行的线性层完成计算，行并行线性层结束时需要 AllReduce 来合并各 rank 的部分和。这是经典的 Megatron-LM 范式，每层固定 2 次 AllReduce，逻辑简单。

但当系统引入 DP Attention（同一 TP 组内不同 rank 处理不同请求的注意力计算）后，问题变得复杂：Attention 阶段各 rank 只持有部分 token，而 MLP 阶段（权重按 full TP 切分）需要所有 token。数据在层内的不同阶段处于不同的"分布状态"，需要在正确的位置插入 AllGather、ReduceScatter 或 AllReduce 来完成状态转换。

`LayerCommunicator` 就是为了解决这个问题而设计的：它用一个统一的抽象来描述数据在层内各阶段的分布状态，并在初始化时自动推导出每个衔接点需要的通信操作。

#### ScatterMode：数据分布的三种状态

理解 `LayerCommunicator` 的关键在于理解 `ScatterMode`。它描述的不是"数据在哪个 rank 上"，而是"同一份数据被多少个 rank 共同持有"。以 TP=4、DP=2（因此 `attn_tp_size=2`）、系统同时处理请求 a, b, c, d 为例：

**`SCATTERED`**（每个 rank 独占一份分片）：rank0 持有 a，rank1 持有 b，rank2 持有 c，rank3 持有 d。对应的 `process_group_size` 为 1——同一份数据只有 1 个 rank 持有。

**`TP_ATTN_FULL`**（attention TP 组内完整）：rank0 和 rank1 同属一个 attention TP 组，它们都持有 ab；rank2 和 rank3 同属另一组，都持有 cd。对应的 `process_group_size` 为 `attn_tp_size`=2——同一份数据有 2 个 rank 持有。

**`FULL`**（所有 rank 完整）：每个 rank 都持有 abcd。对应的 `process_group_size` 为 `tp_size`=4。

`CommunicateContext` 在初始化时把这三种模式和对应的 group size 记录下来：

```python
process_group_sizes = {
    ScatterMode.SCATTERED: 1,
    ScatterMode.TP_ATTN_FULL: attn_tp_size,  # 2
    ScatterMode.FULL: tp_size,                # 4
}
```

之后判断两种模式之间是否需要通信，只需比较它们的 `process_group_size` 是否相等——相等意味着数据已经在对的分布状态上，不需要任何通信。

#### LayerScatterModes：一层之内的五个分布状态

一个 Transformer 层被分成五个阶段，每个阶段的数据需要处于特定的分布状态。`LayerScatterModes` 在初始化时一次性计算出全部五个：

```
layer_input_mode → [prepare_attn] → attn_mode → [prepare_mlp] → mlp_mode → [postprocess_layer] → layer_output_mode
                                                    ↕
                                            middle_residual_mode
```

`attn_mode` 固定为 `TP_ATTN_FULL`（注意力计算需要 TP 组内看到相同的完整 token 序列），其余四个根据配置动态计算。

以 Qwen3 在**标准 TP=4、无 DP Attention** 下为例。Qwen3 是 dense 模型（`is_layer_sparse=False`），`enable_moe_dense_fully_dp()` 默认 False，因此 `mlp_mode` 计算为 `FULL`。无 DP Attention 意味着 `attn_tp_size == tp_size`，所以 `TP_ATTN_FULL` 和 `FULL` 的 `process_group_size` 相等（都是 4）。五个模式的实际值为：

```
layer_input_mode    = TP_ATTN_FULL  (process_group_size=4)
attn_mode           = TP_ATTN_FULL  (process_group_size=4)
mlp_mode            = FULL          (process_group_size=4)
middle_residual_mode= TP_ATTN_FULL  (process_group_size=4)
layer_output_mode   = TP_ATTN_FULL  (process_group_size=4)
```

所有 `process_group_size` 相同，意味着相邻阶段之间不需要任何数据重分布通信——这就是标准 TP 的简单情况。

再看 **TP=4、DP=2** 的情况（`attn_tp_size=2`，`attn_dp_size=2`）：

```
layer_input_mode    = TP_ATTN_FULL  (process_group_size=2)
attn_mode           = TP_ATTN_FULL  (process_group_size=2)
mlp_mode            = FULL          (process_group_size=4)
middle_residual_mode= TP_ATTN_FULL  (process_group_size=2)
layer_output_mode   = TP_ATTN_FULL  (process_group_size=2)
```

此时 `attn_mode`(2) → `mlp_mode`(4) 需要通信（从 TP_ATTN_FULL 扩展到 FULL），`mlp_mode`(4) → `layer_output_mode`(2) 也需要通信（从 FULL 收缩回 TP_ATTN_FULL）。具体是什么通信操作，由三个策略类决定。

#### 三个策略类：初始化时绑定，运行时零判断

`LayerCommunicator` 在 `_post_init_communicate()` 中根据相邻阶段的 mode 组合，为三个衔接点各选定一个静态函数。之后每次 forward 调用时直接执行绑定好的函数，不再做任何条件判断。

##### CommunicateSimpleFn — prepare_attn 的衔接

负责 `layer_input_mode → attn_mode` 的转换。由于 `attn_mode` 固定为 `TP_ATTN_FULL`，只有两种情况：

当 `layer_input_mode` 已经是 `TP_ATTN_FULL`（或 group size 相同），绑定 `_trivial`——直接透传，零开销。标准 TP 走这条路。

当 `layer_input_mode` 是 `SCATTERED`（MoE+A2A 场景，上一层输出是分片的），绑定 `_scattered_to_tp_attn_full`——执行 AllGather，从预分配的 `local_dp_buffer` 收集各 rank 的分片，拼成 TP attention 组内的完整 token 序列：

```python
def _scattered_to_tp_attn_full(hidden_states, forward_batch, context):
    hidden_states, local_hidden_states = get_local_dp_buffer(), hidden_states
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states
```

##### CommunicateWithAllReduceAndLayerNormFn — prepare_mlp 的衔接

这是最复杂的衔接点，因为它同时要完成三件事：(1) 合并 attention 行并行线性层（o_proj）的部分和，(2) 进行数据重分布，(3) 应用 LayerNorm。三者顺序交织，是性能优化的重点。

负责的转换是：`(hidden_states: attn_mode, residual: layer_input_mode) → (hidden_states: mlp_mode, residual: middle_residual_mode)`。

**路径一：`_simple`**。条件：输入输出 group size 不变且 `attn_tp_size==1`（即单卡或纯 DP）。不需要 AllReduce（因为 TP 内只有一个 rank），直接做 LayerNorm：

```python
hidden_states, residual = layernorm(hidden_states, residual)
```

**路径二：`_gather_hidden_states_and_residual`**。条件：`attn_mode=TP_ATTN_FULL → mlp_mode=FULL`。这是 DP Attention 和标准 TP 的主路径。内部再根据是否有 DP Attention 分叉：

在 **标准 TP（无 DP Attention，`attn_dp_size==1`）** 下，attention 的 o_proj 产出的 hidden_states 是行并行的部分和，此处需要 AllReduce 合并。实现上有两个变体：

如果 FlashInfer AllReduce 融合可用（SM90/SM100 架构、batch_size ≤ 2048），调用 `layernorm.forward_with_allreduce_fusion(hidden_states, residual)`，将 NCCL AllReduce 和 RMSNorm 融合为一个 kernel，省去中间的显存读写：

```python
if apply_flashinfer_allreduce_fusion(hidden_states.shape[0]):
    hidden_states, residual = layernorm.forward_with_allreduce_fusion(hidden_states, residual)
```

否则分两步：先 `tensor_model_parallel_all_reduce(hidden_states)` 完成 AllReduce，再 `layernorm(hidden_states, residual)` 做 LayerNorm。这里的 `tensor_model_parallel_all_reduce` 会走到 `GroupCoordinator.all_reduce()`，后者内部有一套多后端 fallback 策略（CustomAllreduce → PyNccl → torch.distributed）。

在 **DP Attention（`attn_dp_size > 1`）** 下，逻辑不同。此时各 attention TP 组只有部分 token（比如 rank0,1 有 ab，rank2,3 有 cd），而 MLP 需要所有 token。处理方式是：先让 `attn_tp_rank==0` 的 rank 将 hidden_states 和 residual 相加（只有 rank0 做加法，避免重复计算），然后通过 `dp_gather_partial` 将各组的 token 收集到每个 rank 上。如果 `attn_tp_size==1`（即 `tp_size==dp_size`，每个 rank 独立处理不同 token），还可以进一步优化：先在小数据量上做 LayerNorm，再做 AllGather，减少 LayerNorm 的计算量：

```python
use_layer_norm_before_gather = context.attn_tp_size == 1
if use_layer_norm_before_gather:
    residual = hidden_states
    hidden_states = layernorm(hidden_states)  # 小数据量 LayerNorm
dp_gather_partial(hidden_states, local_hidden_states, forward_batch)  # AllGather
```

**路径三：`_scatter_hidden_states_and_residual`**。条件：`attn_mode=TP_ATTN_FULL → mlp_mode=SCATTERED`。这是 MoE+A2A 场景——MoE 层的 token dispatch/combine 由外部处理，LayerCommunicator 只需把数据从 TP_ATTN_FULL 收缩到 SCATTERED。实现是 ReduceScatter：将 AllReduce 拆成 ReduceScatter（合并部分和的同时完成分片），然后对已分片的数据做 LayerNorm：

```python
hidden_states = hidden_states.tensor_split(context.attn_tp_size)[context.attn_tp_rank]
attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)  # 归约+分片
hidden_states, residual = layernorm(hidden_states, residual)       # 对小数据做 LayerNorm
```

ReduceScatter 相比 AllReduce 的优势：通信量减半（AllReduce = ReduceScatter + AllGather，如果后续不需要 FULL 数据，就不需要 AllGather 那一步），且 LayerNorm 的计算量也按 `attn_tp_size` 倍缩小。

##### CommunicateSummableTensorPairFn — postprocess_layer 的衔接

负责 `(hidden_states: mlp_mode, residual: middle_residual_mode) → layer_output_mode`。这个衔接点的特殊之处在于 hidden_states 和 residual 是"可加的"——如果需要，可以先将两者相加再做通信，减少一次通信量。

**`_trivial`**：group size 不变，直接透传。标准 TP 走这条路。

**`_scatter_hidden_states`**：`FULL → TP_ATTN_FULL`。这是 DP Attention 的典型场景——MLP 在 FULL 模式下完成计算后，需要把数据散回各 attention TP 组。用 `dp_scatter` 从全局 buffer 中取出当前组对应的分片。如果启用了 padding 模式并允许 ReduceScatter，则用 `dp_reduce_scatter_tensor` 替代——此时 MLP 的 `down_proj` 跳过了 AllReduce（`reduce_results=False`），部分和的归约通过此处的 ReduceScatter 完成，同时顺带完成了数据分片，一举两得：

```python
if allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
    dp_reduce_scatter_tensor(hidden_states, global_hidden_states)  # 归约+分片一步到位
else:
    dp_scatter(hidden_states, global_hidden_states, forward_batch)  # 仅分片
```

**`_gather`**：`SCATTERED → TP_ATTN_FULL`。MoE+A2A 场景的反向操作——MoE 层输出是 SCATTERED 的，下一层 attention 需要 TP_ATTN_FULL，因此做 AllGather。注意它先将 hidden_states 和 residual 相加再通信，只需传输一份数据：

```python
hidden_states += residual
residual = None
attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states)
```

#### 数据在一层中的完整流动

把上面的策略选择和实际数据流动串起来，以 Qwen3 + DP Attention (TP=4, DP=2, attn_tp_size=2) 为例，追踪一层中 hidden_states 在 rank0 上的形态变化：

```
[输入] hidden_states: [N_local, H]  (TP_ATTN_FULL, rank0和rank1持有相同的ab)
  │
  ├─ prepare_attn(): input_layernorm(hidden_states, residual)
  │   CommunicateSimpleFn: _trivial（TP_ATTN_FULL→TP_ATTN_FULL，无通信）
  │
  ├─ self_attn(): QKV列并行 → Attention → o_proj行并行(reduce_results=False)
  │   hidden_states: [N_local, H]  仍是 TP_ATTN_FULL，但是行并行的部分和
  │
  ├─ prepare_mlp(): _gather_hidden_states_and_residual
  │   ├── attn_tp_rank==0 时: hidden_states += residual
  │   ├── dp_gather_partial(): AllGather，收集所有DP组的token
  │   │   hidden_states: [N_local, H] → [N_global, H]  变成 FULL
  │   ├── layernorm(hidden_states)  (或 layernorm 前置到 gather 之前做优化)
  │   └── residual 保持 TP_ATTN_FULL
  │
  ├─ mlp(): gate_up列并行 → 激活 → down_proj行并行(默认AllReduce)
  │   hidden_states: [N_global, H]  FULL 模式
  │
  └─ postprocess_layer(): _scatter_hidden_states
      dp_scatter(): 从FULL取出本attention组对应的分片
      hidden_states: [N_global, H] → [N_local, H]  回到 TP_ATTN_FULL
[输出] hidden_states: [N_local, H]  (TP_ATTN_FULL, 和输入同形态)
```

而在标准 TP（无 DP Attention）下，同一层的数据流动要简单得多——所有通信点都是 trivial 或单纯的 AllReduce：

```
[输入] hidden_states: [N, H]  (TP_ATTN_FULL == FULL, 所有rank持有全部token)
  │
  ├─ prepare_attn(): input_layernorm    （_trivial, 无通信）
  ├─ self_attn(): ... o_proj(reduce_results=False)
  ├─ prepare_mlp(): AllReduce + layernorm（合并o_proj部分和，可能与LayerNorm融合）
  ├─ mlp(): ... down_proj(AllReduce在内部)
  └─ postprocess_layer()                （_trivial, 无通信）
[输出] hidden_states: [N, H]
```

#### attn_tp_input_scattered：一种特殊的 ReduceScatter 优化

除了 `LayerScatterModes` 驱动的通信策略外，`communicator.py` 还有一个独立的优化路径：`AttnTpContext` 管理的 `input_scattered` 模式。

这个优化的思路是：在 prefill（extend）阶段，attention 的输入 token 数量很大，如果 TP 组内所有 rank 都处理全部 token，计算量是冗余的。`input_scattered` 模式将上一层的输出通过 ReduceScatter（而非 AllReduce）分片到各 rank：

```python
### prepare_attn() 开头
if get_attn_tp_context().input_scattered:
    hidden_states, residual = self._tp_reduce_scatter(hidden_states, residual)
```

`_tp_reduce_scatter` 将 `[N, H]` 的 hidden_states 通过 `get_tp_group().reduce_scatter_tensor()` 变成 `[N/tp_size, H]`。每个 rank 只拿到 1/tp_size 的 token，在这个小数据上做 LayerNorm 和 QKV 投影。到了 attention 计算前，再通过 `AttentionInputs.fetch_hidden_states()` 中的 AllGather 恢复完整 token。

等式上 AllReduce = ReduceScatter + AllGather，通信量没变，但好处是 **LayerNorm 和 QKV 投影的计算量缩小了 tp_size 倍**。这对 prefill 阶段的长序列尤其有意义。这个优化目前限于特定条件（CUDA、有 `q_lora_rank`、非 DP Attention、非 MoE A2A 等），通过 `AttnTpContext.init_context()` 中的条件判断控制。

#### 通信操作的实际执行路径

`LayerCommunicator` 中出现的通信操作最终都通过 `GroupCoordinator`（`parallel_state.py`）执行。`GroupCoordinator` 不是对 `torch.distributed` 的简单封装，而是一个多后端路由器：

对于 **AllReduce**（`tensor_model_parallel_all_reduce → GroupCoordinator.all_reduce`），它按优先级尝试：CustomAllreduce（NVLink 上的 IPC 共享内存方案，适合 <8MB 小 tensor）→ PyNccl（直接调用 `ncclAllReduce`）→ `torch.distributed.all_reduce`。在 CUDA Graph 模式下会选择不同的 out-of-place 实现。

对于 **AllGather**（`attn_tp_all_gather_into_tensor → GroupCoordinator.all_gather_into_tensor`），优先用 PyNccl 的 `ncclAllGather`，fallback 到 `torch.distributed.all_gather_into_tensor`。

对于 **ReduceScatter**（`attn_tp_reduce_scatter_tensor → GroupCoordinator.reduce_scatter_tensor`），同样优先 PyNccl 的 `ncclReduceScatter`，fallback 到 `torch.distributed.reduce_scatter_tensor`。

这些底层通信全部在 GPU 上通过 NCCL 完成，走的是 GPU 间的高速互联（NVLink/NVSwitch 或 PCIe），与 Scheduler 之间用 Gloo CPU broadcast 传递请求元数据是完全独立的两条通信通道。

#### LayerScatterModes 的计算逻辑

最后回到 `LayerScatterModes` 的自动推导逻辑。五个 mode 的计算存在依赖关系，核心驱动因素是 `mlp_mode`——它由层是否为 sparse（MoE）以及相关配置决定：

如果是 **dense 层**（Qwen3 的所有层）：当 `moe_dense_tp_size==1`（dense 层完全用 DP 计算）时 `mlp_mode=SCATTERED`，否则 `mlp_mode=FULL`。

如果是 **sparse 层**（MoE）：当使用 A2A 后端或 flashinfer MoE allgather 时 `mlp_mode=SCATTERED`（token 的 dispatch/combine 由 MoE 层自己处理），否则 `mlp_mode=FULL`。

由 `mlp_mode` 向两端推导：`middle_residual_mode` 跟随 `mlp_mode`（SCATTERED→SCATTERED，FULL→TP_ATTN_FULL）。`layer_output_mode` 在非最后一层时也跟随 `mlp_mode`（SCATTERED→SCATTERED，FULL→TP_ATTN_FULL），最后一层固定为 `model_input_output()`（通常 TP_ATTN_FULL）。`layer_input_mode` 等于上一层的 `layer_output_mode`，第 0 层等于 `model_input_output()`。

这意味着对于一个混合了 dense 层和 MoE 层的模型（如 Qwen3-MoE），dense 层和 MoE 层会自动获得不同的 `LayerScatterModes`，衔接处的通信操作也会自动适配——比如 dense 层输出 TP_ATTN_FULL，下一层 MoE 层的 `layer_input_mode` 就是 TP_ATTN_FULL，`prepare_attn` 的 `CommunicateSimpleFn` 就是 `_trivial`；而如果 MoE 层的 `layer_output_mode` 是 SCATTERED，下一个 dense 层的 `layer_input_mode` 就是 SCATTERED，`prepare_attn` 就会绑定 `_scattered_to_tp_attn_full` 来做 AllGather。整个过程由 `LayerScatterModes.init_new()` 在模型初始化时一次性完成，运行时不再有任何分支判断。


