---
title: TBO
type: docs
description: DualPipe 实现
weight: 50
---

{{< svg "/images/why_tbo.drawio.svg" "120%" >}}

### 何时有收益

双流有优势和劣势，只有当 __优势 > 劣势的时候才有收益__

- **优势**：dispatch 和 combine 被计算 overlap 掉的时间
- **劣势**：计算会裂化

### 引擎建模

#### Yield

yield 在这里的语义是 “让权”，如果一个 microbatch 执行 yield，就表示可以让另一个 microbatch 执行了。

以 deepseek 的 decode 阶段多机的 overlap 为例，我们可以在下图中设置 “让权点”：

{{< svg "/images/deepseek_decode_tbo.drawio.svg" "120%" >}}

通过这种方式可以巧妙地实现通用的双流框架，大家可以观察到：

- 两个 microbatch 的编排流程是完全一致的，可以使用一个 变量 来表示两者的交错方式
- 对于通信流，dispatch 和 combine 都是使用异步的调用方式，每次当我们在一个 microbatch 中 **launch 完当前阶段所有需要的 kernel 之后** 即可 让权给 另一个 microbatch
    - 这样可以在单流中实现双流的逻辑，通过异步的 dispatch/combine 实现通信流

```python
return OperationsStrategy(
    deep_gemm_num_sms=None,
    tbo_delta_stages=2,
    operations=[
        layer.op_comm_prepare_attn,     # Stage 0: Attention 准备
        layer.self_attn.op_prepare,     #          计算 Q, 准备 KV cache
        operations.YieldOperation(),    # ──────── Yield: 让权点 1
        
        layer.self_attn.op_core,        # Stage 1: Attention 核心
        layer.op_comm_prepare_mlp,      #          MLP 通信准备
        layer.mlp.op_gate,              #          门控网络
        layer.mlp.op_select_experts,    #          选择专家
        operations.YieldOperation(),    # ──────── Yield: 让权点 2
        
        layer.mlp.op_dispatch_a,        # Stage 2: Dispatch A (第1次 A2A)
        layer.mlp.op_shared_experts,    #          共享专家计算
        operations.YieldOperation(),    # ──────── Yield: 让权点 3
        
        layer.mlp.op_dispatch_b,        # Stage 3: Dispatch B (第2次 A2A)
        layer.mlp.op_experts,           #          Expert GEMM 计算
        layer.mlp.op_combine_a,         #          Combine A (第1次 A2A)
        operations.YieldOperation(),    # ──────── Yield: 让权点 4
        
        layer.mlp.op_combine_b,         # Stage 4: Combine B (第2次 A2A)
        operations.YieldOperation(),    # ──────── Yield: 让权点 5
        
        layer.mlp.op_output,            # Stage 5: 输出处理
        layer.op_comm_postprocess_layer,#          层后处理通信
    ],
)
```


在看一下 DeepSeek prefill 阶段的多机 TBO：

{{< svg "/images/deepseek_prefill_overlap.drawio.svg" "120%" >}}


{{< tabpane persist=false lang="python" >}}
{{< tab "Simple" >}}
def _compute_moe_deepseek_blog_prefill(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=total_num_sms - DeepEPConfig.num_sms,
        tbo_delta_stages=0,
        operations=[
            attn,
            dispatch_a,
            Yield,

            dispatch_b,
            mlp,
            combine_a,
            Yield,

            shared,
            combine_b,
        ],
    )
{{< /tab >}}
{{< tab "Complex" >}}
def _compute_moe_deepseek_blog_prefill(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=total_num_sms - DeepEPConfig.num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )
{{< /tab >}}
{{< /tabpane >}}

```
Normal 模式:
  _a = 准备数据 + 打 event 标记
  _b = 提交通信 + 等通信完成

Low-Latency 模式:
  _a = 提交通信（直接调用 _dispatch_core/_combine_core）
  _b = 等通信完成
```

#### Workflow

{{< svg "/images/tbo_pipeline_flow.svg" "90%" >}}

##### Split

##### Overlapped Execution


```python
def execute_overlapped_operations(inputs_arr, operations_arr, delta_stages):
    executor_a = _StageExecutor("a", stages_a, inputs_a)
    executor_b = _StageExecutor("b", stages_b, inputs_b)
    
    # Phase 1: A 先行 delta_stages 步
    for _ in range(delta_stages):
        executor_a.next()
    
    # Phase 2: 并行执行
    for _ in range(executor_a.num_stages - delta_stages):
        executor_a.next()  # 异步执行 A 的下一个 stage
        executor_b.next()  # 异步执行 B 的下一个 stage
    
    # Phase 3: B 完成
    for _ in range(delta_stages):
        executor_b.next()
    
    return [executor_a.output, executor_b.output]
```


##### Merge
