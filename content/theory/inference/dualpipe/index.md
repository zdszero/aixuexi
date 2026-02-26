---
title: 双流
type: docs
description: DualPipe 实现
weight: 50
---

### operation 策略配置

以 deepseek 为例进行介绍：

```python
# Prefill
return OperationsStrategy(
    deep_gemm_num_sms=deep_gemm_num_sms,
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
```


