---
title: 双流
type: docs
description: DualPipe 实现
weight: 50
---

### 何时有收益

双流有优势和劣势，只有当 __优势 > 劣势的时候才有收益__

- **优势**：dispatch 和 combine 被计算 overlap 掉的时间
- **劣势**：计算会裂化

### sglang TBO config

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


