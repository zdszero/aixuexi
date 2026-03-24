---
title: Gated DeltaNet
type: docs
weight: 30
---

### Gated DeltaNet

#### Memory Update

\[
M_t = \alpha_t \odot M_{t-1} + \beta_t \odot (k_t \otimes v_t)
\]

✔ 含义你已经说对了：

* \( \alpha_t \)：控制**遗忘**
* \( \beta_t \)：控制**写入**
* \( k_t \otimes v_t \)：当前 token 的 rank-1 memory

##### Shape

- \(k_t\): `[bs, q_len, k_heads, hdim]`
- \(q_t\): `[bs, q_len, k_heads, hdim]`
- \(v_t\): `[bs, q_len, v_heads, hdim]`
- \(k_t \otimes v_t\): `[bs, q_len, v_heads, h_dim, h_dim]`
- \(M_{t}\): `[bs, q_len, v_heads, h_dim, h_dim]`
- \(M_{t} \cdot q_t\): `[bs, q_len, v_heads * hdim]`

{{< svg "/images/gated_deltanet_module.drawio.svg" >}}

##### Theory

计算量：核心来自于 \(k_t \otimes v_t\) 这个外积：

- \(M_{t-1}^{\top} k_t\): 计算量为 `bs * q_len * v_heads * h_dim * h_dim`

> **Gated DeltaNet = 用递归方式构建一个带时间衰减的低秩记忆矩阵 \( M \)，再用 \( q \) 一次性读取它，从而近似 attention。**

| 项      | Attention        | GatedDeltaNet      |
| ------ | ---------------- | ------------------ |
| FLOPs  | \(O(n^2 d)\)       | \(O(n d^2)\)         |
| IO     | 读 KV cache（\(O(n)\)） | 只读写 state（\(O(d^2)\)）   |
| 核心瓶颈   | memory-bound     | compute-bound（更可能） |
| decode | 越来越慢             | 恒定                 |

以 Qwen3.5 为例：

在 Linear Attention 中，每个 token 需要存储的 \(M_{t}\) 的 shape 为 `[linear_v_heads, v_hdim, v_hdim]`，具体的元素个数为 \(64 \times 128 \times 128 = 2^{20}\)。

在 GQA 中，每个 token 需要存储的 KV cache 的 shape 为 `[kv_len, kv_heads, qk_hdim + v_hdim]`，具体的元素个数为 \(\text{kv\_len} \times 2 \times (256 + 256) = 1024 \times \text{kv\_len} = 2^{10} \times \text{kv\_len}\)。

当 kv_len ≥ 1024 的时候，gqa 的 kv cache 大小超过 linear attention。

#### Memory Read

\[
y_t = q_t \cdot M_t
\]

👉 这里就是你刚才缺的关键点：

> **q 是“查询压缩记忆 M”的读头（read head）**

展开一下：

\[
y_t = q_t \cdot \left( \sum_i w_i (k_i \otimes v_i) \right)
\]

\[
= \sum_i w_i (q_t \cdot k_i) v_i
\]

---

💡 这一行非常关键：

> 👉 **它本质是在“模拟 attention”**

但注意：

* 没有 softmax
* 没有逐 token 显式访问
* 所有历史已经被压进 \( M \)

### vs Attention

#### Attention

\[
y_t = \sum_i \text{softmax}(q_t \cdot k_i), v_i
\]

---

#### DeltaNet

\[
M_t = \sum_i \left( \beta_i \prod_{j=i+1}^t \alpha_j \right) (k_i \otimes v_i)
\]

\[
y_t = q_t \cdot M_t
\]

---

👉 所以最终是：

\[
y_t = \sum_i w_i (q_t \cdot k_i) v_i
\]

其中：

\[
w_i = \beta_i \prod_{j=i+1}^t \alpha_j
\]

---

💡 关键理解：

> **Attention 的权重 = softmax(q·k)**
> **DeltaNet 的权重 = 时间衰减（α）+ 写入强度（β）**

### Qwen 3.5

{{< svg "/images/gated_deltanet_computation.drawio.svg" >}}

🔷 Step 1：历史读取

\[
\hat{v}_t = M_{t-1}^\top k_t
\]

🔷 Step 2：Delta 计算

\[
\Delta_t = \beta_t \odot (v_t - \hat{v}_t)
\]

🔷 Step 3：状态更新（Gated）

\[
M_t = \alpha_t \odot M_{t-1} + k_t \otimes \Delta_t
\]

其中：

\[
\alpha_t = \exp(g_t)
\]

🔷 Step 4：输出

\[
o_t = M_t^\top q_t
\]

---

总结：

\[
\boxed{
\begin{aligned}
\hat{v}_t &= M_{t-1}^\top k_t \\
\Delta_t &= \beta_t (v_t - \hat{v}_t) \\
M_t &= \alpha_t M_{t-1} + k_t \otimes \Delta_t \\
o_t &= M_t^\top q_t
\end{aligned}
}
\]
