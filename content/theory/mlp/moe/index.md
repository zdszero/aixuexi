---
title: MoE
type: docs
weight: 20
---

**MoE 的本质**是一个**稀疏加权函数族**，其核心形式为：

\[
\mathrm{MoE}(x) = \sum_{e} \text{sparse-gate}_e(x) \cdot f_e(x)
\]

其中，**Gate** 负责决定“用谁”，而**专家**则决定“怎么算”。通过 Top-k 路由，系统实现了**计算稀疏化**，但模型的总参数量仍随专家数增加而显著增长。

---

### Symbols

- **输入向量**：\( x \in \mathbb{R}^{d} \)
- **专家总数**：\( E \)（对应专家 \( e = 1,\dots,E \)）
- **专家函数**：每个专家是一个独立参数化的函数 \( f_e(x; \theta_e) \)，其中 \( \theta_e \) 为第 \( e \) 个专家的私有参数（不共享）。
- **路由稀疏性**：每个 token 仅激活 \( k \) 个专家，且满足 \( k \ll E \)。

---

### Gate

#### Gate score

Gate 通常是一个可学习的线性层，为每个专家生成 logits：
\[
g(x) = W_g x \in \mathbb{R}^{E}, \quad W_g \in \mathbb{R}^{E \times d}
\]

#### Top-k
令 \( \mathcal{S}(x) = \text{TopK}(g(x), k) \) 表示 logits 最高的 \( k \) 个专家索引集合。

#### Softmax

仅在选中的专家上进行 softmax 归一化：
\[
\alpha_e(x) =
\begin{cases}
\displaystyle
\frac{\exp(g_e(x))}{\sum_{j \in \mathcal{S}(x)} \exp(g_j(x))}, & e \in \mathcal{S}(x) \\
0, & \text{otherwise}
\end{cases}
\]

> 🔑 **关键点**：非 Top-k 专家完全不参与计算，Gate 权重仅在 Top-k 内部归一化。

#### Example

假设我们有一个 **MoE 层**，其参数如下：
*   输入维度 `d = 4`
*   专家数量 `E = 5`
*   Top-k 参数 `k = 2`
*   输入向量 `x` 是一个 4 维向量。

我们定义具体的数值：

**1. 输入向量**
\[
x = \begin{bmatrix} 1.0 \\ -0.5 \\ 2.0 \\ 0.5 \end{bmatrix}
\]

**2. Gate 线性层权重** \( W_g \in \mathbb{R}^{5 \times 4} \)
\[
W_g = \begin{bmatrix}
0.1 & -0.2 & 0.3 & 0.0 \\
0.4 & 0.1 & -0.1 & 0.2 \\
-0.3 & 0.2 & 0.1 & 0.4 \\
0.0 & -0.1 & 0.2 & 0.1 \\
0.2 & 0.0 & -0.2 & 0.3
\end{bmatrix}
\]
其中每一行 \( W_g[e] \) 对应一个专家的 Gate 权重。

---

**步骤 1: 计算 Gate Score (Logits)**
\[
g(x) = W_g \cdot x
\]
\[
\begin{aligned}
g_1 &= 0.1*1.0 + (-0.2)*(-0.5) + 0.3*2.0 + 0.0*0.5 = 0.1 + 0.1 + 0.6 + 0.0 = 0.8 \\
g_2 &= 0.4*1.0 + 0.1*(-0.5) + (-0.1)*2.0 + 0.2*0.5 = 0.4 - 0.05 - 0.2 + 0.1 = 0.25 \\
g_3 &= (-0.3)*1.0 + 0.2*(-0.5) + 0.1*2.0 + 0.4*0.5 = -0.3 - 0.1 + 0.2 + 0.2 = 0.0 \\
g_4 &= 0.0*1.0 + (-0.1)*(-0.5) + 0.2*2.0 + 0.1*0.5 = 0.0 + 0.05 + 0.4 + 0.05 = 0.5 \\
g_5 &= 0.2*1.0 + 0.0*(-0.5) + (-0.2)*2.0 + 0.3*0.5 = 0.2 + 0.0 - 0.4 + 0.15 = -0.05
\end{aligned}
\]
所以 Gate Score 向量为：
\[
g(x) = \begin{bmatrix} 0.8 \\ 0.25 \\ 0.0 \\ 0.5 \\ -0.05 \end{bmatrix}
\]

**步骤 2: Top-k 选择**
我们选择 logits 最高的 `k=2` 个专家。
对 \( g(x) \) 排序：`0.8 (专家1)`, `0.5 (专家4)`, `0.25 (专家2)`, `0.0 (专家3)`, `-0.05 (专家5)`。
因此，选中的专家索引集合为：
\[
\mathcal{S}(x) = \{1, 4\}
\]

**步骤 3: 在 Top-k 内进行 Softmax 归一化**
我们只对专家 1 和专家 4 的 logits 计算 softmax。
\[
\begin{aligned}
\sum_{j \in \{1,4\}} \exp(g_j(x)) &= \exp(0.8) + \exp(0.5) \\
&= 2.22554 + 1.64872 \\
&= 3.87426
\end{aligned}
\]
\[
\begin{aligned}
\alpha_1(x) &= \frac{\exp(0.8)}{3.87426} = \frac{2.22554}{3.87426} \approx 0.574 \\
\alpha_4(x) &= \frac{\exp(0.5)}{3.87426} = \frac{1.64872}{3.87426} \approx 0.426
\end{aligned}
\]
未被选中的专家权重为 0。
最终的门控权重向量为：
\[
\alpha(x) = \begin{bmatrix} 0.574 \\ 0 \\ 0 \\ 0.426 \\ 0 \end{bmatrix}
\]

MoE 层的输出是所选专家输出的加权和：
\[
\mathrm{MoE}(x) = \sum_{e \in \mathcal{S}(x)} \alpha_e(x) \cdot f_e(x) = 0.574 \cdot f_1(x) + 0.426 \cdot f_4(x)
\]
其中 \( f_1(x) \) 和 \( f_4(x) \) 分别是专家1和专家4的前馈网络（FFN）的输出。

---

### MoE

最抽象的形式可写为：
\[
\boxed{\mathrm{MoE}(x) = \sum_{e=1}^{E} \alpha_e(x) \cdot f_e(x; \theta_e)}
\]
由于 \( \alpha_e(x) \) 在非 Top-k 专家上为零，上式等价于仅对选中专家求和：
\[
\mathrm{MoE}(x) = \sum_{e \in \mathcal{S}(x)} \alpha_e(x) \cdot f_e(x; \theta_e)
\]

---

在 LLM 中，专家通常实现为前馈网络（FFN）：
\[
f_e(x) = W^{(2)}_e \cdot \sigma\!\left(W^{(1)}_e x\right)
\]
代入 MoE 公式即得：
\[
\mathrm{MoE}(x) = \sum_{e \in \mathcal{S}(x)} \alpha_e(x) \cdot W^{(2)}_e \cdot \sigma\!\left(W^{(1)}_e x\right)
\]

---

对第 \( t \) 个 token，其输出为：
\[
y_t = \sum_{e \in \mathcal{S}(x_t)} \alpha_{t,e} \cdot f_e(x_t)
\]
在实际系统中，**每个 token 独立路由**，同一 batch 内不同 token 激活的专家集合 \( \mathcal{S}(x_t) \) 可能完全不同。

---

为防止 Gate 坍缩到少数专家，常引入辅助损失进行约束。

定义专家使用率与 Gate 概率：
\[
p_e = \frac{1}{N}\sum_{t=1}^N \mathbb{I}[e \in \mathcal{S}(x_t)], \quad
q_e = \frac{1}{N}\sum_{t=1}^N \alpha_{t,e}
\]

典型的负载均衡损失为：
\[
\mathcal{L}_{\text{balance}} = E \sum_{e=1}^E p_e \cdot q_e
\]

### Fine-Grained Expert

当专家数量比较多时，不是通过一次 gate + topk 就选出来 K 个专家，而是通过一种【两阶段专家】的方式：**先组内竞争，再组间竞争**。

比如说对于 deepseek v3，其配置中包含以下专家配置：

- `n_routed_experts`: 256，总共 256 个专家
- `num_experts_per_tok`: 8，每个 token 选择 8 个动态专家
- `n_group`：将动态专家分为 8 组，在不使用冗余专家的情况下每组 32 个
- `topk_group`：在第一阶段选取 4 个专家组，最终每个 token 选取的 8 个专家从这 4 个专家组中诞生

#### 组内筛选与打分
1.  **输入**：Token 表示 `X`
    *   形状：`[num_tokens, num_experts]`
    *   每个 token 对每个专家有一个初始的“兴趣度”分数。

2.  **分数归一化与校正**
    *   `S = sigmoid(X)`：将原始分数映射到 `(0, 1)` 区间，作为基础专家权重。
    *   `S_corrected = S + correction_bias`：引入一个可学习的偏置项，用于调整专家选择的倾向性（例如，缓解专家负载不均衡）。

3.  **分组与组内竞争**
    *   将 `num_experts` 个专家均匀分为 `n_group` 组。
    *   **形状变换**：`S_corrected` 从 `[num_tokens, num_experts]` 变为 `[num_tokens, n_group, num_experts_per_group]`。
    *   **组内打分**：在每个组内，为每个 token 选取 Top-2 的专家，并**将它们的分数求和**，作为该组的“组得分”。
    *   **输出**：组得分矩阵，形状为 `[num_tokens, n_group, 1]`。这代表了每个 token 对每个组的“兴趣度”。

#### 组间竞争与专家选择

4.  **选择候选组**
    *   基于组得分，为每个 token 选择得分最高的 **Top-4 个组**。
    *   输出形状：`[num_tokens, 4]`（存储选中的组索引）。

5.  **生成组掩码并映射到专家**
    *   创建一个组掩码 `group_mask`，形状为 `[num_tokens, n_groups, 1]`。对于每个 token，被选中的组对应位置为 1，否则为 0。
    *   将组掩码扩展（`expand`）到专家维度，得到初始的专家候选掩码 `expert_candidate_mask`，形状为 `[num_tokens, num_experts, 1]`。其逻辑是：**一个专家所属的组若被选中，则该专家进入候选池**。

6.  **最终专家筛选**
    *   在候选专家池（由掩码定义）内，为每个 token 进行**全局 Top-8** 筛选。这确保了最终从所有被选中的组里挑出最优秀的 8 个专家。
    *   **权重计算**：最终这 8 个专家的融合权重，使用**第一阶段计算的原始 `sigmoid` 分数 `S`**（而非校正后的分数或组得分），以确保权重的直接性与可解释性。
