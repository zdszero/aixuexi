---
title: blockwise
type: docs
weight: 60
---

在实际运行模型时，如果输入的序列过长，可能需要将序列切分成若干段，分别计算后再合并结果。这种情况下就会用到分段 attention 技术。该技术主要用于 chunked prefill 和 SP 并行等技术，本节将重点介绍其背后的数学原理。

### Blockwise Softmax

#### LSE

为了更好地进行分段计算，我们引入一个重要的数学工具：**对数求和指数**（Log-Sum-Exp，简称 LSE），其定义为：

\[
\text{LSE}(x_1, x_2, \dots, x_n) = \log\left( \sum_{i=1}^{n} e^{x_i} \right)
\]

> LSE: log of (sum of exp)
>
> For each \(x_i\), we sum \(e^{x_i}\), and then log the sum result.

在计算 softmax 时，直接使用上述公式可能会遇到数值问题：如果某些 \( x_i \) 很大，\( e^{x_i} \) 可能超过浮点数表示范围（上溢出）；如果所有 \( x_i \) 都很小，\( e^{x_i} \) 可能接近零，导致有效数字丢失（下溢出）。

为了解决这个问题，常用的做法是**先减去序列中的最大值**。令 \( M = \max(x_1, x_2, \dots, x_n) \)，则 softmax 可改写为：

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} 
= \frac{e^{x_i - M}}{\sum_{j=1}^{n} e^{x_j - M}}
\]

此时，所有指数部分的输入 \( x_j - M \) 都 ≤ 0，因此 \( e^{x_j - M} \in (0, 1] \)，既避免了上溢出，也缓解了下溢出。

利用 LSE 可以进一步将 softmax 表示为更稳定的形式。首先注意到：

\[
\text{LSE}(x) = \log\left( \sum_{j=1}^{n} e^{x_j} \right) 
= M + \log\left( \sum_{j=1}^{n} e^{x_j - M} \right)
\]

于是，softmax 可写为：

\[
\text{softmax}(x_i) 
= \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} 
= e^{x_i - \log\left( \sum_{j=1}^{n} e^{x_j} \right)} 
= e^{x_i - \text{LSE}(x)}
\]

这个表达式在数值计算中非常实用：先计算 \( \text{LSE}(x) \)（使用减去最大值的稳定方法），再计算 \( x_i - \text{LSE}(x) \)，最后取指数，整个过程数值稳定。

此外，LSE 有一个重要的性质：**它对 \( x_i \) 的导数正好是 softmax**。证明如下：

\[
\frac{\partial}{\partial x_i} \text{LSE}(x) 
= \frac{\partial}{\partial x_i} \log\left( \sum_{j=1}^{n} e^{x_j} \right) 
= \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} 
= \text{softmax}(x_i)
\]

{{< details "导数推导结果" >}}
**链式法则的应用**

令：

\[
S = \sum_{j=1}^{n} e^{x_j}
\]

则：

\[
\text{LSE}(x) = \log S
\]

对 \( x_i \) 求导时，注意 \( S \) 是 \( x_1, \dots, x_n \) 的函数。

\[
\frac{\partial \text{LSE}}{\partial x_i} = \frac{\partial \log S}{\partial S} \cdot \frac{\partial S}{\partial x_i}
\]

**第一步**：对 \(\log S\) 关于 \(S\) 求导

\[
\frac{\partial \log S}{\partial S} = \frac{1}{S}
\]

**第二步**：对 \(S\) 关于 \(x_i\) 求导

\[
S = e^{x_1} + e^{x_2} + \dots + e^{x_n}
\]

其中，只有 \( e^{x_i} \) 这一项依赖于 \( x_i \)，其他 \( e^{x_j} \)（\( j \neq i \)）对 \( x_i \) 的导数为 0。

所以：

\[
\frac{\partial S}{\partial x_i} = e^{x_i}
\]

**第三步**：合并结果

\[
\frac{\partial \text{LSE}}{\partial x_i} = \frac{1}{S} \cdot e^{x_i} = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
\]
{{< /details >}}

记住这几个特性：

\[
\boxed{\text{softmax}(x_i) 
= e^{x_i - \text{LSE}(x)}}
\]

\[
\boxed{\frac{\partial}{\partial x_i} \text{LSE}(x) 
= \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} 
= \text{softmax}(x_i)}
\]

#### 分段计算的方法

在实际计算中，如果向量 \( X \) 很长，我们可以将其分成若干段依次处理。假设将 \( X \) 分为 \( n \) 段：

\[
X = [X_1, X_2, \dots, X_n]
\]

{{< region note >}}
**需要明确的一些概念**

- \(X, X_1, X_2\) 这些都是数组
- \(\text{LSE}(X_1), \text{LSE}(X_2)\) 这种是常量
- \(x_i\) 是一个变量
- \(\text{softmax}_{1}(x_i)\) 这种是一个函数
{{< /region >}}

**第一步：处理第一段 \( X_1 \)**

我们只能基于当前段的信息计算：

\[
\text{lse}_1 = \log\left( \sum_{x \in X_1} e^{x} \right)
\]
\[
\text{softmax}_1(x_i) = e^{x_i - \text{lse}_1}, \quad x_i \in X_1
\]

**第二步：处理第二段 \( X_2 \)**

当读到第二段时，我们需要更新之前的结果。计算第二段的 LSE：

\[
\text{lse}_2 = \log\left( \sum_{x \in X_2} e^{x} \right)
\]

此时，前两段合并后的 softmax 可以表示为：

\[
\text{softmax}_{2}(x_i) = \frac{e^{x_i}}{e^{\text{lse}_1} + e^{\text{lse}_2}} = \frac{e^{x_i - \text{lse}_1}}{1 + e^{\text{lse}_2 - \text{lse}_1}}, \quad x_i \in [X_1, X_2]
\]

这等价于：

\[
\text{softmax}_{2}(x_i) = \text{softmax}_1(x_i) \cdot \frac{1}{1 + e^{\text{lse}_2 - \text{lse}_1}}
\]

若记 \( \sigma(z) = \frac{1}{1 + e^{-z}} \) 为 sigmoid 函数，则上式可写为：

\[
\begin{aligned}
\text{softmax}_{2}(x_i) 
&= \text{softmax}_1(x_i) \cdot \sigma(\text{lse}_1 - \text{lse}_2) \\
&= e^{x_i - \text{lse}_{1}} \cdot \sigma(\text{lse}_1 - \text{lse}_2)
\end{aligned}
\]

#### 递推公式

统一符号定义（关键）

我们约定：

* 第 \(k\) 段数据：\(X_k\)
* **局部 LSE（只看当前段）**：
  \[
  \ell_k := \log \sum_{x \in X_k} e^x
  \]
* **全局 LSE（累积到第 k 段）**：
  \[
  L_k := \log \sum_{x \in \cup_{j=1}^k X_j} e^x
  \]
* **全局 softmax（第 k 步）**：
  \[
  p^{(k)}(x_i)
  \]

其实整个递推可以理解为：

\[
L_k = \log\left(e^{L_{k-1}} + e^{\ell_k}\right),
\]

\[
p^{(k)}(x_i) = 
\begin{cases} 
p^{(k-1)}(x_i) \, \sigma(L_{k-1} - \ell_k), & x_i \in \bigcup_{j<k} X_j \\
e^{x_i - \ell_k} \, \sigma(\ell_k - L_{k-1}), & x_i \in X_k
\end{cases}
\]

### KV Chunked Attention

关于“KV chunked attention”和“Q chunked attention”这两个术语，目前学术界并没有统一的定义，不同文献或讨论中常常出现名称混用，容易造成理解上的混淆。为了便于后续讨论，我先对它们做一个简单的区分说明：

- **KV chunked attention**：  
  在这种方法中，查询（Q）是已知且完整的，但由于计内存限制或其他原因，无法一次性将 Q 与所有历史的 KV 进行注意力计算。因此，将 KV 切分成多个小块，逐块与 Q 进行计算，最后合并各块的结果，从而得到完整的注意力输出。

- **Q chunked attention**：  
  这种方法则是将查询（Q）切分成多个块，然后逐块与完整的键值（KV）进行注意力计算，最后再合并结果。

---

这里首先需要说明 kv chunked prefill，再上文中我们提到的 \(Q\) 可以是若干个 token 的 \(q\) 组成。为了 方便论证，我们先从一个简单的 case 出发，首先 **冻结所有自由度**：当前只算一个 token 的 \(q\)

{{< region note >}}
在整个推导里：

* **唯一的“自变量”是当前的 query 向量 \( q_x \)**
* **\(K, V\) 是已知常量（KV cache）**
* **\(A_i, B_i, \text{attn}_i, \text{LSE}_i\)**
  👉 **全部都是“中间计算结果 / 标量或向量值”**

**没有任何随机变量、没有函数未定项**
{{< /region >}}

我们正在计算的是：

\[
\text{attention}(q_x, K, V)
\]

这里：

* \(x\)：query 的位置（token index）
* **\(q_x\)：一个确定的向量**

**\(K, V\) 是已知的、但太大 → 被分块**

假设：

* 序列长度 = \(\text{seqlen}\)
* 被拆成 \(B_{KV}\) 个 block：

\[
K = [K_1; K_2; \dots; K_{B_{KV}}], \quad
V = [V_1; V_2; \dots; V_{B_{KV}}]
\]

这些都是 **已知常量矩阵**

---

**原始 attention（不拆分）**

\[
\text{attn}(q_x) =
\frac{\sum_{y=1}^{\text{seqlen}} e^{w_{xy}} v_y}
{\sum_{y=1}^{\text{seqlen}} e^{w_{xy}}}
\]

**KV 拆分之后**

我们只是把 **同一个求和** 拆成几段而已：

对 **第 (i) 个 KV block**：

\[
\boxed{
\begin{aligned}
B_i &= \sum_{y \in \text{block } i} e^{q_x k_y^T} \quad\text{（标量）}\\[4pt]
A_i &= \sum_{y \in \text{block } i} e^{q_x k_y^T} v_y \quad\text{（向量）}
\end{aligned}}
\]

👉 注意：

| 符号         | 类型                |
| ---------- | ----------------- |
| \(q_x\)      | 固定向量              |
| \(K_i, V_i\) | 固定矩阵              |
| \(B_i\)      | **算出来的标量**        |
| \(A_i\)      | **算出来的向量（\(d_v\) 维）** |

**完整 attention 就是**

\[
\text{attn}
= \frac{\sum_i A_i}{\sum_i B_i}
\]

定义：

\[
\boxed{\text{attn}_i := \frac{A_i}{B_i}}
\]

它表示：**“只看第 i 个 KV block，算出来的局部 attention”**

接下来，我们将说明如何通过子模块导出全局注意力（Attn）的结果。首先从简单情况入手，假设将全局注意力分成两个子块处理：

那么，将这两个子块合并后的全局注意力可表示为：

\[
\text{Attn} = \frac{A_1 + A_2}{B_1 + B_2}
\]

代入 \(A_i = \text{attn}_i \cdot B_i\)：

\[
\begin{aligned}
\text{attn}_{12}
&= \frac{\text{attn}_1 B_1 + \text{attn}_2 B_2}{B_1 + B_2} \
&= \text{attn}_1 \frac{B_1}{B_{12}} + \text{attn}_2 \frac{B_2}{B_{12}}
\end{aligned}
\]

\[
\text{LSE}_i := \log B_i
\]

---

**合并两个 block 的 LSE：**

\[
\begin{aligned}
\text{LSE}_{12}
&= \log(B_1 + B_2) \
&= \log\left(e^{\text{LSE}_1} + e^{\text{LSE}_2}\right) \
&= \text{LSE}_1 + \log\left(1 + e^{\text{LSE}_2 - \text{LSE}_1}\right)
\end{aligned}
\]

这就是 **log-sum-exp trick**

---

**attn 的加权也用 LSE 表达：**

\[
\frac{B_i}{B_{12}} = e^{\text{LSE}_i - \text{LSE}_{12}}
\]

于是：

\[
\boxed{
\text{attn}_{12}
= \text{attn}_1 e^{\text{LSE}_1 - \text{LSE}_{12}} + \text{attn}_2 e^{\text{LSE}_2 - \text{LSE}_{12}}
}
\]

---

事实上还可以进一步简化，即不需要计算出 \(LSE_{12}\)，而改用 sigmoid 函数实现推导如下：

\[
\begin{aligned}
\text{attn}_{12} &= \text{attn}_1 \cdot e^{LSE_1 - LSE_{12}} + \text{attn}_2 \cdot e^{LSE_2 - LSE_{12}} \\
&= \text{attn}_1 \cdot e^{-\log(1 + e^{LSE_2 - LSE_1})} + \text{attn}_2 \cdot e^{LSE_2 - LSE_1 - \log(1 + e^{LSE_2 - LSE_1})} \\
&= \text{attn}_1 \cdot \frac{1}{1 + e^{LSE_2 - LSE_1}} + \text{attn}_2 \cdot \frac{e^{LSE_2 - LSE_1}}{1 + e^{LSE_2 - LSE_1}} \\
&= \text{attn}_1 \cdot \frac{1}{1 + e^{LSE_2 - LSE_1}} + \text{attn}_2 \cdot \frac{1}{1 + e^{-LSE_2 + LSE_1}} \\
&= \text{attn}_1 \cdot \sigma(LSE_2 - LSE_1) + \text{attn}_2 \cdot \sigma(LSE_1 - LSE_2) \\
&= \text{attn}_1 - (\text{attn}_1 - \text{attn}_2) \cdot \sigma(LSE_1 - LSE_2)
\end{aligned}
\]

其中 \(\sigma(x) = \dfrac{1}{1 + e^{-x}}\) 为 sigmoid 函数，即

\[
\boxed{
\text{attn}_{12}
= \text{attn}_1 - (\text{attn}_1 - \text{attn}_2) \cdot \sigma(LSE_1 - LSE_2)
}
\]

> Streaming Attention 的本质
>
> * 只要能稳定地维护：
>
>   * 当前累计的 \( \text{LSE} \)
>   * 当前累计的 \( \text{attn} \)
>
> 就可以 **一块一块地 streaming 计算 attention**

#### 代码描述

```python
def qk_chunked_attention(query, key_chunks, value_chunks):
    """
    分块计算 attention
    """
    # 初始化
    lse_global = -inf
    output = zeros_like(query @ values)
    
    for k_chunk, v_chunk in zip(key_chunks, value_chunks):
        # 计算当前块的 attention 分数
        scores = query @ k_chunk.T
        
        # 计算局部 LSE
        lse_local = logsumexp(scores, dim=-1)
        
        # 更新全局 LSE
        lse_global = logaddexp(lse_global, lse_local)
        
        # 计算当前块的 attention 权重（部分 softmax）
        attn_weights = exp(scores - lse_local)
        
        # 计算当前块的贡献
        chunk_output = attn_weights @ v_chunk
        
        # 如果是第一个块，直接使用
        # 如果不是第一个块，需要重新缩放之前的输出
        if not first_chunk:
            # 重新缩放之前的输出
            scale = exp(lse_prev_global - lse_global)
            output *= scale
            
            # 缩放当前块的贡献
            chunk_scale = exp(lse_local - lse_global)
            chunk_output *= chunk_scale
        
        # 累加当前块的贡献
        output += chunk_output
        
        lse_prev_global = lse_global
    
    return output
```


### Q Chunked Attention

__chunk 1__

|       | \(k_0\) | \(k_1\) | \(k_2\) | \(k_3\) |
|-------|----|----|----|----|
| \(q_0\)    | 1  | -  | -  | -  |
| \(q_1\)    | 1  | 1  | -  | -  |
| \(q_2\)    | 1  | 1  | 1  | -  |
| \(q_3\)    | 1  | 1  | 1  | 1  |

__chunk 2__

|       | \(k_0\) | \(k_1\) | \(k_2\) | \(k_3\) | \(k_4\) | \(k_5\) | \(k_6\) | \(k_7\) |
|-------|----|----|----|----|----|----|----|----|
| \(q_4\)    | 1  | 1  | 1  | 1  | 1  | -  | -  | -  |
| \(q_5\)    | 1  | 1  | 1  | 1  | 1  | 1  | -  | -  |
| \(q_6\)    | 1  | 1  | 1  | 1  | 1  | 1  | 1  | -  |
| \(q_7\)    | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  |

__chunk 3__ 

|       | \(k_0\) | \(k_1\) | \(k_2\) | \(k_3\) | \(k_4\) | \(k_5\) | \(k_6\) | \(k_7\) | \(k_8\) | \(k_9\) | \(k_{10}\) | \(k_{11}\) |
|-------|----|----|----|----|----|----|----|----|----|----|-----|-----|
| \(q_8\)    | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | -  | -   | -   |
| \(q_9\)    | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | -   | -   |
| \(q_{10}\)   | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1   | -   |
| \(q_{11}\)   | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1   | 1   |
