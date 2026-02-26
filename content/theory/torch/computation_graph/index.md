---
title: 计算图
type: docs
description: 计算图，包含动态图和静态图
weight: 30
---

### 定义

- node: tensor, computation unit
- edge: function, computation function
- graph: computation process expressed in abstract graph

### 实现模型

|           | 静态图 | 动态图  |
| --------- | --- | ---- |
| 图构建时间     | 执行前 | 执行时  |
| Python 参与 | 很少  | 每一步  |
| 控制流       | 受限  | 完全支持 |
| 调试        | 困难  | 简单   |

__静态图相比动态图少了哪部分时间？__

Python 执行和调用 C++ library 的时间，使用动态图：

```
第一阶段：静态图 vs 动态图之争（2016）
第二阶段：动态图 + 编译器融合（2020）
第三阶段：动态图 + 编译器 + CUDA Graph（大模型时代）
```

静态图就是类似 C 语言，编译完一次性执行，动态图类似解析器，边执行边解释。

需要注意的是，即使使用动态图，kernel launch 的时间仍然无法被节省，这是发生在 cpu 和 gpu 之间。

### Example

{{< svg "/images/computation_graph_ex.svg" >}}

下文通过一个计算图示例，展示了如何计算输出 \( g \) 对图中每个节点的偏导数（梯度）。它先直接计算了靠近输出节点的几个梯度（如 \(\frac{\partial g}{\partial f}\)、\(\frac{\partial g}{\partial o}\)），然后逐步向输入方向推进，并指出直接重复计算效率低下，从而引入链式法则来复用已计算的梯度。最后，针对具有多个下游分支的节点，说明需要使用多变量链式法则，即沿所有路径的梯度乘积求和。

**基于计算图梯度计算的核心**  

梯度计算的核心是**反向传播**：从输出节点开始，沿计算图反向逐层计算梯度，并利用链式法则将已计算的梯度向后传递，避免重复计算。这本质上是一种动态规划策略。

**链式法则与多变量链式法则**  

1. **链式法则**：用于单一路径的梯度传递。例如计算 \(\frac{\partial g}{\partial d}\) 时，利用已算出的 \(\frac{\partial g}{\partial e}\)：  
   \[
   \frac{\partial g}{\partial d} = \frac{\partial g}{\partial e} \cdot \frac{\partial e}{\partial d}
   \]  
   这样只需计算局部偏导 \(\frac{\partial e}{\partial d}\)，再与上游梯度相乘即可。
2. **多变量链式法则**：当节点有多个下游路径时，需汇总所有路径的贡献。例如节点 \(b\) 同时影响 \(c\) 和 \(d\)，则：  
   \[
   \frac{\partial g}{\partial b} = \frac{\partial g}{\partial c} \cdot \frac{\partial c}{\partial b} + \frac{\partial g}{\partial d} \cdot \frac{\partial d}{\partial b}
   \]  
   即沿所有从 \(g\) 到 \(b\) 的路径，将路径上的梯度乘积相加。

---

**该例子的计算过程简述**  

1. 先直接计算输出 \(g\) 对邻近节点 \(f, o\) 的梯度：\(\frac{\partial g}{\partial f}=3\)，\(\frac{\partial g}{\partial o}=-1\)。  
2. 计算 \(\frac{\partial g}{\partial e}\) 时，可展开表达式，也得 \(6\)。  
3. 计算 \(\frac{\partial g}{\partial d}\) 时，若直接展开会重复之前步骤；改为使用链式法则：  
   \[
   \frac{\partial g}{\partial d} = \frac{\partial g}{\partial e} \cdot \frac{\partial e}{\partial d} = 6 \cdot 2d = 12d
   \]  
   其中 \(\frac{\partial e}{\partial d} = 2d\) 是局部偏导。  
4. 计算 \(\frac{\partial g}{\partial c}\) 类似：  
   \[
   \frac{\partial g}{\partial c} = \frac{\partial g}{\partial e} \cdot \frac{\partial e}{\partial c} = 6 \cdot 7 = 42
   \]  
5. 计算 \(\frac{\partial g}{\partial b}\) 时，因 \(b\) 同时指向 \(c\) 和 \(d\)，采用多变量链式法则：  
   \[
   \frac{\partial g}{\partial b} = \frac{\partial g}{\partial c} \cdot \frac{\partial c}{\partial b} + \frac{\partial g}{\partial d} \cdot \frac{\partial d}{\partial b}
   \]  
   代入已知值：  
   \[
   = 42 \cdot 2 + 12d \cdot 5 = 84 + 60d
   \]  
   再根据 \(d = 5b+6\) 可进一步用 \(b\) 或 \(a\) 表示。  
6. 最终可继续计算 \(\frac{\partial g}{\partial a}\)，方法类似。

整个过程体现了反向传播中梯度的高效复用与多路径累加的思想。
