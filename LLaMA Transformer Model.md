```
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                        LLaMA Transformer Model                  │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                  输入
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    Token Embedding Layer                        │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                 Rotary Position Encoding(RoPE)                  │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    Transformer Blocks (× n_layers)              │
                                    ├─────────────────────────────────────────────────────────────────┤
                                    │  ┌────────────────────────────────────────────────────────────┐ │
                                    │  │                    Transformer Block                       │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │              Residual Connection 1                  │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                  RMSNorm (Attention)                │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                Multi-Head Attention                 │   │ │
                                    │  │  │  ┌─────────────┬─────────────┬─────────────┐        │   │ │
                                    │  │  │  │      Wq     │      Wk     │      Wv     │        │   │ │
                                    │  │  │  │ (ColumnPar) │ (ColumnPar) │ (ColumnPar) │        │   │ │
                                    │  │  │  └─────────────┴─────────────┴─────────────┘        │   │ │
                                    │  │  │         ↓             ↓             ↓               │   │ │
                                    │  │  │    ┌─────────────────────────┐      ↓               │   │ │
                                    │  │  │    │   Apply Rotary Embed    │      ↓               │   │ │
                                    │  │  │    └─────────────────────────┘      ↓               │   │ │
                                    │  │  │                ↓                    ↓               │   │ │
                                    │  │  │    ┌────────────────────────┐       ↓               │   │ │
                                    │  │  │    │ scores = torch.matmul  │       ↓               │   │ │
                                    │  │  │    └────────────────────────┘       ↓               │   │ │
                                    │  │  │             mask ↓                  ↓               │   │ │
                                    │  │  │    ┌────────────────────────┐       ↓               │   │ │
                                    │  │  │    │scores = softmax(scores)│       ↓               │   │ │
                                    │  │  │    └────────────────────────┘       ↓               │   │ │
                                    │  │  │                   ↓                 ↓               │   │ │
                                    │  │  │            ┌──────────────────────────────┐         │   │ │
                                    │  │  │            │    output = torch.matmul     │         │   │ │
                                    │  │  │            └──────────────────────────────┘         │   │ │
                                    │  │  │                           ↓                         │   │ │
                                    │  │  │  ┌───────────────────────────────────────────────┐  │   │ │
                                    │  │  │  │                  Wo (RowPar)                  │  │   │ │
                                    │  │  │  └───────────────────────────────────────────────┘  │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │              Residual Connection 2                  │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                  RMSNorm (FFN)                      │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │               FeedForward (SwiGLU)                  │   │ │
                                    │  │  │  ┌─────────────┐   ┌─────────────┐   ┌───────────┐  │   │ │
                                    │  │  │  │     W1      │   │     W3      │   │    W2     │  │   │ │
                                    │  │  │  │ (ColumnPar) │   │ (ColumnPar) │   │ (RowPar)  │  │   │ │
                                    │  │  │  └─────────────┘   └─────────────┘   └───────────┘  │   │ │
                                    │  │  │        ↓               ↓                   ↑        │   │ │
                                    │  │  │  ┌─────────┐     ┌─────────┐               │        │   │ │
                                    │  │  │  │  SiLU   │     │  Linear │               │        │   │ │
                                    │  │  │  └─────────┘     └─────────┘               │        │   │ │
                                    │  │  │        ↓               ↓                   │        │   │ │
                                    │  │  │  ┌─────────────────────────┐               │        │   │ │
                                    │  │  │  │   Element-wise Multiply │───────────────┘        │   │ │
                                    │  │  │  └─────────────────────────┘                        │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                      output                         │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  └────────────────────────────────────────────────────────────┘ │
                                    │                            × n_layers                           │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                      Final RMSNorm                              │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    Output Projection                            │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                                                  输出
                                                          (仅最后一个token的logits)
```



***

# 源码解读
# 大语言模型核心组件源码解读

## 1. RMSNorm 归一化

RMSNorm（Root Mean Square Layer Normalization）是一种简化的归一化方法，与传统的 LayerNorm 相比，它移除了均值中心化步骤，仅保留了方差缩放部分，在保持模型性能的同时降低了计算成本。

### 1.1 数学原理

RMSNorm 的计算分为两步：



1. 计算输入张量的均方根（RMS）

$\text{rms}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$

其中$d$是特征维度，$\epsilon$是防止除零的小常数



1. 归一化并应用缩放参数

$\text{RMSNorm}(x) = \left( \frac{x}{\text{rms}(x)} \right) \cdot \gamma$

其中$\gamma$是可学习的缩放参数

### 1.2 代码解析



```
class RMSNorm(torch.nn.Module):

&#x20;   def \_\_init\_\_(self, dim: int, eps: float = 1e-6):

&#x20;       super().\_\_init\_\_()

&#x20;       self.eps = eps  # 防止除零的小常数

&#x20;       self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

&#x20;   def \_norm(self, x):

&#x20;       # 计算均方根并归一化

&#x20;       return x \* torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

&#x20;   def forward(self, x):

&#x20;       # 转换为float计算以保持精度，再转换回原类型

&#x20;       output = self.\_norm(x.float()).type\_as(x)

&#x20;       # 应用缩放参数

&#x20;       return output \* self.weight
```

与 LayerNorm 相比，RMSNorm 减少了均值计算步骤，在 Transformer 架构中能提供相似的正则化效果，但计算效率更高，因此在许多现代大语言模型中被采用。

## 2. RoPE 位置编码

RoPE（Rotary Position Embedding）是一种位置编码方法，通过对特征进行旋转操作来注入位置信息，具有良好的长度外推性和相对位置编码特性。

### 2.1 数学原理

RoPE 的核心思想是将位置信息编码为复数平面上的旋转操作：



1. 对于维度为$d$的特征，将其分为$d/2$对，每对表示复数的实部和虚部

2. 位置$m$的旋转角度为：$\theta_k = 10000^{-2(k-1)/d}$，其中$k=1,2,...,d/2$

3. 旋转操作：

$\begin{bmatrix} q_{m,2k-1} \\ q_{m,2k} \end{bmatrix}  = \begin{bmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{bmatrix}
\begin{bmatrix} q_{2k-1} \\ q_{2k} \end{bmatrix}$

这种旋转操作确保了注意力分数仅依赖于相对位置，满足$e^{i(m-n)\theta} = e^{im\theta} \cdot e^{-in\theta}$的性质。

### 2.2 代码解析



```
def precompute\_freqs\_cis(dim: int, end: int, theta: float = 10000.0):

&#x20;   # 计算频率参数 θ\_k = 1 / (theta^(2k/d))

&#x20;   freqs = 1.0 / (theta \*\* (torch.arange(0, dim, 2)\[: (dim // 2)].float() / dim))

&#x20;   # 生成位置索引 t = 0, 1, ..., end-1

&#x20;   t = torch.arange(end, device=freqs.device)

&#x20;   # 计算每个位置的角度: m\*θ\_k

&#x20;   freqs = torch.outer(t, freqs).float()

&#x20;   # 转换为复数形式: cos(mθ\_k) + i\*sin(mθ\_k)

&#x20;   freqs\_cis = torch.polar(torch.ones\_like(freqs), freqs)  # complex64

&#x20;   return freqs\_cis
```

该函数预计算了所有位置和维度的旋转因子，以复数形式存储，在注意力计算时通过复数乘法高效应用旋转操作。

## 3. Attention 注意力机制

注意力机制是 Transformer 的核心组件，能够让模型动态关注输入序列的不同部分。以下实现采用了多头注意力机制，并结合了 RoPE 位置编码。

### 3.1 数学原理

多头注意力的计算过程：



1. 将输入通过线性变换得到查询 (Q)、键 (K)、值 (V)

   $Q = W_q X, \quad K = W_k X, \quad V = W_v X$

2. 应用位置编码（RoPE）

   $\hat{Q} = \text{RoPE}(Q), \quad \hat{K} = \text{RoPE}(K)$

3. 计算注意力分数

   $\text{scores} = \frac{\hat{Q} \hat{K}^T}{\sqrt{d_k}} + \text{mask}$

   其中$d_k$是头维度，mask 用于防止关注未来位置

4. 计算注意力权重和输出

   $\text{attn} = \text{softmax}(\text{scores})$

   $\text{output} = \text{attn} \cdot V$

   $\text{MultiHead}(X) = W_o [\text{output}_1; \text{output}_2; ...; \text{output}_h]$

### 3.2 代码解析



```
class Attention(nn.Module):

&#x20;   def \_\_init\_\_(self, args: ModelArgs):

&#x20;       super().\_\_init\_\_()

&#x20;       # 计算本地头数（考虑模型并行）

&#x20;       self.n\_local\_heads = args.n\_heads // fs\_init.get\_model\_parallel\_world\_size()

&#x20;       self.head\_dim = args.dim // args.n\_heads  # 每个头的维度

&#x20;       # Q、K、V投影矩阵（列并行）

&#x20;       self.wq = ColumnParallelLinear(

&#x20;           args.dim,

&#x20;           args.n\_heads \* self.head\_dim,

&#x20;           bias=False,

&#x20;           gather\_output=False,

&#x20;           init\_method=lambda x: x,

&#x20;       )

&#x20;       self.wk = ColumnParallelLinear(

&#x20;           args.dim,

&#x20;           args.n\_heads \* self.head\_dim,

&#x20;           bias=False,

&#x20;           gather\_output=False,

&#x20;           init\_method=lambda x: x,

&#x20;       )

&#x20;       self.wv = ColumnParallelLinear(

&#x20;           args.dim,

&#x20;           args.n\_heads \* self.head\_dim,

&#x20;           bias=False,

&#x20;           gather\_output=False,

&#x20;           init\_method=lambda x: x,

&#x20;       )

&#x20;      &#x20;

&#x20;       # 输出投影矩阵（行并行）

&#x20;       self.wo = RowParallelLinear(

&#x20;           args.n\_heads \* self.head\_dim,

&#x20;           args.dim,

&#x20;           bias=False,

&#x20;           input\_is\_parallel=True,

&#x20;           init\_method=lambda x: x,

&#x20;       )

&#x20;       # 缓存K和V以加速推理（自回归生成时）

&#x20;       self.cache\_k = torch.zeros(

&#x20;           (args.max\_batch\_size, args.max\_seq\_len, self.n\_local\_heads, self.head\_dim)

&#x20;       ).cuda()

&#x20;       self.cache\_v = torch.zeros(

&#x20;           (args.max\_batch\_size, args.max\_seq\_len, self.n\_local\_heads, self.head\_dim)

&#x20;       ).cuda()

&#x20;   def forward(self, x: torch.Tensor, start\_pos: int, freqs\_cis: torch.Tensor, mask: Optional\[torch.Tensor]):

&#x20;       bsz, seqlen, \_ = x.shape

&#x20;       # 计算Q、K、V

&#x20;       xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

&#x20;       # 重塑为多头格式

&#x20;       xq = xq.view(bsz, seqlen, self.n\_local\_heads, self.head\_dim)

&#x20;       xk = xk.view(bsz, seqlen, self.n\_local\_heads, self.head\_dim)

&#x20;       xv = xv.view(bsz, seqlen, self.n\_local\_heads, self.head\_dim)

&#x20;       # 应用RoPE位置编码

&#x20;       xq, xk = apply\_rotary\_emb(xq, xk, freqs\_cis=freqs\_cis)

&#x20;       # 更新缓存

&#x20;       self.cache\_k = self.cache\_k.to(xq)

&#x20;       self.cache\_v = self.cache\_v.to(xq)

&#x20;       self.cache\_k\[:bsz, start\_pos : start\_pos + seqlen] = xk

&#x20;       self.cache\_v\[:bsz, start\_pos : start\_pos + seqlen] = xv

&#x20;       # 获取所有相关的K和V（包括缓存的历史信息）

&#x20;       keys = self.cache\_k\[:bsz, : start\_pos + seqlen]

&#x20;       values = self.cache\_v\[:bsz, : start\_pos + seqlen]

&#x20;       # 转置以适应注意力计算 (bs, heads, seqlen, head\_dim)

&#x20;       xq = xq.transpose(1, 2)

&#x20;       keys = keys.transpose(1, 2)

&#x20;       values = values.transpose(1, 2)

&#x20;      &#x20;

&#x20;       # 计算注意力分数并缩放

&#x20;       scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head\_dim)

&#x20;       # 应用mask（防止关注未来位置）

&#x20;       if mask is not None:

&#x20;           scores = scores + mask  # (bs, n\_local\_heads, slen, cache\_len + slen)

&#x20;      &#x20;

&#x20;       # 计算注意力权重

&#x20;       scores = F.softmax(scores.float(), dim=-1).type\_as(xq)

&#x20;       # 计算注意力输出

&#x20;       output = torch.matmul(scores, values)  # (bs, n\_local\_heads, slen, head\_dim)

&#x20;      &#x20;

&#x20;       # 重塑为原始格式

&#x20;       output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

&#x20;       # 输出投影

&#x20;       return self.wo(output)
```

该实现包含了模型并行（ColumnParallelLinear 和 RowParallelLinear）和 KV 缓存机制，适合大模型训练和高效推理。

## 4. FFN 前馈网络

前馈网络是 Transformer 中的另一个关键组件，负责对注意力输出进行非线性变换和特征提取。

### 4.1 数学原理

前馈网络的计算过程：

$\text{FFN}(x) = W_2 \cdot \text{Silu}(W_1 x) \odot W_3 x$

其中$\text{Silu}(x) = x \cdot \sigma(x)$是 Sigmoid 线性单元激活函数，$\odot$表示元素 - wise 乘法。

这种 "geglu" 结构相比传统的两层线性变换加激活函数的结构，能在相同参数量下提供更强的表达能力。

### 4.2 代码解析



```
class FeedForward(nn.Module):

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       dim: int,

&#x20;       hidden\_dim: int,

&#x20;       multiple\_of: int,

&#x20;   ):

&#x20;       super().\_\_init\_\_()

&#x20;       # 计算隐藏层维度（遵循特定比例和对齐要求）

&#x20;       hidden\_dim = int(2 \* hidden\_dim / 3)

&#x20;       hidden\_dim = multiple\_of \* ((hidden\_dim + multiple\_of - 1) // multiple\_of)

&#x20;       # 三层线性变换（使用模型并行）

&#x20;       self.w1 = ColumnParallelLinear(

&#x20;           dim, hidden\_dim, bias=False, gather\_output=False, init\_method=lambda x: x

&#x20;       )

&#x20;       self.w2 = RowParallelLinear(

&#x20;           hidden\_dim, dim, bias=False, input\_is\_parallel=True, init\_method=lambda x: x

&#x20;       )

&#x20;       self.w3 = ColumnParallelLinear(

&#x20;           dim, hidden\_dim, bias=False, gather\_output=False, init\_method=lambda x: x

&#x20;       )

&#x20;   def forward(self, x):

&#x20;       # 应用GEGLU结构：w2(silu(w1(x)) \* w3(x))

&#x20;       return self.w2(F.silu(self.w1(x)) \* self.w3(x))
```

该实现采用了 GEGLU（Gated Exponential Linear Unit）变体，通过引入门控机制增强了模型的表达能力。

## 5. Transformer Block 与整体架构

Transformer Block 是模型的基本构建单元，由注意力机制和前馈网络组成，通过残差连接和归一化增强训练稳定性。

### 5.1 代码解析

#### Transformer Block



```
class TransformerBlock(nn.Module):

&#x20;   def \_\_init\_\_(self, layer\_id: int, args: ModelArgs):

&#x20;       super().\_\_init\_\_()

&#x20;       self.n\_heads = args.n\_heads

&#x20;       self.dim = args.dim

&#x20;       self.head\_dim = args.dim // args.n\_heads

&#x20;       self.attention = Attention(args)  # 注意力模块

&#x20;       self.feed\_forward = FeedForward(  # 前馈网络

&#x20;           dim=args.dim, hidden\_dim=4 \* args.dim, multiple\_of=args.multiple\_of

&#x20;       )

&#x20;       self.layer\_id = layer\_id

&#x20;       self.attention\_norm = RMSNorm(args.dim, eps=args.norm\_eps)  # 注意力前的归一化

&#x20;       self.ffn\_norm = RMSNorm(args.dim, eps=args.norm\_eps)  # 前馈网络前的归一化

&#x20;   def forward(self, x: torch.Tensor, start\_pos: int, freqs\_cis: torch.Tensor, mask: Optional\[torch.Tensor]):

&#x20;       # 注意力子层：残差连接 + 归一化 + 注意力

&#x20;       h = x + self.attention.forward(self.attention\_norm(x), start\_pos, freqs\_cis, mask)

&#x20;       # 前馈子层：残差连接 + 归一化 + 前馈网络

&#x20;       out = h + self.feed\_forward.forward(self.ffn\_norm(h))

&#x20;       return out
```

#### 完整 Transformer 模型



```
class Transformer(nn.Module):

&#x20;   def \_\_init\_\_(self, params: ModelArgs):

&#x20;       super().\_\_init\_\_()

&#x20;       self.params = params

&#x20;       self.vocab\_size = params.vocab\_size

&#x20;       self.n\_layers = params.n\_layers

&#x20;       # 词嵌入层（并行实现）

&#x20;       self.tok\_embeddings = ParallelEmbedding(

&#x20;           params.vocab\_size, params.dim, init\_method=lambda x: x

&#x20;       )

&#x20;       # 堆叠多个Transformer Block

&#x20;       self.layers = torch.nn.ModuleList()

&#x20;       for layer\_id in range(params.n\_layers):

&#x20;           self.layers.append(TransformerBlock(layer\_id, params))

&#x20;       # 最终归一化层

&#x20;       self.norm = RMSNorm(params.dim, eps=params.norm\_eps)

&#x20;       # 输出层（映射到词汇表空间）

&#x20;       self.output = ColumnParallelLinear(

&#x20;           params.dim, params.vocab\_size, bias=False, init\_method=lambda x: x

&#x20;       )

&#x20;       # 预计算RoPE频率

&#x20;       self.freqs\_cis = precompute\_freqs\_cis(

&#x20;           self.params.dim // self.params.n\_heads, self.params.max\_seq\_len \* 2

&#x20;       )

&#x20;   @torch.inference\_mode()  # 推理模式，禁用梯度计算

&#x20;   def forward(self, tokens: torch.Tensor, start\_pos: int):

&#x20;       \_bsz, seqlen = tokens.shape

&#x20;       # 词嵌入

&#x20;       h = self.tok\_embeddings(tokens)

&#x20;       # 确保频率参数在正确设备上

&#x20;       self.freqs\_cis = self.freqs\_cis.to(h.device)

&#x20;       # 获取当前序列对应的RoPE频率

&#x20;       freqs\_cis = self.freqs\_cis\[start\_pos : start\_pos + seqlen]

&#x20;       # 构建掩码（仅在序列长度>1时需要）

&#x20;       mask = None

&#x20;       if seqlen > 1:

&#x20;           mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)

&#x20;           # 上三角掩码，防止关注未来位置

&#x20;           mask = torch.triu(mask, diagonal=start\_pos + 1).type\_as(h)

&#x20;       # 经过所有Transformer Block

&#x20;       for layer in self.layers:

&#x20;           h = layer(h, start\_pos, freqs\_cis, mask)

&#x20;       # 最终归一化

&#x20;       h = self.norm(h)

&#x20;       # 仅输出最后一个位置的logits（适用于自回归生成）

&#x20;       output = self.output(h\[:, -1, :])

&#x20;       return output.float()
```

### 5.2 整体架构说明



1. **输入处理**：将输入 token 通过词嵌入层转换为向量表示

2. **核心网络**：堆叠多个 Transformer Block，每个 Block 包含：

* 注意力子层（带 RMSNorm 和残差连接）

* 前馈网络子层（带 RMSNorm 和残差连接）

1. **输出处理**：通过最终的归一化和线性层，将最后一层输出映射到词汇表空间

该实现针对大模型训练和推理进行了优化：



* 使用模型并行（ColumnParallelLinear、RowParallelLinear）拆分计算

* 采用 KV 缓存机制加速自回归生成

* 结合 RoPE 位置编码提供更好的位置感知能力

* 使用 RMSNorm 提高计算效率

这些设计使得模型能够有效处理长序列输入，并在保持性能的同时提高计算效率。

> （注：文档部分内容可能由 AI 生成）
















































## RMSNorm


<details>
  <summary>RMSNorm代码实现</summary>
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

</details>




## RoPE



```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
```




  
## Transformer Block (* n_layers )

### Attention
```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)
```




### FFN
```python
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```


```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```



```python
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
```








