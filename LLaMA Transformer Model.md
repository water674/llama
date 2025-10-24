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



## RMSNorm（Root Mean Square Layer Normalization）

### 数学原理

- 计算输入张量的均方根（RMS）

$$
\text{rms}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

其中 $d$ 是特征维度， $\epsilon$ 是防止除零的小常数



- 归一化并应用缩放参数

$$
\text{RMSNorm}(x) = \left( \frac{x}{\text{rms}(x)} \right) \cdot \gamma
$$

  其中 $\gamma$ 是可学习的缩放参数


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




## RoPE（Rotary Position Embedding）

### 数学原理

- 对于维度为 $d$ 的特征，将其分为 $d/2$ 对，每对表示复数的实部和虚部

- 位置 $m$ 的旋转角度为： $\theta_k = 10000^{-2(k-1)/d}$ ，其中 $k=1,2,...,d/2$ 

- 旋转操作：

```math
\begin{bmatrix} q_{m,2k-1} \\ q_{m,2k} \end{bmatrix}  = \begin{bmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{bmatrix} \begin{bmatrix} q_{2k-1} \\ q_{2k} \end{bmatrix}
```


这种旋转操作确保了注意力分数仅依赖于相对位置，满足 $e^{i(m-n)\theta} = e^{im\theta} \cdot e^{-in\theta}$ 的性质。

<details>
<summary>RoPE代码实现</summary>
  
  ```python
  def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
      t = torch.arange(end, device=freqs.device)  # type: ignore
      freqs = torch.outer(t, freqs).float()  # type: ignore
      freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
      return freqs_cis 
  ```
</details>







## Transformer Block (* n_layers )

### Attention

#### 数学原理

- 将输入通过线性变换得到查询 (Q)、键 (K)、值 (V)

$$
Q = W_q X, \quad K = W_k X, \quad V = W_v X
$$

- 应用位置编码（RoPE）

$$
\hat{Q} = \text{RoPE}(Q), \quad \hat{K} = \text{RoPE}(K)
$$

- 计算注意力分数

$$
\text{scores} = \frac{\hat{Q} \hat{K}^T}{\sqrt{d_k}} + \text{mask}
$$

   其中 $d_k$ 是头维度，mask 用于防止关注未来位置

- 计算注意力权重和输出

$$
\text{attn} = \text{softmax}(\text{scores})
$$

$$
\text{output} = \text{attn} \cdot V
$$

$$
\text{MultiHead}(X) = W_o [\text{output}_1; \text{output}_2; ...; \text{output}_h]
$$




<details>
<summary>Attention代码实现</summary>
  
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
</details>










### FFN

#### 数学原理

前馈网络的计算过程：

$$
\text{FFN}(x) = W_2 \cdot \text{Silu}(W_1 x) \odot W_3 x
$$

其中 $\text{Silu}(x) = x \cdot \sigma(x)$ 是 Sigmoid 线性单元激活函数， $\odot$ 表示逐个元素乘法。

<details>
<summary>FFN代码实现</summary>
  
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
</details>


















## Transformer Block 与整体架构

Transformer Block 是模型的基本构建单元，由注意力机制和前馈网络组成，通过残差连接和归一化增强训练稳定性。

<details>
<summary>Transformer块代码实现</summary>
  
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
</details>
















n_layer个Transformer块构成Transformer整个核心部分。






<details>
<summary>Transformer代码实现</summary>
  
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
</details>



