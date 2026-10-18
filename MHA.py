import torch
from einops import rearrange
from RoPE import RotaryPositionalEmbedding
def softmax(x,i):
    max_val=torch.max(x,dim=i,keepdim=True).values
    x=x-max_val
    x=torch.exp(x)
    sum_x=torch.sum(x, dim=i, keepdim=True)
    return x/sum_x 
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    实现缩放点积注意力
    q, k: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    mask: 可选布尔张量 (seq_len, seq_len)
          mask=True 代表该位置允许参与注意力权重计算, False 权重置零
    return: (batch_size, ..., seq_len, d_v)
    """
    scores=q@k.transpose(-2,-1)
    scores/=(q.shape[-1]**0.5)
    if mask is not None:
        scores=scores.masked_fill(mask==False,-torch.inf)
    scores=softmax(scores,-1)
    return scores@v
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_theta: float) -> None:
        super().__init__()
        assert d_model %num_heads==0
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  注意力头数
        max_seq_len: int  RoPE 最大序列长度
        rope_theta: float  RoPE 基数频率
        '''
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads
        self.w_qkv=torch.nn.Linear(d_model,3*d_model)
        self.rope=(RotaryPositionalEmbedding(theta=rope_theta,d_k=self.d_head,max_seq_len=max_seq_len)
                   if rope_theta > 0 else None)
        self.w_output=torch.nn.Linear(d_model,d_model)
        pass

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        '''
        x: (batch_size, seq_len, d_model)  输入张量
        token_positions: (batch_size, seq_len) 或 None  各 token 的位置下标
        返回形状: (batch_size, seq_len, d_model)
        '''
        qkv=self.w_qkv(x)
        q,k,v=torch.split(qkv,self.d_model,dim=-1)
        q=rearrange(q,"... seq (num_heads d_head)->... num_heads seq d_head",num_heads=self.num_heads,d_head=self.d_head)
        k=rearrange(k,"... seq (num_heads d_head)->... num_heads seq d_head",num_heads=self.num_heads,d_head=self.d_head)
        v=rearrange(v,"... seq (num_heads d_head)->... num_heads seq d_head",num_heads=self.num_heads,d_head=self.d_head)
        seq_len=x.shape[1]
        if self.rope is not None:
            if token_positions is  None:
                token_positions=torch.arange(seq_len,device=x.device)
            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)
        mask=torch.tril(torch.ones((seq_len,seq_len),device=q.device)).bool()
        output_head=scaled_dot_product_attention(q,k,v,mask)
        output_head=rearrange(output_head,"... num_heads seq d_head->... seq (num_heads d_head)")
        output=self.w_output(output_head)
        return output
