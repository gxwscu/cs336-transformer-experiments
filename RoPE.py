import torch
import torch.nn as nn
from einops import rearrange
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        '''
        theta: float  RoPE 的 Θ 值
        d_k: int  待编码的向量维度（一般指多头注意力层里 query 和 key 向量的维度）
        max_seq_len: int  输入的最大序列长度
        device: torch.device | None = None  存储 buffer 的设备
        '''
        assert d_k%2==0
        positions=torch.arange(max_seq_len,device=device)
        dim_indices=torch.arange(0,d_k,2,device=device)
        freqs=theta**-(dim_indices/d_k)
        angles = positions.reshape(-1, 1).float() @ freqs.reshape(1, -1)
        cos_table=torch.cos(angles)
        sin_table=torch.sin(angles)
        self.register_buffer("cos_table",cos_table)
        self.register_buffer("sin_table",sin_table)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        x: (..., seq_len, d_k) 输入张量，支持任意批量维度
        token_positions: (..., seq_len) 存储每个 token 的位置下标
        return: (..., seq_len, d_k) 旋转位置编码后的张量，与输入形状完全一致
        '''
        cos=self.cos_table[token_positions]
        sin=self.sin_table[token_positions]
        x1,x2=rearrange(x,"... (half_d xy)->... half_d xy",xy=2).unbind(dim=-1)
        x1_rot=x1*cos-x2*sin
        x2_rot=x1*sin+x2*cos
        x_rot=torch.stack([x1_rot,x2_rot],dim=-1)
        x_rot=rearrange(x_rot,"... half_d xy->... (half_d xy)")
        return x_rot