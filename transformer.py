import torch
import torch.nn as nn
from FFN import FFN
from embedding import embedding
from MHA import MultiHeadSelfAttention
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, rope_theta: float, use_rmsnorm: bool = True,
                 post_norm: bool = False, ffn_gated: bool = True) -> None:
        super().__init__()
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  注意力头数
        d_ff: int  前馈网络中间层维度
        max_seq_len: int  RoPE 最大序列长度
        rope_theta: float  RoPE 基数频率
        '''
        self.post_norm = post_norm
        self.norm1=torch.nn.RMSNorm(d_model,eps=1e-5) if use_rmsnorm else nn.Identity()
        self.MultiheadAttention=MultiHeadSelfAttention(d_model,num_heads,max_seq_len,rope_theta)
        self.norm2=torch.nn.RMSNorm(d_model,eps=1e-5) if use_rmsnorm else nn.Identity()
        self.ffn=FFN(d_model,d_ff,gated=ffn_gated)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        输入形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, seq_len, d_model)
        '''
        if self.post_norm:
            x = self.norm1(x + self.MultiheadAttention(x))
            return self.norm2(x + self.ffn(x))
        x=x+self.MultiheadAttention(self.norm1(x))
        x=x+self.ffn(self.norm2(x))
        return x
class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        ffn_gated: bool = True,
        tie_embedding: bool = False,
        embed_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        '''
        d_model: int  模型隐藏层维度
        num_heads: int  注意力头数
        d_ff: int  前馈网络中间层维度
        vocab_size: int  词表大小
        context_length: int  最大上下文长度
        num_layers: int  Transformer Block 层数
        rope_theta: float  RoPE 基数频率
        ''' 
        if embed_init_std <= 0:
            raise ValueError("embed_init_std must be positive")
        self.embedding=embedding(vocab_size,d_model)
        if embed_init_std != 0.02:
            nn.init.trunc_normal_(self.embedding.w, std=embed_init_std,
                                  a=-3*embed_init_std, b=3*embed_init_std)
        self.transformerBlocks=nn.Sequential(
            *[TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,
                               use_rmsnorm,post_norm,ffn_gated) for _ in range(num_layers)]
        )
        self.final_norm=nn.RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.llm_head=nn.Linear(d_model,vocab_size)
        if tie_embedding:
            self.llm_head.weight = self.embedding.w
        pass

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''
        tokens: (batch_size, seq_len) 整数 token IDs
        return: (batch_size, seq_len, vocab_size) next-token logits
        '''
        x_emded=self.embedding(tokens)
        x_blocks=self.transformerBlocks(x_emded)
        llm_out=self.llm_head(self.final_norm(x_blocks))
        return llm_out
