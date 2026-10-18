import torch
class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = 512, device=None, dtype=None, gated: bool = True):
        super().__init__()
        '''
        d_model: int  模型隐藏层维度
        d_ff: int  前馈网络中间层维度
        device: torch.device | None = None  参数存放设备
        dtype: torch.dtype | None = None  参数数据类型
        '''
        pass
        self.d_model=d_model
        self.d_ff=d_ff
        self.gated=gated
        self.w1=torch.nn.Linear(d_model,d_ff)
        if gated:
            self.w3=torch.nn.Linear(d_model,d_ff)
        self.w2=torch.nn.Linear(d_ff,d_model)  


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        FFN(x) = W2(SiLU(W1*x) ⊙ W3*x)，⊙ 代表逐元素相乘
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        '''
        y1=self.w1(x)
        y13=y1*torch.sigmoid(y1)
        if self.gated:
            y13=y13*self.w3(x)
        return self.w2(y13)