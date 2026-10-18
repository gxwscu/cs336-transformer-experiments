import torch
@torch.inference_mode()
def decode(model, prompt, eottokenid=None, max_new_tokens=256, temperature=0.5, top_p=0.9, context_length=256, stop_token_sequences=None) -> list[int]:
    """
    从语言模型采样生成，返回新生成的 token ID（包含 prompt）。
    model: 输入 (batch, seq_len)，输出 (batch, seq_len, vocab_size)
    prompt: 已编码的 token ID 列表
    eottokenid: 单 token 结束符的 ID；None 表示不检查单 token 结束符
    stop_token_sequences: 多 token 结束符列表；完整匹配后停止，返回值保留结束符
    max_new_tokens: 最多新生成的 token 数
    temperature: 温度参数
    top_p: Top-p 核采样阈值
    context_length: 模型最大上下文长度
    """
    stop_sequences = [list(seq) for seq in (stop_token_sequences or [])]
    if any(not seq for seq in stop_sequences):
        raise ValueError("结束符 token 序列不能为空")
    ids=list(prompt)
    prompt_length = len(ids)
    device = next(model.parameters()).device
    x=torch.tensor(ids,dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    for i in range(0,max_new_tokens):
        x_input=x[:,-context_length:]
        pred_y=model(x_input)[:,-1,:]
        probs = torch.softmax(pred_y / temperature, dim=-1).squeeze(0)
        sorted_probs, sorted_indices = probs.sort(descending=True) # 同时返回 (排序后的序列， 排序后的序列元素的原始下标/ token id)
        cumsum = sorted_probs.cumsum(dim=-1) # (vocab_size,)
        cutoff_idx = torch.searchsorted(cumsum, top_p)   # 找到第一个使前缀和 >= p 的下标，假设是 j；
        sorted_mask = torch.arange(cumsum.shape[-1], device=probs.device) <= cutoff_idx # 构建掩码数组，0~j-1 号元素为 true，剩余元素为 false
        probs[sorted_indices[~sorted_mask]] = 0
        probs = probs / probs.sum()                  # 丢掉尾部之后要重新归一化
        next_token_id = torch.multinomial(probs, num_samples=1).item() # 按给定概率 probs 采样 num_samples 个元素（默认不放回采样，即采样出的多个元素互不相同）
        ids.append(next_token_id) # 将新生成的 token 加入到输入序列中（列表形式）
        x = torch.cat([x, torch.tensor([[next_token_id]], device=x.device)], dim=1) # 将新生成的 token 加入到输入序列中（张量形式）
        if eottokenid is not None and next_token_id == eottokenid: # 生成了结束符，解码提前终止
            break
        # 只检查新生成的内容，避免把提示词末尾误当成结束符。
        if any(
            len(ids) - prompt_length >= len(seq) and ids[-len(seq):] == seq
            for seq in stop_sequences
        ):
            break
    return ids