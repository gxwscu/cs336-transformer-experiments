"""使用本项目已训练的 TinyStoriesV2-GPT4 检查点生成英文故事。"""

import argparse
from pathlib import Path

import torch

from bpe_tokenizer import Tokenizer
from transformer import TransformerLM
from 文本生成 import decode


BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_DIR = BASE_DIR / "results" / "TinyStoriesV2-GPT4-train"
DEFAULT_CHECKPOINT = (
    BASE_DIR / "results" / "checkpoints" / "TinyStoriesV2-GPT4"
    / "TinyStoriesV2-GPT4-bs32-lr6.0e-04_6.0e-05-s42" / "iter005000.pt"
)

# 与这个已训练检查点的训练记录一致。旧检查点没有 config.json。
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
ROPE_THETA = 10000.0
D_FF = 64 * (D_MODEL * 8 // 3 // 64)
EOT_TEXT = "<|endoftext|>"


def parse_args():
    parser = argparse.ArgumentParser(description="使用 TinyStories 已训练模型生成英文故事")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="默认使用验证损失较低的 5000 步检查点")
    parser.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily.",
                        help="英文故事开头")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.temperature <= 0:
        raise ValueError("--temperature 必须大于 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p 必须在 (0, 1] 内")

    checkpoint = args.checkpoint.resolve()
    for path in (checkpoint, TOKENIZER_DIR / "vocab.json", TOKENIZER_DIR / "merge.txt"):
        require_file(path)

    tokenizer = Tokenizer()
    # 训练时注册的是另一个拼写；训练语料真正使用的分隔符由 3 个普通 token 组成。
    tokenizer.from_files(TOKENIZER_DIR / "vocab.json", TOKENIZER_DIR / "merge.txt",
                         ["<|end_of_text|>"])
    if len(tokenizer.vocab) != VOCAB_SIZE:
        raise ValueError(f"分词器词表大小 {len(tokenizer.vocab)} 与检查点 {VOCAB_SIZE} 不一致")
    eot_ids = tokenizer.encode(EOT_TEXT)
    if not eot_ids:
        raise ValueError("TinyStories 结束符编码为空")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 没有可用的 CUDA；请改用 --device cpu")

    # 旧检查点只保存了 model、optim 和 iter_num，模型结构取自当时的训练记录。
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        use_rmsnorm=True,
        post_norm=False,
        ffn_gated=True,
        tie_embedding=False,
        embed_init_std=0.02,
    )
    model.load_state_dict(state["model"])
    iter_num = int(state["iter_num"])
    del state
    model.to(device).eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("提示词编码后为空")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    out_ids = decode(
        model, prompt_ids, eottokenid=None,
        stop_token_sequences=[eot_ids],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=CONTEXT_LENGTH,
    )
    if out_ids[-len(eot_ids):] == eot_ids:
        out_ids = out_ids[:-len(eot_ids)]

    print(f"已载入 TinyStories 第 {iter_num} 步检查点 | "
          f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | 设备 {device}")
    print("\n生成结果：\n")
    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
