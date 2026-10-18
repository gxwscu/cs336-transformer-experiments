"""OWT experiment launcher for this project's train_llm.py. No training on import."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def make_plan(args):
    common = dict(batch_size=args.batch_size, total_iters=args.total_iters,
                  warmup_iters=max(1, args.total_iters // 10),
                  val_interval=max(1, args.total_iters // 100),
                  save_interval=max(1, args.total_iters // 4),
                  max_lr=args.max_lr, min_lr=args.max_lr / 10)
    plan = []
    def add(name, amp=True, **changes):
        cfg = dict(common, **changes)
        cfg['cosine_cycle_iters'] = cfg['total_iters']
        cfg['amp'] = amp
        cfg['name'] = name
        plan.append(cfg)
    group = args.group
    if group == 'smoke':
        add('smoke', total_iters=3, warmup_iters=1, val_interval=1, save_interval=3)
    if group in ('lr', 'all'):
        for lr in args.learning_rates:
            add('lr_sweep', max_lr=lr, min_lr=lr / 10)
    if group in ('bs', 'all'):
        # All runs see exactly the same number of sampled training tokens.
        samples = args.total_iters * args.batch_size
        for bs in args.batch_sizes:
            if samples % bs:
                raise ValueError(f'total_iters * batch_size ({samples}) must be divisible by {bs}')
            steps = samples // bs
            add('bs_sweep', batch_size=bs, total_iters=steps,
                warmup_iters=max(1, steps // 10), val_interval=max(1, steps // 100),
                save_interval=max(1, steps // 4))
    if group == 'amp':
        add('fp32', amp=False)
        add('amp', amp=True)
    if group in ('ablation', 'all'):
        add('ablation_base')
        add('ablation_no_rmsnorm', no_rmsnorm=True, ablation_name='no_rmsnorm')
        add('ablation_post_norm', post_norm=True, ablation_name='post_norm')
        add('ablation_nope', rope_theta=0, ablation_name='nope')
        add('ablation_no_gated', no_gated_ffn=True, ablation_name='no_gated')
        add('ablation_tied', tie_embedding=True, ablation_name='tied')
    if group in ('accel', 'all'):
        add('accel_base')
        add('accel_tied', tie_embedding=True)
        add('accel_muon', optimizer='muon')
        add('accel_muon_tied', optimizer='muon', tie_embedding=True)
    return plan


def resume_plan(args):
    if not args.resume_from:
        raise ValueError("resume 需要 --resume-from 检查点路径")
    checkpoint = args.resume_from.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少 {config_path}；请使用原训练命令直接传 --resume_from")
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    if previous.get("dataset_name") != "owt":
        raise ValueError("这个实验入口只支持 OWT 数据集")
    keys = (
        "batch_size", "total_iters", "warmup_iters", "cosine_cycle_iters",
        "val_interval", "save_interval", "max_lr", "min_lr", "vocab_size",
        "context_length", "d_model", "num_layers", "num_heads", "rope_theta",
        "weight_decay", "beta1", "beta2", "max_l2_norm", "seed",
        "val_batch_size", "val_batches", "val_seed", "embed_init_std",
        "no_rmsnorm", "post_norm", "no_gated_ffn", "tie_embedding",
        "optimizer", "wandb_project", "dataset_name",
        "phase_start_iter", "phase_total_iters", "phase_max_lr",
        "phase_min_lr", "phase_warmup_iters",
    )
    cfg = {key: previous[key] for key in keys if key in previous}
    cfg.update(name="resume", amp=bool(previous.get("amp", False)),
               resume_from=str(checkpoint))
    return [cfg]


def extend_plan(args):
    if not args.extend_from or args.additional_iters is None:
        raise ValueError("extend 需要 --extend-from 检查点 和 --additional-iters 新增步数")
    if args.additional_iters <= 0:
        raise ValueError("--additional-iters 必须为正数")
    checkpoint = args.extend_from.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Reuse the original model/optimizer configuration; only the new phase schedule changes.
    source = type("Source", (), {"resume_from": checkpoint})()
    cfg = resume_plan(source)[0]
    cfg.pop("resume_from")
    if not checkpoint.stem.startswith("iter") or not checkpoint.stem[4:].isdigit():
        raise ValueError("检查点文件名应为 iter000000.pt")
    step = int(checkpoint.stem[4:])
    if step != cfg["total_iters"]:
        raise ValueError("请选择原阶段最后一步的检查点；阶段中途请使用 resume")
    old_min_lr = cfg["min_lr"]
    cfg.update(name="extend", extend_from=str(checkpoint),
               phase_start_iter=step, phase_total_iters=args.additional_iters,
               phase_max_lr=old_min_lr, phase_min_lr=old_min_lr / 10,
               phase_warmup_iters=0, total_iters=step + args.additional_iters)
    return [cfg]


def command(args, root, cfg, name, save_dir):
    values = dict(run_name=name, dataset_name='owt', wandb_project='easy-llm-owt',
                  wandb_mode=args.wandb_mode, vocab_size=32000, context_length=256,
                  d_model=512, num_layers=4, num_heads=16, rope_theta=10000,
                  weight_decay=0.1, beta1=0.9, beta2=0.95, max_l2_norm=1.0,
                  val_batch_size=args.val_batch_size, val_batches=args.val_batches,
                  val_seed=args.val_seed, seed=args.seed, train_data_path=str(root/'data/OWT/owt_train.npy'),
                  val_data_path=str(root/'data/OWT/owt_valid.npy'),
                  save_dir=str(save_dir), device='cuda', optimizer='adamw')
    values.update({k: v for k, v in cfg.items() if k not in ('name', 'amp')})
    cmd = [sys.executable, '-u', str(root/'train_llm.py')]
    for k, v in values.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k}')
        else:
            cmd.extend([f'--{k}', str(v)])
    if cfg['amp']:
        cmd.append('--amp')
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('group', nargs='?', default='smoke', choices=['smoke', 'lr', 'bs', 'ablation', 'accel', 'amp', 'all', 'resume', 'extend'])
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--total-iters', type=int, default=16000)
    p.add_argument('--max-lr', type=float, default=6e-4)
    p.add_argument('--learning-rates', type=float, nargs='+', default=[1e-4, 3e-4, 6e-4, 1.2e-3])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--val-batch-size', type=int, default=16)
    p.add_argument('--val-batches', type=int, default=16)
    p.add_argument('--val-seed', type=int, default=2026)
    p.add_argument('--wandb-mode', choices=['offline', 'online', 'disabled'], default=os.getenv('WANDB_MODE', 'offline'))
    p.add_argument('--resume-from', type=Path, help='resume a single run from a checkpoint')
    p.add_argument('--extend-from', type=Path, help='last checkpoint from a completed phase')
    p.add_argument('--additional-iters', type=int, help='training steps in the new phase')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print commands without training')
    args = p.parse_args()
    if min(args.batch_size, args.total_iters, *args.batch_sizes) < 1 or min(args.max_lr, *args.learning_rates) <= 0:
        p.error('batch sizes, iterations and learning rates must be positive')
    if min(args.val_batch_size, args.val_batches) < 1 or args.val_seed < 0:
        p.error('validation batch sizes must be positive and val-seed nonnegative')
    root = args.project_root.resolve()
    if not (root/'train_llm.py').is_file():
        p.error(f'train_llm.py not found in {root}; use --project-root')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA is unavailable in this Python environment')
    if not torch.cuda.is_bf16_supported():
        p.error('This launcher uses the existing BF16 training path; BF16 CUDA support is required')
    print(f'Python: {sys.executable}\nGPU: {torch.cuda.get_device_name(0)}', flush=True)
    arrays = []
    for name in ('owt_train.npy', 'owt_valid.npy'):
        path = root/'data'/'OWT'/name
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        if a.ndim != 1 or len(a) <= 256 or a.dtype.kind not in 'iu':
            p.error(f'Invalid token array: {path}')
        # Check samples at the beginning, middle and end; this is not a full data scan.
        for start in (0, max(0, len(a)//2-2048), max(0, len(a)-4096)):
            chunk = a[start:start+4096]
            if int(chunk.min()) < 0 or int(chunk.max()) >= 32000:
                p.error(f'Token IDs out of range in {path}')
        print(f'{name}: {len(a):,} tokens, {a.dtype}', flush=True)
        arrays.append(a)
    if args.group == "resume":
        if args.extend_from or args.additional_iters is not None:
            p.error("resume 模式只接受 --resume-from")
        try:
            plan = resume_plan(args)
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))
    elif args.group == "extend":
        if args.resume_from:
            p.error("extend 模式只接受 --extend-from")
        try:
            plan = extend_plan(args)
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))
    else:
        if args.resume_from or args.extend_from or args.additional_iters is not None:
            p.error("断点或新阶段参数只能用于 resume 或 extend 模式")
        plan = make_plan(args)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out = root/'results'/'experiment_batches'/stamp
    records = []
    for i, cfg in enumerate(plan, 1):
        name = f"{i:02d}-{cfg['name']}-lr{cfg['max_lr']:.1e}-bs{cfg['batch_size']}-s{args.seed}"
        cmd = command(args, root, cfg, name, out/'checkpoints')
        steps_this_run = cfg['phase_total_iters'] if args.group == 'extend' else cfg['total_iters']
        tokens = cfg['batch_size'] * steps_this_run * 256
        records.append(dict(name=name, config=cfg, command=cmd, train_tokens=tokens, status='pending'))
        print(f'\n[{i}/{len(plan)}] {name}: {tokens:,} training tokens; '
              f'{tokens/len(arrays[0]):.4f} x dataset size (random sampling)', flush=True)
        print(subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        print('\nDRY RUN: no training started and no experiment files created.')
        return
    out.mkdir(parents=True)
    manifest = out/'manifest.json'
    def save():
        manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    for record in records:
        print(f"\nSTART {record['name']}\nResults: {out}", flush=True)
        record['status'] = 'running'
        start = time.perf_counter()
        save()
        proc = None
        try:
            with (out/f"{record['name']}.log").open('w', encoding='utf-8') as log:
                proc = subprocess.Popen(record['command'], cwd=root, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace')
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
            record.update(exit_code=code, status='completed' if code == 0 else 'failed')
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait()
            record['status'] = 'interrupted_or_error'
            raise
        finally:
            record['elapsed_seconds'] = time.perf_counter() - start
            save()
        if code:
            raise SystemExit(f"Experiment failed (exit {code}); remaining runs stopped. See {out}")
    print(f'\nAll experiments completed. Logs and checkpoints: {out}')


if __name__ == '__main__':
    main()
