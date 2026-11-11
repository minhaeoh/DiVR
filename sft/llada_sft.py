#!/usr/bin/env python3
"""GSM8K SFT of LLaDA-8B-Base (the SFT generator of the paper).

Recipe: LoRA r 128 / alpha 256 / dropout 0.05 on every linear projection, learning rate 1e-5 (linear, 50 warmup
steps), batch 1 x 8 accumulation, 2 epochs, bf16, seed 42; GSM8K train partitioned by seed 42 into 6,973 training
and 500 held-out questions; targets EOS-filled to a multiple of 32; one held-out evaluation and checkpoint per
epoch; the final epoch's adapter is the generator.

    python -m sft.llada_sft --output_dir checkpoints/llada-base-gsm8k-sft
    python -m sft.llada_sft --output_dir /tmp/sft_smoke --max_steps 4 --limit 32 --n_eval 8      # smoke test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from divr.data import load_gsm8k_split
from divr.family import LLADA
from sft.sft_common import DiffusionCollator, DiffusionTrainer, SFTDataset, enable_llada_activation_checkpointing, freeze_eval_noise, preprocess_gsm8k

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=128); ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=640)
    ap.add_argument("--eos_fill_block", type=int, default=32)
    ap.add_argument("--split_seed", type=int, default=42); ap.add_argument("--n_holdout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N training examples")
    ap.add_argument("--n_eval", type=int, default=0, help="smoke: first N held-out examples")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_rows, hold_rows, split_meta = load_gsm8k_split(args.split_seed, args.n_holdout)
    processed, stats = preprocess_gsm8k(train_rows, tok, args.max_length, args.eos_fill_block)
    eval_processed = freeze_eval_noise(preprocess_gsm8k(hold_rows, tok, args.max_length, args.eos_fill_block)[0])
    if args.limit:
        processed = processed[:args.limit]
    if args.n_eval:
        eval_processed = eval_processed[:args.n_eval]
    assert tok.eos_token_id in LLADA.eos_ids, (tok.eos_token_id, LLADA.eos_ids)
    print(f"[sft] {len(processed)} train / {len(eval_processed)} held out; {stats['n_dropped_too_long']} dropped; "
          f"len mean {stats['kept_len_mean']:.0f} max {stats['kept_len_max']}", flush=True)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    enable_llada_activation_checkpointing(model)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                                             target_modules=TARGET_MODULES, task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()

    steps_per_epoch = max(1, len(processed) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, max_grad_norm=1.0, lr_scheduler_type="linear", warmup_steps=50,
        bf16=True, seed=args.seed, logging_steps=20,
        save_strategy="steps", save_steps=steps_per_epoch, eval_strategy="steps", eval_steps=steps_per_epoch,
        per_device_eval_batch_size=args.per_device_train_batch_size, save_total_limit=None,
        report_to=[], remove_unused_columns=False, prediction_loss_only=True, gradient_checkpointing=False,
        dataloader_num_workers=2)
    trainer = DiffusionTrainer(model=model, args=targs, train_dataset=SFTDataset(processed), eval_dataset=SFTDataset(eval_processed),
                               data_collator=DiffusionCollator(tokenizer=tok, mask_token_id=LLADA.mask_id))
    t0 = time.time()
    trainer.train()
    ev = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    meta = {"model": args.model, "split": split_meta, "data_stats": stats, "n_train": len(processed), "n_eval": len(eval_processed),
            "final_eval": ev, "hours": round((time.time() - t0) / 3600, 2), **vars(args), "lora_target_modules": TARGET_MODULES}
    json.dump(meta, open(pathlib.Path(args.output_dir) / "train_meta.json", "w"), indent=1, default=str)
    print(f"[sft] done in {meta['hours']} h; held-out eval {ev}", flush=True)


if __name__ == "__main__":
    main()
