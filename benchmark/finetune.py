"""Noise-aware fine-tuning of the cross-encoder re-ranker (plain PyTorch training loop).

Three variants share the same queries, negatives procedure, step budget and model selection;
they differ ONLY in which surface form of each training query the model sees:

  clean  — original text every time            (control: in-domain adaptation, no noise)
  noisy  — the ASR transcript every time       (noise only)
  mixed  — clean or ASR, re-drawn 50/50 per query per epoch (noise as data augmentation)
  distill — consistency distillation (Part 2): a frozen copy of the base model scores each
            candidate list for the CLEAN query (teacher); the student is trained so that its
            scores for the NOISY transcript, and for the clean query, match the teacher's
            distribution (KL divergence, temperature T). No relevance labels are used, so label
            noise from incomplete judgments cannot enter, and the target is the zero-shot
            model's own clean-query ranking — which noise-only training failed to beat.

Objective: pointwise binary cross-entropy on (query, passage) logits — the same objective the
base MS MARCO checkpoint was trained with — over 1 positive + K BM25 hard negatives per query.
Optimiser: AdamW (decoupled weight decay 0.01, none on biases/LayerNorm), linear warm-up 10%
then linear decay, gradient-norm clipping at 1.0, dropout 0.1 left on from the checkpoint.
Model selection, two protocols reported side by side:
  A. early stopping — validation nDCG@10 (mean over clean and ASR forms of 120 held-out
     TRAIN-split queries) at step 0 and after every epoch; best checkpoint saved as models/<name>;
  B. fixed budget — the weights after the last epoch, saved as models/<name>-last.

Part 3 options (defaults reproduce Parts 1-2 exactly): --base picks the student checkpoint
(e.g. bge-base), --teacher a different, frozen teacher for distillation (e.g. bge-m3),
--noisy-form which transcript field is the "noisy" query (Part 3 HeySQuAD: real | synth),
--candidates bm25 | hybrid hard negatives, --fp16 mixed precision for large models, and
--trainset may be given several times (training sets are concatenated).

Usage: python -I finetune.py --variant mixed --seed 0
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, metrics, speech  # noqa: E402
from bench.first_stage import BM25  # noqa: E402
from bench.rerankers import resolve  # noqa: E402

VAL_DEPTH = 25
MODELS_DIR = HERE / "models"
LOG_DIR = HERE / "results" / "training"


def pick_form(variant: str, example: dict, epoch: int, seed: int) -> str:
    if variant in ("clean", "noisy"):
        return variant
    return "noisy" if zlib.crc32(f"{seed}:{epoch}:{example['dataset']}:{example['qid']}".encode()) % 2 else "clean"


def epoch_pairs(examples, variant, epoch, seed, negatives, rng, noisy_form="noisy", neg_key="negatives"):
    pairs = []
    for ex in examples:
        form = pick_form(variant, ex, epoch, seed)
        field = noisy_form if form == "noisy" else "clean"
        positive = rng.choice(sorted(ex["positives"]))
        pool = ex[neg_key][field]
        pairs.append((ex["dataset"], ex[field], positive, 1.0))
        for doc in rng.sample(pool, min(negatives, len(pool))):
            pairs.append((ex["dataset"], ex[field], doc, 0.0))
    rng.shuffle(pairs)
    return pairs


@torch.no_grad()
def validate(model, tok, device, val_runs, corpora, max_len) -> dict[str, float]:
    model.eval()
    out = {}
    for form in ("clean", "noisy"):
        scores = []
        for ds_name, query, run, rel in val_runs[form]:
            if not run:
                scores.append(0.0)
                continue
            texts = [corpora[ds_name].doc_text(d) for d in run]
            logits = []
            for i in range(0, len(texts), 64):
                enc = tok([query] * len(texts[i:i + 64]), texts[i:i + 64], padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt").to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and AMP):
                    logits.append(model(**enc).logits.view(-1).float().cpu())
            order = torch.cat(logits).argsort(descending=True).tolist()
            scores.append(metrics.evaluate([run[i] for i in order], rel)["ndcg@10"])
        out[form] = float(np.mean(scores))
    out["mean"] = (out["clean"] + out["noisy"]) / 2
    model.train()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("clean", "noisy", "mixed", "distill"), required=True)
    p.add_argument("--temperature", type=float, default=1.0, help="distill only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--negatives", type=int, default=7)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--device", default=data.best_device())
    p.add_argument("--trainset", type=Path, action="append", default=None)
    p.add_argument("--base", default="ce-l6", help="student checkpoint tag (bench.rerankers)")
    p.add_argument("--teacher", default="", help="distill: frozen teacher tag (default: copy of the base)")
    p.add_argument("--noisy-form", default="noisy", help="training-set field used as the noisy query")
    p.add_argument("--candidates", choices=("bm25", "hybrid"), default="bm25")
    p.add_argument("--fp16", action="store_true", help="mixed-precision training (large models)")
    p.add_argument("--last-only", action="store_true",
                   help="save only the fixed-budget (protocol B) weights; validation is still logged")
    p.add_argument("--name", default="", help="override the output name (smoke tests)")
    args = p.parse_args()

    global AMP
    AMP = args.fp16
    args.trainset = args.trainset or [data.CACHE / "trainset.json"]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from sentence_transformers import CrossEncoder
    base_ckpt, MAX_LEN = resolve(args.base)
    ce = CrossEncoder(base_ckpt, max_length=MAX_LEN, device=args.device)
    model, tok, device = ce.model, ce.tokenizer, ce.model.device
    nf, neg_key = args.noisy_form, ("negatives_hybrid" if args.candidates == "hybrid" else "negatives")

    examples = [e for path in args.trainset for e in data.load_json(path)["examples"]]
    train_all = [e for e in examples if e["role"] == "train"]
    # Training-data hygiene: an ASR hallucination ("prunes" -> "I'm going to go to the bathroom")
    # no longer expresses the information need, so pairing it with the relevant passage is label
    # noise. Drop such queries from training for EVERY variant (same query set across variants).
    # Validation and test data are never filtered.
    train = [e for e in train_all if not speech.is_hallucination(e["clean"], e[nf])]
    val = [e for e in examples if e["role"] == "val"]
    corpora = {name: data.load(name) for name in {e["dataset"] for e in examples}}
    bm25 = {name: BM25(ds) for name, ds in corpora.items()}
    val_runs = {form: [(e["dataset"], e[field], [d for d, _ in bm25[e["dataset"]].search(e[field], VAL_DEPTH)],
                        e["positives"]) for e in val] for form, field in (("clean", "clean"), ("noisy", nf))}

    group = 1 + args.negatives
    per_step = max(1, args.batch // group)              # distill: queries per step (same pairs/step)
    steps_per_epoch = (math.ceil(len(train) / per_step) if args.variant == "distill"
                       else math.ceil(len(train) * group / args.batch))
    total = steps_per_epoch * args.epochs
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [{"params": [w for n, w in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": args.weight_decay},
              {"params": [w for n, w in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    optim = torch.optim.AdamW(groups, lr=args.lr)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    name = args.name or f"ft-{args.variant}-s{args.seed}"
    log = {"hallucinations_dropped": len(train_all) - len(train), "variant": args.variant, "seed": args.seed, "args": {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()}, "train_queries": len(train),
           "val_queries": len(val), "steps_per_epoch": steps_per_epoch, "loss": [], "val": []}
    best = validate(model, tok, device, val_runs, corpora, MAX_LEN)
    log["val"].append({"epoch": 0, "step": 0, **best})
    print(f"[{name}] step 0 (base model) val nDCG@10 clean={best['clean']:.4f} noisy={best['noisy']:.4f}", flush=True)
    best_mean, best_epoch = best["mean"], 0
    if not args.last_only:
        ce.save(str(MODELS_DIR / name))                  # the base is a valid "best" until beaten

    rng = random.Random(args.seed)
    teacher, t_tok, t_len = None, tok, MAX_LEN
    if args.variant == "distill":
        if args.teacher and args.teacher != args.base:
            t_ckpt, t_len = resolve(args.teacher)
            t_ce = CrossEncoder(t_ckpt, max_length=t_len, device=args.device)
            teacher, t_tok = t_ce.model.eval(), t_ce.tokenizer
            if device.type == "cuda":
                teacher.half()
        else:
            teacher = copy.deepcopy(model).eval()
        for w in teacher.parameters():
            w.requires_grad_(False)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and device.type == "cuda")
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    tau = args.temperature

    def logits(net, queries, docs, tokenizer=tok, max_len=MAX_LEN):
        enc = tokenizer(queries, docs, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and AMP):
            return net(**enc).logits.view(-1, group).float()

    def batches(epoch):
        if args.variant != "distill":
            pairs = epoch_pairs(train, args.variant, epoch, args.seed, args.negatives, rng, nf, neg_key)
            return [pairs[i:i + args.batch] for i in range(0, len(pairs), args.batch)]
        order = train[:]
        rng.shuffle(order)
        out = []
        for i in range(0, len(order), per_step):
            chunk = []
            for ex in order[i:i + per_step]:
                pool = ex[neg_key][nf]
                negs = rng.sample(pool, min(args.negatives, len(pool)))
                negs += [negs[-1]] * (args.negatives - len(negs)) if negs else []
                docs = [rng.choice(sorted(ex["positives"]))] + negs
                if len(docs) == group:
                    chunk.append((ex["dataset"], ex["clean"], ex[nf], docs))
            if chunk:
                out.append(chunk)
        return out

    model.train()
    step, started, window = 0, time.time(), []
    for epoch in range(1, args.epochs + 1):
        for batch in batches(epoch):
            if args.variant == "distill":
                docs = [corpora[ds].doc_text(d) for ds, _, _, ds_docs in batch for d in ds_docs]
                clean_q = [c for _, c, _, ds_docs in batch for _ in ds_docs]
                noisy_q = [n for _, _, n, ds_docs in batch for _ in ds_docs]
                with torch.no_grad():
                    target = torch.softmax(logits(teacher, clean_q, docs, t_tok, t_len) / tau, dim=-1)
                loss = (kl(torch.log_softmax(logits(model, noisy_q, docs) / tau, dim=-1), target)
                        + kl(torch.log_softmax(logits(model, clean_q, docs) / tau, dim=-1), target)) * tau * tau
            else:
                enc = tok([q for _, q, _, _ in batch], [corpora[ds].doc_text(d) for ds, _, d, _ in batch],
                          padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
                labels = torch.tensor([y for *_, y in batch], dtype=torch.float32, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and AMP):
                    out = model(**enc).logits.view(-1)
                loss = loss_fn(out.float(), labels)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); sched.step(); step += 1
            window.append(loss.item())
            if step % 20 == 0:
                log["loss"].append({"step": step, "loss": float(np.mean(window)), "lr": sched.get_last_lr()[0]})
                print(f"[{name}] epoch {epoch} step {step}/{total} loss {np.mean(window):.4f} "
                      f"({step / (time.time() - started):.2f} steps/s)", flush=True)
                window = []
        v = validate(model, tok, device, val_runs, corpora, MAX_LEN)
        log["val"].append({"epoch": epoch, "step": step, **v})
        improved = v["mean"] > best_mean
        print(f"[{name}] epoch {epoch} val nDCG@10 clean={v['clean']:.4f} noisy={v['noisy']:.4f}"
              f"{'  <- best, saved' if improved else ''}", flush=True)
        if improved:
            best_mean, best_epoch = v["mean"], epoch
            if not args.last_only:
                ce.save(str(MODELS_DIR / name))
    # Protocol B (fixed budget): keep the final-epoch weights as well as the early-stopped ones,
    # so the test comparison is not decided by near-ties on a 120-query validation set.
    ce.save(str(MODELS_DIR / f"{name}-last"))
    log["best_epoch"], log["best_val_mean"] = best_epoch, best_mean
    log["train_seconds"] = time.time() - started
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data.save_json(LOG_DIR / f"{name}.json", log)
    print(f"[{name}] done in {log['train_seconds'] / 60:.1f} min; best epoch {best_epoch} "
          f"(val mean nDCG@10 {best_mean:.4f})", flush=True)


AMP = False

if __name__ == "__main__":
    main()
