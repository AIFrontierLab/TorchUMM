import os, csv, random
import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Optional, List

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from umm.post_training.unigame.janus_pro.framework import JanusProAdvFramework


LOGGER = logging.getLogger("umm.unigame.janus_pro")

try:
    # Reuse the same official answer processor used by VQA eval scripts.
    from eval.vlm.eval.vqa.textvqa_eval import EvalAIAnswerProcessor  # type: ignore
except Exception:
    EvalAIAnswerProcessor = None

try:
    # Keep validation metric identical to the official evaluator script.
    from eval.vlm.eval.vqa.textvqa_eval import TextVQAAccuracyEvaluator  # type: ignore
except Exception:
    TextVQAAccuracyEvaluator = None


_NORM_RE = re.compile(r"[^a-z0-9\s]")
_OFFICIAL_PROMPT_SUFFIX = "Answer the question using a single word or phrase."


def _normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = _NORM_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def _token_f1(pred: str, gold: str) -> float:
    """Word-level F1 between normalised prediction and ground-truth."""
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = sum(1 for t in pred_toks if t in gold_toks)
    if common == 0:
        return 0.0
    prec = common / len(pred_toks)
    rec = common / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def _bleu1(pred: str, gold: str) -> float:
    """Unigram BLEU with brevity penalty (no external deps)."""
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    from collections import Counter
    ref_counts = Counter(gold_toks)
    clipped = 0
    for tok, cnt in Counter(pred_toks).items():
        clipped += min(cnt, ref_counts.get(tok, 0))
    prec = clipped / len(pred_toks)
    bp = min(1.0, len(pred_toks) / len(gold_toks))   # brevity penalty
    import math
    return prec * math.exp(1 - 1 / bp) if bp < 1 else prec


def _majority_answer_from_list(answers_list: list[dict]) -> str:
    votes: list[str] = []
    for item in answers_list or []:
        ans = _normalize_answer(str(item.get("answer", "")))
        if ans:
            votes.append(ans)
    if not votes:
        return ""
    return max(set(votes), key=votes.count)


def _vqa_soft_score(pred: str, answers_list: list[dict]) -> float:
    pred_n = _normalize_answer(pred)
    if not pred_n:
        return 0.0
    matches = 0
    for item in answers_list or []:
        if _normalize_answer(str(item.get("answer", ""))) == pred_n:
            matches += 1
    return min(1.0, matches / 3.0)


def _official_process_answer(text: str) -> str:
    if EvalAIAnswerProcessor is not None:
        return EvalAIAnswerProcessor()(text)
    return _normalize_answer(text)


def _vqa_soft_score_official(pred: str, answers_list: list[dict]) -> float:
    """Official-style VQA soft score with EvalAI answer processing.

    This mirrors the per-sample scoring used by common VQA evaluators where
    each annotator is compared against the remaining answers.
    """
    pred_n = _official_process_answer(pred or "")
    if not pred_n:
        return 0.0

    answers = []
    for item in answers_list or []:
        ans = _official_process_answer(str(item.get("answer", "")))
        if ans:
            answers.append(ans)
    if not answers:
        return 0.0

    unique_answer_scores: dict[str, float] = {}
    for unique_answer in set(answers):
        accs = []
        for i in range(len(answers)):
            other_answers = answers[:i] + answers[i + 1 :]
            matches = sum(1 for a in other_answers if a == unique_answer)
            accs.append(min(1.0, matches / 3.0))
        unique_answer_scores[unique_answer] = sum(accs) / max(1, len(accs))

    return float(unique_answer_scores.get(pred_n, 0.0))


def _answers_to_str_list(answers: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(answers, list):
        return out
    for item in answers:
        if isinstance(item, dict):
            v = item.get("answer")
            if v is None:
                v = item.get("text")
            if v is not None:
                s = str(v).strip()
                if s:
                    out.append(s)
        else:
            s = str(item).strip()
            if s:
                out.append(s)
    return out


def _ensure_len10(answers: list[str]) -> list[str]:
    if len(answers) == 10:
        return answers
    if len(answers) == 0:
        return [""] * 10
    if len(answers) > 10:
        return answers[:10]
    return answers + [answers[-1]] * (10 - len(answers))


def _clean_pred_answer(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    s = s.replace("Ġ", " ").replace("▁", " ")
    s = " ".join(s.split())

    low = s.lower()
    for p in ("answer:", "answer is", "the answer is", "final answer:"):
        if low.startswith(p):
            s = s[len(p) :].strip(" :")
            low = s.lower()

    marker = _OFFICIAL_PROMPT_SUFFIX.lower()
    pos = low.find(marker)
    if pos >= 0:
        s = s[:pos].strip(" .,:;")

    for sep in ("\n", ". ", "! ", "? "):
        if sep in s:
            s = s.split(sep, 1)[0].strip(" .,:;")
            break

    return " ".join(s.split())

class HardBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self.data: List[Dict] = []

    def __len__(self): 
        return len(self.data)

    def push_many(self, items: List[Dict]):
        if not items:
            return
        self.data.extend(items)
        self.data.sort(key=lambda x: float(x["H"]), reverse=True)
        if len(self.data) > self.capacity:
            self.data = self.data[: self.capacity]

    def sample(self, n: int, temperature: float = 2.0, pop: bool = False) -> List[Dict]:
        if len(self.data) == 0:
            return []
        k = min(int(n), len(self.data))

        idx = torch.arange(len(self.data), dtype=torch.float32)
        inv_rank = (len(self.data) - 1) - idx
        probs = torch.softmax(inv_rank / max(1e-6, float(temperature)), dim=0)
        idxs = torch.multinomial(probs, num_samples=k, replacement=False).tolist()
        items = [self.data[i] for i in idxs]
        if pop:
            for i in sorted(idxs, reverse=True):
                self.data.pop(i)
        return items


class CLIPScorer:
    def __init__(self, device, name="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        import open_clip
        self.model, _, self.pp = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=device
        )
        self.tok = open_clip.get_tokenizer(name)
        self.device = device

    @torch.no_grad()
    def score(self, images: torch.Tensor, texts: List[str], micro_bs: int = 32) -> torch.Tensor:

        assert images.dim() == 4 and images.size(1) == 3
        N = images.size(0)
        target = 224

        import torch.nn.functional as F

        _, _, H, W = images.shape
        scale = target / min(H, W)
        newH, newW = int(round(H * scale)), int(round(W * scale))
        x = F.interpolate(images, size=(newH, newW), mode="bicubic", align_corners=False, antialias=True)
        y0 = max(0, (newH - target) // 2)
        x0 = max(0, (newW - target) // 2)
        x = x[:, :, y0:y0+target, x0:x0+target]

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        tok = self.tok(texts)           
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()} if isinstance(tok, dict) else tok.to(self.device)

        outs = []
        for i in range(0, N, max(1, micro_bs)):
            xi = x[i:i+micro_bs]
            ti = {k: v[i:i+micro_bs] for k, v in tok.items()} if isinstance(tok, dict) else tok[i:i+micro_bs]
            fe_i = F.normalize(self.model.encode_image(xi), dim=-1)
            fe_t = F.normalize(self.model.encode_text(ti), dim=-1)
            outs.append((fe_i * fe_t).sum(dim=-1))
        return torch.cat(outs, 0)


class AdvTrainer:
    def __init__(self, config: Dict):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_distributed else 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}" if self.is_distributed else "cuda")
            if self.is_distributed:
                torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.cfg = config

        _base = JanusProAdvFramework(
            model_path=config.get("model_path", "deepseek-ai/Janus-Pro-7B"),
            num_answers=int(config["num_answers"]),
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            finetune_last_k=config.get("finetune_last_k", 0),
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device=self.device,
            eps_max=config.get("eps_max", 0.02),
            low_cpu_mem_usage=bool(config.get("low_cpu_mem_usage", False)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )
        self.base = _base
    
        if self.is_distributed:
            backend = dist.get_backend() if dist.is_initialized() else "unknown"
            LOGGER.info(
                "before DDP wrap | backend=%s rank=%d local_rank=%d trainable_disc=%d trainable_gen=%d",
                backend,
                self.rank,
                self.local_rank,
                sum(1 for p in _base.mm.language_model.parameters() if p.requires_grad),
                len(_base.gen_params),
            )

            ddp_kwargs = {
                "find_unused_parameters": bool(self.cfg.get("ddp_find_unused_parameters", True)),
                "broadcast_buffers": bool(self.cfg.get("ddp_broadcast_buffers", False)),
                "gradient_as_bucket_view": bool(self.cfg.get("ddp_gradient_as_bucket_view", False)),
            }
            # gloo + CUDA with explicit device_ids can segfault in some torch/python combos.
            if backend == "nccl" and self.device.type == "cuda":
                ddp_kwargs.update({
                    "device_ids": [self.local_rank],
                    "output_device": self.local_rank,
                })

            self.model = DDP(_base, **ddp_kwargs)
            LOGGER.info("after DDP wrap | rank=%d local_rank=%d", self.rank, self.local_rank)
        else:
            self.model = _base

        self.base.img_gen_size    = int(config.get("img_gen_size", 256))
        self.base.img_patch_size  = int(config.get("img_patch_size", 16))
        self.base.cfg_weight      = float(config.get("cfg_weight", 5.0))
        self.base.gen_temperature = float(config.get("gen_temperature", 1.0))
        self.base.max_answer_tokens = int(config.get("max_answer_tokens", 0))
        self.max_answer_chars = int(config.get("max_answer_chars", 0))
        self.skip_oom_batches = bool(config.get("skip_oom_batches", True))
        self.filtered_batch_count = 0
        self.oom_skipped_count = 0

        self._disc_params = [p for p in self.base.mm.language_model.parameters() if p.requires_grad]
        self._gen_params  = list(self.base.gen_params)

        if len(self._disc_params) == 0 and len(self._gen_params) == 0:
            raise RuntimeError(
                "No trainable parameters found before DDP init. "
                "Check LoRA attachment and model load consistency across ranks."
            )

        self.disc_opt = torch.optim.AdamW(
            self._disc_params,
            lr=float(config.get("disc_lr", 3e-6)),
            weight_decay=float(config.get("weight_decay", 0.01)),
            betas=(0.9, 0.999),
        )
        self.gen_opt = torch.optim.AdamW(
            self._gen_params,
            lr=float(config.get("gen_lr", 3e-2)),
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )

        amp_enabled = bool(config.get("use_amp", True))
        lm_dtype = (self.model.module.mm.language_model.dtype
                    if hasattr(self.model, "module") else self.model.mm.language_model.dtype)
        if lm_dtype in (torch.float16, torch.bfloat16):
            amp_enabled = False
        self.use_amp = amp_enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.workdir = config.get("logdir", "logs_januspro")
        self.ckpt_dir = os.path.join(self.workdir, "checkpoints")
        self.metrics_dir = os.path.join(self.workdir, "metrics")
        for d in [self.workdir, self.ckpt_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        self.log_stride    = int(config.get("log_stride"))
        self.save_every    = int(config.get("save_interval"))
        self.csv_log_stride = int(config.get("csv_log_stride", self.log_stride))
        self.step_debug_log_stride = int(config.get("step_debug_log_stride", 1))
        self.pre_step_log_stride = int(config.get("pre_step_log_stride", 20))

        self.train_csv_path = os.path.join(self.metrics_dir, "train_log.csv")
        if (self.rank == 0) and (not os.path.exists(self.train_csv_path)):
            with open(self.train_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "total", "gen_CE_adv", "disc_CE", "eps"])

        self.disc_clip = float(config.get("disc_clip_norm", 0.0) or 0.0)
        self.gen_clip  = float(config.get("gen_clip_norm",  0.0) or 0.0)
        self.eps_max = float(config.get("eps_max", 0.02))

        self.step = 0
        self._loss_buf = defaultdict(lambda: {"sum": 0.0, "n": 0})

        self.gdbg_csv_path = os.path.join(self.metrics_dir, "gdbg_log.csv")
        if self.rank == 0 and (not os.path.exists(self.gdbg_csv_path)):
            with open(self.gdbg_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step","||∇logeps||","Δlogeps","eps","CE_adv@G","ΔJ_D","ΔJ_G"])

        self.d_updates = 1
        self.g_updates = 1

        self.gen_lambda = float(config.get("gen_lambda", 120.0))

        self.use_clean_dpass    = bool(config.get("use_clean_dpass", True))
        self.d_clean_weight     = float(config.get("d_clean_weight", 1.0)) 

        self.clip_tau    = float(config.get("clip_tau", 0.30))
        self.clip_lambda = float(config.get("clip_lambda", 0.2))
        self.cand_K      = int(config.get("cand_K", 3))
        self.hard_topk   = int(config.get("hard_topk", 1))
        self.buffer_size = int(config.get("buffer_size", 4096))
        self.buffer_temp = float(config.get("buffer_temp", 2.0))
        self.hard_bs     = int(config.get("hard_bs", 8))
        self.decoded_micro_bs = int(config.get("decoded_micro_bs", 2))
        self.hard_push_max_per_step = int(config.get("hard_push_max_per_step", 0))
        self.mine_stride = int(config.get("mine_stride", 1))       

        self.hard_buf = HardBuffer(self.buffer_size)
        self.clip = None
        if self.cand_K > 0:
            LOGGER.info("initializing CLIP scorer | model=%s ckpt=%s", config.get("clip_name", "ViT-B-16"), config.get("clip_ckpt", "laion2b_s34b_b88k"))
            self.clip = CLIPScorer(
                self.device,
                name=config.get("clip_name", "ViT-B-16"),
                pretrained=config.get("clip_ckpt", "laion2b_s34b_b88k"),
            )
            LOGGER.info("CLIP scorer ready")
        else:
            LOGGER.info("skip CLIP scorer init because cand_K=%d", self.cand_K)

        self.base.reg_enable = bool(config.get("reg_enable", True))
        self.base.reg_l2     = float(config.get("reg_l2", 0.0))
        self.base.reg_tv     = float(config.get("reg_tv", 0.0))
        self.base.reg_eps    = float(config.get("reg_eps", 0.0))

        self.hard_csv = os.path.join(self.metrics_dir, "hard_samples.csv")
        if (self.rank == 0) and (not os.path.exists(self.hard_csv)):
            with open(self.hard_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "step","global_idx","qid","cand_k","H","CE","CLIP","tau","lambda",
                    "th_mode","th_value","picked","img_path","question","answer"
                ])
        self.hard_thresh_mode   = str(config.get("hard_thresh_mode", "quantile")).lower()
        self.hard_thresh_q      = float(config.get("hard_thresh_q", 0.80))
        self.hard_thresh_value  = float(config.get("hard_thresh_value", 0.0))

        LOGGER.info(
            "AdvTrainer init | model_path=%s | eps_max=%.4f | disc_lr=%s | gen_lr=%s",
            config.get("model_path", ""),
            self.eps_max,
            config.get("disc_lr", 3e-6),
            config.get("gen_lr", 3e-2),
        )
        LOGGER.info(
            "sample guard | max_answer_chars=%d max_answer_tokens=%d skip_oom_batches=%s",
            self.max_answer_chars,
            int(getattr(self.base, "max_answer_tokens", 0)),
            self.skip_oom_batches,
        )


    @staticmethod
    def _quantile(x: torch.Tensor, q: float) -> float:
        q = min(0.999, max(0.001, float(q)))
        return float(torch.quantile(x.detach().float(), q).item())
    
    def _hard_threshold(self, H: torch.Tensor) -> float:
        if self.hard_thresh_mode == "absolute":
            return float(self.hard_thresh_value)
        return self._quantile(H, self.hard_thresh_q)

    def _read_eps(self) -> Optional[float]:
        try:
            mod = getattr(self.base, "perturb", None)
            if mod is None or getattr(mod, "log_eps", None) is None:
                return None
            with torch.no_grad():
                eps01 = torch.sigmoid(mod.log_eps.detach().float())
                eps = float((mod.eps_max * eps01).item())
                if not (0.0 <= eps <= max(1.0, float(mod.eps_max) * 1.5)):
                    print(f"[WARN] eps out of expected range: {eps} (eps_max={mod.eps_max})")
                    eps = max(0.0, min(float(mod.eps_max), eps))
                return eps
        except Exception:
            return None


    @torch.no_grad()
    def _mine_and_buffer(self, batch: Dict):
        imgs_batch = batch["image"]
        assert torch.is_tensor(imgs_batch) and imgs_batch.dim() == 4 and imgs_batch.size(1) == 3, \
            "Expect batch['image'] as Tensor[B,3,H,W]"

        qs  = batch["question"]
        ans = batch["answer_text"]
        B   = len(qs)
        if B == 0 or self.cand_K <= 0:
            return

        Ht, Wt = imgs_batch.shape[-2], imgs_batch.shape[-1] 

        prompts, cand_meta = [], []
        for i in range(B):
            for k in range(self.cand_K):
                prompts.append(qs[i])
                cand_meta.append((i, k))
        
        cand_imgs = self.base._official_generate_batch(
            prompts, inject_adv=True,
            seed=114514 + self.step * 97,
            img_size=self.base.img_gen_size,
            patch_size=self.base.img_patch_size,
            cfg_weight=self.base.cfg_weight,
        )

        if cand_imgs.shape[-2:] != (Ht, Wt):
            cand_imgs = torch.nn.functional.interpolate(
                cand_imgs, size=(Ht, Wt), mode="bilinear", align_corners=False
            )

        cand_qs = [qs[i]  for (i, _) in cand_meta]
        cand_as = [ans[i] for (i, _) in cand_meta]
        N = cand_imgs.size(0)
        
        if self.clip is None:
            s_clip = torch.zeros(N, device=self.device, dtype=torch.float32)
        else:
            s_clip = self.clip.score(cand_imgs, cand_qs, micro_bs=self.decoded_micro_bs)

        ce_vals = []
        mb = max(1, self.decoded_micro_bs)
        tmp_batch = {"image": cand_imgs, "question": cand_qs, "answer_text": cand_as}
        for i in range(0, N, mb):
            sub = {k: (v[i:i+mb] if torch.is_tensor(v) else v[i:i+mb]) for k, v in tmp_batch.items()}
            ce, _ = self.model(sub)
            ce_vals.extend([float(ce.detach())] * min(mb, N - i))
        ce_vals = torch.tensor(ce_vals, device=self.device, dtype=torch.float32)

        clip_hinge = torch.clamp(self.clip_tau - s_clip, min=0.0)
        H = ce_vals + self.clip_lambda * clip_hinge
        th = self._hard_threshold(H)
        mask = (H >= th) | (s_clip <= self.clip_tau)
        chosen_idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()

        by_src = {}
        for idx in chosen_idx:
            i0, k0 = cand_meta[idx]
            by_src.setdefault(i0, []).append(idx)
        picked_final = []
        for i0, lst in by_src.items():
            if not lst: continue
            lst = sorted(lst, key=lambda t: float(H[t]), reverse=True)[:max(1, self.hard_topk)]
            picked_final.extend(lst)

        items = []
        if self.rank == 0:
            import csv as _csv

        for j in picked_final:
            i0, k0 = cand_meta[j]
            H_j  = float(H[j].item())
            CE_j = float(ce_vals[j].item())
            C_j  = float(s_clip[j].item())

            items.append({
                "image": cand_imgs[j].detach().cpu(),  
                "question": cand_qs[j],
                "answer_text": cand_as[j],
                "H": H_j,
            })
            if self.rank == 0:
                with open(self.hard_csv, "a", newline="") as f:
                    _csv.writer(f).writerow([
                        self.step + 1, j, i0, k0, H_j, CE_j, C_j,
                        float(self.clip_tau), float(self.clip_lambda),
                        self.hard_thresh_mode,
                        (self.hard_thresh_value if self.hard_thresh_mode=="absolute" else self.hard_thresh_q),
                        1, "",
                        cand_qs[j], cand_as[j]
                    ])

        if self.rank == 0:
            remained = set(range(N)) - set(picked_final)
            for j in list(remained):
                i0, k0 = cand_meta[j]
                H_j  = float(H[j].item()); CE_j = float(ce_vals[j].item()); C_j = float(s_clip[j].item())
                with open(self.hard_csv, "a", newline="") as f:
                    _csv.writer(f).writerow([
                        self.step + 1, j, i0, k0, H_j, CE_j, C_j,
                        float(self.clip_tau), float(self.clip_lambda),
                        self.hard_thresh_mode,
                        (self.hard_thresh_value if self.hard_thresh_mode=="absolute" else self.hard_thresh_q),
                        0, "", cand_qs[j], cand_as[j]
                    ])

        if self.hard_push_max_per_step > 0 and len(items) > self.hard_push_max_per_step:
            items.sort(key=lambda d: float(d["H"]), reverse=True)
            items = items[: self.hard_push_max_per_step]
        self.hard_buf.push_many(items)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              ckpt_eval_loader: Optional[DataLoader] = None):
        num_epochs = int(self.cfg.get("num_epochs", 1))
        lam = self.gen_lambda
        LOGGER.info(
            "train start | epochs=%d | train_steps_per_epoch=%d | save_every=%d",
            num_epochs,
            len(train_loader),
            self.save_every,
        )

        for epoch in range(num_epochs):
            if self.is_distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            pbar = tqdm(total=len(train_loader), desc=f"Train[{epoch+1}/{num_epochs}]", dynamic_ncols=True) if self.rank == 0 else None

            for batch in train_loader:
                try:
                    if self.rank == 0 and ((self.step + 1) % max(1, self.pre_step_log_stride) == 0):
                        LOGGER.info("step=%d epoch=%d begin", self.step + 1, epoch + 1)

                    batch, dropped = self._filter_oversized_batch(batch)
                    if dropped > 0 and self.rank == 0:
                        self.filtered_batch_count += 1
                        LOGGER.warning(
                            "filtered oversized samples | step=%d dropped=%d max_answer_chars=%d",
                            self.step + 1,
                            dropped,
                            self.max_answer_chars,
                        )
                    if batch is None:
                        if self.rank == 0 and pbar:
                            pbar.set_postfix_str("skip=oversized")
                            pbar.update(1)
                        continue

                    batch = self._to_device(batch)

                    self.disc_opt.zero_grad(set_to_none=True)
                    self.gen_opt.zero_grad(set_to_none=True)
                    for p in self._gen_params:  p.requires_grad = True
                    for p in self._disc_params: p.requires_grad = False

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        self.base._adv_flag = True
                        loss_adv_G, aux = self.model(batch)
                        self.base._adv_flag = False
                        reg_term = aux.get("reg", torch.tensor(0.0, device=self.device, dtype=loss_adv_G.dtype))
                        gen_loss = - lam * loss_adv_G + reg_term

                    old_gen_params = [p.data.detach().clone().float() for p in self._gen_params]
                    self.scaler.scale(gen_loss).backward()
                    self.scaler.unscale_(self.gen_opt)
                    if self.gen_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._gen_params, self.gen_clip)
                
                    g_logeps = getattr(self.base.perturb, "log_eps", None)
                    grad_norm = (0.0 if (g_logeps is None or g_logeps.grad is None)
                                 else float(g_logeps.grad.detach().abs().mean().item()))
                    logeps_old = float(self.base.perturb.log_eps.detach().item()) if hasattr(self.base.perturb, "log_eps") else 0.0

                    gen_grads_before = []
                    for p in self._gen_params:
                        if p.grad is None:
                            gen_grads_before.append(None)
                        else:
                            gen_grads_before.append((-1.0 / max(lam, 1e-12)) * p.grad.detach().float().clone())

                    self.scaler.step(self.gen_opt)
                    self.scaler.update()

                    deltaG = 0.0
                    for p, gJ, old in zip(self._gen_params, gen_grads_before, old_gen_params):
                        if gJ is None: continue
                        d = (p.data.detach().float() - old)
                        deltaG += float((gJ * d).sum().item())

                    logeps_new = float(self.base.perturb.log_eps.detach().item()) if hasattr(self.base.perturb, "log_eps") else logeps_old
                    dlogeps = (logeps_new - logeps_old)
                    eps_now = self._read_eps() or 0.0
                    gen_ce_this = float(loss_adv_G.detach())

                    if self.rank == 0:
                        with open(self.gdbg_csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([
                                self.step + 1, float(grad_norm), float(dlogeps),
                                float(eps_now), float(loss_adv_G.detach()),
                                0.0, float(deltaG)
                            ])

                    if (self.step + 1) % max(1, self.mine_stride) == 0:
                        self._mine_and_buffer(batch)

                    disc_ce_this = 0.0
                    deltaJ_D_this = 0.0

                    if self.use_clean_dpass:
                        self.disc_opt.zero_grad(set_to_none=True)
                        self.gen_opt.zero_grad(set_to_none=True)
                        for p in self._gen_params:
                            p.requires_grad = False
                        for p in self._disc_params:
                            p.requires_grad = True

                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            self.base._adv_flag = False
                            loss_clean, _ = self.model(batch)
                            self.base._adv_flag = False
                            disc_loss_clean = self.d_clean_weight * loss_clean

                        old_disc_params_clean = [p.data.detach().clone().float() for p in self._disc_params]
                        self.scaler.scale(disc_loss_clean).backward()
                        self.scaler.unscale_(self.disc_opt)
                        if self.disc_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                        disc_grads_before_clean = [
                            p.grad.detach().float().clone() if p.grad is not None else None
                            for p in self._disc_params
                        ]
                        self.scaler.step(self.disc_opt)
                        self.scaler.update()

                        delta_cln = 0.0
                        for p, g, old in zip(self._disc_params, disc_grads_before_clean, old_disc_params_clean):
                            if g is None:
                                continue
                            d = (p.data.detach().float() - old)
                            delta_cln += float((g * d).sum().item())
                        deltaJ_D_this += delta_cln

                        disc_ce_this += float(loss_clean.detach())
                        self._loss_buf["disc_ce_clean_ma"]["sum"] += float(loss_clean.detach())
                        self._loss_buf["disc_ce_clean_ma"]["n"] += 1
                    
                    hard_items = self.hard_buf.sample(n=self.hard_bs, temperature=self.buffer_temp, pop=True)

                    if hard_items:
                        if torch.is_tensor(batch["image"]):
                            Ht, Wt = batch["image"].shape[-2], batch["image"].shape[-1]
                        else:
                            im0 = batch["image"][0]
                            if torch.is_tensor(im0):
                                Ht, Wt = im0.shape[-2], im0.shape[-1]
                            else:
                                Wt, Ht = im0.size

                        resized_imgs = []
                        for it in hard_items:
                            t = it["image"].detach().float().clamp(0, 1).cpu()
                            if t.dim() == 3 and t.size(0) == 1:
                                t = t.expand(3, -1, -1)
                            if t.dim() == 3 and t.size(0) == 4:
                                t = t[:3, ...]
                            if t.shape[-2:] != (Ht, Wt):
                                t = F.interpolate(
                                    t.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False
                                ).squeeze(0)
                            resized_imgs.append(t)

                        hb = {
                            "image": torch.stack(resized_imgs, 0).to(self.device),
                            "question": [it["question"] for it in hard_items],
                            "answer_text": [it["answer_text"] for it in hard_items],
                        }

                        self.disc_opt.zero_grad(set_to_none=True)
                        for p in self._gen_params:
                            p.requires_grad = False
                        for p in self._disc_params:
                            p.requires_grad = True

                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            self.base._adv_flag = False
                            loss_hard, _ = self.model(hb)
                            self.base._adv_flag = False
                            disc_loss_hard = loss_hard

                        old_disc_params_hard = [p.data.detach().clone().float() for p in self._disc_params]
                        self.scaler.scale(disc_loss_hard).backward()
                        self.scaler.unscale_(self.disc_opt)
                        if self.disc_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                        disc_grads_before_hard = [
                            p.grad.detach().float().clone() if p.grad is not None else None
                            for p in self._disc_params
                        ]
                        self.scaler.step(self.disc_opt)
                        self.scaler.update()

                        delta_h = 0.0
                        for p, g, old in zip(self._disc_params, disc_grads_before_hard, old_disc_params_hard):
                            if g is None:
                                continue
                            d = (p.data.detach().float() - old)
                            delta_h += float((g * d).sum().item())
                        deltaJ_D_this += delta_h

                        disc_ce_this += float(loss_hard.detach())
                        self._loss_buf["disc_ce_hard_ma"]["sum"] += float(loss_hard.detach())
                        self._loss_buf["disc_ce_hard_ma"]["n"] += 1

                    for p in self._disc_params: p.requires_grad = True

                    eps_val = self._read_eps()

                    d_pass_cnt = int(self.use_clean_dpass) + (1 if hard_items else 0)
                    d_pass_cnt = max(1, d_pass_cnt)
                
                    self._loss_buf["gen_ce_adv_ma"]["sum"] += float(gen_ce_this)
                    self._loss_buf["gen_ce_adv_ma"]["n"]   += 1
                    self._loss_buf["disc_ce_ma"]["sum"]    += (disc_ce_this / d_pass_cnt)
                    self._loss_buf["disc_ce_ma"]["n"]      += 1

                    if (self.step + 1) % self.log_stride == 0:
                        for _, b in self._loss_buf.items():
                            if b["n"] > 0:
                                b["sum"], b["n"] = 0.0, 0

                    self.step += 1
                    if self.rank == 0 and pbar:
                        pbar.set_postfix_str(
                            f"D(CE)={(disc_ce_this/d_pass_cnt):.4f} | "
                            f"G(CE)={gen_ce_this:.4f} | "
                            f"eps={(eps_val if eps_val is not None else 0.0):.4f}/{self.eps_max:g} | "
                            f"buf={len(self.hard_buf)}"
                        )
                        pbar.update(1)

                    do_ckpt = (self.step % self.save_every == 0)
                    if do_ckpt and self.rank == 0:
                        self.save_ckpt(step_tag=f"step_{self.step}")
                        # Evaluate on external ckpt eval loader (e.g. VQAv2)
                        if ckpt_eval_loader is not None:
                            ckpt_eval_max = int(self.cfg.get("ckpt_eval_max_batches", 50))
                            LOGGER.info("ckpt eval start | step=%d max_batches=%d", self.step, ckpt_eval_max)
                            ckpt_m = self.evaluate(
                                ckpt_eval_loader,
                                split_name=f"ckpt_eval_step_{self.step}",
                                max_batches=ckpt_eval_max,
                            )
                            LOGGER.info("ckpt eval done | step=%d metrics=%s", self.step, ckpt_m)
                        # Evaluate on training val split
                        if val_loader is not None and bool(self.cfg.get("ckpt_eval_on_val", False)):
                            # Build a non-distributed loader so rank-0 evaluates ALL val samples,
                            # not just 1/N of them from DistributedSampler.
                            _eval_vl = val_loader
                            if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'num_replicas'):
                                from torch.utils.data import DataLoader as _DL
                                _vl_kwargs = dict(
                                    dataset=val_loader.dataset,
                                    batch_size=val_loader.batch_size,
                                    sampler=None,
                                    shuffle=False,
                                    num_workers=val_loader.num_workers,
                                    collate_fn=val_loader.collate_fn,
                                )
                                if val_loader.num_workers > 0:
                                    _vl_kwargs.update(pin_memory=True, persistent_workers=True, prefetch_factor=2)
                                _eval_vl = _DL(**_vl_kwargs)
                            val_eval_max = int(self.cfg.get("ckpt_eval_max_batches", 50))
                            LOGGER.info("ckpt val eval start | step=%d max_batches=%d full_dataset=%d", self.step, val_eval_max, len(_eval_vl.dataset))
                            val_m = self.evaluate(
                                _eval_vl,
                                split_name=f"ckpt_val_step_{self.step}",
                                max_batches=val_eval_max,
                            )
                            LOGGER.info("ckpt val eval done | step=%d metrics=%s", self.step, val_m)
                    if do_ckpt and self.is_distributed and (ckpt_eval_loader is not None or bool(self.cfg.get("ckpt_eval_on_val", False))):
                        dist.barrier()

                    if self.rank == 0 and ((self.step + 1) % self.csv_log_stride == 0):
                        with open(self.train_csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([
                                self.step + 1,
                                float(disc_ce_this + gen_ce_this),
                                float(gen_ce_this),
                                float(disc_ce_this / d_pass_cnt),
                                ("" if eps_val is None else float(eps_val)),
                            ])

                    if self.rank == 0 and (self.step % max(1, self.step_debug_log_stride) == 0):
                        LOGGER.info(
                            "step=%d epoch=%d D_CE=%.6f G_CE=%.6f eps=%.6f buf=%d deltaJ_D=%.6f deltaJ_G=%.6f",
                            self.step,
                            epoch + 1,
                            float(disc_ce_this / d_pass_cnt),
                            float(gen_ce_this),
                            float(eps_val if eps_val is not None else 0.0),
                            len(self.hard_buf),
                            float(deltaJ_D_this),
                            float(deltaG),
                        )
                except Exception as exc:
                    if self.skip_oom_batches and isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                        self.oom_skipped_count += 1
                        LOGGER.warning(
                            "oom batch skipped | epoch=%d step=%d skipped_total=%d",
                            epoch + 1,
                            self.step + 1,
                            self.oom_skipped_count,
                        )
                        self.disc_opt.zero_grad(set_to_none=True)
                        self.gen_opt.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if self.rank == 0 and pbar:
                            pbar.set_postfix_str(f"skip=oom({self.oom_skipped_count})")
                            pbar.update(1)
                        continue
                    LOGGER.exception(
                        "train step failed | epoch=%d step=%d",
                        epoch + 1,
                        self.step + 1,
                    )
                    raise

            if pbar: pbar.close()

    def save_ckpt(self, step_tag: str):
        if self.rank != 0: return
        path = os.path.join(self.ckpt_dir, f"{step_tag}.pt")
        to_save = {
            "step": self.step,
            "model": self.base.state_dict(),
            "perturb_only": self.base.perturb.state_dict(),
            "disc_opt": self.disc_opt.state_dict(),
            "gen_opt": self.gen_opt.state_dict(),
            "cfg": self.cfg,
        }
        torch.save(to_save, path)
        print(f"[CKPT] saved => {path}")

    def load_ckpt(self, ckpt_path: str, load_optim: bool = True, strict: bool = False) -> dict[str, Any]:
        """Load training checkpoint for resume.

        Loads model weights into `self.base`, restores optimizer states when
        available, and sets `self.step` from checkpoint metadata.
        """
        p = os.path.expanduser(str(ckpt_path))
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if not os.path.exists(p):
            raise FileNotFoundError(f"resume checkpoint not found: {p}")

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model", ckpt)
        missing, unexpected = self.base.load_state_dict(model_state, strict=bool(strict))

        disc_opt_loaded = False
        gen_opt_loaded = False
        if bool(load_optim):
            if isinstance(ckpt, dict) and "disc_opt" in ckpt:
                self.disc_opt.load_state_dict(ckpt["disc_opt"])
                disc_opt_loaded = True
            if isinstance(ckpt, dict) and "gen_opt" in ckpt:
                self.gen_opt.load_state_dict(ckpt["gen_opt"])
                gen_opt_loaded = True

        if isinstance(ckpt, dict) and "step" in ckpt:
            self.step = int(ckpt["step"])

        info = {
            "path": p,
            "step": int(self.step),
            "missing": int(len(missing)),
            "unexpected": int(len(unexpected)),
            "disc_opt_loaded": bool(disc_opt_loaded),
            "gen_opt_loaded": bool(gen_opt_loaded),
        }
        if self.rank == 0:
            LOGGER.info(
                "resume loaded | path=%s step=%d missing=%d unexpected=%d disc_opt=%s gen_opt=%s",
                info["path"],
                info["step"],
                info["missing"],
                info["unexpected"],
                info["disc_opt_loaded"],
                info["gen_opt_loaded"],
            )
        return info

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        split_name: str,
        max_batches: int = 0,
    ) -> dict[str, float]:
        """Evaluate generative VQA on a held-out split.

        Metrics:
        - exact_match: normalized prediction equals majority answer.
        - vqa_soft: VQA-style score min(1, matches/3) over annotator answers.
        - vqa_soft_official: official TextVQAAccuracyEvaluator score.
        """
        if self.rank != 0:
            return {
                "samples": 0.0,
                "exact_match": 0.0,
                "vqa_soft": 0.0,
                "vqa_soft_official": 0.0,
            }

        model_was_training = self.base.mm.training
        self.base.mm.eval()
        self.base._adv_flag = False

        total = 0
        labeled_total = 0
        exact = 0.0
        soft = 0.0
        soft_official = 0.0
        token_f1_sum = 0.0
        bleu1_sum = 0.0
        pred_char_total = 0
        pred_sig_parts: list[str] = []
        pred_list_official: list[dict[str, Any]] = []
        sample_preds: list[tuple[str, str]] = []   # first N (pred, gt) pairs for diagnostics

        eval_prompt_suffix = str(self.cfg.get("eval_prompt_suffix", "")).strip()
        if not eval_prompt_suffix and bool(self.cfg.get("eval_use_official_prompt", True)):
            eval_prompt_suffix = _OFFICIAL_PROMPT_SUFFIX

        LOGGER.info(
            "eval config | split=%s eval_prompt_suffix=%r eval_max_new_tokens=%d eval_temperature=%.2f",
            split_name,
            eval_prompt_suffix if eval_prompt_suffix else "(none)",
            int(self.cfg.get("eval_max_new_tokens", 32)),
            float(self.cfg.get("eval_temperature", 0.0)),
        )

        out_path = os.path.join(self.metrics_dir, f"eval_{split_name}.csv")
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "split", "idx", "pred", "gt_majority", "soft_score"])

            total_batches = len(eval_loader) if hasattr(eval_loader, "__len__") else 0
            eval_pbar = tqdm(total=total_batches, desc=f"Eval[{split_name}]", dynamic_ncols=True)

            for bi, batch in enumerate(eval_loader):
                if max_batches > 0 and bi >= max_batches:
                    break

                batch = self._to_device(batch)
                eval_questions = batch["question"]
                if eval_prompt_suffix:
                    eval_questions = [
                        q
                        if eval_prompt_suffix.lower() in str(q).lower()
                        else f"{q} {eval_prompt_suffix}".strip()
                        for q in eval_questions
                    ]
                preds = self.base.infer_answers_batch(
                    images=batch["image"],
                    questions=eval_questions,
                    max_new_tokens=int(self.cfg.get("eval_max_new_tokens", 32)),
                    temperature=float(self.cfg.get("eval_temperature", 0.0)),
                )

                # Some model/tokenizer combinations may return fewer generations
                # than requested when batched with `inputs_embeds`. Fallback to
                # per-sample generation to keep evaluation accounting correct.
                if len(preds) != len(eval_questions):
                    LOGGER.warning(
                        "eval pred count mismatch | split=%s batch=%d preds=%d expected=%d; fallback to per-sample infer",
                        split_name,
                        bi,
                        len(preds),
                        len(eval_questions),
                    )
                    imgs = batch["image"]
                    preds = []
                    for i in range(len(eval_questions)):
                        if torch.is_tensor(imgs):
                            img_i = imgs[i : i + 1]
                        else:
                            img_i = [imgs[i]]
                        p_i = self.base.infer_answers_batch(
                            images=img_i,
                            questions=[eval_questions[i]],
                            max_new_tokens=int(self.cfg.get("eval_max_new_tokens", 32)),
                            temperature=float(self.cfg.get("eval_temperature", 0.0)),
                        )
                        preds.append(p_i[0] if p_i else "")

                raw_answers = batch.get("answers", [])
                for i, pred in enumerate(preds):
                    answers_i = raw_answers[i] if i < len(raw_answers) else []
                    gt_majority = _majority_answer_from_list(answers_i)
                    pred_clean = _clean_pred_answer(pred)
                    pred_norm = _normalize_answer(pred_clean)
                    gt_norm = _normalize_answer(gt_majority)
                    s_soft = _vqa_soft_score(pred_clean, answers_i)
                    s_soft_official = _vqa_soft_score_official(pred_clean, answers_i)
                    s_token_f1 = _token_f1(pred_clean, gt_majority)
                    s_bleu1 = _bleu1(pred_clean, gt_majority)
                    has_labels = bool(answers_i)

                    total += 1
                    pred_char_total += len(pred_clean)
                    if len(pred_sig_parts) < 128:
                        pred_sig_parts.append(pred_clean)
                    if len(sample_preds) < 5:
                        sample_preds.append((pred_clean, gt_majority))
                    if has_labels:
                        labeled_total += 1
                        if pred_norm and pred_norm == gt_norm:
                            exact += 1.0
                        soft += s_soft
                        soft_official += s_soft_official
                        token_f1_sum += s_token_f1
                        bleu1_sum += s_bleu1
                        pred_list_official.append({
                            "pred_answer": pred_clean,
                            "gt_answers": _ensure_len10(_answers_to_str_list(answers_i)),
                        })

                    writer.writerow([
                        self.step,
                        split_name,
                        total,
                        pred_clean,
                        gt_majority,
                        f"{s_soft:.6f}",
                    ])

                eval_pbar.update(1)

            eval_pbar.close()

        official_eval_score = 0.0
        if labeled_total > 0 and pred_list_official:
            if TextVQAAccuracyEvaluator is not None:
                official_eval_score = float(
                    TextVQAAccuracyEvaluator().eval_pred_list(pred_list_official, disable_tqdm=True)
                )
            else:
                LOGGER.warning(
                    "TextVQAAccuracyEvaluator is unavailable; fallback to approximate official soft score."
                )
                official_eval_score = (soft_official / labeled_total)

        metrics = {
            "samples": float(total),
            "labeled_samples": float(labeled_total),
            "exact_match": (exact / labeled_total) if labeled_total > 0 else 0.0,
            "vqa_soft": (soft / labeled_total) if labeled_total > 0 else 0.0,
            "vqa_soft_official": official_eval_score,
            "token_f1": (token_f1_sum / labeled_total) if labeled_total > 0 else 0.0,
            "bleu1": (bleu1_sum / labeled_total) if labeled_total > 0 else 0.0,
            "pred_avg_chars": (pred_char_total / total) if total > 0 else 0.0,
        }
        pred_sig = hashlib.md5("\n".join(pred_sig_parts).encode("utf-8", errors="ignore")).hexdigest()[:12]
        LOGGER.info(
            "eval done | split=%s samples=%d labeled=%d exact=%.4f soft=%.4f "
            "soft_official=%.4f token_f1=%.4f bleu1=%.4f pred_avg_chars=%.2f pred_sig=%s",
            split_name,
            total,
            labeled_total,
            metrics["exact_match"],
            metrics["vqa_soft"],
            metrics["vqa_soft_official"],
            metrics["token_f1"],
            metrics["bleu1"],
            metrics["pred_avg_chars"],
            pred_sig,
        )
        for si, (sp, sg) in enumerate(sample_preds):
            LOGGER.info("eval sample[%d] pred=%r  gt=%r", si, sp[:200], sg[:200])

        if model_was_training:
            self.base.mm.train()
        return metrics

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        return out

    @staticmethod
    def _subset_batch(batch: Dict, keep_idx: list[int]) -> Dict:
        out: Dict = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v[keep_idx]
            elif isinstance(v, list):
                out[k] = [v[i] for i in keep_idx]
            else:
                out[k] = v
        return out

    def _filter_oversized_batch(self, batch: Dict) -> tuple[Dict | None, int]:
        max_chars = int(self.max_answer_chars)
        if max_chars <= 0:
            return batch, 0

        answers = batch.get("answer_text")
        if not isinstance(answers, list):
            return batch, 0

        keep_idx = []
        for i, a in enumerate(answers):
            if len(str(a)) <= max_chars:
                keep_idx.append(i)

        dropped = len(answers) - len(keep_idx)
        if dropped <= 0:
            return batch, 0
        if len(keep_idx) == 0:
            return None, dropped

        return self._subset_batch(batch, keep_idx), dropped
