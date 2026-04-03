from __future__ import annotations

import csv
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from umm.post_training.unigame.bagel.framework import BagelUniGameFramework


LOGGER = logging.getLogger("umm.unigame.bagel")
_NORM_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = _NORM_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


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


class HardBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self.data: List[Dict] = []

    def __len__(self) -> int:
        return len(self.data)

    def push_many(self, items: List[Dict]) -> None:
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
    def __init__(self, device, name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k"):
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
        self.tok = open_clip.get_tokenizer(name)
        self.device = device

    @torch.no_grad()
    def score(self, images: torch.Tensor, texts: List[str], micro_bs: int = 32) -> torch.Tensor:
        assert images.dim() == 4 and images.size(1) == 3
        N = images.size(0)
        target = 224

        _, _, h, w = images.shape
        scale = target / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        x = F.interpolate(images, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        y0 = max(0, (new_h - target) // 2)
        x0 = max(0, (new_w - target) // 2)
        x = x[:, :, y0 : y0 + target, x0 : x0 + target]

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        tok = self.tok(texts)
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()} if isinstance(tok, dict) else tok.to(self.device)

        outs = []
        for i in range(0, N, max(1, micro_bs)):
            xi = x[i : i + micro_bs]
            ti = {k: v[i : i + micro_bs] for k, v in tok.items()} if isinstance(tok, dict) else tok[i : i + micro_bs]
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
        _base = BagelUniGameFramework(
            model_path=str(config.get("model_path", "")),
            bagel_root=config.get("bagel_root"),
            max_mem_per_gpu=str(config.get("max_mem_per_gpu", "80GiB")),
            offload_folder=str(config.get("offload_folder", "./tmp/offload")),
            distributed_single_gpu=bool(config.get("distributed_single_gpu", False)),
            eps_max=float(config.get("eps_max", 0.02)),
            finetune_last_k=int(config.get("finetune_last_k", 0)),
            lora_r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            use_lora=bool(config.get("use_lora", True)),
            train_lm_head=bool(config.get("train_lm_head", False)),
            train_norms_only=bool(config.get("train_norms_only", False)),
            train_last_k_norms=int(config.get("train_last_k_norms", 0)),
        )
        self.base = _base

        trainable_param_count = sum(1 for p in _base.parameters() if p.requires_grad)
        LOGGER.info("rank=%d local_rank=%d trainable_param_count=%d", self.rank, self.local_rank, trainable_param_count)

        if self.is_distributed:
            self.model = DDP(
                _base,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            self.model = _base

        self.base.cfg_weight = float(config.get("cfg_weight", 4.0))
        self.base.gen_temperature = float(config.get("gen_temperature", 1.0))

        self._disc_params = [p for p in self.base.mm.language_model.parameters() if p.requires_grad]
        self._gen_params = list(self.base.gen_params)
        if len(self._disc_params) == 0:
            raise ValueError(
                "No trainable discriminator parameters found for Bagel UniGame. "
                "Enable `use_lora: true` or set `train_lm_head: true` or `finetune_last_k > 0`."
            )

        self.disc_opt = torch.optim.AdamW(
            self._disc_params,
            lr=float(config.get("disc_lr", 3e-6)),
            weight_decay=float(config.get("weight_decay", 0.01)),
            betas=(0.9, 0.999),
            foreach=False,
        )
        self.gen_opt = torch.optim.AdamW(
            self._gen_params,
            lr=float(config.get("gen_lr", 3e-2)),
            weight_decay=0.0,
            betas=(0.9, 0.999),
            foreach=False,
        )

        amp_enabled = bool(config.get("use_amp", True)) and torch.cuda.is_available()
        self.use_amp = amp_enabled
        self.amp_dtype = torch.bfloat16 if self.use_amp else None
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        self.workdir = str(config.get("logdir", "logs_bagel_unigame"))
        self.ckpt_dir = os.path.join(self.workdir, "checkpoints")
        self.metrics_dir = os.path.join(self.workdir, "metrics")
        for d in [self.workdir, self.ckpt_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        self.log_stride = int(config.get("log_stride", 10))
        self.save_every = int(config.get("save_interval", 500))
        self.csv_log_stride = int(config.get("csv_log_stride", self.log_stride))
        self.step_debug_log_stride = int(config.get("step_debug_log_stride", 1))

        self.train_csv_path = os.path.join(self.metrics_dir, "train_log.csv")
        if (self.rank == 0) and (not os.path.exists(self.train_csv_path)):
            with open(self.train_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "total", "gen_CE_adv", "disc_CE", "eps"])

        self.disc_clip = float(config.get("disc_clip_norm", 0.0) or 0.0)
        self.gen_clip = float(config.get("gen_clip_norm", 0.0) or 0.0)
        self.eps_max = float(config.get("eps_max", 0.02))

        self.step = 0
        self._loss_buf = defaultdict(lambda: {"sum": 0.0, "n": 0})

        self.gdbg_csv_path = os.path.join(self.metrics_dir, "gdbg_log.csv")
        if self.rank == 0 and (not os.path.exists(self.gdbg_csv_path)):
            with open(self.gdbg_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "||grad_logeps||", "delta_logeps", "eps", "CE_adv@G", "deltaJ_D", "deltaJ_G"])

        self.gen_lambda = float(config.get("gen_lambda", 120.0))
        self.use_clean_dpass = bool(config.get("use_clean_dpass", True))
        self.d_clean_weight = float(config.get("d_clean_weight", 1.0))

        self.clip_tau = float(config.get("clip_tau", 0.30))
        self.clip_lambda = float(config.get("clip_lambda", 0.2))
        self.cand_K = int(config.get("cand_K", 3))
        self.hard_topk = int(config.get("hard_topk", 1))
        self.buffer_size = int(config.get("buffer_size", 4096))
        self.buffer_temp = float(config.get("buffer_temp", 2.0))
        self.hard_bs = int(config.get("hard_bs", 8))
        self.decoded_micro_bs = int(config.get("decoded_micro_bs", 2))
        self.hard_push_max_per_step = int(config.get("hard_push_max_per_step", 0))
        self.mine_stride = int(config.get("mine_stride", 1))

        self.hard_buf = HardBuffer(self.buffer_size)
        self.clip = CLIPScorer(
            self.device,
            name=str(config.get("clip_name", "ViT-B-16")),
            pretrained=str(config.get("clip_ckpt", "laion2b_s34b_b88k")),
        )

        self.base.reg_enable = bool(config.get("reg_enable", True))
        self.base.reg_l2 = float(config.get("reg_l2", 0.0))
        self.base.reg_tv = float(config.get("reg_tv", 0.0))
        self.base.reg_eps = float(config.get("reg_eps", 0.0))

        self.hard_csv = os.path.join(self.metrics_dir, "hard_samples.csv")
        if (self.rank == 0) and (not os.path.exists(self.hard_csv)):
            with open(self.hard_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "step",
                        "global_idx",
                        "qid",
                        "cand_k",
                        "H",
                        "CE",
                        "CLIP",
                        "tau",
                        "lambda",
                        "th_mode",
                        "th_value",
                        "picked",
                        "img_path",
                        "question",
                        "answer",
                    ]
                )
        self.hard_thresh_mode = str(config.get("hard_thresh_mode", "quantile")).lower()
        self.hard_thresh_q = float(config.get("hard_thresh_q", 0.80))
        self.hard_thresh_value = float(config.get("hard_thresh_value", 0.0))

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
                return max(0.0, min(float(mod.eps_max), eps))
        except Exception:
            return None

    @torch.no_grad()
    def _mine_and_buffer(self, batch: Dict) -> None:
        imgs_batch = batch["image"]
        assert torch.is_tensor(imgs_batch) and imgs_batch.dim() == 4 and imgs_batch.size(1) == 3

        qs = batch["question"]
        ans = batch["answer_text"]
        B = len(qs)
        if B == 0 or self.cand_K <= 0:
            return

        Ht, Wt = imgs_batch.shape[-2], imgs_batch.shape[-1]

        prompts, cand_meta = [], []
        for i in range(B):
            for k in range(self.cand_K):
                prompts.append(qs[i])
                cand_meta.append((i, k))

        cand_imgs = self.base._official_generate_batch(prompts, inject_adv=True, seed=114514 + self.step * 97)
        if cand_imgs.shape[-2:] != (Ht, Wt):
            cand_imgs = F.interpolate(cand_imgs, size=(Ht, Wt), mode="bilinear", align_corners=False)

        cand_qs = [qs[i] for (i, _) in cand_meta]
        cand_as = [ans[i] for (i, _) in cand_meta]
        N = cand_imgs.size(0)

        s_clip = self.clip.score(cand_imgs.to(self.device), cand_qs, micro_bs=self.decoded_micro_bs)

        ce_vals = []
        mb = max(1, self.decoded_micro_bs)
        tmp_batch = {"image": cand_imgs.to(self.device), "question": cand_qs, "answer_text": cand_as}
        for i in range(0, N, mb):
            sub = {k: (v[i : i + mb] if torch.is_tensor(v) else v[i : i + mb]) for k, v in tmp_batch.items()}
            ce, _ = self.model(sub)
            ce_vals.extend([float(ce.detach())] * min(mb, N - i))
        ce_vals = torch.tensor(ce_vals, device=self.device, dtype=torch.float32)

        clip_hinge = torch.clamp(self.clip_tau - s_clip, min=0.0)
        H = ce_vals + self.clip_lambda * clip_hinge
        th = self._hard_threshold(H)
        mask = (H >= th) | (s_clip <= self.clip_tau)
        chosen_idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()

        by_src: dict[int, list[int]] = {}
        for idx in chosen_idx:
            i0, _ = cand_meta[idx]
            by_src.setdefault(i0, []).append(idx)

        picked_final = []
        for i0, lst in by_src.items():
            if not lst:
                continue
            lst = sorted(lst, key=lambda t: float(H[t]), reverse=True)[: max(1, self.hard_topk)]
            picked_final.extend(lst)

        items = []
        for j in picked_final:
            items.append(
                {
                    "image": cand_imgs[j].detach().cpu(),
                    "question": cand_qs[j],
                    "answer_text": cand_as[j],
                    "H": float(H[j].item()),
                }
            )

        if self.hard_push_max_per_step > 0 and len(items) > self.hard_push_max_per_step:
            items.sort(key=lambda d: float(d["H"]), reverse=True)
            items = items[: self.hard_push_max_per_step]
        self.hard_buf.push_many(items)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        num_epochs = int(self.cfg.get("num_epochs", 1))
        lam = self.gen_lambda

        for epoch in range(num_epochs):
            if self.is_distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            pbar = tqdm(total=len(train_loader), desc=f"Train[{epoch+1}/{num_epochs}]", dynamic_ncols=True) if self.rank == 0 else None

            for batch in train_loader:
                batch = self._to_device(batch)

                self.disc_opt.zero_grad(set_to_none=True)
                self.gen_opt.zero_grad(set_to_none=True)
                for p in self._gen_params:
                    p.requires_grad = True
                for p in self._disc_params:
                    p.requires_grad = False

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    self.base._adv_flag = True
                    loss_adv_G, aux = self.model(batch)
                    self.base._adv_flag = False
                    reg_term = aux.get("reg", torch.tensor(0.0, device=self.device, dtype=loss_adv_G.dtype))
                    gen_loss = -lam * loss_adv_G + reg_term

                old_gen = [p.data.detach().clone().float() for p in self._gen_params]
                if self.use_grad_scaler:
                    self.scaler.scale(gen_loss).backward()
                    self.scaler.unscale_(self.gen_opt)
                else:
                    gen_loss.backward()
                if self.gen_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._gen_params, self.gen_clip)

                g_logeps = getattr(self.base.perturb, "log_eps", None)
                grad_norm = 0.0 if (g_logeps is None or g_logeps.grad is None) else float(g_logeps.grad.detach().abs().mean().item())
                logeps_old = float(self.base.perturb.log_eps.detach().item())

                gen_grads_before = []
                for p in self._gen_params:
                    if p.grad is None:
                        gen_grads_before.append(None)
                    else:
                        gen_grads_before.append((-1.0 / max(lam, 1e-12)) * p.grad.detach().float().clone())

                if self.use_grad_scaler:
                    self.scaler.step(self.gen_opt)
                    self.scaler.update()
                else:
                    self.gen_opt.step()

                deltaG = 0.0
                for p, gJ, old in zip(self._gen_params, gen_grads_before, old_gen):
                    if gJ is None:
                        continue
                    d = p.data.detach().float() - old
                    deltaG += float((gJ * d).sum().item())

                logeps_new = float(self.base.perturb.log_eps.detach().item())
                dlogeps = logeps_new - logeps_old
                eps_now = self._read_eps() or 0.0
                gen_ce_this = float(loss_adv_G.detach())

                if self.rank == 0:
                    with open(self.gdbg_csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([self.step + 1, grad_norm, dlogeps, eps_now, gen_ce_this, 0.0, deltaG])

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

                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        self.base._adv_flag = False
                        loss_clean, _ = self.model(batch)
                        disc_loss_clean = self.d_clean_weight * loss_clean

                    old_disc = [p.data.detach().clone().float() for p in self._disc_params]
                    if self.use_grad_scaler:
                        self.scaler.scale(disc_loss_clean).backward()
                        self.scaler.unscale_(self.disc_opt)
                    else:
                        disc_loss_clean.backward()
                    if self.disc_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                    disc_grads_before = [p.grad.detach().float().clone() if p.grad is not None else None for p in self._disc_params]
                    if self.use_grad_scaler:
                        self.scaler.step(self.disc_opt)
                        self.scaler.update()
                    else:
                        self.disc_opt.step()

                    delta_cln = 0.0
                    for p, g, old in zip(self._disc_params, disc_grads_before, old_disc):
                        if g is None:
                            continue
                        d = p.data.detach().float() - old
                        delta_cln += float((g * d).sum().item())
                    deltaJ_D_this += delta_cln

                    disc_ce_this += float(loss_clean.detach())

                hard_items = self.hard_buf.sample(n=self.hard_bs, temperature=self.buffer_temp, pop=True)
                if hard_items:
                    if torch.is_tensor(batch["image"]):
                        Ht, Wt = batch["image"].shape[-2], batch["image"].shape[-1]
                    else:
                        im0 = batch["image"][0]
                        Ht, Wt = im0.shape[-2], im0.shape[-1]

                    resized_imgs = []
                    for it in hard_items:
                        t = it["image"].detach().float().clamp(0, 1).cpu()
                        if t.shape[-2:] != (Ht, Wt):
                            t = F.interpolate(t.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False).squeeze(0)
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

                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        self.base._adv_flag = False
                        loss_hard, _ = self.model(hb)
                        disc_loss_hard = loss_hard

                    old_disc_h = [p.data.detach().clone().float() for p in self._disc_params]
                    if self.use_grad_scaler:
                        self.scaler.scale(disc_loss_hard).backward()
                        self.scaler.unscale_(self.disc_opt)
                    else:
                        disc_loss_hard.backward()
                    if self.disc_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                    disc_grads_h = [p.grad.detach().float().clone() if p.grad is not None else None for p in self._disc_params]
                    if self.use_grad_scaler:
                        self.scaler.step(self.disc_opt)
                        self.scaler.update()
                    else:
                        self.disc_opt.step()

                    delta_h = 0.0
                    for p, g, old in zip(self._disc_params, disc_grads_h, old_disc_h):
                        if g is None:
                            continue
                        d = p.data.detach().float() - old
                        delta_h += float((g * d).sum().item())
                    deltaJ_D_this += delta_h
                    disc_ce_this += float(loss_hard.detach())

                for p in self._disc_params:
                    p.requires_grad = True

                eps_val = self._read_eps()
                d_pass_cnt = int(self.use_clean_dpass) + (1 if hard_items else 0)
                d_pass_cnt = max(1, d_pass_cnt)

                self.step += 1
                if self.rank == 0 and pbar:
                    pbar.set_postfix_str(
                        f"D(CE)={(disc_ce_this / d_pass_cnt):.4f} | "
                        f"G(CE)={gen_ce_this:.4f} | "
                        f"eps={(eps_val if eps_val is not None else 0.0):.4f}/{self.eps_max:g} | "
                        f"buf={len(self.hard_buf)}"
                    )
                    pbar.update(1)

                if (self.step % self.save_every == 0) and self.rank == 0:
                    self.save_ckpt(step_tag=f"step_{self.step}")

                if self.rank == 0 and ((self.step + 1) % self.csv_log_stride == 0):
                    with open(self.train_csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [
                                self.step + 1,
                                float(disc_ce_this + gen_ce_this),
                                float(gen_ce_this),
                                float(disc_ce_this / d_pass_cnt),
                                ("" if eps_val is None else float(eps_val)),
                            ]
                        )

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

            if pbar:
                pbar.close()

    def save_ckpt(self, step_tag: str) -> None:
        if self.rank != 0:
            return
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
    def evaluate(self, eval_loader: DataLoader, split_name: str, max_batches: int = 0) -> dict[str, float]:
        if self.rank != 0:
            return {"samples": 0.0, "exact_match": 0.0, "vqa_soft": 0.0}

        total = 0
        labeled_total = 0
        exact = 0.0
        soft = 0.0

        out_path = os.path.join(self.metrics_dir, f"eval_{split_name}.csv")
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "split", "idx", "pred", "gt_majority", "soft_score"])

            for bi, batch in enumerate(eval_loader):
                if max_batches > 0 and bi >= max_batches:
                    break

                batch = self._to_device(batch)
                preds = self.base.infer_answers_batch(
                    images=batch["image"],
                    questions=batch["question"],
                    max_new_tokens=int(self.cfg.get("eval_max_new_tokens", 32)),
                    temperature=float(self.cfg.get("eval_temperature", 0.0)),
                )

                raw_answers = batch.get("answers", [])
                for i, pred in enumerate(preds):
                    answers_i = raw_answers[i] if i < len(raw_answers) else []
                    gt_majority = _majority_answer_from_list(answers_i)
                    pred_norm = _normalize_answer(pred)
                    gt_norm = _normalize_answer(gt_majority)
                    s_soft = _vqa_soft_score(pred, answers_i)
                    has_labels = bool(answers_i)

                    total += 1
                    if has_labels:
                        labeled_total += 1
                        if pred_norm and pred_norm == gt_norm:
                            exact += 1.0
                        soft += s_soft

                    writer.writerow([self.step, split_name, total, pred, gt_majority, f"{s_soft:.6f}"])

        metrics = {
            "samples": float(total),
            "labeled_samples": float(labeled_total),
            "exact_match": (exact / labeled_total) if labeled_total > 0 else 0.0,
            "vqa_soft": (soft / labeled_total) if labeled_total > 0 else 0.0,
        }
        LOGGER.info(
            "eval done | split=%s samples=%d labeled=%d exact=%.4f soft=%.4f",
            split_name,
            total,
            labeled_total,
            metrics["exact_match"],
            metrics["vqa_soft"],
        )
        return metrics

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        return out
