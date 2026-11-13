#!/usr/bin/env python
"""
Grid search for the best Real-Time Chunking (RTC, guided mode) settings on a recorded dataset.

The script simulates what happens during `lerobot-rollout --inference.type=rtc` offline:

    time t-s : chunk P is predicted from obs[t-s]               (previous chunk)
    time t   : queue has `queue_threshold` actions left -> leftover = P[s:]   (s = chunk_size - queue_threshold)
               new chunk N is predicted from obs[t] WITH RTC (leftover, inference_delay d)
               while inference runs, the robot executes leftover[:d]; afterwards N[d:]

For every RTC configuration (schedule x max_guidance_weight x execution_horizon x queue_threshold
x inference_delay) the following metrics are computed in the normalized action space:

    jump    : L2 distance at the switch point |N[d] - leftover[d-1]| minus the natural
              ground-truth step |GT[d] - GT[d-1]|  (lower = smoother transition)
    prefix  : mean L2 distance between N[d:h] and leftover[d:h] (consistency with the old plan)
    gt_err  : MSE between N[d:d+s] and the recorded dataset actions (lower = task fidelity)
    score   : gt_err + jump_weight * max(jump, 0)

The same noise is used for every config of a sample, so differences come only from RTC.
A no-RTC baseline is reported as well.

Usage:
    uv run python examples/rtc/find_best_rtc_settings.py \
        --policy_path=outputs/train/2026-11-11/14-39-17_smolVLA/checkpoints/012000/pretrained_model \
        --dataset=/home/student/.cache/huggingface/lerobot/Schrunk/so101_tea_strainer_tea_dispenser_20261111_114004 \
        --num_samples=30 --device=cuda
"""

import argparse
import csv
import itertools
import logging
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging


def parse_list(s: str, typ):
    return [typ(x) for x in s.split(",") if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    episode: int
    frame: int
    obs_prev: dict  # preprocessed observation at t - s (one per queue_threshold)
    obs_now: dict  # preprocessed observation at t
    gt: torch.Tensor  # (chunk, action_dim) normalized ground-truth actions starting at t
    gt_valid: torch.Tensor  # (chunk,) bool, False where padded
    noise: torch.Tensor


class RTCTuner:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        self.policy_cfg.pretrained_path = args.policy_path
        self.policy_cfg.device = self.device

        meta = LeRobotDatasetMetadata(args.dataset)
        self.fps = meta.fps
        delta_timestamps = resolve_delta_timestamps(self.policy_cfg, meta)
        self.dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
        logging.info(f"Dataset: {self.dataset.num_episodes} episodes, {len(self.dataset)} frames, {self.fps} fps")

        self.preprocessor, _ = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=args.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

        policy_cls = get_policy_class(self.policy_cfg.type)
        self.policy = policy_cls.from_pretrained(args.policy_path, config=self.policy_cfg)
        self.policy.to(self.device).eval()

        self.chunk_size = self.policy.config.chunk_size
        self.action_dim = self.policy.config.output_features[ACTION].shape[0]
        self.ep_from = self.dataset.meta.episodes["dataset_from_index"]
        self.ep_to = self.dataset.meta.episodes["dataset_to_index"]

    # ------------------------------------------------------------------ helpers
    def _set_rtc(self, enabled: bool, schedule=RTCAttentionSchedule.EXP, weight=10.0, horizon=10):
        self.policy.config.rtc_config = RTCConfig(
            enabled=enabled,
            mode="guided",
            prefix_attention_schedule=schedule,
            max_guidance_weight=weight,
            execution_horizon=horizon,
        )
        self.policy.init_rtc_processor()

    def _get_obs(self, global_idx: int) -> dict:
        item = self.dataset[global_idx]
        batch = default_collate([item])
        return self.preprocessor(batch)

    @torch.no_grad()
    def _predict(self, obs: dict, noise: torch.Tensor, **rtc_kwargs) -> torch.Tensor:
        self.policy.reset()
        obs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        out = self.policy.predict_action_chunk(obs, noise=noise.clone(), **rtc_kwargs)
        return out.squeeze(0)[:, : self.action_dim].float()

    def _sample_noise(self) -> torch.Tensor:
        size = (1, self.chunk_size, self.policy.config.max_action_dim)
        return self.policy.model.sample_noise(size, self.device)

    # ------------------------------------------------------------------ latency
    def measure_inference_delay(self, obs: dict, noise: torch.Tensor) -> int:
        self._set_rtc(True, horizon=10)
        leftover = torch.zeros(self.chunk_size // 2, self.action_dim, device=self.device)
        for _ in range(2):  # warmup
            self._predict(obs, noise, inference_delay=3, prev_chunk_left_over=leftover, execution_horizon=10)
        times = []
        for _ in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._predict(obs, noise, inference_delay=3, prev_chunk_left_over=leftover, execution_horizon=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        lat = float(np.median(times))
        delay = max(1, math.ceil(lat * self.fps))
        logging.info(f"Measured RTC inference latency: {lat * 1000:.1f} ms -> inference_delay ≈ {delay} steps")
        return delay

    # ------------------------------------------------------------------ samples
    def build_samples(self, queue_thresholds: list[int]) -> list[Sample]:
        rng = random.Random(self.args.seed)
        max_s = max(self.chunk_size - q for q in queue_thresholds)
        samples = []
        tries = 0
        while len(samples) < self.args.num_samples and tries < self.args.num_samples * 20:
            tries += 1
            ep = rng.randrange(self.dataset.num_episodes)
            start, end = int(self.ep_from[ep]), int(self.ep_to[ep])
            length = end - start
            if length < max_s + self.chunk_size:
                continue
            frame = rng.randrange(max_s, length - self.chunk_size // 2)
            obs_now = self._get_obs(start + frame)
            obs_prev = {q: self._get_obs(start + frame - (self.chunk_size - q)) for q in queue_thresholds}
            gt = obs_now[ACTION].squeeze(0)[:, : self.action_dim].float()
            pad_key = f"{ACTION}_is_pad"
            gt_valid = ~obs_now[pad_key].squeeze(0).bool() if pad_key in obs_now else torch.ones(len(gt), dtype=torch.bool)
            obs_now = {k: v for k, v in obs_now.items() if not k.startswith(ACTION)}
            obs_prev = {q: {k: v for k, v in o.items() if not k.startswith(ACTION)} for q, o in obs_prev.items()}
            samples.append(Sample(ep, frame, obs_prev, obs_now, gt, gt_valid.to(self.device), self._sample_noise()))
        logging.info(f"Built {len(samples)} evaluation samples")
        return samples

    # ------------------------------------------------------------------ metrics
    def _metrics(self, new: torch.Tensor, leftover: torch.Tensor, s: Sample, d: int, h: int, exec_len: int):
        d = min(d, len(leftover))
        # switch point: last executed old action -> first executed new action
        jump = torch.linalg.norm(new[d] - leftover[d - 1]).item() if d > 0 else 0.0
        gt_step = torch.linalg.norm(s.gt[d] - s.gt[d - 1]).item() if d > 0 else 0.0
        jump_excess = jump - gt_step
        h_eff = min(h, len(leftover))
        prefix = torch.linalg.norm(new[d:h_eff] - leftover[d:h_eff], dim=-1).mean().item() if h_eff > d else 0.0
        sl = slice(d, min(d + exec_len, self.chunk_size))
        valid = s.gt_valid[sl]
        err = ((new[sl] - s.gt[sl]) ** 2).mean(-1)
        gt_err = err[valid].mean().item() if valid.any() else float("nan")
        # smoothness of the actually executed trajectory leftover[:d] + new[d:]
        executed = torch.cat([leftover[:d], new[d : d + exec_len]])
        jerk = torch.diff(executed, n=2, dim=0).norm(dim=-1).mean().item() if len(executed) > 2 else 0.0
        return {"jump": jump, "jump_excess": jump_excess, "prefix_dev": prefix, "gt_err": gt_err, "jerk": jerk}

    # ------------------------------------------------------------------ main
    def run(self):
        a = self.args
        set_seed(a.seed)
        queue_thresholds = parse_list(a.queue_thresholds, int)
        samples = self.build_samples(queue_thresholds)
        if not samples:
            raise RuntimeError("No valid samples (episodes too short?)")

        if a.inference_delays == "auto":
            delays = [self.measure_inference_delay(samples[0].obs_now, samples[0].noise)]
            delays = sorted({max(1, delays[0] - 1), delays[0], delays[0] + 2})
        else:
            delays = parse_list(a.inference_delays, int)

        schedules = [RTCAttentionSchedule(x.upper()) for x in parse_list(a.schedules, str)]
        weights = parse_list(a.guidance_weights, float)
        horizons = parse_list(a.execution_horizons, int)

        # previous chunks (no RTC) and no-RTC baselines, cached
        self._set_rtc(False)
        prev_chunks = {}
        baselines = {}
        for i, s in enumerate(samples):
            for q in queue_thresholds:
                prev = self._predict(s.obs_prev[q], self._sample_noise())
                prev_chunks[(i, q)] = prev[self.chunk_size - q :]
            baselines[i] = self._predict(s.obs_now, s.noise)

        grid = list(itertools.product(queue_thresholds, delays, schedules, weights, horizons))
        grid = [g for g in grid if g[4] <= g[0] and g[1] < g[4]]  # horizon <= leftover len, delay < horizon
        logging.info(f"Evaluating {len(grid)} RTC configs x {len(samples)} samples")

        rows = []
        # baseline rows
        for q, d in itertools.product(queue_thresholds, delays):
            exec_len = self.chunk_size - q
            ms = [self._metrics(baselines[i], prev_chunks[(i, q)], s, d, 0, exec_len) for i, s in enumerate(samples)]
            rows.append(self._aggregate(ms, q, d, "NO_RTC", 0.0, 0))

        t_start = time.time()
        for n, (q, d, sched, w, h) in enumerate(grid):
            self._set_rtc(True, sched, w, h)
            exec_len = self.chunk_size - q
            ms = []
            for i, s in enumerate(samples):
                leftover = prev_chunks[(i, q)]
                new = self._predict(
                    s.obs_now, s.noise, inference_delay=d, prev_chunk_left_over=leftover, execution_horizon=h
                )
                ms.append(self._metrics(new, leftover, s, d, h, exec_len))
            rows.append(self._aggregate(ms, q, d, sched.value, w, h))
            el = time.time() - t_start
            eta = el / (n + 1) * (len(grid) - n - 1)
            r = rows[-1]
            logging.info(
                f"[{n + 1}/{len(grid)}] q={q} d={d} {sched.value:6s} w={w:5.1f} h={h:2d} | "
                f"score={r['score']:.4f} gt_err={r['gt_err']:.4f} jump={r['jump']:.4f} "
                f"prefix={r['prefix_dev']:.4f}  (ETA {eta / 60:.1f} min)"
            )

        self._report(rows)

    def _aggregate(self, ms, q, d, sched, w, h):
        agg = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        agg["score"] = agg["gt_err"] + self.args.jump_weight * max(agg["jump_excess"], 0.0)
        return {"queue_threshold": q, "inference_delay": d, "schedule": sched, "max_guidance_weight": w,
                "execution_horizon": h, **agg}

    def _report(self, rows):
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, "rtc_grid_results.csv")
        rows_sorted = sorted(rows, key=lambda r: r["score"])
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
            writer.writeheader()
            writer.writerows(rows_sorted)

        # inference_delay is a property of the hardware, not a setting -> average over all tested delays
        metric_keys = ["score", "gt_err", "jump", "jump_excess", "prefix_dev", "jerk"]
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            if r["schedule"] == "NO_RTC":
                continue
            key = (r["queue_threshold"], r["schedule"], r["max_guidance_weight"], r["execution_horizon"])
            groups.setdefault(key, []).append(r)
        delays_str = ",".join(str(d) for d in sorted({r["inference_delay"] for r in rows}))
        robust = []
        for (q, sched, w, h), rs in groups.items():
            robust.append({"queue_threshold": q, "inference_delay": delays_str, "schedule": sched,
                           "max_guidance_weight": w, "execution_horizon": h,
                           **{k: float(np.mean([r[k] for r in rs])) for k in metric_keys}})
        robust.sort(key=lambda r: r["score"])
        robust_path = os.path.join(self.args.output_dir, "rtc_grid_results_avg_over_delays.csv")
        with open(robust_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(robust[0]))
            writer.writeheader()
            writer.writerows(robust)

        hdr = f"{'#':>3} {'q':>3} {'d':>7} {'schedule':8} {'w':>5} {'h':>3} {'score':>8} {'gt_err':>8} {'jump':>7} {'jumpEx':>7} {'prefix':>7} {'jerk':>7}"
        print("\n" + "=" * len(hdr))
        print(f"Top RTC configurations, averaged over inference delays {delays_str} (lower score = better)")
        print("=" * len(hdr))
        print(hdr)
        for i, r in enumerate(robust[: self.args.top_k]):
            print(
                f"{i + 1:>3} {r['queue_threshold']:>3} {r['inference_delay']:>7} {r['schedule']:8} "
                f"{r['max_guidance_weight']:>5.1f} {r['execution_horizon']:>3} {r['score']:>8.4f} "
                f"{r['gt_err']:>8.4f} {r['jump']:>7.4f} {r['jump_excess']:>7.4f} {r['prefix_dev']:>7.4f} {r['jerk']:>7.4f}"
            )
        print("\nNo-RTC baselines:")
        for r in rows:
            if r["schedule"] == "NO_RTC":
                print(f"    q={r['queue_threshold']} d={r['inference_delay']}: score={r['score']:.4f} "
                      f"gt_err={r['gt_err']:.4f} jump={r['jump']:.4f} jerk={r['jerk']:.4f}")

        best = robust[0]
        print(f"\nFull results: {path}\nAveraged over delays: {robust_path}")
        print("\nRecommended rollout flags:")
        print(
            "    --inference.type=rtc \\\n"
            "    --inference.rtc.mode=guided \\\n"
            f"    --inference.queue_threshold={best['queue_threshold']} \\\n"
            f"    --inference.rtc.execution_horizon={best['execution_horizon']} \\\n"
            f"    --inference.rtc.max_guidance_weight={best['max_guidance_weight']} \\\n"
            f"    --inference.rtc.prefix_attention_schedule={best['schedule']}"
        )
        print(f"  (robust over inference_delay={best['inference_delay']} steps at {self.fps} fps)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy_path", required=True)
    p.add_argument("--dataset", required=True, help="repo_id or local dataset path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedules", default="LINEAR,EXP,ONES,ZEROS")
    p.add_argument("--guidance_weights", default="1,2.5,5,10,20")
    p.add_argument("--execution_horizons", default="5,10,15,20,25,30")
    p.add_argument("--queue_thresholds", default="20,30")
    p.add_argument("--inference_delays", default="auto", help="'auto' (measure latency) or e.g. '3,5,8'")
    p.add_argument("--jump_weight", type=float, default=1.0, help="weight of the boundary jump in the score")
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--output_dir", default="outputs/rtc_tuning")
    args = p.parse_args()

    init_logging()
    RTCTuner(args).run()


if __name__ == "__main__":
    main()
