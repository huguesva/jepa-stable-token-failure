"""Evaluate PushT planning with a matched corner-patch nuisance.

Adapted from the upstream ``eval.py`` at 8edfeb3.  Changes:

1. **API port** to stable-worldmodel 0.0.5: ``world.evaluate_from_dataset(...)``
   instead of ``world.evaluate(dataset=...)``, and ``swm.policy.AutoCostModel``
   instead of ``swm.wm.utils.load_pretrained``.
2. **Nuisance** applied to the observation *and* the goal image through
   :class:`nuisance.eval_policy.NuisanceEvalPolicy`.
3. Results are written as JSON (per-episode successes retained) next to a
   human-readable dump, so the figure script never has to parse free text.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from nuisance.eval_policy import NuisanceEvalPolicy
from nuisance.nuisance import NuisanceRenderer, NuisanceSpec


def img_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=input_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=dataset_path
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, (
        "Planning horizon must be smaller than or equal to eval_budget"
    )

    spec = NuisanceSpec(**OmegaConf.to_container(cfg.nuisance, resolve=True))

    # The environment always renders the raw scene; the nuisance may then enlarge
    # it, and the model input size follows the nuisance, not the renderer.
    render_size = cfg.eval.render_size
    input_size = spec.output_size(render_size)
    print(
        f"[nuisance-eval] condition={spec.condition} mode={spec.mode} "
        f"border={spec.border} render_size={render_size} input_size={input_size}"
    )

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(render_size, render_size))

    transform = {"pixels": img_transform(input_size), "goal": img_transform(input_size)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- policy
    model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cfg.cache_dir)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )
    if spec.paints_border:
        policy = NuisanceEvalPolicy(
            policy, NuisanceRenderer(spec), goal_mode=cfg.eval.goal_nuisance
        )

    results_path = Path(cfg.output.dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # -- sample evaluation start points from the PushT dataset
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    chosen = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    chosen = np.sort(valid_indices[chosen])

    eval_episodes = dataset.get_row_data(chosen)[col_name]
    eval_start_idx = dataset.get_row_data(chosen)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=bool(cfg.output.save_video),
        video_path=results_path,
    )
    elapsed = time.time() - start_time
    print(metrics["success_rate"], "success rate")

    record = {
        "tag": cfg.output.tag,
        "policy": str(cfg.policy),
        "task": cfg.eval.dataset_name,
        "condition": spec.condition,
        "nuisance": OmegaConf.to_container(cfg.nuisance, resolve=True),
        "goal_nuisance": cfg.eval.goal_nuisance,
        "eval_seed": int(cfg.seed),
        "num_eval": int(cfg.eval.num_eval),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "eval_budget": int(cfg.eval.eval_budget),
        "render_size": int(render_size),
        "input_size": int(input_size),
        "plan_config": OmegaConf.to_container(cfg.plan_config, resolve=True),
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": [bool(x) for x in metrics["episode_successes"]],
        "episode_ids": [int(x) for x in eval_episodes.tolist()],
        "start_steps": [int(x) for x in eval_start_idx.tolist()],
        "eval_seconds": elapsed,
    }

    out_json = results_path / f"{cfg.output.tag}.json"
    out_json.write_text(json.dumps(record, indent=2))
    print(f"[nuisance-eval] wrote {out_json}")

    with (results_path / cfg.output.filename).open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: success_rate={metrics['success_rate']}\n")
        f.write(f"evaluation_time: {elapsed} seconds\n")


if __name__ == "__main__":
    run()
