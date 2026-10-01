#!/usr/bin/env python3
"""Collect paired PushT evaluations and reproduce the planning-success figure."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import t


ARMS = (
    ("grey", "Grey control", "#64748b"),
    ("stable", "Episode-stable color", "#d55e00"),
    ("unpredictable", "Unpredictable color", "#0072b2"),
)
DEFAULT_SEEDS = (4711, 8123, 2659, 6011, 7297, 8837, 9187, 10427, 11717, 12829)
DEFAULT_EVAL_SEEDS = (42, 4242)


def read_csv(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open()))
    seeds = [int(row["seed"]) for row in rows]
    scores = {
        arm: np.asarray([float(row[arm]) for row in rows], dtype=np.float64)
        for arm, _label, _color in ARMS
    }
    return seeds, scores


def validate_report(report: dict, arm: str, eval_seed: int) -> float:
    successes = report["episode_successes"]
    measured = 100.0 * sum(successes) / len(successes)
    if len(successes) != report["num_eval"] or len(successes) != 50:
        raise ValueError("every evaluation must contain exactly 50 episodes")
    if abs(measured - report["success_rate"]) > 1e-10:
        raise ValueError("success rate does not match the retained episode outcomes")
    if report["eval_seed"] != eval_seed or report["goal_nuisance"] != "matched":
        raise ValueError("unexpected evaluation seed or goal-nuisance mode")

    nuisance = report["nuisance"]
    expected = {"geometry": "patch", "n_patches": 1, "patch_size": 14, "border": 14}
    if any(nuisance[key] != value for key, value in expected.items()):
        raise ValueError(f"report does not use the one-token patch: {report['tag']}")
    if arm == "grey":
        valid = nuisance["condition"] == "neutral"
    else:
        rho = 1.0 if arm == "stable" else 0.0
        valid = nuisance["condition"] == "markov" and nuisance["rho"] == rho
    if not valid:
        raise ValueError(f"report is assigned to the wrong arm: {report['tag']}")
    return measured


def read_evaluations(
    directory: Path,
    eval_seeds: tuple[int, ...],
    allow_incomplete: bool,
) -> tuple[list[int], dict[str, np.ndarray]]:
    pattern = re.compile(r"^(grey|stable|unpredictable)_seed(\d+)_eval(\d+)\.json$")
    found_seeds = {
        int(match.group(2))
        for path in directory.glob("*.json")
        if (match := pattern.match(path.name))
    }
    requested = list(DEFAULT_SEEDS)
    if allow_incomplete:
        requested = [seed for seed in requested if seed in found_seeds]

    complete = []
    by_arm: dict[str, list[float]] = {arm: [] for arm, _label, _color in ARMS}
    for seed in requested:
        paths = {
            arm: [directory / f"{arm}_seed{seed}_eval{eval_seed}.json" for eval_seed in eval_seeds]
            for arm, _label, _color in ARMS
        }
        if not all(path.is_file() for arm_paths in paths.values() for path in arm_paths):
            if allow_incomplete:
                continue
            raise FileNotFoundError(f"seed {seed} is missing one or more paired evaluations")
        complete.append(seed)
        for arm, _label, _color in ARMS:
            values = [
                validate_report(json.loads(path.read_text()), arm, eval_seed)
                for path, eval_seed in zip(paths[arm], eval_seeds, strict=True)
            ]
            by_arm[arm].append(float(np.mean(values)))

    if len(complete) < 2:
        raise ValueError("at least two complete paired seeds are required")
    return complete, {arm: np.asarray(values) for arm, values in by_arm.items()}


def mean_summary(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half_width = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sample_sd": sd, "ci95": [mean - half_width, mean + half_width]}


def paired_summary(left: np.ndarray, right: np.ndarray) -> dict:
    differences = left - right
    result = mean_summary(differences)
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    result["differences"] = differences.tolist()
    result["exact_two_sided_sign_flip_p"] = float(
        np.mean(np.asarray(null) >= observed - 1e-12)
    )
    return result


def save_scores(path: Path, seeds: list[int], scores: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "grey", "stable", "unpredictable"))
        for index, seed in enumerate(seeds):
            writer.writerow((seed, *(scores[arm][index] for arm, _label, _color in ARMS)))


def plot(path: Path, scores: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    figure, axis = plt.subplots(figsize=(7.8, 4.3), layout="constrained")
    n = len(next(iter(scores.values())))
    offsets = np.linspace(-0.20, 0.20, n)

    for index, (arm, label, color) in enumerate(ARMS):
        values = scores[arm]
        summary = mean_summary(values)
        mean = summary["mean"]
        low, high = summary["ci95"]
        axis.bar(index, mean, width=0.58, color=color, alpha=0.16, zorder=2)
        axis.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            ecolor=color,
            elinewidth=2,
            capsize=6,
            capthick=2,
            zorder=4,
        )
        axis.plot([index - 0.29, index + 0.29], [mean, mean], color=color, lw=2.2, zorder=4)
        axis.scatter(
            index + offsets,
            values,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        label_y = min(101.0, max(float(values.max()), high) + 4.0)
        axis.text(index, label_y, f"{mean:.1f}%", ha="center", color=color, fontweight="bold", fontsize=16)

    axis.set_xticks(range(3), [label for _arm, label, _color in ARMS])
    axis.tick_params(axis="x", length=0, pad=10)
    axis.set_xlim(-0.58, 2.58)
    axis.set_ylim(0, 106)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.set_ylabel("PushT planning success")
    axis.grid(axis="y", color="#e2e8f0", lw=0.7)
    axis.set_axisbelow(True)
    axis.text(0.01, 0.98, f"{n} training seeds", transform=axis.transAxes, va="top", color="#475569")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("results/reference_scores.csv"))
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/planning_success"))
    args = parser.parse_args()

    if args.eval_dir:
        seeds, scores = read_evaluations(args.eval_dir, tuple(args.eval_seeds), args.allow_incomplete)
        save_scores(args.output.with_name(args.output.name + "_scores").with_suffix(".csv"), seeds, scores)
    else:
        seeds, scores = read_csv(args.scores)

    if len({len(values) for values in scores.values()}) != 1:
        raise ValueError("arms have different numbers of training seeds")
    plot(args.output, scores)

    summary = {arm: mean_summary(values) for arm, values in scores.items()}
    summary["paired_contrasts"] = {
        "stable_minus_grey": paired_summary(scores["stable"], scores["grey"]),
        "unpredictable_minus_stable": paired_summary(scores["unpredictable"], scores["stable"]),
        "unpredictable_minus_grey": paired_summary(scores["unpredictable"], scores["grey"]),
    }
    record = {"seeds": seeds, "scores": {key: value.tolist() for key, value in scores.items()}, "statistics": summary}
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
