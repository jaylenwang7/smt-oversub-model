"""Compute the 10% iso-physical-core oversubscription-ratio breakeven.

This script reuses the updated 02c declarative config instead of reimplementing
the model. For each candidate no-SMT R, it rebuilds the config so resource
scaling is re-resolved at that R.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import json5

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from smt_oversub_model.declarative import AnalysisConfig, DeclarativeAnalysisEngine


Metric = Literal["carbon", "tco"]

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "oversub_analysis"
    / "genoa"
    / "scheduling_input_sensitivity"
    / "iso_physical_core"
    / "resource_scaling"
    / "util_10_pct.jsonc"
)
CONFIG_DIR = CONFIG_PATH.parent

HEADLINE_SMT_R = 3.32
HEADLINE_NOSMT_R = 5.58
HEADLINE_VCPU_MULTIPLIER = 0.75


@lru_cache(maxsize=None)
def load_base_config() -> dict:
    """Load the 10% iso-physical-core config and convert it to a compare."""
    with CONFIG_PATH.open() as f:
        config = json5.load(f)

    config["analysis"] = {
        "type": "compare",
        "baseline": "smt_oversub",
        "scenarios": ["smt_oversub", "nosmt_oversub"],
    }
    config["output_dir"] = None
    return config


@lru_cache(maxsize=None)
def evaluate_case(
    smt_r: float,
    nosmt_r: float,
    vcpu_multiplier: float = HEADLINE_VCPU_MULTIPLIER,
) -> tuple[dict, dict]:
    """Evaluate SMT and no-SMT for one R pair."""
    config = copy.deepcopy(load_base_config())
    config["scenarios"]["smt_oversub"]["oversub_ratio"] = smt_r
    config["scenarios"]["nosmt_oversub"]["oversub_ratio"] = nosmt_r
    config["scenarios"]["nosmt_oversub"]["vcpu_demand_multiplier"] = vcpu_multiplier

    analysis_config = AnalysisConfig.from_dict(config, base_path=CONFIG_DIR)
    result = DeclarativeAnalysisEngine().run(analysis_config)
    return (
        result.scenario_results["smt_oversub"],
        result.scenario_results["nosmt_oversub"],
    )


def metric_key(metric: Metric) -> str:
    return {
        "carbon": "total_carbon_kg",
        "tco": "total_cost_usd",
    }[metric]


def pct_diff(
    metric: Metric,
    *,
    smt_r: float,
    nosmt_r: float,
    vcpu_multiplier: float = HEADLINE_VCPU_MULTIPLIER,
) -> float:
    """Return no-SMT percentage difference versus SMT."""
    smt, nosmt = evaluate_case(smt_r, nosmt_r, vcpu_multiplier)
    key = metric_key(metric)
    return (nosmt[key] - smt[key]) / smt[key] * 100.0


def find_breakeven(
    metric: Metric,
    *,
    smt_r: float,
    low: float = 0.5,
    high: float = 12.0,
    iterations: int = 24,
) -> tuple[float, float]:
    """Find the first no-SMT R where the target metric is no worse than SMT."""
    if pct_diff(metric, smt_r=smt_r, nosmt_r=low) <= 0:
        return low, pct_diff(metric, smt_r=smt_r, nosmt_r=low)
    if pct_diff(metric, smt_r=smt_r, nosmt_r=high) > 0:
        raise ValueError(f"No {metric} breakeven found below no-SMT R={high}")

    for _ in range(iterations):
        mid = (low + high) / 2.0
        if pct_diff(metric, smt_r=smt_r, nosmt_r=mid) > 0:
            low = mid
        else:
            high = mid

    return high, pct_diff(metric, smt_r=smt_r, nosmt_r=high)


def print_current_headline() -> None:
    smt, nosmt = evaluate_case(HEADLINE_SMT_R, HEADLINE_NOSMT_R)
    print("Updated 02c iso-physical-core 10% resource-scaling setup")
    print(f"- SMT R: {HEADLINE_SMT_R:.2f}")
    print(f"- no-SMT R: {HEADLINE_NOSMT_R:.2f}")
    print(f"- no-SMT / SMT R ratio: {HEADLINE_NOSMT_R / HEADLINE_SMT_R:.3f}x")
    print(f"- SMT : no-SMT R ratio: 1 : {HEADLINE_NOSMT_R / HEADLINE_SMT_R:.3f}")
    print(f"- vCPU demand multiplier: {HEADLINE_VCPU_MULTIPLIER:.2f}")
    print(f"- SMT servers: {smt['num_servers']}")
    print(f"- no-SMT servers: {nosmt['num_servers']}")
    print(f"- Carbon change: {pct_diff('carbon', smt_r=HEADLINE_SMT_R, nosmt_r=HEADLINE_NOSMT_R):+.2f}%")
    print(f"- TCO change: {pct_diff('tco', smt_r=HEADLINE_SMT_R, nosmt_r=HEADLINE_NOSMT_R):+.2f}%")


def print_fixed_smt_breakeven() -> None:
    print("\nBreakeven with SMT fixed at the headline R")
    print("| Metric | no-SMT breakeven R | no-SMT / SMT ratio | SMT : no-SMT | Diff at threshold |")
    print("|---|---:|---:|---:|---:|")
    for metric in ("carbon", "tco"):
        r, diff = find_breakeven(metric, smt_r=HEADLINE_SMT_R)
        print(
            f"| {metric.upper()} | {r:.3f} | {r / HEADLINE_SMT_R:.3f}x | "
            f"1 : {r / HEADLINE_SMT_R:.3f} | {diff:+.3f}% |"
        )


def print_same_r_sanity() -> None:
    print(f"\nSame-R sanity check at R={HEADLINE_SMT_R:.2f}")
    print("| no-SMT vCPU multiplier | SMT servers | no-SMT servers | Carbon change | TCO change |")
    print("|---:|---:|---:|---:|---:|")
    for multiplier in (1.0, HEADLINE_VCPU_MULTIPLIER):
        smt, nosmt = evaluate_case(HEADLINE_SMT_R, HEADLINE_SMT_R, multiplier)
        print(
            f"| {multiplier:.2f} | {smt['num_servers']} | {nosmt['num_servers']} | "
            f"{pct_diff('carbon', smt_r=HEADLINE_SMT_R, nosmt_r=HEADLINE_SMT_R, vcpu_multiplier=multiplier):+.2f}% | "
            f"{pct_diff('tco', smt_r=HEADLINE_SMT_R, nosmt_r=HEADLINE_SMT_R, vcpu_multiplier=multiplier):+.2f}% |"
        )


def print_same_ratio_r_progression() -> None:
    print(f"\nWhy same relative R flips between R=1.00 and R={HEADLINE_SMT_R:.2f}")
    print("| Same R on both sides | Server ratio | no-SMT / SMT per-server embodied carbon | no-SMT / SMT fleet embodied carbon | no-SMT / SMT per-server power | no-SMT / SMT fleet power | Carbon change | TCO change |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r_value in (1.0, HEADLINE_SMT_R):
        smt, nosmt = evaluate_case(r_value, r_value)
        server_ratio = nosmt["num_servers"] / smt["num_servers"]
        smt_embodied_per_server = smt["embodied_carbon_kg"] / smt["num_servers"]
        nosmt_embodied_per_server = nosmt["embodied_carbon_kg"] / nosmt["num_servers"]
        smt_fleet_power = smt["num_servers"] * smt["power_per_server_w"]
        nosmt_fleet_power = nosmt["num_servers"] * nosmt["power_per_server_w"]
        print(
            f"| {r_value:.2f} | {server_ratio:.3f}x | "
            f"{nosmt_embodied_per_server / smt_embodied_per_server:.3f}x | "
            f"{nosmt['embodied_carbon_kg'] / smt['embodied_carbon_kg']:.3f}x | "
            f"{nosmt['power_per_server_w'] / smt['power_per_server_w']:.3f}x | "
            f"{nosmt_fleet_power / smt_fleet_power:.3f}x | "
            f"{pct_diff('carbon', smt_r=r_value, nosmt_r=r_value):+.2f}% | "
            f"{pct_diff('tco', smt_r=r_value, nosmt_r=r_value):+.2f}% |"
        )


def main() -> None:
    print_current_headline()
    print_fixed_smt_breakeven()
    print_same_r_sanity()
    print_same_ratio_r_progression()


if __name__ == "__main__":
    main()
