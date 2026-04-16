#!/usr/bin/env python3
"""Generate and optionally run mixed-fleet (SMT + no-SMT) partitioning analyses.

This script generates analyses that explore what happens when a fleet is split
into two pools — one running SMT, one running no-SMT — and demand is modeled as
self-selecting between them based on vCPU demand discount. The split point is
determined by the per-workload breakeven: workloads whose discount is below the
breakeven are treated as choosing no-SMT, while those above are treated as
staying on SMT.

It uses the updated scheduling-input R values from the iso-LP and
iso-physical-core experimental calibrations (see 02c).

Configs are written to:

    configs/oversub_analysis/genoa/mixed_fleet/

Results are written to:

    results/oversub_analysis/genoa/mixed_fleet/

Usage:

    MPLCONFIGDIR=/tmp/mpl python tools/generate_mixed_fleet_analysis.py
    python tools/generate_mixed_fleet_analysis.py --generate-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from smt_oversub_model.declarative import run_analysis
from smt_oversub_model.output import OutputWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "oversub_analysis" / "genoa" / "mixed_fleet"
RESULTS_ROOT = REPO_ROOT / "results" / "oversub_analysis" / "genoa" / "mixed_fleet"

UTILS = [0.10, 0.20, 0.30]

# vCPU discount trait distribution: uniform from 0.50 to 1.00 in 0.05 steps.
# Value 0.50 = strong no-SMT advantage, 1.00 = no advantage.
TRAIT_BINS = [
    {"value": round(0.50 + i * 0.05, 2), "vcpu_fraction": round(1 / 11, 4)}
    for i in range(11)
]

# Split-point sweep range for the sweep configs.
SPLIT_SWEEP_VALUES = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01,
]

BREAKEVEN_METRICS = ("carbon", "tco")

# Scheduling-input bases: R values from the iso-LP and iso-physical-core
# experimental calibrations (02c).
INPUT_BASES: dict[str, dict[str, Any]] = {
    "iso_lp": {
        "label": "Iso-LP",
        "description": (
            "Interpolated Go-CPU max-safe VP/LP rates from the baseline "
            "8 LP / 2 VM VP-constrained SMT regime versus No-SMT."
        ),
        "ratios": {
            0.10: {"smt": 2.59, "nosmt": 5.58},
            0.20: {"smt": 1.29, "nosmt": 2.79},
            0.30: {"smt": 0.86, "nosmt": 1.86},
        },
        # Carbon breakeven multipliers from scheduling_input_sensitivity results.
        # Used as x_axis_markers on sweep plots.
        "breakeven_multipliers": {
            "resource_scaling": {0.10: 0.87, 0.20: 0.88, 0.30: 0.89},
            "resource_constraints": {0.10: 0.90, 0.20: 0.89, 0.30: 0.83},
        },
    },
    "iso_physical_core": {
        "label": "Iso-Physical-Core",
        "description": (
            "Interpolated Go-CPU max-safe VP/LP rates from the pool-adjusted "
            "16 LP / 4 VM SMT regime on the same physical-core budget, versus No-SMT."
        ),
        "ratios": {
            0.10: {"smt": 3.32, "nosmt": 5.58},
            0.20: {"smt": 1.66, "nosmt": 2.79},
            0.30: {"smt": 1.11, "nosmt": 1.86},
        },
        "breakeven_multipliers": {
            "resource_scaling": {0.10: 0.85, 0.20: 0.83, 0.30: 0.82},
            "resource_constraints": {0.10: 0.90, 0.20: 0.87, 0.30: 0.77},
        },
    },
}

RESOURCE_MODES: dict[str, dict[str, Any]] = {
    "resource_scaling": {
        "label": "Purpose-Built No-SMT (Resource Scaling)",
        "short_label": "Purpose-Built",
        "nosmt_processor": "nosmt_linear",
        "smt_scenario_patch": {
            "resource_scaling": {"scale_with_vcpus": ["memory", "ssd"]},
        },
        "nosmt_scenario_patch": {
            "resource_scaling": {"scale_with_vcpus": ["memory", "ssd"]},
        },
    },
    "resource_constraints": {
        "label": "Same Hardware SMT-Off (Resource Constraints)",
        "short_label": "Same HW",
        "nosmt_processor": "nosmt_smt_hw_linear",
        "smt_scenario_patch": {
            "resource_constraints": {
                "memory_gb": {"capacity_per_thread": 4.8, "demand_per_vcpu": 4.0},
                "ssd_gb": {"capacity_per_thread": 75.0, "demand_per_vcpu": 50.0},
            },
        },
        "nosmt_scenario_patch": {
            "resource_constraints": {
                "memory_gb": {"capacity_per_thread": 9.6, "demand_per_vcpu": 4.0},
                "ssd_gb": {"capacity_per_thread": 150.0, "demand_per_vcpu": 50.0},
            },
        },
    },
}


def modeled_ratio(value: float) -> float:
    """Cap inferred safe rates below 1.0 to no-oversubscription."""
    return max(1.0, value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def repo_rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def output_rel(path: Path) -> str:
    return f"./{repo_rel(path)}"


def build_compare_config(
    basis_key: str,
    mode: str,
    util: float,
) -> tuple[Path, dict[str, Any]]:
    """Build a compare config with metric-specific mixed-fleet split points."""
    basis = INPUT_BASES[basis_key]
    raw_ratios = basis["ratios"][util]
    smt_r = modeled_ratio(raw_ratios["smt"])
    nosmt_r = modeled_ratio(raw_ratios["nosmt"])
    util_pct = int(util * 100)
    mode_info = RESOURCE_MODES[mode]

    config_path = CONFIG_ROOT / basis_key / mode / f"util_{util_pct}_pct_compare.jsonc"
    output_dir = RESULTS_ROOT / basis_key / mode / f"util_{util_pct}pct_compare"

    # SMT homogeneous scenario
    smt_homo = {"processor": "smt", "oversub_ratio": smt_r}
    smt_homo.update(json.loads(json.dumps(mode_info["smt_scenario_patch"])))

    # No-SMT homogeneous scenario (fleet-average discount 0.75)
    nosmt_homo = {
        "processor": mode_info["nosmt_processor"],
        "oversub_ratio": nosmt_r,
        "vcpu_demand_multiplier": 0.75,
    }
    nosmt_homo.update(json.loads(json.dumps(mode_info["nosmt_scenario_patch"])))

    # SMT pool sub-scenario (for composite)
    smt_pool = {"processor": "smt", "oversub_ratio": smt_r}
    smt_pool.update(json.loads(json.dumps(mode_info["smt_scenario_patch"])))

    # No-SMT pool sub-scenario (for composite)
    nosmt_pool = {
        "processor": mode_info["nosmt_processor"],
        "oversub_ratio": nosmt_r,
        "vcpu_demand_multiplier": 1.0,
    }
    nosmt_pool.update(json.loads(json.dumps(mode_info["nosmt_scenario_patch"])))

    mixed_fleet_scenarios: dict[str, dict[str, Any]] = {}
    mixed_fleet_labels: dict[str, str] = {}
    mixed_fleet_analysis_names: list[str] = []

    for metric in BREAKEVEN_METRICS:
        scenario_name = f"mixed_fleet_{metric}"
        mixed_fleet_scenarios[scenario_name] = {
            "composite": {
                "nosmt_pool": {
                    "allocation": "below_split",
                    "parameter_effects": {"vcpu_demand_multiplier": "weighted_average"},
                },
                "smt_pool": {
                    "allocation": "above_split",
                    "parameter_effects": {"vcpu_demand_multiplier": 1.0},
                },
            },
            "split_trait": "vcpu_discount",
            "split_point": {
                "auto_breakeven": {
                    "baseline_scenario": "smt_pool",
                    "target_scenario": "nosmt_pool",
                    "target_parameter": "vcpu_demand_multiplier",
                    "match_metric": metric,
                    "search_bounds": [0.5, 1.0],
                }
            },
        }
        mixed_fleet_labels[scenario_name] = f"Mixed Fleet\n({metric.upper()} split)"
        mixed_fleet_analysis_names.append(scenario_name)

    description = (
        f"At {util_pct}% utilization: Comparison of homogeneous SMT "
        f"(R={smt_r:.2f}), homogeneous no-SMT (R={nosmt_r:.2f}, discount=0.75), "
        f"and mixed fleet with metric-specific auto-breakeven splits. {basis['label']} basis, "
        f"{mode_info['short_label']} resource model."
    )

    return config_path, {
        "name": f"{basis_key}_{mode}_util_{util_pct}pct_compare",
        "description": description,
        "scenarios": {
            "smt_homogeneous": smt_homo,
            "nosmt_homogeneous": nosmt_homo,
            "smt_pool": smt_pool,
            "nosmt_pool": nosmt_pool,
            **mixed_fleet_scenarios,
        },
        "workload": {
            "total_vcpus": 100000,
            "avg_util": util,
            "traits": {
                "vcpu_discount": {
                    "type": "discrete",
                    "bins": TRAIT_BINS,
                },
            },
        },
        "analysis": {
            "type": "compare",
            "baseline": "smt_homogeneous",
            "scenarios": ["smt_homogeneous", "nosmt_homogeneous", *mixed_fleet_analysis_names],
            "separate_metric_plots": True,
            "show_plot_title": False,
            "labels": {
                "smt_homogeneous": f"SMT\n(R={smt_r:.2f})",
                "nosmt_homogeneous": f"No-SMT\n(R={nosmt_r:.2f})",
                **mixed_fleet_labels,
            },
        },
        "processor": {
            "smt": "configs/shared/genoa_processors.jsonc:genoa_smt",
            "nosmt_linear": "configs/shared/genoa_processors.jsonc:genoa_nosmt_linear",
            "nosmt_smt_hw_linear": (
                "configs/shared/genoa_processors.jsonc:genoa_nosmt_smt_hw_linear"
            ),
        },
        "cost": {
            "carbon_intensity_g_kwh": 175,
            "electricity_cost_usd_kwh": 0.28,
            "lifetime_years": 6,
        },
        "power_curve": {"type": "polynomial"},
        "output_dir": output_rel(output_dir),
    }


def build_sweep_config(
    basis_key: str,
    mode: str,
    util: float,
) -> tuple[Path, dict[str, Any]]:
    """Build a split-point sweep config."""
    basis = INPUT_BASES[basis_key]
    raw_ratios = basis["ratios"][util]
    smt_r = modeled_ratio(raw_ratios["smt"])
    nosmt_r = modeled_ratio(raw_ratios["nosmt"])
    util_pct = int(util * 100)
    mode_info = RESOURCE_MODES[mode]

    config_path = CONFIG_ROOT / basis_key / mode / f"util_{util_pct}_pct_sweep.jsonc"
    output_dir = RESULTS_ROOT / basis_key / mode / f"util_{util_pct}pct_sweep"

    # Breakeven multiplier for x-axis marker
    breakeven = basis["breakeven_multipliers"][mode].get(util)

    # SMT baseline
    smt_baseline = {"processor": "smt", "oversub_ratio": smt_r}
    smt_baseline.update(json.loads(json.dumps(mode_info["smt_scenario_patch"])))

    # SMT pool
    smt_pool = {"processor": "smt", "oversub_ratio": smt_r}
    smt_pool.update(json.loads(json.dumps(mode_info["smt_scenario_patch"])))

    # No-SMT pool
    nosmt_pool = {
        "processor": mode_info["nosmt_processor"],
        "oversub_ratio": nosmt_r,
        "vcpu_demand_multiplier": 1.0,
    }
    nosmt_pool.update(json.loads(json.dumps(mode_info["nosmt_scenario_patch"])))

    # No-SMT homogeneous
    nosmt_homo = {
        "processor": mode_info["nosmt_processor"],
        "oversub_ratio": nosmt_r,
        "vcpu_demand_multiplier": 0.75,
    }
    nosmt_homo.update(json.loads(json.dumps(mode_info["nosmt_scenario_patch"])))

    # Mixed fleet composite
    mixed_fleet = {
        "composite": {
            "nosmt_pool": {
                "allocation": "below_split",
                "parameter_effects": {"vcpu_demand_multiplier": "weighted_average"},
            },
            "smt_pool": {
                "allocation": "above_split",
                "parameter_effects": {"vcpu_demand_multiplier": 1.0},
            },
        },
        "split_trait": "vcpu_discount",
        "split_point": breakeven if breakeven else 0.85,
    }

    analysis_spec: dict[str, Any] = {
        "type": "compare_sweep",
        "baseline": "smt_baseline",
        "sweep_scenarios": ["nosmt_homogeneous", "mixed_fleet"],
        "sweep_parameter": "split_point",
        "sweep_parameter_label": "vCPU Discount Split Point",
        "sweep_values": SPLIT_SWEEP_VALUES,
        "show_breakeven_marker": True,
        "separate_metric_plots": True,
        "show_plot_title": False,
        "labels": {
            "smt_baseline": f"SMT ({basis['label']}, R={smt_r:.2f})",
            "nosmt_homogeneous": f"No-SMT Homogeneous (R={nosmt_r:.2f}, discount=0.75)",
            "mixed_fleet": f"Mixed Fleet ({basis['label']})",
        },
    }
    if breakeven:
        analysis_spec["x_axis_markers"] = [breakeven]
        analysis_spec["x_axis_marker_labels"] = [f"breakeven ({breakeven:.2f})"]

    description = (
        f"At {util_pct}% utilization: Sweep split point for mixed fleet vs "
        f"homogeneous no-SMT vs SMT baseline. {basis['label']} basis, "
        f"{mode_info['short_label']} resource model. "
        f"SMT R={smt_r:.2f}, No-SMT R={nosmt_r:.2f}."
    )

    return config_path, {
        "name": f"{basis_key}_{mode}_util_{util_pct}pct_sweep",
        "description": description,
        "scenarios": {
            "smt_baseline": smt_baseline,
            "smt_pool": smt_pool,
            "nosmt_pool": nosmt_pool,
            "nosmt_homogeneous": nosmt_homo,
            "mixed_fleet": mixed_fleet,
        },
        "workload": {
            "total_vcpus": 100000,
            "avg_util": util,
            "traits": {
                "vcpu_discount": {
                    "type": "discrete",
                    "bins": TRAIT_BINS,
                },
            },
        },
        "analysis": analysis_spec,
        "processor": {
            "smt": "configs/shared/genoa_processors.jsonc:genoa_smt",
            "nosmt_linear": "configs/shared/genoa_processors.jsonc:genoa_nosmt_linear",
            "nosmt_smt_hw_linear": (
                "configs/shared/genoa_processors.jsonc:genoa_nosmt_smt_hw_linear"
            ),
        },
        "cost": {
            "carbon_intensity_g_kwh": 175,
            "electricity_cost_usd_kwh": 0.28,
            "lifetime_years": 6,
        },
        "power_curve": {"type": "polynomial"},
        "output_dir": output_rel(output_dir),
    }


def generate_configs() -> list[Path]:
    """Generate the full config set."""
    generated: list[Path] = []
    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                path, payload = build_compare_config(basis_key, mode, util)
                write_json(path, payload)
                generated.append(path)

                path, payload = build_sweep_config(basis_key, mode, util)
                write_json(path, payload)
                generated.append(path)
    return generated


def generated_config_paths() -> list[Path]:
    """Return generated configs in execution order (compares before sweeps)."""
    paths: list[Path] = []
    # Run compares first (they resolve auto_breakeven)
    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                util_pct = int(util * 100)
                paths.append(CONFIG_ROOT / basis_key / mode / f"util_{util_pct}_pct_compare.jsonc")
    # Then sweeps
    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                util_pct = int(util * 100)
                paths.append(CONFIG_ROOT / basis_key / mode / f"util_{util_pct}_pct_sweep.jsonc")
    return paths


def run_generated_configs(config_paths: list[Path]) -> None:
    """Run generated configs and write their outputs."""
    for path in config_paths:
        print(f"Running {repo_rel(path)} ...")
        result = run_analysis(path)
        if result.config.output_dir:
            OutputWriter(result.config.output_dir).write(result)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_compare_savings(basis_key: str, mode: str, util: float) -> dict[str, Any]:
    """Extract % change values from a compare result."""
    util_pct = int(util * 100)
    result_path = RESULTS_ROOT / basis_key / mode / f"util_{util_pct}pct_compare" / "results.json"
    result = load_json(result_path)

    scenarios = result["scenario_results"]
    baseline = scenarios["smt_homogeneous"]
    nosmt = scenarios["nosmt_homogeneous"]
    mixed_carbon = scenarios["mixed_fleet_carbon"]
    mixed_tco = scenarios["mixed_fleet_tco"]

    def pct_change(a: float, b: float) -> float:
        return (a - b) / b * 100 if b != 0 else 0.0

    return {
        "basis": basis_key,
        "resource_mode": mode,
        "avg_util_pct": util_pct,
        "smt_r": modeled_ratio(INPUT_BASES[basis_key]["ratios"][util]["smt"]),
        "nosmt_r": modeled_ratio(INPUT_BASES[basis_key]["ratios"][util]["nosmt"]),
        "smt_servers": baseline["num_servers"],
        "nosmt_servers": nosmt["num_servers"],
        "nosmt_carbon_pct": pct_change(nosmt["total_carbon_kg"], baseline["total_carbon_kg"]),
        "nosmt_tco_pct": pct_change(nosmt["total_cost_usd"], baseline["total_cost_usd"]),
        "nosmt_server_pct": pct_change(nosmt["num_servers"], baseline["num_servers"]),
        "carbon_split_point": mixed_carbon.get("auto_resolved_split_point"),
        "tco_split_point": mixed_tco.get("auto_resolved_split_point"),
        "mixed_carbon_pct_at_carbon_split": pct_change(
            mixed_carbon["total_carbon_kg"], baseline["total_carbon_kg"]
        ),
        "mixed_tco_pct_at_carbon_split": pct_change(
            mixed_carbon["total_cost_usd"], baseline["total_cost_usd"]
        ),
        "mixed_server_pct_at_carbon_split": pct_change(
            mixed_carbon["num_servers"], baseline["num_servers"]
        ),
        "mixed_carbon_servers": mixed_carbon["num_servers"],
        "mixed_carbon_nosmt_pool_servers": mixed_carbon.get("sub_results", {}).get("nosmt_pool", {}).get("num_servers", 0),
        "mixed_carbon_smt_pool_servers": mixed_carbon.get("sub_results", {}).get("smt_pool", {}).get("num_servers", 0),
        "mixed_carbon_pct_at_tco_split": pct_change(
            mixed_tco["total_carbon_kg"], baseline["total_carbon_kg"]
        ),
        "mixed_tco_pct_at_tco_split": pct_change(
            mixed_tco["total_cost_usd"], baseline["total_cost_usd"]
        ),
        "mixed_server_pct_at_tco_split": pct_change(
            mixed_tco["num_servers"], baseline["num_servers"]
        ),
        "mixed_tco_servers": mixed_tco["num_servers"],
        "mixed_tco_nosmt_pool_servers": mixed_tco.get("sub_results", {}).get("nosmt_pool", {}).get("num_servers", 0),
        "mixed_tco_smt_pool_servers": mixed_tco.get("sub_results", {}).get("smt_pool", {}).get("num_servers", 0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def fmt_float(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}"


def write_summary_markdown(rows: list[dict[str, Any]]) -> None:
    """Write human-readable markdown summary."""
    lines = [
        "# Mixed Fleet Partitioning Summary",
        "",
        "Comparison at each utilization point: homogeneous SMT (baseline), "
        "homogeneous no-SMT (geomean discount 0.75), mixed fleet with a carbon-selected split, "
        "and mixed fleet with a TCO-selected split.",
        "",
        "% change values are relative to SMT homogeneous baseline.",
        "",
    ]

    for mode in RESOURCE_MODES:
        mode_info = RESOURCE_MODES[mode]
        lines.extend(["", f"## {mode_info['label']}", ""])

        for basis_key in INPUT_BASES:
            basis = INPUT_BASES[basis_key]
            mode_rows = [
                r for r in rows
                if r["basis"] == basis_key and r["resource_mode"] == mode
            ]
            lines.extend([f"### {basis['label']}", ""])
            lines.append(
                "| Util % | SMT R | No-SMT R | Carbon Split | TCO Split | "
                "No-SMT Carbon | Mixed Carbon (C split) | Mixed Carbon (T split) | "
                "No-SMT TCO | Mixed TCO (C split) | Mixed TCO (T split) | "
                "No-SMT Servers | Mixed Servers (C) | Mixed Servers (T) |"
            )
            lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for r in mode_rows:
                lines.append(
                    f"| {r['avg_util_pct']} "
                    f"| {r['smt_r']:.2f} | {r['nosmt_r']:.2f} "
                    f"| {fmt_float(r['carbon_split_point'])} "
                    f"| {fmt_float(r['tco_split_point'])} "
                    f"| {fmt_pct(r['nosmt_carbon_pct'])} "
                    f"| {fmt_pct(r['mixed_carbon_pct_at_carbon_split'])} "
                    f"| {fmt_pct(r['mixed_carbon_pct_at_tco_split'])} "
                    f"| {fmt_pct(r['nosmt_tco_pct'])} "
                    f"| {fmt_pct(r['mixed_tco_pct_at_carbon_split'])} "
                    f"| {fmt_pct(r['mixed_tco_pct_at_tco_split'])} "
                    f"| {fmt_pct(r['nosmt_server_pct'])} "
                    f"| {fmt_pct(r['mixed_server_pct_at_carbon_split'])} "
                    f"| {fmt_pct(r['mixed_server_pct_at_tco_split'])} |"
                )
            lines.append("")

    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def plot_cross_utilization_summary(rows: list[dict[str, Any]]) -> None:
    """Create grouped bar charts showing savings across utilization points."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping summary plots")
        return

    plots_dir = RESULTS_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            mode_rows = [
                r for r in rows
                if r["basis"] == basis_key and r["resource_mode"] == mode
            ]
            mode_rows.sort(key=lambda r: r["avg_util_pct"])
            utils = [r["avg_util_pct"] for r in mode_rows]

            for metric, metric_label in [("carbon", "Carbon"), ("tco", "TCO")]:
                nosmt_vals = [r[f"nosmt_{metric}_pct"] for r in mode_rows]
                mixed_vals = [
                    r["mixed_carbon_pct_at_carbon_split"]
                    if metric == "carbon"
                    else r["mixed_tco_pct_at_tco_split"]
                    for r in mode_rows
                ]

                fig, ax = plt.subplots(figsize=(7, 4.5))
                x = np.arange(len(utils))
                width = 0.32

                bars1 = ax.bar(
                    x - width / 2, nosmt_vals, width,
                    label="No-SMT Homogeneous", color="#2196F3", alpha=0.85,
                )
                bars2 = ax.bar(
                    x + width / 2, mixed_vals, width,
                    label="Mixed Fleet", color="#FF9800", alpha=0.85,
                )

                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        va = "bottom" if height >= 0 else "top"
                        offset = 0.3 if height >= 0 else -0.3
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + offset,
                            f"{height:.1f}%",
                            ha="center", va=va, fontsize=8, fontweight="bold",
                        )

                ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
                ax.set_xlabel("Average Utilization (%)", fontsize=11)
                ax.set_ylabel(f"{metric_label} Change vs SMT Baseline (%)", fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{u}%" for u in utils])
                ax.legend(loc="best", fontsize=9)

                basis_label = INPUT_BASES[basis_key]["label"]
                mode_label = RESOURCE_MODES[mode]["short_label"]
                ax.set_title(
                    f"{metric_label} Savings: {basis_label}, {mode_label}",
                    fontsize=12, fontweight="bold",
                )

                plt.tight_layout()
                fname = f"{basis_key}_{mode}_{metric}_summary.png"
                fig.savefig(plots_dir / fname, dpi=150)
                plt.close(fig)
                print(f"  Saved {fname}")

    # Also create combined cross-basis comparison plots
    for mode in RESOURCE_MODES:
        for metric, metric_label in [("carbon", "Carbon"), ("tco", "TCO")]:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(UTILS))
            n_groups = len(INPUT_BASES) * 2  # 2 per basis (nosmt, mixed)
            total_width = 0.7
            bar_width = total_width / n_groups

            colors = {
                "iso_lp": {"nosmt": "#2196F3", "mixed": "#64B5F6"},
                "iso_physical_core": {"nosmt": "#FF9800", "mixed": "#FFB74D"},
            }

            for i, basis_key in enumerate(INPUT_BASES):
                basis_rows = [
                    r for r in rows
                    if r["basis"] == basis_key and r["resource_mode"] == mode
                ]
                basis_rows.sort(key=lambda r: r["avg_util_pct"])
                basis_label = INPUT_BASES[basis_key]["label"]

                nosmt_vals = [r[f"nosmt_{metric}_pct"] for r in basis_rows]
                mixed_vals = [
                    r["mixed_carbon_pct_at_carbon_split"]
                    if metric == "carbon"
                    else r["mixed_tco_pct_at_tco_split"]
                    for r in basis_rows
                ]

                offset_nosmt = -total_width / 2 + bar_width * (i * 2) + bar_width / 2
                offset_mixed = -total_width / 2 + bar_width * (i * 2 + 1) + bar_width / 2

                ax.bar(
                    x + offset_nosmt, nosmt_vals, bar_width,
                    label=f"{basis_label} No-SMT Homo.",
                    color=colors[basis_key]["nosmt"], alpha=0.85,
                )
                ax.bar(
                    x + offset_mixed, mixed_vals, bar_width,
                    label=f"{basis_label} Mixed Fleet",
                    color=colors[basis_key]["mixed"], alpha=0.85,
                )

            ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
            ax.set_xlabel("Average Utilization (%)", fontsize=11)
            ax.set_ylabel(f"{metric_label} Change vs SMT Baseline (%)", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(u * 100)}%" for u in UTILS])
            ax.legend(loc="best", fontsize=8)

            mode_label = RESOURCE_MODES[mode]["short_label"]
            ax.set_title(
                f"{metric_label} Savings: Basis Comparison ({mode_label})",
                fontsize=12, fontweight="bold",
            )

            plt.tight_layout()
            fname = f"basis_comparison_{mode}_{metric}_summary.png"
            fig.savefig(plots_dir / fname, dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")


def plot_server_breakdown(rows: list[dict[str, Any]]) -> None:
    """Create stacked bar charts showing server type breakdown per scenario."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping server breakdown plots")
        return

    plots_dir = RESULTS_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_views = [
        ("carbon", "mixed_carbon_nosmt_pool_servers", "mixed_carbon_smt_pool_servers", "mixed_carbon_servers"),
        ("tco", "mixed_tco_nosmt_pool_servers", "mixed_tco_smt_pool_servers", "mixed_tco_servers"),
    ]

    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            mode_rows = [
                r for r in rows
                if r["basis"] == basis_key and r["resource_mode"] == mode
            ]
            mode_rows.sort(key=lambda r: r["avg_util_pct"])
            utils = [r["avg_util_pct"] for r in mode_rows]

            for metric, nosmt_key, smt_key, total_key in metric_views:
                fig, ax = plt.subplots(figsize=(8, 5))
                n_scenarios = 3  # SMT homo, no-SMT homo, mixed fleet
                x = np.arange(len(utils))
                total_width = 0.75
                bar_width = total_width / n_scenarios

                for r in mode_rows:
                    util_idx = utils.index(r["avg_util_pct"])

                    ax.bar(
                        util_idx - bar_width,
                        r["smt_servers"], bar_width,
                        color="#5C6BC0", alpha=0.85,
                        label="SMT Homogeneous" if util_idx == 0 else None,
                    )

                    ax.bar(
                        util_idx,
                        r["nosmt_servers"], bar_width,
                        color="#26A69A", alpha=0.85,
                        label="No-SMT Homogeneous" if util_idx == 0 else None,
                    )

                    nosmt_pool = r[nosmt_key]
                    smt_pool = r[smt_key]
                    ax.bar(
                        util_idx + bar_width,
                        nosmt_pool, bar_width,
                        color="#FF7043", alpha=0.85,
                        label="Mixed: No-SMT Pool" if util_idx == 0 else None,
                    )
                    ax.bar(
                        util_idx + bar_width,
                        smt_pool, bar_width,
                        bottom=nosmt_pool,
                        color="#FFA726", alpha=0.85,
                        label="Mixed: SMT Pool" if util_idx == 0 else None,
                    )

                    for offset, count in [
                        (-bar_width, r["smt_servers"]),
                        (0, r["nosmt_servers"]),
                        (bar_width, r[total_key]),
                    ]:
                        ax.text(
                            util_idx + offset, count + 5,
                            str(count), ha="center", va="bottom",
                            fontsize=8, fontweight="bold",
                        )

                    if nosmt_pool + smt_pool > 0:
                        if nosmt_pool > 30:
                            ax.text(
                                util_idx + bar_width, nosmt_pool / 2,
                                str(nosmt_pool), ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold",
                            )
                        if smt_pool > 30:
                            ax.text(
                                util_idx + bar_width, nosmt_pool + smt_pool / 2,
                                str(smt_pool), ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold",
                            )

                ax.set_xlabel("Average Utilization (%)", fontsize=11)
                ax.set_ylabel("Server Count", fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{u}%" for u in utils])
                ax.legend(loc="upper left", fontsize=8)

                basis_label = INPUT_BASES[basis_key]["label"]
                mode_label = RESOURCE_MODES[mode]["short_label"]
                ax.set_title(
                    f"Server Breakdown ({metric.upper()} split): {basis_label}, {mode_label}",
                    fontsize=12, fontweight="bold",
                )

                plt.tight_layout()
                fname = f"{basis_key}_{mode}_server_breakdown_{metric}.png"
                fig.savefig(plots_dir / fname, dpi=150)
                plt.close(fig)
                print(f"  Saved {fname}")


def write_summary_files() -> None:
    """Write consolidated result summaries and plots."""
    rows: list[dict[str, Any]] = []
    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                try:
                    rows.append(extract_compare_savings(basis_key, mode, util))
                except FileNotFoundError as e:
                    print(f"  Warning: missing result for {basis_key}/{mode}/{int(util*100)}%: {e}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_ROOT / "summary.csv", rows)
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_summary_markdown(rows)
    plot_cross_utilization_summary(rows)
    plot_server_breakdown(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only write configs; do not run analyses or refresh summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = generate_configs()
    print(f"Generated {len(generated)} configs under {repo_rel(CONFIG_ROOT)}")

    if args.generate_only:
        return

    run_generated_configs(generated_config_paths())
    write_summary_files()
    print(f"\nResults written to {repo_rel(RESULTS_ROOT)}")


if __name__ == "__main__":
    main()
