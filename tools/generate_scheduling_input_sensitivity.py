#!/usr/bin/env python3
"""Generate and optionally run scheduling-input sensitivity analyses.

This script adds a separate analysis branch for comparing three scheduling-input
bases:

1. ``legacy_existing``: the currently checked-in model calibration
2. ``iso_lp``: updated interpolated Go-CPU rates from the 8 LP / 2 VM baseline
3. ``iso_physical_core``: updated interpolated Go-CPU rates from the
   pool-adjusted 16 LP / 4 VM SMT regime

It writes new configs under:

    configs/oversub_analysis/genoa/scheduling_input_sensitivity/

and results under:

    results/oversub_analysis/genoa/scheduling_input_sensitivity/

By default it also runs the generated configs and emits consolidated CSV/JSON/MD
summaries that are easier to cite from the analysis docs.
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
CONFIG_ROOT = REPO_ROOT / "configs" / "oversub_analysis" / "genoa" / "scheduling_input_sensitivity"
RESULTS_ROOT = REPO_ROOT / "results" / "oversub_analysis" / "genoa" / "scheduling_input_sensitivity"

UTILS = [0.10, 0.20, 0.30]
SWEEP_VALUES = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    1.05, 1.10, 1.15, 1.20,
]
MARKER_VALUES = [1.00, 0.75]
MARKER_LABELS = ["no_discount", "geomean"]

INPUT_BASES: dict[str, dict[str, Any]] = {
    "iso_lp": {
        "label": "Iso-LP (8 LP / 2 VM)",
        "description": (
            "Interpolated Go-CPU max-safe VP/LP rates from the baseline 8 LP / 2 VM "
            "VP-constrained SMT regime versus No-SMT."
        ),
        "source_doc": (
            "/Users/jaylenw/Code/atf-benchmarking/scripts/docs/"
            "scheduling_constraints_smt_iso_physical_core.md"
        ),
        "ratios": {
            0.10: {"smt": 2.59, "nosmt": 5.58},
            0.20: {"smt": 1.29, "nosmt": 2.79},
            0.30: {"smt": 0.86, "nosmt": 1.86},
        },
    },
    "iso_physical_core": {
        "label": "Iso-Physical-Core (16 LP / 4 VM SMT)",
        "description": (
            "Interpolated Go-CPU max-safe VP/LP rates from the pool-adjusted "
            "16 LP / 4 VM SMT regime on the same physical-core budget, versus No-SMT."
        ),
        "source_doc": (
            "/Users/jaylenw/Code/atf-benchmarking/scripts/docs/"
            "scheduling_constraints_smt_iso_physical_core.md"
        ),
        "ratios": {
            0.10: {"smt": 3.32, "nosmt": 5.58},
            0.20: {"smt": 1.66, "nosmt": 2.79},
            0.30: {"smt": 1.11, "nosmt": 1.86},
        },
    },
}

LEGACY_EXISTING_INPUTS: dict[str, dict[float, dict[str, float]]] = {
    "baseline": {
        0.10: {"smt": 2.34, "nosmt": 5.00},
        0.20: {"smt": 1.33, "nosmt": 2.34},
        0.30: {"smt": 1.14, "nosmt": 1.60},
    },
    "resource_scaling": {
        0.10: {"smt": 2.34, "nosmt": 5.00},
        0.20: {"smt": 1.33, "nosmt": 2.34},
        0.30: {"smt": 1.14, "nosmt": 1.60},
    },
    "resource_constraints": {
        0.10: {"smt": 3.00, "nosmt": 5.00},
        0.20: {"smt": 1.33, "nosmt": 2.34},
        0.30: {"smt": 1.00, "nosmt": 1.67},
    },
}

RESOURCE_MODES: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Baseline (No Resource Model)",
        "description_suffix": "no extra resource scaling or fixed-capacity constraints",
        "nosmt_processor_alias": "nosmt_linear",
        "processor_key": "nosmt_linear",
        "scenario_patch": {},
    },
    "resource_scaling": {
        "label": "Purpose-Built No-SMT (Resource Scaling)",
        "description_suffix": "scaled memory/SSD that grow with packed vCPU count",
        "nosmt_processor_alias": "nosmt_linear",
        "processor_key": "nosmt_linear",
        "scenario_patch": {
            "resource_scaling": {"scale_with_vcpus": ["memory", "ssd"]},
        },
    },
    "resource_constraints": {
        "label": "Same Hardware SMT-Off (Resource Constraints)",
        "description_suffix": "fixed SMT-hardware memory/SSD capacities shared by both modes",
        "nosmt_processor_alias": "nosmt_smt_hw_linear",
        "processor_key": "nosmt_smt_hw_linear",
        "scenario_patch": {
            "resource_constraints": {
                "memory_gb": {
                    "capacity_per_thread": None,
                    "demand_per_vcpu": 4.0,
                },
                "ssd_gb": {
                    "capacity_per_thread": None,
                    "demand_per_vcpu": 50.0,
                },
            },
        },
    },
}

LEGACY_RESULT_PATHS: dict[str, dict[float, Path]] = {
    "baseline": {
        0.10: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_10pct_linear/results.json",
        0.20: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_20pct_linear/results.json",
        0.30: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_30pct_linear/results.json",
    },
    "resource_scaling": {
        0.10: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_10pct_linear_resource_scaling/results.json",
        0.20: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_20pct_linear_resource_scaling/results.json",
        0.30: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_30pct_linear_resource_scaling/results.json",
    },
    "resource_constraints": {
        0.10: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_10pct_linear_resource_constraints/results.json",
        0.20: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_20pct_linear_resource_constraints/results.json",
        0.30: REPO_ROOT / "results/oversub_analysis/genoa/linear/util_30pct_linear_resource_constraints/results.json",
    },
}


def repo_rel(path: Path) -> str:
    """Return a repo-relative posix path."""
    return path.relative_to(REPO_ROOT).as_posix()


def output_rel(path: Path) -> str:
    """Return an output_dir string rooted at the repo."""
    return f"./{repo_rel(path)}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def format_pct(value: float | None) -> str:
    """Format percent-like scalar for markdown tables."""
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_ratio(value: float) -> str:
    """Format oversubscription ratio."""
    return f"{value:.2f}"


def modeled_ratio(value: float) -> float:
    """Cap inferred safe rates below 1.0 to no-oversubscription."""
    return max(1.0, value)


def patch_for_mode(mode: str, processor: str) -> dict[str, Any]:
    """Build per-scenario resource-mode patch."""
    patch = json.loads(json.dumps(RESOURCE_MODES[mode]["scenario_patch"]))
    if mode != "resource_constraints":
        return patch

    # Same physical server: 768 GB memory, 12 TB SSD.
    if processor == "smt":
        memory_per_thread = 4.8
        ssd_per_thread = 75.0
    else:
        memory_per_thread = 9.6
        ssd_per_thread = 150.0

    patch["resource_constraints"]["memory_gb"]["capacity_per_thread"] = memory_per_thread
    patch["resource_constraints"]["ssd_gb"]["capacity_per_thread"] = ssd_per_thread
    return patch


def build_compare_sweep_config(
    basis_key: str,
    util: float,
    mode: str,
) -> tuple[Path, dict[str, Any]]:
    """Build one generated compare_sweep config."""
    basis = INPUT_BASES[basis_key]
    raw_ratios = basis["ratios"][util]
    ratios = {
        "smt": modeled_ratio(raw_ratios["smt"]),
        "nosmt": modeled_ratio(raw_ratios["nosmt"]),
    }
    util_pct = int(util * 100)

    config_path = (
        CONFIG_ROOT
        / basis_key
        / mode
        / f"util_{util_pct}_pct.jsonc"
    )
    output_dir = (
        RESULTS_ROOT
        / basis_key
        / mode
        / f"util_{util_pct}pct"
    )

    smt_scenario = {
        "processor": "smt",
        "oversub_ratio": ratios["smt"],
        "util_overhead": 0.0,
    }
    smt_scenario.update(patch_for_mode(mode, "smt"))

    nosmt_alias = RESOURCE_MODES[mode]["nosmt_processor_alias"]
    nosmt_scenario = {
        "processor": nosmt_alias,
        "oversub_ratio": ratios["nosmt"],
        "vcpu_demand_multiplier": 1.0,
    }
    nosmt_scenario.update(patch_for_mode(mode, "nosmt"))

    description = (
        f"At {util_pct}% utilization: {basis['label']} scheduling-input calibration "
        f"with SMT R={ratios['smt']:.2f} and No-SMT R={ratios['nosmt']:.2f}, using "
        f"{RESOURCE_MODES[mode]['description_suffix']}."
    )
    if raw_ratios != ratios:
        description += (
            " Raw interpolated rates below 1.0 are capped to 1.0 because a "
            "safe oversubscription ratio below 1 implies no oversubscription."
        )

    return config_path, {
        "name": f"{basis_key}_{mode}_util_{util_pct}pct",
        "description": description,
        "scenarios": {
            "smt_oversub": smt_scenario,
            "nosmt_oversub": nosmt_scenario,
        },
        "workload": {
            "total_vcpus": 100000,
            "avg_util": util,
        },
        "analysis": {
            "type": "compare_sweep",
            "baseline": "smt_oversub",
            "sweep_scenario": "nosmt_oversub",
            "sweep_parameter": "vcpu_demand_multiplier",
            "sweep_parameter_label": "vCPU Demand Discount",
            "sweep_values": SWEEP_VALUES,
            "show_breakeven_marker": True,
            "separate_metric_plots": True,
            "show_plot_title": False,
            "x_axis_markers": [0.65, 0.75, 0.85],
            "x_axis_marker_labels": ["low", "geomean", "high"],
            "labels": {
                "smt_oversub": f"SMT ({basis['label']}, R={ratios['smt']:.2f})",
                "nosmt_oversub": f"No-SMT ({basis['label']}, R={ratios['nosmt']:.2f})",
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
        "power_curve": {
            "type": "polynomial",
        },
        "output_dir": output_rel(output_dir),
    }


def build_savings_curve_config(mode: str) -> tuple[Path, dict[str, Any]]:
    """Build a basis-comparison savings curve config."""
    config_path = CONFIG_ROOT / f"basis_comparison_{mode}_savings_curve.jsonc"
    output_dir = RESULTS_ROOT / f"basis_comparison_{mode}_savings_curve"

    config_sets = []
    for basis_key, basis in INPUT_BASES.items():
        config_sets.append(
            {
                "label": basis["label"],
                "configs": [
                    repo_rel(CONFIG_ROOT / basis_key / mode / f"util_{int(util * 100)}_pct.jsonc")
                    for util in UTILS
                ],
            }
        )

    return config_path, {
        "name": f"basis_comparison_{mode}_savings_curve",
        "description": (
            f"Compare SMT-vs-no-SMT savings across scheduling-input bases for the "
            f"{RESOURCE_MODES[mode]['label']} deployment model."
        ),
        "analysis": {
            "type": "savings_curve",
            "config_sets": config_sets,
            "x_parameter": "workload.avg_util",
            "x_display_multiplier": 100,
            "x_label": "Average Utilization (%)",
            "y_label": "Savings vs SMT Baseline (%)",
            "marker_values": MARKER_VALUES,
            "marker_labels": MARKER_LABELS,
            "metrics": ["carbon", "tco"],
            "legend_title": "Scheduling Input Basis",
            "plot": {
                "figsize": [8, 5],
            },
        },
        "output_dir": output_rel(output_dir),
    }


def build_breakeven_curve_config(mode: str) -> tuple[Path, dict[str, Any]]:
    """Build a basis-comparison breakeven curve config."""
    config_path = CONFIG_ROOT / f"basis_comparison_{mode}_breakeven_curve.jsonc"
    output_dir = RESULTS_ROOT / f"basis_comparison_{mode}_breakeven_curve"

    series = []
    for basis_key, basis in INPUT_BASES.items():
        series.append(
            {
                "label": basis["label"],
                "configs": [
                    repo_rel(CONFIG_ROOT / basis_key / mode / f"util_{int(util * 100)}_pct.jsonc")
                    for util in UTILS
                ],
            }
        )

    return config_path, {
        "name": f"basis_comparison_{mode}_breakeven_curve",
        "description": (
            f"Compare carbon breakeven multipliers across scheduling-input bases for the "
            f"{RESOURCE_MODES[mode]['label']} deployment model."
        ),
        "analysis": {
            "type": "breakeven_curve",
            "series": series,
            "x_parameter": "workload.avg_util",
            "x_display_multiplier": 100,
            "breakeven_metric": "carbon",
            "x_label": "Average Utilization (%)",
            "y_label": "Breakeven vCPU Demand Multiplier",
            "y_axis_markers": [0.65, 0.75, 0.85],
            "y_axis_marker_labels": ["low", "geomean", "high"],
        },
        "output_dir": output_rel(output_dir),
    }


def generated_config_paths() -> list[Path]:
    """Return generated configs in execution order."""
    paths: list[Path] = []
    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                paths.append(CONFIG_ROOT / basis_key / mode / f"util_{int(util * 100)}_pct.jsonc")
    for mode in RESOURCE_MODES:
        paths.append(CONFIG_ROOT / f"basis_comparison_{mode}_savings_curve.jsonc")
        paths.append(CONFIG_ROOT / f"basis_comparison_{mode}_breakeven_curve.jsonc")
    return paths


def generate_configs() -> list[Path]:
    """Generate the full config set."""
    generated: list[Path] = []

    for basis_key in INPUT_BASES:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                path, payload = build_compare_sweep_config(basis_key, util, mode)
                write_json(path, payload)
                generated.append(path)

    for mode in RESOURCE_MODES:
        path, payload = build_savings_curve_config(mode)
        write_json(path, payload)
        generated.append(path)

        path, payload = build_breakeven_curve_config(mode)
        write_json(path, payload)
        generated.append(path)

    return generated


def extract_marker(compare_sweep_results: list[dict[str, Any]], value: float) -> dict[str, Any]:
    """Return the exact sweep entry for a marker value."""
    for entry in compare_sweep_results:
        if abs(entry["parameter_value"] - value) < 1e-9:
            return entry
    raise KeyError(f"Missing marker value {value}")


def find_breakeven(compare_sweep_results: list[dict[str, Any]], metric: str) -> float | None:
    """Linearly interpolate the zero crossing for one compare_sweep metric."""
    metric_key = f"{metric}_diff_pct"
    for idx in range(len(compare_sweep_results) - 1):
        left = compare_sweep_results[idx]
        right = compare_sweep_results[idx + 1]
        y1 = left[metric_key]
        y2 = right[metric_key]
        if (y1 <= 0 <= y2) or (y2 <= 0 <= y1):
            x1 = left["parameter_value"]
            x2 = right["parameter_value"]
            if y1 == y2:
                return x1
            return x1 + (-y1 / (y2 - y1)) * (x2 - x1)
    return None


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk."""
    return json.loads(path.read_text())


def legacy_summary_row(mode: str, util: float) -> dict[str, Any]:
    """Build one summary row from an existing checked-in result."""
    result = load_json(LEGACY_RESULT_PATHS[mode][util])
    compare = result["compare_sweep_results"]
    at_no_discount = extract_marker(compare, 1.00)
    at_geomean = extract_marker(compare, 0.75)
    return {
        "basis": "legacy_existing",
        "basis_label": "Legacy Existing Repo Configs",
        "resource_mode": mode,
        "resource_mode_label": RESOURCE_MODES[mode]["label"],
        "avg_util_pct": int(util * 100),
        "smt_r": LEGACY_EXISTING_INPUTS[mode][util]["smt"],
        "nosmt_r": LEGACY_EXISTING_INPUTS[mode][util]["nosmt"],
        "carbon_diff_pct_at_1_00": at_no_discount["carbon_diff_pct"],
        "tco_diff_pct_at_1_00": at_no_discount["tco_diff_pct"],
        "server_diff_pct_at_1_00": at_no_discount["server_diff_pct"],
        "carbon_diff_pct_at_0_75": at_geomean["carbon_diff_pct"],
        "tco_diff_pct_at_0_75": at_geomean["tco_diff_pct"],
        "server_diff_pct_at_0_75": at_geomean["server_diff_pct"],
        "carbon_breakeven_multiplier": find_breakeven(compare, "carbon"),
        "tco_breakeven_multiplier": find_breakeven(compare, "tco"),
    }


def generated_summary_row(basis_key: str, mode: str, util: float) -> dict[str, Any]:
    """Build one summary row from a generated result."""
    result_path = RESULTS_ROOT / basis_key / mode / f"util_{int(util * 100)}pct" / "results.json"
    result = load_json(result_path)
    compare = result["compare_sweep_results"]
    at_no_discount = extract_marker(compare, 1.00)
    at_geomean = extract_marker(compare, 0.75)
    raw_ratios = INPUT_BASES[basis_key]["ratios"][util]
    ratios = {
        "smt": modeled_ratio(raw_ratios["smt"]),
        "nosmt": modeled_ratio(raw_ratios["nosmt"]),
    }
    return {
        "basis": basis_key,
        "basis_label": INPUT_BASES[basis_key]["label"],
        "resource_mode": mode,
        "resource_mode_label": RESOURCE_MODES[mode]["label"],
        "avg_util_pct": int(util * 100),
        "smt_r": ratios["smt"],
        "nosmt_r": ratios["nosmt"],
        "carbon_diff_pct_at_1_00": at_no_discount["carbon_diff_pct"],
        "tco_diff_pct_at_1_00": at_no_discount["tco_diff_pct"],
        "server_diff_pct_at_1_00": at_no_discount["server_diff_pct"],
        "carbon_diff_pct_at_0_75": at_geomean["carbon_diff_pct"],
        "tco_diff_pct_at_0_75": at_geomean["tco_diff_pct"],
        "server_diff_pct_at_0_75": at_geomean["server_diff_pct"],
        "carbon_breakeven_multiplier": find_breakeven(compare, "carbon"),
        "tco_breakeven_multiplier": find_breakeven(compare, "tco"),
    }


def summary_rows() -> list[dict[str, Any]]:
    """Build all consolidated summary rows."""
    rows: list[dict[str, Any]] = []
    for mode in RESOURCE_MODES:
        for util in UTILS:
            rows.append(legacy_summary_row(mode, util))
        for basis_key in INPUT_BASES:
            for util in UTILS:
                rows.append(generated_summary_row(basis_key, mode, util))
    return rows


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build pairwise deltas for the two comparison steps of interest."""
    index = {
        (row["basis"], row["resource_mode"], row["avg_util_pct"]): row
        for row in rows
    }

    deltas: list[dict[str, Any]] = []
    comparisons = [
        ("legacy_existing", "iso_lp", "Updated 8LP Input vs Legacy Existing"),
        ("iso_lp", "iso_physical_core", "Iso-Physical-Core vs Updated 8LP"),
    ]
    metrics = [
        "carbon_diff_pct_at_1_00",
        "tco_diff_pct_at_1_00",
        "carbon_diff_pct_at_0_75",
        "tco_diff_pct_at_0_75",
        "carbon_breakeven_multiplier",
        "tco_breakeven_multiplier",
    ]

    for left_basis, right_basis, label in comparisons:
        for mode in RESOURCE_MODES:
            for util in UTILS:
                left = index[(left_basis, mode, int(util * 100))]
                right = index[(right_basis, mode, int(util * 100))]
                row = {
                    "comparison": label,
                    "left_basis": left_basis,
                    "right_basis": right_basis,
                    "resource_mode": mode,
                    "avg_util_pct": int(util * 100),
                    "left_smt_r": left["smt_r"],
                    "right_smt_r": right["smt_r"],
                    "nosmt_r": right["nosmt_r"],
                }
                for metric in metrics:
                    left_value = left[metric]
                    right_value = right[metric]
                    if left_value is None or right_value is None:
                        row[f"delta_{metric}"] = None
                    else:
                        row[f"delta_{metric}"] = right_value - left_value
                deltas.append(row)

    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with stable field ordering."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table_for_rows(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    """Render a simple markdown table."""
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key]
            if isinstance(value, float):
                if "breakeven" in key:
                    values.append(format_pct(value))
                elif key.endswith("_r"):
                    values.append(format_ratio(value))
                else:
                    values.append(format_pct(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_summary_markdown(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    """Write a human-readable markdown summary."""
    lines = [
        "# Scheduling Input Sensitivity Summary",
        "",
        "This directory is the additive follow-up analysis for the SMT vs no-SMT "
        "scheduling-input basis question.",
        "",
        "It compares three input sets:",
        "",
        "- `legacy_existing`: the current checked-in repo calibration",
        "- `iso_lp`: updated interpolated Go-CPU rates from the 8 LP / 2 VM baseline",
        "- `iso_physical_core`: updated interpolated Go-CPU rates from the 16 LP / 4 VM "
        "pool-adjusted SMT regime",
        "",
        "## Input Ratios",
        "",
    ]

    input_rows: list[dict[str, Any]] = []
    for mode in RESOURCE_MODES:
        for util in UTILS:
            input_rows.append(
                {
                    "resource_mode": mode,
                    "avg_util_pct": int(util * 100),
                    "legacy_smt_r": LEGACY_EXISTING_INPUTS[mode][util]["smt"],
                    "legacy_nosmt_r": LEGACY_EXISTING_INPUTS[mode][util]["nosmt"],
                    "iso_lp_smt_r": modeled_ratio(INPUT_BASES["iso_lp"]["ratios"][util]["smt"]),
                    "iso_lp_nosmt_r": modeled_ratio(INPUT_BASES["iso_lp"]["ratios"][util]["nosmt"]),
                    "iso_physical_smt_r": modeled_ratio(INPUT_BASES["iso_physical_core"]["ratios"][util]["smt"]),
                    "iso_physical_nosmt_r": modeled_ratio(INPUT_BASES["iso_physical_core"]["ratios"][util]["nosmt"]),
                }
            )

    lines.extend(
        table_for_rows(
            input_rows,
            [
                ("Mode", "resource_mode"),
                ("Util %", "avg_util_pct"),
                ("Legacy SMT R", "legacy_smt_r"),
                ("Legacy No-SMT R", "legacy_nosmt_r"),
                ("Iso-LP SMT R", "iso_lp_smt_r"),
                ("Iso-LP No-SMT R", "iso_lp_nosmt_r"),
                ("Iso-Phys SMT R", "iso_physical_smt_r"),
                ("Iso-Phys No-SMT R", "iso_physical_nosmt_r"),
            ],
        )
    )
    lines.extend([
        "",
        "Modeled-R note: if an interpolated safe VP/LP rate falls below `1.0`, the model caps it to `1.0`.",
        "A value below `1.0` would imply under-subscription; operationally you would stop at no oversubscription instead.",
    ])

    for mode in RESOURCE_MODES:
        lines.extend(["", f"## {RESOURCE_MODES[mode]['label']}", ""])
        mode_rows = [row for row in rows if row["resource_mode"] == mode]
        lines.extend(
            table_for_rows(
                mode_rows,
                [
                    ("Basis", "basis"),
                    ("Util %", "avg_util_pct"),
                    ("SMT R", "smt_r"),
                    ("No-SMT R", "nosmt_r"),
                    ("Carbon @1.00", "carbon_diff_pct_at_1_00"),
                    ("TCO @1.00", "tco_diff_pct_at_1_00"),
                    ("Carbon @0.75", "carbon_diff_pct_at_0_75"),
                    ("TCO @0.75", "tco_diff_pct_at_0_75"),
                    ("Carbon Breakeven", "carbon_breakeven_multiplier"),
                ],
            )
        )

        lines.extend(["", f"### Delta Summary: {RESOURCE_MODES[mode]['label']}", ""])
        mode_deltas = [row for row in deltas if row["resource_mode"] == mode]
        lines.extend(
            table_for_rows(
                mode_deltas,
                [
                    ("Comparison", "comparison"),
                    ("Util %", "avg_util_pct"),
                    ("d Carbon @1.00", "delta_carbon_diff_pct_at_1_00"),
                    ("d TCO @1.00", "delta_tco_diff_pct_at_1_00"),
                    ("d Carbon @0.75", "delta_carbon_diff_pct_at_0_75"),
                    ("d TCO @0.75", "delta_tco_diff_pct_at_0_75"),
                    ("d Carbon Breakeven", "delta_carbon_breakeven_multiplier"),
                ],
            )
        )

    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def write_summary_files() -> None:
    """Write consolidated result summaries."""
    rows = summary_rows()
    deltas = build_delta_rows(rows)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_ROOT / "basis_summary.csv", rows)
    write_csv(RESULTS_ROOT / "basis_delta_summary.csv", deltas)
    (RESULTS_ROOT / "basis_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RESULTS_ROOT / "basis_delta_summary.json").write_text(json.dumps(deltas, indent=2) + "\n")
    write_summary_markdown(rows, deltas)


def run_generated_configs(config_paths: list[Path]) -> None:
    """Run generated configs and write their outputs."""
    for path in config_paths:
        result = run_analysis(path)
        if result.config.output_dir:
            OutputWriter(result.config.output_dir).write(result)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only write configs; do not run analyses or refresh summaries.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    generated = generate_configs()
    if args.generate_only:
        return

    run_generated_configs(generated_config_paths())
    write_summary_files()


if __name__ == "__main__":
    main()
