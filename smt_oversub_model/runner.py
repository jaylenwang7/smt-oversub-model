"""
Experiment runner for executing configs and producing results.

Orchestrates config -> model evaluation -> structured output.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from .config import ExperimentConfig, validate_config
from .model import (
    OverssubModel, PowerCurve, ProcessorConfig,
    ScenarioParams, WorkloadParams, CostParams, ScenarioResult
)


VERSION = "1.0.0"


def _scenario_result_to_dict(result: ScenarioResult) -> dict:
    """Convert ScenarioResult to JSON-serializable dict."""
    return asdict(result)


def _format_timestamp() -> str:
    """Return ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SingleRunResult:
    """Result from a single (non-sweep) experiment run."""
    breakeven_carbon: Optional[float]
    breakeven_tco: Optional[float]
    scenarios: Dict[str, dict]  # baseline, smt_oversub, nosmt_breakeven
    savings: Dict[str, float]   # Relative metrics


@dataclass
class SweepPointResult:
    """Result for a single point in a parameter sweep."""
    parameter_value: float
    breakeven_carbon: Optional[float]
    breakeven_tco: Optional[float]
    scenarios: Dict[str, dict]
    savings: Dict[str, float]


@dataclass
class RunResult:
    """
    Complete result from running an experiment.

    Contains metadata, echoed config, and results.
    """
    meta: Dict[str, Any]
    config: dict
    # For single runs:
    results: Optional[SingleRunResult] = None
    # For sweeps:
    sweep_results: Optional[List[SweepPointResult]] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {
            "meta": self.meta,
            "config": self.config,
        }
        if self.results:
            d["results"] = {
                "breakeven_carbon": self.results.breakeven_carbon,
                "breakeven_tco": self.results.breakeven_tco,
                "scenarios": self.results.scenarios,
                "savings": self.results.savings,
            }
        if self.sweep_results:
            d["sweep_results"] = [
                {
                    "parameter_value": p.parameter_value,
                    "breakeven_carbon": p.breakeven_carbon,
                    "breakeven_tco": p.breakeven_tco,
                    "scenarios": p.scenarios,
                    "savings": p.savings,
                }
                for p in self.sweep_results
            ]
        return d


class Runner:
    """
    Experiment runner that executes configs and produces structured results.

    Example:
        config = load_config("configs/my_experiment.json")
        runner = Runner(config)
        result = runner.run()
        save_result(result, "results/my_experiment_2026-01-08.json")
    """

    def __init__(self, config: ExperimentConfig, config_path: Optional[str] = None):
        """
        Initialize runner with experiment config.

        Args:
            config: Experiment configuration
            config_path: Optional path to config file (for metadata)
        """
        self.config = config
        self.config_path = config_path

        # Validate config
        errors = validate_config(config)
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

    def _build_model_components(self):
        """Build model objects from config."""
        cfg = self.config
        curve_fn = cfg.power_curve.to_callable()

        # Build processor configs from named processors
        smt_spec = cfg.processor.smt
        nosmt_spec = cfg.processor.nosmt

        # SMT processor
        smt_power = PowerCurve(
            p_idle=smt_spec.power_idle_w,
            p_max=smt_spec.power_max_w,
            curve_fn=curve_fn,
        )
        smt_proc = ProcessorConfig(
            physical_cores=smt_spec.physical_cores,
            threads_per_core=smt_spec.threads_per_core,
            power_curve=smt_power,
            thread_overhead=smt_spec.thread_overhead,
        )

        # Non-SMT processor
        nosmt_power = PowerCurve(
            p_idle=nosmt_spec.power_idle_w,
            p_max=nosmt_spec.power_max_w,
            curve_fn=curve_fn,
        )
        nosmt_proc = ProcessorConfig(
            physical_cores=nosmt_spec.physical_cores,
            threads_per_core=nosmt_spec.threads_per_core,
            power_curve=nosmt_power,
            thread_overhead=nosmt_spec.thread_overhead,
        )

        # Workload
        workload = WorkloadParams(
            total_vcpus=cfg.workload.total_vcpus,
            avg_util=cfg.workload.avg_util,
        )

        # Cost
        cost = CostParams(
            embodied_carbon_kg=cfg.cost.embodied_carbon_kg,
            server_cost_usd=cfg.cost.server_cost_usd,
            carbon_intensity_g_kwh=cfg.cost.carbon_intensity_g_kwh,
            electricity_cost_usd_kwh=cfg.cost.electricity_cost_usd_kwh,
            lifetime_hours=cfg.cost.lifetime_years * 8760,
        )

        return smt_proc, nosmt_proc, workload, cost

    def _compute_single(self, smt_proc, nosmt_proc, workload, cost) -> SingleRunResult:
        """Compute results for a single (non-sweep) run."""
        cfg = self.config
        model = OverssubModel(workload, cost)

        # Baseline: SMT, no oversub
        baseline_scenario = ScenarioParams(smt_proc, oversub_ratio=1.0, util_overhead=0.0)
        baseline_result = model.evaluate_scenario(baseline_scenario)

        # SMT with oversub
        smt_oversub_scenario = ScenarioParams(
            smt_proc,
            oversub_ratio=cfg.oversubscription.smt_ratio,
            util_overhead=cfg.oversubscription.smt_util_overhead,
        )
        smt_result = model.evaluate_scenario(smt_oversub_scenario)

        # Find breakeven for non-SMT
        breakeven_carbon = model.find_breakeven_oversub(
            smt_result, nosmt_proc, cfg.oversubscription.nosmt_util_overhead, metric='carbon'
        )
        breakeven_tco = model.find_breakeven_oversub(
            smt_result, nosmt_proc, cfg.oversubscription.nosmt_util_overhead, metric='tco'
        )

        # Evaluate non-SMT at breakeven (use carbon breakeven if available)
        nosmt_breakeven_result = None
        if breakeven_carbon:
            nosmt_scenario = ScenarioParams(
                nosmt_proc, breakeven_carbon, cfg.oversubscription.nosmt_util_overhead
            )
            nosmt_breakeven_result = model.evaluate_scenario(nosmt_scenario)

        scenarios = {
            "baseline": _scenario_result_to_dict(baseline_result),
            "smt_oversub": _scenario_result_to_dict(smt_result),
        }
        if nosmt_breakeven_result:
            scenarios["nosmt_breakeven"] = _scenario_result_to_dict(nosmt_breakeven_result)

        # Compute savings
        savings = {
            "smt_carbon_savings_vs_baseline_pct": (
                (1 - smt_result.total_carbon_kg / baseline_result.total_carbon_kg) * 100
            ),
            "smt_tco_savings_vs_baseline_pct": (
                (1 - smt_result.total_cost_usd / baseline_result.total_cost_usd) * 100
            ),
            "smt_server_reduction_pct": (
                (1 - smt_result.num_servers / baseline_result.num_servers) * 100
            ),
        }

        return SingleRunResult(
            breakeven_carbon=breakeven_carbon,
            breakeven_tco=breakeven_tco,
            scenarios=scenarios,
            savings=savings,
        )

    def _apply_sweep_value(self, param: str, value: float):
        """Apply a sweep parameter value to the config (modifies in place, returns old value)."""
        cfg = self.config

        # Map parameter names to config locations
        param_map = {
            "avg_util": ("workload", "avg_util"),
            "total_vcpus": ("workload", "total_vcpus"),
            "smt_ratio": ("oversubscription", "smt_ratio"),
            "smt_util_overhead": ("oversubscription", "smt_util_overhead"),
            "nosmt_util_overhead": ("oversubscription", "nosmt_util_overhead"),
            "embodied_carbon_kg": ("cost", "embodied_carbon_kg"),
            "server_cost_usd": ("cost", "server_cost_usd"),
            "carbon_intensity_g_kwh": ("cost", "carbon_intensity_g_kwh"),
            "electricity_cost_usd_kwh": ("cost", "electricity_cost_usd_kwh"),
            "lifetime_years": ("cost", "lifetime_years"),
            "nosmt_physical_cores": ("processor.nosmt", "physical_cores"),
            "nosmt_power_ratio": ("processor.nosmt", "power_ratio"),
            "nosmt_idle_ratio": ("processor.nosmt", "idle_ratio"),
            "smt_physical_cores": ("processor.smt", "physical_cores"),
            "smt_p_idle": ("processor.smt", "power_idle_w"),
            "smt_p_max": ("processor.smt", "power_max_w"),
        }

        if param not in param_map:
            raise ValueError(f"Unknown sweep parameter: {param}")

        location, attr = param_map[param]

        # Navigate to the right config object
        if location == "workload":
            obj = cfg.workload
        elif location == "oversubscription":
            obj = cfg.oversubscription
        elif location == "cost":
            obj = cfg.cost
        elif location == "processor.nosmt":
            obj = cfg.processor.nosmt
        elif location == "processor.smt":
            obj = cfg.processor.smt
        else:
            raise ValueError(f"Unknown config location: {location}")

        old_value = getattr(obj, attr)
        setattr(obj, attr, value)
        return old_value

    def _compute_sweep(self) -> List[SweepPointResult]:
        """Compute results for a parameter sweep."""
        sweep = self.config.sweep
        results = []

        for value in sweep.values:
            # Apply sweep value
            old_value = self._apply_sweep_value(sweep.parameter, value)

            # Rebuild model and compute
            smt_proc, nosmt_proc, workload, cost = self._build_model_components()
            single = self._compute_single(smt_proc, nosmt_proc, workload, cost)

            results.append(SweepPointResult(
                parameter_value=value,
                breakeven_carbon=single.breakeven_carbon,
                breakeven_tco=single.breakeven_tco,
                scenarios=single.scenarios,
                savings=single.savings,
            ))

            # Restore original value
            self._apply_sweep_value(sweep.parameter, old_value)

        return results

    def run(self) -> RunResult:
        """
        Execute the experiment and return results.

        Returns:
            RunResult containing metadata, config echo, and results
        """
        # Build metadata
        meta = {
            "timestamp": _format_timestamp(),
            "version": VERSION,
            "config_file": self.config_path,
            "experiment_name": self.config.name,
        }

        # Echo config
        config_dict = self.config.to_dict()

        # Run experiment
        if self.config.is_sweep():
            sweep_results = self._compute_sweep()
            return RunResult(
                meta=meta,
                config=config_dict,
                sweep_results=sweep_results,
            )
        else:
            smt_proc, nosmt_proc, workload, cost = self._build_model_components()
            single = self._compute_single(smt_proc, nosmt_proc, workload, cost)
            return RunResult(
                meta=meta,
                config=config_dict,
                results=single,
            )


def save_result(result: RunResult, path: str | Path) -> None:
    """
    Save a run result to JSON file.

    Args:
        result: RunResult to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)


def load_result(path: str | Path) -> dict:
    """
    Load a previous run result from JSON file.

    Args:
        path: Path to result file

    Returns:
        Dict containing the result data
    """
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)


def generate_output_filename(config: ExperimentConfig, timestamp: Optional[str] = None) -> str:
    """
    Generate a default output filename for a config.

    Format: {name}_{timestamp}.json

    Args:
        config: Experiment config
        timestamp: Optional timestamp string (uses current time if not provided)

    Returns:
        Filename string (not full path)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # Clean up ISO timestamp for filename
        timestamp = timestamp.replace(":", "").replace("-", "")[:15]

    safe_name = config.name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}_{timestamp}.json"
