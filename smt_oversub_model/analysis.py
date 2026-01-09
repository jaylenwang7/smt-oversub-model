"""
Flexible analysis utilities for comparing processor configurations.

This module provides a high-level API for comparing arbitrary scenarios
without being locked into the breakeven-finding workflow.

Example: Compare SMT vs non-SMT at the same oversubscription ratio
    from smt_oversub_model.analysis import compare_smt_vs_nosmt

    result = compare_smt_vs_nosmt(
        total_vcpus=10000,
        oversub_ratio=1.0,  # No oversubscription
        avg_util=0.3,
    )
    print(result['summary'])

Example: Fully custom scenario comparison
    from smt_oversub_model.analysis import ScenarioBuilder, compare_scenarios

    builder = ScenarioBuilder()
    smt = builder.build_scenario("SMT Config", smt=True, oversub_ratio=1.0)
    nosmt = builder.build_scenario("Non-SMT Config", smt=False, oversub_ratio=1.0)

    result = compare_scenarios([smt, nosmt], baseline_idx=0)
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Callable

from .model import (
    OverssubModel, PowerCurve, ProcessorConfig,
    ScenarioParams, WorkloadParams, CostParams, ScenarioResult
)


@dataclass
class ProcessorDefaults:
    """Default processor parameters (can be overridden)."""
    # SMT processor
    smt_physical_cores: int = 48
    smt_threads_per_core: int = 2
    smt_power_idle_w: float = 100.0
    smt_power_max_w: float = 400.0

    # Non-SMT processor
    nosmt_physical_cores: int = 48
    nosmt_threads_per_core: int = 1
    nosmt_power_ratio: float = 0.85  # Relative to SMT max power
    nosmt_idle_ratio: float = 0.9    # Relative to SMT idle power


@dataclass
class CostDefaults:
    """Default cost parameters (can be overridden)."""
    embodied_carbon_kg: float = 1000.0
    server_cost_usd: float = 10000.0
    carbon_intensity_g_kwh: float = 400.0
    electricity_cost_usd_kwh: float = 0.10
    lifetime_years: float = 5.0


@dataclass
class ScenarioSpec:
    """Specification for a single scenario to evaluate."""
    name: str
    processor: ProcessorConfig
    oversub_ratio: float
    util_overhead: float = 0.0


class ScenarioBuilder:
    """
    Builder for creating scenario specifications with sensible defaults.

    Allows easy customization of processor and cost parameters.
    """

    def __init__(
        self,
        processor_defaults: Optional[ProcessorDefaults] = None,
        cost_defaults: Optional[CostDefaults] = None,
        power_curve_fn: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize builder with optional custom defaults.

        Args:
            processor_defaults: Custom processor defaults
            cost_defaults: Custom cost defaults
            power_curve_fn: Custom power curve function (util -> factor)
        """
        self.proc = processor_defaults or ProcessorDefaults()
        self.cost = cost_defaults or CostDefaults()
        self.power_curve_fn = power_curve_fn or (lambda u: u)  # Linear default

    def build_smt_processor(self, **overrides) -> ProcessorConfig:
        """Build an SMT processor config with optional overrides."""
        physical_cores = overrides.get('physical_cores', self.proc.smt_physical_cores)
        threads_per_core = overrides.get('threads_per_core', self.proc.smt_threads_per_core)
        power_idle = overrides.get('power_idle_w', self.proc.smt_power_idle_w)
        power_max = overrides.get('power_max_w', self.proc.smt_power_max_w)

        power_curve = PowerCurve(
            p_idle=power_idle,
            p_max=power_max,
            curve_fn=self.power_curve_fn,
        )

        return ProcessorConfig(
            physical_cores=physical_cores,
            threads_per_core=threads_per_core,
            power_curve=power_curve,
        )

    def build_nosmt_processor(self, **overrides) -> ProcessorConfig:
        """Build a non-SMT processor config with optional overrides."""
        physical_cores = overrides.get('physical_cores', self.proc.nosmt_physical_cores)
        threads_per_core = overrides.get('threads_per_core', self.proc.nosmt_threads_per_core)

        # Power derived from SMT power with ratios
        power_idle = overrides.get(
            'power_idle_w',
            self.proc.smt_power_idle_w * self.proc.nosmt_idle_ratio
        )
        power_max = overrides.get(
            'power_max_w',
            self.proc.smt_power_max_w * self.proc.nosmt_power_ratio
        )

        power_curve = PowerCurve(
            p_idle=power_idle,
            p_max=power_max,
            curve_fn=self.power_curve_fn,
        )

        return ProcessorConfig(
            physical_cores=physical_cores,
            threads_per_core=threads_per_core,
            power_curve=power_curve,
        )

    def build_scenario(
        self,
        name: str,
        smt: bool = True,
        oversub_ratio: float = 1.0,
        util_overhead: float = 0.0,
        **processor_overrides,
    ) -> ScenarioSpec:
        """
        Build a scenario specification.

        Args:
            name: Display name for the scenario
            smt: Whether to use SMT (True) or non-SMT (False) processor
            oversub_ratio: Oversubscription ratio (vCPU:pCPU)
            util_overhead: Additional utilization overhead
            **processor_overrides: Processor parameter overrides

        Returns:
            ScenarioSpec ready for evaluation
        """
        if smt:
            processor = self.build_smt_processor(**processor_overrides)
        else:
            processor = self.build_nosmt_processor(**processor_overrides)

        return ScenarioSpec(
            name=name,
            processor=processor,
            oversub_ratio=oversub_ratio,
            util_overhead=util_overhead,
        )

    def build_cost_params(self, **overrides) -> CostParams:
        """Build cost parameters with optional overrides."""
        return CostParams(
            embodied_carbon_kg=overrides.get('embodied_carbon_kg', self.cost.embodied_carbon_kg),
            server_cost_usd=overrides.get('server_cost_usd', self.cost.server_cost_usd),
            carbon_intensity_g_kwh=overrides.get('carbon_intensity_g_kwh', self.cost.carbon_intensity_g_kwh),
            electricity_cost_usd_kwh=overrides.get('electricity_cost_usd_kwh', self.cost.electricity_cost_usd_kwh),
            lifetime_hours=overrides.get('lifetime_years', self.cost.lifetime_years) * 8760,
        )

    def build_workload_params(self, total_vcpus: int, avg_util: float) -> WorkloadParams:
        """Build workload parameters."""
        return WorkloadParams(
            total_vcpus=total_vcpus,
            avg_util=avg_util,
        )


def evaluate_scenarios(
    scenarios: List[ScenarioSpec],
    workload: WorkloadParams,
    cost: CostParams,
) -> List[Dict[str, Any]]:
    """
    Evaluate a list of scenarios and return results.

    Args:
        scenarios: List of ScenarioSpec to evaluate
        workload: Workload parameters
        cost: Cost parameters

    Returns:
        List of result dicts with scenario data
    """
    model = OverssubModel(workload, cost)
    results = []

    for spec in scenarios:
        scenario_params = ScenarioParams(
            processor=spec.processor,
            oversub_ratio=spec.oversub_ratio,
            util_overhead=spec.util_overhead,
        )
        result = model.evaluate_scenario(scenario_params)

        results.append({
            'name': spec.name,
            'oversub_ratio': spec.oversub_ratio,
            'util_overhead': spec.util_overhead,
            'is_smt': spec.processor.threads_per_core > 1,
            'physical_cores': spec.processor.physical_cores,
            'threads_per_core': spec.processor.threads_per_core,
            **asdict(result),
        })

    return results


def compare_scenarios(
    scenarios: List[ScenarioSpec],
    workload: WorkloadParams,
    cost: CostParams,
    baseline_idx: int = 0,
) -> Dict[str, Any]:
    """
    Compare multiple scenarios and compute relative metrics.

    Args:
        scenarios: List of ScenarioSpec to compare
        workload: Workload parameters
        cost: Cost parameters
        baseline_idx: Index of baseline scenario for comparisons

    Returns:
        Dict with scenarios, comparisons, and summary
    """
    results = evaluate_scenarios(scenarios, workload, cost)

    if not results:
        return {'scenarios': [], 'comparisons': {}, 'summary': {}}

    baseline = results[baseline_idx]

    # Compute comparisons
    comparisons = {}
    for i, r in enumerate(results):
        if i == baseline_idx:
            continue

        name = r['name']
        comparisons[name] = {
            'carbon_diff_pct': _pct_diff(r['total_carbon_kg'], baseline['total_carbon_kg']),
            'tco_diff_pct': _pct_diff(r['total_cost_usd'], baseline['total_cost_usd']),
            'server_diff_pct': _pct_diff(r['num_servers'], baseline['num_servers']),
            'carbon_diff_abs': r['total_carbon_kg'] - baseline['total_carbon_kg'],
            'tco_diff_abs': r['total_cost_usd'] - baseline['total_cost_usd'],
            'server_diff_abs': r['num_servers'] - baseline['num_servers'],
        }

    # Summary
    summary = {
        'baseline': baseline['name'],
        'total_vcpus': workload.total_vcpus,
        'avg_util': workload.avg_util,
        'lifetime_years': cost.lifetime_hours / 8760,
    }

    return {
        'scenarios': results,
        'comparisons': comparisons,
        'summary': summary,
    }


def _pct_diff(value: float, baseline: float) -> float:
    """Calculate percentage difference from baseline."""
    if baseline == 0:
        return 0.0
    return ((value - baseline) / baseline) * 100


def compare_smt_vs_nosmt(
    total_vcpus: int,
    oversub_ratio: float = 1.0,
    avg_util: float = 0.3,
    smt_util_overhead: float = 0.0,
    nosmt_util_overhead: float = 0.0,
    # Processor overrides
    smt_physical_cores: int = 48,
    nosmt_physical_cores: int = 48,
    smt_power_idle_w: float = 100.0,
    smt_power_max_w: float = 400.0,
    nosmt_power_ratio: float = 0.85,
    nosmt_idle_ratio: float = 0.9,
    # Cost overrides
    embodied_carbon_kg: float = 1000.0,
    server_cost_usd: float = 10000.0,
    carbon_intensity_g_kwh: float = 400.0,
    electricity_cost_usd_kwh: float = 0.10,
    lifetime_years: float = 5.0,
    # Power curve
    power_curve_fn: Optional[Callable[[float], float]] = None,
) -> Dict[str, Any]:
    """
    Compare SMT vs non-SMT processors at the same oversubscription ratio.

    This is a convenience function for the common use case of comparing
    SMT-enabled vs SMT-disabled configurations side by side.

    Args:
        total_vcpus: Total vCPU demand to serve
        oversub_ratio: Oversubscription ratio for both configs (default 1.0 = no oversub)
        avg_util: Average VM utilization (default 0.3)
        smt_util_overhead: Utilization overhead for SMT
        nosmt_util_overhead: Utilization overhead for non-SMT
        smt_physical_cores: Physical cores per SMT server
        nosmt_physical_cores: Physical cores per non-SMT server
        smt_power_idle_w: SMT server idle power (watts)
        smt_power_max_w: SMT server max power (watts)
        nosmt_power_ratio: Non-SMT max power as ratio of SMT
        nosmt_idle_ratio: Non-SMT idle power as ratio of SMT
        embodied_carbon_kg: Embodied carbon per server (kg CO2e)
        server_cost_usd: Server cost (USD)
        carbon_intensity_g_kwh: Grid carbon intensity (g CO2/kWh)
        electricity_cost_usd_kwh: Electricity price ($/kWh)
        lifetime_years: Server lifetime (years)
        power_curve_fn: Custom power curve function

    Returns:
        Dict with scenarios, comparisons, and summary
    """
    proc_defaults = ProcessorDefaults(
        smt_physical_cores=smt_physical_cores,
        smt_power_idle_w=smt_power_idle_w,
        smt_power_max_w=smt_power_max_w,
        nosmt_physical_cores=nosmt_physical_cores,
        nosmt_power_ratio=nosmt_power_ratio,
        nosmt_idle_ratio=nosmt_idle_ratio,
    )

    cost_defaults = CostDefaults(
        embodied_carbon_kg=embodied_carbon_kg,
        server_cost_usd=server_cost_usd,
        carbon_intensity_g_kwh=carbon_intensity_g_kwh,
        electricity_cost_usd_kwh=electricity_cost_usd_kwh,
        lifetime_years=lifetime_years,
    )

    builder = ScenarioBuilder(proc_defaults, cost_defaults, power_curve_fn)

    # Build scenarios
    smt_scenario = builder.build_scenario(
        name="SMT",
        smt=True,
        oversub_ratio=oversub_ratio,
        util_overhead=smt_util_overhead,
    )
    nosmt_scenario = builder.build_scenario(
        name="Non-SMT",
        smt=False,
        oversub_ratio=oversub_ratio,
        util_overhead=nosmt_util_overhead,
    )

    workload = builder.build_workload_params(total_vcpus, avg_util)
    cost = builder.build_cost_params()

    return compare_scenarios([smt_scenario, nosmt_scenario], workload, cost, baseline_idx=0)


def compare_oversub_ratios(
    total_vcpus: int,
    oversub_ratios: List[float],
    smt: bool = True,
    avg_util: float = 0.3,
    util_overhead: float = 0.0,
    # Cost/processor kwargs
    **kwargs,
) -> Dict[str, Any]:
    """
    Compare different oversubscription ratios for the same processor type.

    Args:
        total_vcpus: Total vCPU demand to serve
        oversub_ratios: List of oversubscription ratios to compare
        smt: Whether to use SMT processor (default True)
        avg_util: Average VM utilization (default 0.3)
        util_overhead: Utilization overhead
        **kwargs: Additional processor/cost parameters

    Returns:
        Dict with scenarios, comparisons, and summary
    """
    proc_defaults = ProcessorDefaults(
        smt_physical_cores=kwargs.get('smt_physical_cores', 48),
        smt_power_idle_w=kwargs.get('smt_power_idle_w', 100.0),
        smt_power_max_w=kwargs.get('smt_power_max_w', 400.0),
        nosmt_physical_cores=kwargs.get('nosmt_physical_cores', 48),
        nosmt_power_ratio=kwargs.get('nosmt_power_ratio', 0.85),
        nosmt_idle_ratio=kwargs.get('nosmt_idle_ratio', 0.9),
    )

    cost_defaults = CostDefaults(
        embodied_carbon_kg=kwargs.get('embodied_carbon_kg', 1000.0),
        server_cost_usd=kwargs.get('server_cost_usd', 10000.0),
        carbon_intensity_g_kwh=kwargs.get('carbon_intensity_g_kwh', 400.0),
        electricity_cost_usd_kwh=kwargs.get('electricity_cost_usd_kwh', 0.10),
        lifetime_years=kwargs.get('lifetime_years', 5.0),
    )

    builder = ScenarioBuilder(proc_defaults, cost_defaults, kwargs.get('power_curve_fn'))

    # Build scenarios for each ratio
    proc_type = "SMT" if smt else "Non-SMT"
    scenarios = []
    for ratio in oversub_ratios:
        scenarios.append(builder.build_scenario(
            name=f"{proc_type} R={ratio:.1f}",
            smt=smt,
            oversub_ratio=ratio,
            util_overhead=util_overhead,
        ))

    workload = builder.build_workload_params(total_vcpus, avg_util)
    cost = builder.build_cost_params()

    return compare_scenarios(scenarios, workload, cost, baseline_idx=0)
