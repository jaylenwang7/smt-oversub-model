"""
Generalized declarative analysis framework for SMT oversubscription model.

This module provides a config-driven analysis API where breakeven can be found
for any numeric parameter, with support for compound conditions and structured output.

Example usage:
    from smt_oversub_model.declarative import DeclarativeAnalysisEngine

    engine = DeclarativeAnalysisEngine()
    result = engine.run_from_file("configs/vcpu_demand_breakeven.json")
    print(result.summary)

CLI usage:
    python -m smt_oversub_model.declarative configs/analysis.json
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import copy
import json
import re

try:
    import json5
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False

from .model import (
    OverssubModel, PowerCurve, ProcessorConfig,
    ScenarioParams, WorkloadParams, CostParams, ScenarioResult,
    ComponentBreakdown, EmbodiedBreakdown,
    PowerComponentCurve, build_composite_power_curve,
)
from .analysis import ScenarioSpec, ScenarioBuilder, ProcessorDefaults, CostDefaults
from .config import make_power_curve_fn


def _find_repo_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the repository root by looking for .git or pyproject.toml.

    Args:
        start_path: Directory to start searching from (default: cwd)

    Returns:
        Path to repo root, or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()
    while current != current.parent:
        if (current / '.git').exists() or (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return None


class CostMode(Enum):
    """Mode for cost specification."""
    RAW = "raw"  # Direct specification of all cost parameters
    RATIO_BASED = "ratio_based"  # Specify ratios, derive raw parameters


class ParameterPath:
    """
    Resolve dot-notation paths for nested parameter access.

    Supports paths like:
    - Direct: 'oversub_ratio', 'util_overhead', 'vcpu_demand_multiplier'
    - Nested: 'processor.physical_cores', 'processor.power_curve.p_max'
    - Workload: 'workload.avg_util', 'workload.total_vcpus'
    - Cost: 'cost.embodied_carbon_kg', 'cost.carbon_intensity_g_kwh'
    """

    def __init__(self, path: str):
        """
        Initialize with a dot-notation path.

        Args:
            path: Parameter path like 'oversub_ratio' or 'processor.physical_cores'
        """
        self.path = path
        self.parts = path.split('.')

    def get(self, obj: Any) -> Any:
        """
        Get value from object at this path.

        Args:
            obj: Object to traverse (dict, dataclass, or object with attributes)

        Returns:
            Value at the path

        Raises:
            KeyError/AttributeError: If path doesn't exist
        """
        current = obj
        for part in self.parts:
            if isinstance(current, dict):
                current = current[part]
            elif hasattr(current, '__dataclass_fields__'):
                current = getattr(current, part)
            else:
                current = getattr(current, part)
        return current

    def set(self, obj: Any, value: Any) -> Any:
        """
        Return new object with value set at path (immutable for dataclasses).

        For dicts, modifies in place. For dataclasses, returns new instance.

        Args:
            obj: Object to modify
            value: Value to set

        Returns:
            Modified object (new instance for dataclasses)
        """
        if len(self.parts) == 1:
            return self._set_direct(obj, self.parts[0], value)

        # For nested paths, we need to rebuild from the leaf up
        return self._set_nested(obj, self.parts, value)

    def _set_direct(self, obj: Any, key: str, value: Any) -> Any:
        """Set a single key on an object."""
        if isinstance(obj, dict):
            obj = obj.copy()
            obj[key] = value
            return obj
        elif hasattr(obj, '__dataclass_fields__'):
            return replace(obj, **{key: value})
        else:
            obj_copy = copy.copy(obj)
            setattr(obj_copy, key, value)
            return obj_copy

    def _set_nested(self, obj: Any, parts: List[str], value: Any) -> Any:
        """Recursively set a nested path."""
        if len(parts) == 1:
            return self._set_direct(obj, parts[0], value)

        # Get current value at first part
        first = parts[0]
        if isinstance(obj, dict):
            current = obj.get(first)
        else:
            current = getattr(obj, first, None)

        # Recursively set on nested object
        new_nested = self._set_nested(current, parts[1:], value)

        # Set the first part to the new nested value
        return self._set_direct(obj, first, new_nested)

    def __repr__(self) -> str:
        return f"ParameterPath({self.path!r})"


class MatchType(Enum):
    """Types of matching conditions for breakeven search."""
    EQUAL = "match"           # Within tolerance
    LESS_OR_EQUAL = "<="
    GREATER_OR_EQUAL = ">="
    WITHIN_PERCENT = "within"  # e.g., "within_5%"


@dataclass
class SimpleCondition:
    """
    A single condition for comparing scenarios.

    Examples:
        SimpleCondition("carbon", MatchType.EQUAL)  # Match carbon exactly
        SimpleCondition("tco", MatchType.WITHIN_PERCENT, percent=5.0)  # Within 5%
    """
    metric: str  # 'carbon', 'tco', 'num_servers', 'total_carbon_kg', 'total_cost_usd'
    match_type: MatchType
    tolerance: float = 0.01
    percent: Optional[float] = None

    def _get_metric_value(self, result: Union[ScenarioResult, Dict[str, Any]]) -> float:
        """Extract metric value from result."""
        if isinstance(result, dict):
            # Handle various metric name formats
            if self.metric == 'carbon':
                return result.get('total_carbon_kg', 0)
            elif self.metric == 'tco':
                return result.get('total_cost_usd', 0)
            else:
                return result.get(self.metric, 0)
        else:
            if self.metric == 'carbon':
                return result.total_carbon_kg
            elif self.metric == 'tco':
                return result.total_cost_usd
            elif hasattr(result, self.metric):
                return getattr(result, self.metric)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

    def evaluate(
        self,
        target: Union[ScenarioResult, Dict[str, Any]],
        reference: Union[ScenarioResult, Dict[str, Any]]
    ) -> bool:
        """
        Check if target matches reference according to this condition.

        Args:
            target: Result being tested
            reference: Reference result to compare against

        Returns:
            True if condition is satisfied
        """
        target_val = self._get_metric_value(target)
        ref_val = self._get_metric_value(reference)

        if ref_val == 0:
            return target_val == 0

        if self.match_type == MatchType.EQUAL:
            return abs(target_val - ref_val) / ref_val <= self.tolerance

        elif self.match_type == MatchType.LESS_OR_EQUAL:
            return target_val <= ref_val * (1 + self.tolerance)

        elif self.match_type == MatchType.GREATER_OR_EQUAL:
            return target_val >= ref_val * (1 - self.tolerance)

        elif self.match_type == MatchType.WITHIN_PERCENT:
            pct = self.percent if self.percent is not None else 5.0
            return abs(target_val - ref_val) / ref_val <= pct / 100

        return False

    def get_error(
        self,
        target: Union[ScenarioResult, Dict[str, Any]],
        reference: Union[ScenarioResult, Dict[str, Any]]
    ) -> float:
        """
        Get signed error: target - reference (normalized).

        Positive means target is higher than reference.

        Returns:
            Normalized error (target - reference) / reference
        """
        target_val = self._get_metric_value(target)
        ref_val = self._get_metric_value(reference)

        if ref_val == 0:
            return 0.0 if target_val == 0 else float('inf')

        return (target_val - ref_val) / ref_val

    @classmethod
    def from_string(cls, spec: str) -> 'SimpleCondition':
        """
        Parse condition from string spec.

        Examples:
            "match" -> EQUAL with default tolerance
            "within_5%" -> WITHIN_PERCENT with 5%
            "<=" -> LESS_OR_EQUAL
        """
        if spec == "match":
            return cls(metric="", match_type=MatchType.EQUAL)
        elif spec.startswith("within_"):
            match = re.match(r"within_(\d+(?:\.\d+)?)%?", spec)
            if match:
                pct = float(match.group(1))
                return cls(metric="", match_type=MatchType.WITHIN_PERCENT, percent=pct)
        elif spec == "<=":
            return cls(metric="", match_type=MatchType.LESS_OR_EQUAL)
        elif spec == ">=":
            return cls(metric="", match_type=MatchType.GREATER_OR_EQUAL)

        raise ValueError(f"Unknown condition spec: {spec}")


@dataclass
class CompoundCondition:
    """
    Multiple conditions that must all be satisfied.

    Example:
        {"carbon": "match", "tco": "within_5%"}
    """
    conditions: Dict[str, SimpleCondition]

    @classmethod
    def from_dict(cls, spec: Dict[str, str]) -> 'CompoundCondition':
        """
        Create from dict spec like {"carbon": "match", "tco": "within_5%"}.
        """
        conditions = {}
        for metric, cond_str in spec.items():
            cond = SimpleCondition.from_string(cond_str)
            cond.metric = metric
            conditions[metric] = cond
        return cls(conditions=conditions)

    def evaluate(
        self,
        target: Union[ScenarioResult, Dict[str, Any]],
        reference: Union[ScenarioResult, Dict[str, Any]]
    ) -> bool:
        """All conditions must pass."""
        return all(c.evaluate(target, reference) for c in self.conditions.values())

    def get_primary_error(
        self,
        target: Union[ScenarioResult, Dict[str, Any]],
        reference: Union[ScenarioResult, Dict[str, Any]]
    ) -> float:
        """
        Get error from primary metric (first EQUAL condition, or first condition).

        Used for binary search direction.
        """
        # Prefer EQUAL conditions for search
        for cond in self.conditions.values():
            if cond.match_type == MatchType.EQUAL:
                return cond.get_error(target, reference)

        # Fall back to first condition
        if self.conditions:
            first = next(iter(self.conditions.values()))
            return first.get_error(target, reference)

        return 0.0


@dataclass
class SearchHistoryEntry:
    """Single step in breakeven search."""
    value: float
    error: float
    metric_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class BreakevenResult:
    """Result of a breakeven search."""
    breakeven_value: Optional[float]
    achieved: bool
    iterations: int
    search_history: List[SearchHistoryEntry]
    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str] = None


class GeneralizedBreakevenFinder:
    """
    Find breakeven values for any numeric parameter using binary search.

    This generalizes the original find_breakeven_oversub to work with
    any parameter path and arbitrary match conditions.
    """

    def __init__(
        self,
        model: OverssubModel,
        max_iterations: int = 50,
    ):
        """
        Initialize finder.

        Args:
            model: OverssubModel instance for evaluations
            max_iterations: Maximum binary search iterations
        """
        self.model = model
        self.max_iterations = max_iterations

    def find_breakeven(
        self,
        base_scenario: ScenarioParams,
        reference_result: ScenarioResult,
        vary_parameter: str,
        match_condition: Union[SimpleCondition, CompoundCondition],
        search_bounds: Tuple[float, float],
        tolerance: float = 0.001,
    ) -> BreakevenResult:
        """
        Binary search to find parameter value where target matches reference.

        Args:
            base_scenario: Base scenario to modify
            reference_result: Reference result to match
            vary_parameter: Parameter path to vary (e.g., 'oversub_ratio')
            match_condition: Condition(s) to satisfy
            search_bounds: (low, high) bounds for search
            tolerance: Convergence tolerance

        Returns:
            BreakevenResult with breakeven value and search history
        """
        param_path = ParameterPath(vary_parameter)
        low, high = search_bounds
        history: List[SearchHistoryEntry] = []

        # Determine if we're using simple or compound condition
        if isinstance(match_condition, SimpleCondition):
            get_error = lambda t, r: match_condition.get_error(t, r)
            check_satisfied = lambda t, r: match_condition.evaluate(t, r)
        else:
            get_error = lambda t, r: match_condition.get_primary_error(t, r)
            check_satisfied = lambda t, r: match_condition.evaluate(t, r)

        def evaluate_at(value: float) -> Tuple[ScenarioResult, float]:
            """Evaluate scenario at given parameter value."""
            modified = param_path.set(base_scenario, value)
            result = self.model.evaluate_scenario(modified)
            error = get_error(result, reference_result)
            return result, error

        # Check bounds
        result_low, error_low = evaluate_at(low)
        result_high, error_high = evaluate_at(high)

        history.append(SearchHistoryEntry(low, error_low, self._extract_metrics(result_low)))
        history.append(SearchHistoryEntry(high, error_high, self._extract_metrics(result_high)))

        # Check if solution exists (errors should have opposite signs or one is zero)
        if error_low * error_high > 0 and abs(error_low) > tolerance and abs(error_high) > tolerance:
            # Both same sign and far from zero - no solution in bounds
            return BreakevenResult(
                breakeven_value=None,
                achieved=False,
                iterations=2,
                search_history=history,
                final_result=None,
                error_message=f"No solution in bounds [{low}, {high}]: errors {error_low:.4f} and {error_high:.4f} have same sign"
            )

        # Binary search
        iterations = 2
        for _ in range(self.max_iterations):
            if high - low <= tolerance:
                break

            mid = (low + high) / 2
            result_mid, error_mid = evaluate_at(mid)
            history.append(SearchHistoryEntry(mid, error_mid, self._extract_metrics(result_mid)))
            iterations += 1

            if abs(error_mid) <= tolerance:
                # Found solution
                if check_satisfied(result_mid, reference_result):
                    return BreakevenResult(
                        breakeven_value=mid,
                        achieved=True,
                        iterations=iterations,
                        search_history=history,
                        final_result=asdict(result_mid),
                    )

            # Determine which half to search
            # We want to find where error crosses zero
            if error_mid * error_low < 0:
                high = mid
                error_high = error_mid
            else:
                low = mid
                error_low = error_mid

        # Return best result
        final_value = (low + high) / 2
        final_result, final_error = evaluate_at(final_value)
        achieved = check_satisfied(final_result, reference_result)

        return BreakevenResult(
            breakeven_value=final_value if achieved else None,
            achieved=achieved,
            iterations=iterations,
            search_history=history,
            final_result=asdict(final_result),
            error_message=None if achieved else f"Did not converge: final error {final_error:.4f}"
        )

    def _extract_metrics(self, result: ScenarioResult) -> Dict[str, float]:
        """Extract key metrics from result."""
        return {
            'carbon': result.total_carbon_kg,
            'tco': result.total_cost_usd,
            'num_servers': result.num_servers,
        }


# --- Config Dataclasses ---

@dataclass
class ScenarioConfig:
    """Configuration for a single scenario."""
    processor: str  # "smt" or "nosmt"
    oversub_ratio: float = 1.0
    util_overhead: float = 0.0
    vcpu_demand_multiplier: float = 1.0
    overrides: Optional[Dict[str, Any]] = None  # processor param overrides

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfig':
        return cls(
            processor=data.get('processor', 'smt'),
            oversub_ratio=data.get('oversub_ratio', 1.0),
            util_overhead=data.get('util_overhead', 0.0),
            vcpu_demand_multiplier=data.get('vcpu_demand_multiplier', 1.0),
            overrides=data.get('overrides'),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'processor': self.processor,
            'oversub_ratio': self.oversub_ratio,
            'util_overhead': self.util_overhead,
            'vcpu_demand_multiplier': self.vcpu_demand_multiplier,
        }
        if self.overrides:
            d['overrides'] = self.overrides
        return d


@dataclass
class AnalysisSpec:
    """Specification for the analysis to perform."""
    type: str  # "find_breakeven", "compare", "sweep", "compare_sweep"
    baseline: Optional[str] = None
    reference: Optional[str] = None
    target: Optional[str] = None
    scenarios: Optional[List[str]] = None  # For "compare" type
    vary_parameter: Optional[str] = None
    match_metric: Optional[Union[str, Dict[str, str]]] = None
    search_bounds: Optional[List[float]] = None
    sweep_parameter: Optional[str] = None  # For "sweep" and "compare_sweep" types
    sweep_values: Optional[List[float]] = None
    sweep_scenario: Optional[str] = None  # For "compare_sweep": which scenario to apply sweep to
    sweep_scenarios: Optional[List[str]] = None  # For "compare_sweep": multiple scenarios to sweep (multi-line)
    show_breakeven_marker: bool = True  # For "compare_sweep": show breakeven point marker on plot
    labels: Optional[Dict[str, str]] = None  # Display labels for scenarios (e.g., {"nosmt_r1": "No-SMT R=1.0"})
    sweep_parameter_label: Optional[str] = None  # Display label for sweep parameter (e.g., "vCPU Demand Discount")
    show_plot_title: bool = True  # For line plots: show title (default True)
    x_axis_markers: Optional[List[float]] = None  # For line plots: draw vertical lines at these x-values and label intersections
    x_axis_marker_labels: Optional[List[str]] = None  # For line plots: labels for x_axis_markers (same length as x_axis_markers)
    separate_metric_plots: bool = False  # For compare_sweep: generate separate plots for carbon and TCO instead of combined

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSpec':
        return cls(
            type=data.get('type', 'compare'),
            baseline=data.get('baseline'),
            reference=data.get('reference'),
            target=data.get('target'),
            scenarios=data.get('scenarios'),
            vary_parameter=data.get('vary_parameter'),
            match_metric=data.get('match_metric'),
            search_bounds=data.get('search_bounds'),
            sweep_parameter=data.get('sweep_parameter'),
            sweep_values=data.get('sweep_values'),
            sweep_scenario=data.get('sweep_scenario'),
            sweep_scenarios=data.get('sweep_scenarios'),
            show_breakeven_marker=data.get('show_breakeven_marker', True),
            labels=data.get('labels'),
            sweep_parameter_label=data.get('sweep_parameter_label'),
            show_plot_title=data.get('show_plot_title', True),
            x_axis_markers=data.get('x_axis_markers'),
            x_axis_marker_labels=data.get('x_axis_marker_labels'),
            separate_metric_plots=data.get('separate_metric_plots', False),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {'type': self.type}
        if self.baseline:
            d['baseline'] = self.baseline
        if self.reference:
            d['reference'] = self.reference
        if self.target:
            d['target'] = self.target
        if self.scenarios:
            d['scenarios'] = self.scenarios
        if self.vary_parameter:
            d['vary_parameter'] = self.vary_parameter
        if self.match_metric:
            d['match_metric'] = self.match_metric
        if self.search_bounds:
            d['search_bounds'] = self.search_bounds
        if self.sweep_parameter:
            d['sweep_parameter'] = self.sweep_parameter
        if self.sweep_values:
            d['sweep_values'] = self.sweep_values
        if self.sweep_scenario:
            d['sweep_scenario'] = self.sweep_scenario
        if self.sweep_scenarios:
            d['sweep_scenarios'] = self.sweep_scenarios
        if not self.show_breakeven_marker:
            d['show_breakeven_marker'] = False
        if self.labels:
            d['labels'] = self.labels
        if self.sweep_parameter_label:
            d['sweep_parameter_label'] = self.sweep_parameter_label
        if not self.show_plot_title:
            d['show_plot_title'] = False
        if self.x_axis_markers:
            d['x_axis_markers'] = self.x_axis_markers
        if self.x_axis_marker_labels:
            d['x_axis_marker_labels'] = self.x_axis_marker_labels
        if self.separate_metric_plots:
            d['separate_metric_plots'] = True
        return d


@dataclass
class EmbodiedComponentSpec:
    """Config-level specification for per-core/per-server embodied carbon or cost.

    Used in processor specs and global cost specs to define structured
    cost/carbon breakdowns that scale with core count.

    Example JSON:
        {
            "per_core": {"cpu_die": 10.0, "dram": 2.0},
            "per_server": {"chassis": 100.0, "network": 50.0}
        }

    Result: total_per_server = (10.0 + 2.0) * hw_threads + (100.0 + 50.0)
    """
    per_core: Dict[str, float] = field(default_factory=dict)
    per_server: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbodiedComponentSpec':
        return cls(
            per_core=dict(data.get('per_core', {})),
            per_server=dict(data.get('per_server', {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        if self.per_core:
            d['per_core'] = dict(self.per_core)
        if self.per_server:
            d['per_server'] = dict(self.per_server)
        return d

    def resolve_total(self, physical_cores: int, threads_per_core: int) -> float:
        """Compute flat per-server value from components."""
        hw_threads = physical_cores * threads_per_core
        return sum(self.per_core.values()) * hw_threads + sum(self.per_server.values())

    def to_component_breakdown(self, physical_cores: int, threads_per_core: int) -> ComponentBreakdown:
        """Convert to a resolved ComponentBreakdown for use in model evaluation."""
        return ComponentBreakdown(
            per_core=dict(self.per_core),
            per_server=dict(self.per_server),
            physical_cores=physical_cores,
            threads_per_core=threads_per_core,
        )


@dataclass
class PowerComponentSpec:
    """Config-level specification for a single power component.

    Each component has idle/max power and an optional power curve.
    If not specified, the global power curve is used (when building from config);
    if no global is provided, defaults to polynomial.

    Example JSON:
        {"idle_w": 23, "max_w": 153, "power_curve": {"type": "specpower"}}
    """
    idle_w: float
    max_w: float
    power_curve: Optional['PowerCurveSpec'] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PowerComponentSpec':
        power_curve = None
        if 'power_curve' in data and data['power_curve'] is not None:
            power_curve = PowerCurveSpec.from_dict(data['power_curve'])
        return cls(
            idle_w=data['idle_w'],
            max_w=data['max_w'],
            power_curve=power_curve,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {'idle_w': self.idle_w, 'max_w': self.max_w}
        if self.power_curve is not None:
            d['power_curve'] = self.power_curve.to_dict()
        return d

    def to_power_component_curve(
        self,
        default_curve_spec: Optional['PowerCurveSpec'] = None,
    ) -> PowerComponentCurve:
        """Convert to a PowerComponentCurve for use in model evaluation.

        Fallback order: component power_curve -> default_curve_spec (global) -> polynomial.
        """
        if self.power_curve is not None:
            curve_fn = self.power_curve.to_callable()
        elif default_curve_spec is not None:
            curve_fn = default_curve_spec.to_callable()
        else:
            curve_fn = PowerCurveSpec(type='polynomial').to_callable()
        return PowerComponentCurve(
            idle_w=self.idle_w,
            max_w=self.max_w,
            curve_fn=curve_fn,
        )


@dataclass
class ProcessorSpec:
    """Processor configuration with explicit SMT thread count.

    Args:
        physical_cores: Number of physical CPU cores
        threads_per_core: SMT threads per core (1 = no SMT, 2+ = SMT enabled)
        power_idle_w: Idle power consumption in watts
        power_max_w: Maximum power consumption in watts
        core_overhead: Number of pCPUs reserved for host (not available for VMs)
        embodied_carbon_kg: Optional per-server embodied carbon (overrides cost.embodied_carbon_kg)
        server_cost_usd: Optional per-server cost (overrides cost.server_cost_usd)
        power_curve: Optional per-processor power curve (overrides global power_curve)
    """
    physical_cores: int = 48
    threads_per_core: int = 1  # 1 = no SMT, 2 = SMT (hyperthreading)
    power_idle_w: float = 100.0
    power_max_w: float = 400.0
    core_overhead: int = 0
    # Optional cost overrides - if set, these override the values in CostSpec
    embodied_carbon_kg: Optional[float] = None
    server_cost_usd: Optional[float] = None
    # Optional structured cost/carbon breakdowns (take priority over flat values)
    embodied_carbon: Optional[EmbodiedComponentSpec] = None
    server_cost: Optional[EmbodiedComponentSpec] = None
    # Optional power curve override - if set, overrides the global power_curve
    power_curve: Optional['PowerCurveSpec'] = None
    # Optional per-component power breakdown - when present, overrides power_idle_w/power_max_w
    power_breakdown: Optional[Dict[str, PowerComponentSpec]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessorSpec':
        power_curve = None
        if 'power_curve' in data and data['power_curve'] is not None:
            power_curve = PowerCurveSpec.from_dict(data['power_curve'])

        # Parse structured embodied carbon/cost
        embodied_carbon = None
        if 'embodied_carbon' in data and isinstance(data['embodied_carbon'], dict):
            embodied_carbon = EmbodiedComponentSpec.from_dict(data['embodied_carbon'])

        server_cost = None
        if 'server_cost' in data and isinstance(data['server_cost'], dict):
            server_cost = EmbodiedComponentSpec.from_dict(data['server_cost'])

        # Parse power breakdown
        power_breakdown = None
        if 'power_breakdown' in data and isinstance(data['power_breakdown'], dict):
            power_breakdown = {
                name: PowerComponentSpec.from_dict(comp)
                for name, comp in data['power_breakdown'].items()
            }

        return cls(
            physical_cores=data.get('physical_cores', 48),
            threads_per_core=data.get('threads_per_core', 1),
            power_idle_w=data.get('power_idle_w', 100.0),
            power_max_w=data.get('power_max_w', 400.0),
            core_overhead=data.get('core_overhead', 0),
            embodied_carbon_kg=data.get('embodied_carbon_kg'),
            server_cost_usd=data.get('server_cost_usd'),
            embodied_carbon=embodied_carbon,
            server_cost=server_cost,
            power_curve=power_curve,
            power_breakdown=power_breakdown,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'physical_cores': self.physical_cores,
            'threads_per_core': self.threads_per_core,
            'power_idle_w': self.power_idle_w,
            'power_max_w': self.power_max_w,
            'core_overhead': self.core_overhead,
        }
        # Only include optional fields if set
        # Structured format takes priority in serialization
        if self.embodied_carbon is not None:
            d['embodied_carbon'] = self.embodied_carbon.to_dict()
        elif self.embodied_carbon_kg is not None:
            d['embodied_carbon_kg'] = self.embodied_carbon_kg
        if self.server_cost is not None:
            d['server_cost'] = self.server_cost.to_dict()
        elif self.server_cost_usd is not None:
            d['server_cost_usd'] = self.server_cost_usd
        if self.power_curve is not None:
            d['power_curve'] = self.power_curve.to_dict()
        if self.power_breakdown is not None:
            d['power_breakdown'] = {
                name: comp.to_dict()
                for name, comp in self.power_breakdown.items()
            }
        return d

    @property
    def is_smt(self) -> bool:
        """Return True if this processor has SMT enabled."""
        return self.threads_per_core > 1


@dataclass
class ProcessorConfigSpec:
    """Container for named processor configurations.

    Processors are stored in a dict keyed by name, allowing arbitrary
    processor definitions (not limited to 'smt'/'nosmt').

    Supports multiple loading modes:

    1. **Load all from file**: `"processor": "./processors.json"`
       Loads all processors from the external file.

    2. **Inline definitions**: Define processors directly in the config.
       ```json
       "processor": {
         "smt": { "physical_cores": 48, ... },
         "nosmt": { ... }
       }
       ```

    3. **Mixed mode**: Combine inline definitions with selective imports.
       ```json
       "processor": {
         "custom_a": "./processors.json:smt",  // String reference: file:processor_name
         "custom_b": { "file": "./processors.json", "name": "nosmt" },  // Object reference
         "my_custom": { "physical_cores": 32, ... }  // Inline definition
       }
       ```

    Paths are resolved relative to the config file directory.
    """
    processors: Dict[str, ProcessorSpec] = field(default_factory=dict)
    source_file: Optional[str] = None  # Track where processors were loaded from (if all from one file)

    def __post_init__(self):
        # Ensure defaults for backward compatibility
        if not self.processors:
            self.processors = {
                "smt": ProcessorSpec(physical_cores=48, threads_per_core=2,
                                    power_idle_w=100.0, power_max_w=400.0),
                "nosmt": ProcessorSpec(physical_cores=48, threads_per_core=1,
                                      power_idle_w=90.0, power_max_w=340.0),
            }

    @classmethod
    def _load_external_file(cls, file_path: Path) -> Dict[str, Any]:
        """Load and cache external processor file.

        Args:
            file_path: Resolved path to the processor config file

        Returns:
            Dict of processor data from the file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Processor config file not found: {file_path}")

        with open(file_path, 'r') as f:
            if _HAS_JSON5:
                return json5.load(f)
            else:
                return json.load(f)

    @classmethod
    def _resolve_path(cls, path_str: str, base_path: Optional[Path]) -> Path:
        """Resolve a path string relative to repo root or config directory.

        Resolution order:
        1. If path starts with configs/ or is repo-relative, resolve from repo root
        2. Otherwise try repo root first, then fall back to config directory

        This allows both repo-relative paths (e.g., "configs/shared/processors.jsonc")
        and config-relative paths (e.g., "../shared/processors.jsonc") to work.
        """
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()

        repo_root = _find_repo_root(base_path)

        # Try repo root first
        if repo_root:
            repo_path = repo_root / path
            if repo_path.exists():
                return repo_path.resolve()

        # Fall back to config file directory
        if base_path:
            config_path = base_path / path
            if config_path.exists():
                return config_path.resolve()

        # If neither exists, return repo-relative path (will fail with clearer error)
        if repo_root:
            return (repo_root / path).resolve()
        elif base_path:
            return (base_path / path).resolve()
        return path.resolve()

    @classmethod
    def _parse_reference(
        cls,
        ref: Union[str, Dict[str, Any]],
        base_path: Optional[Path],
    ) -> Tuple[Path, str]:
        """Parse a processor reference to get file path and processor name.

        Args:
            ref: Either "file:name" string or {"file": "...", "name": "..."}
            base_path: Base directory for resolving relative paths

        Returns:
            Tuple of (resolved_file_path, processor_name)
        """
        if isinstance(ref, str):
            # String format: "path/to/file.json:processor_name"
            if ':' not in ref:
                raise ValueError(
                    f"Invalid processor reference '{ref}': "
                    "expected 'file:name' format (e.g., './processors.json:smt')"
                )
            # Handle Windows paths with drive letters (e.g., C:\path\file.json:name)
            # Split from the right to handle this case
            last_colon = ref.rfind(':')
            file_part = ref[:last_colon]
            name_part = ref[last_colon + 1:]
            return cls._resolve_path(file_part, base_path), name_part
        else:
            # Object format: {"file": "...", "name": "..."}
            if 'file' not in ref or 'name' not in ref:
                raise ValueError(
                    f"Invalid processor reference object: "
                    "expected {{'file': '...', 'name': '...'}} format"
                )
            return cls._resolve_path(ref['file'], base_path), ref['name']

    @classmethod
    def _is_processor_reference(cls, spec_data: Any) -> bool:
        """Check if spec_data is a reference to an external processor.

        A reference is either:
        - A string containing ':' (file:name format)
        - A dict with 'file' and 'name' keys
        """
        if isinstance(spec_data, str):
            return ':' in spec_data
        if isinstance(spec_data, dict):
            return 'file' in spec_data and 'name' in spec_data
        return False

    @classmethod
    def _is_inline_processor(cls, spec_data: Any) -> bool:
        """Check if spec_data is an inline processor definition.

        An inline processor has processor-specific fields like physical_cores,
        threads_per_core, power_idle_w, etc.
        """
        if not isinstance(spec_data, dict):
            return False
        # Check for at least one processor field
        processor_fields = {'physical_cores', 'threads_per_core', 'power_idle_w', 'power_max_w',
                            'core_overhead', 'embodied_carbon', 'server_cost', 'power_breakdown'}
        return bool(processor_fields & set(spec_data.keys()))

    @classmethod
    def from_dict(
        cls,
        data: Union[Dict[str, Any], str],
        base_path: Optional[Path] = None,
    ) -> 'ProcessorConfigSpec':
        """Load processor config from dict or file path.

        Supports three modes:
        1. String path: Load all processors from external file
        2. Dict of inline specs: Define processors directly
        3. Mixed: Each processor can be inline or a reference to external file

        Args:
            data: Either a dict of processor specs, or a string path to a JSON file
            base_path: Base directory for resolving relative paths (usually config file dir)

        Returns:
            ProcessorConfigSpec with loaded processors

        Examples:
            # Mode 1: Load all from file
            ProcessorConfigSpec.from_dict("./processors.json", base_path)

            # Mode 2: Inline definitions
            ProcessorConfigSpec.from_dict({
                "smt": {"physical_cores": 48, "threads_per_core": 2, ...},
                "nosmt": {"physical_cores": 48, "threads_per_core": 1, ...}
            }, base_path)

            # Mode 3: Mixed
            ProcessorConfigSpec.from_dict({
                "smt": "./processors.json:smt",  # Reference from file
                "nosmt": {"file": "./processors.json", "name": "nosmt"},  # Object reference
                "custom": {"physical_cores": 32, "threads_per_core": 1, ...}  # Inline
            }, base_path)
        """
        source_file = None

        # Mode 1: String path - load ALL processors from external file
        if isinstance(data, str):
            processor_path = cls._resolve_path(data, base_path)

            if not processor_path.exists():
                raise FileNotFoundError(f"Processor config file not found: {processor_path}")

            file_data = cls._load_external_file(processor_path)
            source_file = str(processor_path)

            processors = {}
            for name, spec_data in file_data.items():
                processors[name] = ProcessorSpec.from_dict(spec_data)
            return cls(processors=processors, source_file=source_file)

        # Mode 2 & 3: Dict - could be inline, references, or mixed
        # Cache loaded files to avoid re-reading
        file_cache: Dict[str, Dict[str, Any]] = {}
        processors = {}

        for name, spec_data in data.items():
            if cls._is_processor_reference(spec_data):
                # Load from external file reference
                file_path, proc_name = cls._parse_reference(spec_data, base_path)
                file_key = str(file_path)

                # Load file if not cached
                if file_key not in file_cache:
                    file_cache[file_key] = cls._load_external_file(file_path)

                # Get the specific processor from the file
                if proc_name not in file_cache[file_key]:
                    available = list(file_cache[file_key].keys())
                    raise KeyError(
                        f"Processor '{proc_name}' not found in {file_path}. "
                        f"Available: {available}"
                    )

                processors[name] = ProcessorSpec.from_dict(file_cache[file_key][proc_name])

            elif cls._is_inline_processor(spec_data):
                # Inline processor definition
                processors[name] = ProcessorSpec.from_dict(spec_data)

            else:
                raise ValueError(
                    f"Invalid processor spec for '{name}': "
                    f"expected inline definition (with physical_cores, threads_per_core, etc.) "
                    f"or reference ('file:name' string or {{'file': '...', 'name': '...'}} object)"
                )

        return cls(processors=processors, source_file=source_file)

    def to_dict(self, include_source: bool = False) -> Dict[str, Any]:
        """Convert to dict representation.

        Args:
            include_source: If True and loaded from file, return the source path
                          instead of inline specs. Default False returns inline specs.
        """
        if include_source and self.source_file:
            return self.source_file
        return {name: spec.to_dict() for name, spec in self.processors.items()}

    def get(self, name: str) -> ProcessorSpec:
        """Get a processor by name."""
        if name not in self.processors:
            raise KeyError(f"Unknown processor: {name}. Available: {list(self.processors.keys())}")
        return self.processors[name]

    # Backward compatibility properties
    @property
    def smt(self) -> ProcessorSpec:
        """Get the 'smt' processor (backward compatibility)."""
        return self.get("smt")

    @property
    def nosmt(self) -> ProcessorSpec:
        """Get the 'nosmt' processor (backward compatibility)."""
        return self.get("nosmt")


@dataclass
class WorkloadSpec:
    """Workload configuration."""
    total_vcpus: int = 10000
    avg_util: float = 0.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkloadSpec':
        return cls(
            total_vcpus=data.get('total_vcpus', 10000),
            avg_util=data.get('avg_util', 0.3),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CostSpec:
    """Cost configuration.

    Supports two modes:
    - RAW: Direct specification of all cost parameters (default)
    - RATIO_BASED: Specify operational/embodied ratios, derive raw parameters

    For ratio-based mode:
    - Specify `mode: "ratio_based"` and `reference_scenario`
    - Provide `operational_carbon_fraction` and/or `operational_cost_fraction`
    - Either provide `embodied_carbon_kg`/`server_cost_usd` (embodied anchor)
      OR `total_carbon_kg`/`total_cost_usd` (total anchor)
    - The system evaluates the reference scenario and derives the operational
      parameters that achieve the target ratio
    """
    # Common parameters
    embodied_carbon_kg: float = 1000.0
    server_cost_usd: float = 10000.0
    lifetime_years: float = 5.0

    # Raw mode parameters (optional in ratio mode)
    carbon_intensity_g_kwh: Optional[float] = 400.0
    electricity_cost_usd_kwh: Optional[float] = 0.10

    # Ratio-based mode parameters
    mode: CostMode = CostMode.RAW
    reference_scenario: Optional[str] = None
    operational_carbon_fraction: Optional[float] = None  # e.g., 0.75 = 75% operational
    operational_cost_fraction: Optional[float] = None    # e.g., 0.6 = 60% operational

    # Total anchor mode parameters (alternative to embodied anchor)
    total_carbon_kg: Optional[float] = None
    total_cost_usd: Optional[float] = None

    # Optional structured cost/carbon breakdowns (global level, resolved per-processor)
    embodied_carbon: Optional[EmbodiedComponentSpec] = None
    server_cost: Optional[EmbodiedComponentSpec] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostSpec':
        # Parse mode
        mode_str = data.get('mode', 'raw')
        if mode_str == 'ratio_based':
            mode = CostMode.RATIO_BASED
        else:
            mode = CostMode.RAW

        # Parse structured embodied carbon/cost
        embodied_carbon = None
        if 'embodied_carbon' in data and isinstance(data['embodied_carbon'], dict):
            embodied_carbon = EmbodiedComponentSpec.from_dict(data['embodied_carbon'])

        server_cost = None
        if 'server_cost' in data and isinstance(data['server_cost'], dict):
            server_cost = EmbodiedComponentSpec.from_dict(data['server_cost'])

        return cls(
            embodied_carbon_kg=data.get('embodied_carbon_kg', 1000.0),
            server_cost_usd=data.get('server_cost_usd', 10000.0),
            carbon_intensity_g_kwh=data.get('carbon_intensity_g_kwh', 400.0 if mode == CostMode.RAW else None),
            electricity_cost_usd_kwh=data.get('electricity_cost_usd_kwh', 0.10 if mode == CostMode.RAW else None),
            lifetime_years=data.get('lifetime_years', 5.0),
            mode=mode,
            reference_scenario=data.get('reference_scenario'),
            operational_carbon_fraction=data.get('operational_carbon_fraction'),
            operational_cost_fraction=data.get('operational_cost_fraction'),
            total_carbon_kg=data.get('total_carbon_kg'),
            total_cost_usd=data.get('total_cost_usd'),
            embodied_carbon=embodied_carbon,
            server_cost=server_cost,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'embodied_carbon_kg': self.embodied_carbon_kg,
            'server_cost_usd': self.server_cost_usd,
            'lifetime_years': self.lifetime_years,
        }
        if self.carbon_intensity_g_kwh is not None:
            d['carbon_intensity_g_kwh'] = self.carbon_intensity_g_kwh
        if self.electricity_cost_usd_kwh is not None:
            d['electricity_cost_usd_kwh'] = self.electricity_cost_usd_kwh
        if self.mode != CostMode.RAW:
            d['mode'] = self.mode.value
        if self.reference_scenario is not None:
            d['reference_scenario'] = self.reference_scenario
        if self.operational_carbon_fraction is not None:
            d['operational_carbon_fraction'] = self.operational_carbon_fraction
        if self.operational_cost_fraction is not None:
            d['operational_cost_fraction'] = self.operational_cost_fraction
        if self.total_carbon_kg is not None:
            d['total_carbon_kg'] = self.total_carbon_kg
        if self.total_cost_usd is not None:
            d['total_cost_usd'] = self.total_cost_usd
        if self.embodied_carbon is not None:
            d['embodied_carbon'] = self.embodied_carbon.to_dict()
        if self.server_cost is not None:
            d['server_cost'] = self.server_cost.to_dict()
        return d

    def is_ratio_based(self) -> bool:
        """Check if this cost spec uses ratio-based mode."""
        return self.mode == CostMode.RATIO_BASED

    def uses_total_anchor(self) -> bool:
        """Check if ratio mode uses total anchor (vs embodied anchor)."""
        return self.total_carbon_kg is not None or self.total_cost_usd is not None

    def validate_ratio_mode(self) -> None:
        """Validate ratio-based mode configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.is_ratio_based():
            return

        if not self.reference_scenario:
            raise ValueError("ratio_based mode requires 'reference_scenario'")

        has_carbon_ratio = self.operational_carbon_fraction is not None
        has_cost_ratio = self.operational_cost_fraction is not None

        if not has_carbon_ratio and not has_cost_ratio:
            raise ValueError(
                "ratio_based mode requires at least one of "
                "'operational_carbon_fraction' or 'operational_cost_fraction'"
            )

        # Validate fraction ranges
        if has_carbon_ratio:
            if not 0 < self.operational_carbon_fraction < 1:
                raise ValueError(
                    f"operational_carbon_fraction must be in (0, 1), "
                    f"got {self.operational_carbon_fraction}"
                )

        if has_cost_ratio:
            if not 0 < self.operational_cost_fraction < 1:
                raise ValueError(
                    f"operational_cost_fraction must be in (0, 1), "
                    f"got {self.operational_cost_fraction}"
                )


@dataclass
class ReferenceScenarioResult:
    """Result from evaluating a reference scenario for ratio resolution."""
    num_servers: int
    energy_kwh: float
    embodied_carbon_kg: float
    embodied_cost_usd: float


class CostResolver:
    """Resolves ratio-based cost specifications to raw parameters.

    Given a reference scenario and target ratios, this class evaluates the
    scenario to determine server count and energy consumption, then derives
    the raw cost parameters (carbon_intensity_g_kwh, electricity_cost_usd_kwh)
    that would achieve the target operational/embodied ratio.

    Math (Embodied Anchor):
        Given: operational_carbon_fraction (f_op), embodied_carbon_kg, reference scenario
        1. embodied_total = num_servers × embodied_carbon_kg
        2. operational_total = embodied_total × f_op / (1 - f_op)
        3. carbon_intensity_g_kwh = operational_total × 1000 / energy_kwh

    Math (Total Anchor):
        Given: operational_carbon_fraction (f_op), total_carbon_kg, reference scenario
        1. operational_total = total_carbon_kg × f_op
        2. embodied_total = total_carbon_kg × (1 - f_op)
        3. embodied_carbon_kg = embodied_total / num_servers
        4. carbon_intensity_g_kwh = operational_total × 1000 / energy_kwh
    """

    def resolve(
        self,
        cost_spec: CostSpec,
        ref_result: ReferenceScenarioResult,
    ) -> CostSpec:
        """Resolve ratio-based cost spec to raw parameters.

        Args:
            cost_spec: The ratio-based cost specification
            ref_result: Result from evaluating the reference scenario

        Returns:
            A new CostSpec with derived raw parameters
        """
        cost_spec.validate_ratio_mode()

        if cost_spec.uses_total_anchor():
            return self._resolve_with_total_anchor(cost_spec, ref_result)
        else:
            return self._resolve_with_embodied_anchor(cost_spec, ref_result)

    def _resolve_with_embodied_anchor(
        self,
        cost_spec: CostSpec,
        ref_result: ReferenceScenarioResult,
    ) -> CostSpec:
        """Derive operational params from embodied anchor.

        Uses the specified embodied_carbon_kg and server_cost_usd as anchors,
        then derives carbon_intensity_g_kwh and electricity_cost_usd_kwh to
        achieve the target operational fractions.
        """
        # Use existing values or defaults
        carbon_intensity = cost_spec.carbon_intensity_g_kwh if cost_spec.carbon_intensity_g_kwh is not None else 400.0
        electricity_cost = cost_spec.electricity_cost_usd_kwh if cost_spec.electricity_cost_usd_kwh is not None else 0.10

        # Derive carbon intensity if carbon fraction specified
        if cost_spec.operational_carbon_fraction is not None:
            carbon_intensity = self._derive_carbon_intensity(
                operational_fraction=cost_spec.operational_carbon_fraction,
                embodied_carbon_kg=cost_spec.embodied_carbon_kg,
                num_servers=ref_result.num_servers,
                energy_kwh=ref_result.energy_kwh,
            )

        # Derive electricity cost if cost fraction specified
        if cost_spec.operational_cost_fraction is not None:
            electricity_cost = self._derive_electricity_cost(
                operational_fraction=cost_spec.operational_cost_fraction,
                server_cost_usd=cost_spec.server_cost_usd,
                num_servers=ref_result.num_servers,
                energy_kwh=ref_result.energy_kwh,
            )

        return CostSpec(
            mode=CostMode.RAW,  # Mark as resolved
            embodied_carbon_kg=cost_spec.embodied_carbon_kg,
            server_cost_usd=cost_spec.server_cost_usd,
            carbon_intensity_g_kwh=carbon_intensity,
            electricity_cost_usd_kwh=electricity_cost,
            lifetime_years=cost_spec.lifetime_years,
            # Clear ratio fields
            reference_scenario=None,
            operational_carbon_fraction=None,
            operational_cost_fraction=None,
            total_carbon_kg=None,
            total_cost_usd=None,
        )

    def _resolve_with_total_anchor(
        self,
        cost_spec: CostSpec,
        ref_result: ReferenceScenarioResult,
    ) -> CostSpec:
        """Derive both embodied and operational params from total anchor.

        Uses the specified total_carbon_kg and/or total_cost_usd as anchors,
        then derives all parameters to achieve the target split.
        """
        # Start with existing values or defaults
        embodied_carbon = cost_spec.embodied_carbon_kg
        server_cost = cost_spec.server_cost_usd
        carbon_intensity = cost_spec.carbon_intensity_g_kwh if cost_spec.carbon_intensity_g_kwh is not None else 400.0
        electricity_cost = cost_spec.electricity_cost_usd_kwh if cost_spec.electricity_cost_usd_kwh is not None else 0.10

        # Derive carbon parameters from total carbon anchor
        if cost_spec.total_carbon_kg is not None and cost_spec.operational_carbon_fraction is not None:
            f_op = cost_spec.operational_carbon_fraction
            total = cost_spec.total_carbon_kg

            operational_total = total * f_op
            embodied_total = total * (1 - f_op)

            embodied_carbon = embodied_total / ref_result.num_servers
            carbon_intensity = self._operational_to_intensity(
                operational_total, ref_result.energy_kwh
            )

        # Derive cost parameters from total cost anchor
        if cost_spec.total_cost_usd is not None and cost_spec.operational_cost_fraction is not None:
            f_op = cost_spec.operational_cost_fraction
            total = cost_spec.total_cost_usd

            operational_total = total * f_op
            embodied_total = total * (1 - f_op)

            server_cost = embodied_total / ref_result.num_servers
            electricity_cost = self._operational_to_rate(
                operational_total, ref_result.energy_kwh
            )

        return CostSpec(
            mode=CostMode.RAW,  # Mark as resolved
            embodied_carbon_kg=embodied_carbon,
            server_cost_usd=server_cost,
            carbon_intensity_g_kwh=carbon_intensity,
            electricity_cost_usd_kwh=electricity_cost,
            lifetime_years=cost_spec.lifetime_years,
            # Clear ratio fields
            reference_scenario=None,
            operational_carbon_fraction=None,
            operational_cost_fraction=None,
            total_carbon_kg=None,
            total_cost_usd=None,
        )

    def _derive_carbon_intensity(
        self,
        operational_fraction: float,
        embodied_carbon_kg: float,
        num_servers: int,
        energy_kwh: float,
    ) -> float:
        """Derive carbon intensity to achieve target operational fraction.

        Given:
            f_op = operational / (operational + embodied)

        Solve for operational:
            operational = embodied * f_op / (1 - f_op)

        Then:
            carbon_intensity_g_kwh = operational * 1000 / energy_kwh
        """
        if energy_kwh <= 0:
            raise ValueError("Cannot derive carbon intensity: energy_kwh must be positive")

        embodied_total = num_servers * embodied_carbon_kg
        operational_total = embodied_total * operational_fraction / (1 - operational_fraction)

        return self._operational_to_intensity(operational_total, energy_kwh)

    def _derive_electricity_cost(
        self,
        operational_fraction: float,
        server_cost_usd: float,
        num_servers: int,
        energy_kwh: float,
    ) -> float:
        """Derive electricity cost to achieve target operational fraction.

        Same math as carbon intensity but for costs.
        """
        if energy_kwh <= 0:
            raise ValueError("Cannot derive electricity cost: energy_kwh must be positive")

        embodied_total = num_servers * server_cost_usd
        operational_total = embodied_total * operational_fraction / (1 - operational_fraction)

        return self._operational_to_rate(operational_total, energy_kwh)

    def _operational_to_intensity(self, operational_kg: float, energy_kwh: float) -> float:
        """Convert operational carbon (kg) to intensity (g/kWh)."""
        # operational_kg = energy_kwh * carbon_intensity_g_kwh / 1000
        # => carbon_intensity_g_kwh = operational_kg * 1000 / energy_kwh
        return operational_kg * 1000 / energy_kwh

    def _operational_to_rate(self, operational_usd: float, energy_kwh: float) -> float:
        """Convert operational cost (USD) to rate (USD/kWh)."""
        # operational_usd = energy_kwh * electricity_cost_usd_kwh
        # => electricity_cost_usd_kwh = operational_usd / energy_kwh
        return operational_usd / energy_kwh


@dataclass
class PowerCurveSpec:
    """Power curve configuration."""
    type: str = "specpower"
    exponent: Optional[float] = None
    freq_mhz: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PowerCurveSpec':
        return cls(
            type=data.get('type', 'specpower'),
            exponent=data.get('exponent'),
            freq_mhz=data.get('freq_mhz'),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {'type': self.type}
        if self.exponent is not None:
            d['exponent'] = self.exponent
        if self.freq_mhz is not None:
            d['freq_mhz'] = self.freq_mhz
        return d

    def to_callable(self) -> Callable[[float], float]:
        return make_power_curve_fn(self.type, self.exponent, self.freq_mhz)


@dataclass
class AnalysisConfig:
    """
    Complete declarative analysis configuration.

    This is the top-level config loaded from JSON.

    Supports loading processor configs from external files:
    - "processor": "./processors.json" - path to processor config file
    - "processor_file": "./processors.json" - alternative key for external file
    - "processor": {...} - inline processor definitions (original behavior)

    Paths are resolved relative to the config file directory.
    """
    name: str
    scenarios: Dict[str, ScenarioConfig]
    analysis: AnalysisSpec
    processor: ProcessorConfigSpec = field(default_factory=ProcessorConfigSpec)
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    cost: CostSpec = field(default_factory=CostSpec)
    power_curve: PowerCurveSpec = field(default_factory=PowerCurveSpec)
    output_dir: Optional[str] = None
    description: str = ""

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        base_path: Optional[Path] = None,
    ) -> 'AnalysisConfig':
        """Load config from dict.

        Args:
            data: Config dict
            base_path: Base directory for resolving relative paths (e.g., processor_file)

        Returns:
            AnalysisConfig instance
        """
        scenarios = {}
        for name, spec in data.get('scenarios', {}).items():
            scenarios[name] = ScenarioConfig.from_dict(spec)

        # Handle processor config - can be inline dict, string path, or via processor_file key
        processor_data = data.get('processor', {})
        processor_file = data.get('processor_file')

        if processor_file:
            # processor_file key takes precedence
            processor_spec = ProcessorConfigSpec.from_dict(processor_file, base_path)
        elif isinstance(processor_data, str):
            # processor is a path string
            processor_spec = ProcessorConfigSpec.from_dict(processor_data, base_path)
        else:
            # processor is inline dict (original behavior)
            processor_spec = ProcessorConfigSpec.from_dict(processor_data, base_path)

        return cls(
            name=data.get('name', 'unnamed'),
            description=data.get('description', ''),
            scenarios=scenarios,
            analysis=AnalysisSpec.from_dict(data.get('analysis', {})),
            processor=processor_spec,
            workload=WorkloadSpec.from_dict(data.get('workload', {})),
            cost=CostSpec.from_dict(data.get('cost', {})),
            power_curve=PowerCurveSpec.from_dict(data.get('power_curve', {})),
            output_dir=data.get('output_dir'),
        )

    @classmethod
    def from_json(cls, path: Path) -> 'AnalysisConfig':
        """Load config from JSON file. Supports comments if json5 is installed.

        Relative paths in the config (e.g., processor_file) are resolved
        relative to the config file's directory.
        """
        path = Path(path).resolve()
        with open(path, 'r') as f:
            if _HAS_JSON5:
                data = json5.load(f)
            else:
                data = json.load(f)
        return cls.from_dict(data, base_path=path.parent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'scenarios': {k: v.to_dict() for k, v in self.scenarios.items()},
            'analysis': self.analysis.to_dict(),
            'processor': self.processor.to_dict(),
            'workload': self.workload.to_dict(),
            'cost': self.cost.to_dict(),
            'power_curve': self.power_curve.to_dict(),
            'output_dir': self.output_dir,
        }


@dataclass
class AnalysisResult:
    """Result of running a declarative analysis."""
    config: AnalysisConfig
    analysis_type: str
    scenario_results: Dict[str, Dict[str, Any]]
    comparisons: Dict[str, Dict[str, float]]
    breakeven: Optional[BreakevenResult] = None
    sweep_results: Optional[List[Dict[str, Any]]] = None
    compare_sweep_results: Optional[List[Dict[str, Any]]] = None  # For compare_sweep analysis
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'config': self.config.to_dict(),
            'analysis_type': self.analysis_type,
            'scenario_results': self.scenario_results,
            'comparisons': self.comparisons,
            'summary': self.summary,
        }
        if self.breakeven:
            result['breakeven'] = {
                'breakeven_value': self.breakeven.breakeven_value,
                'achieved': self.breakeven.achieved,
                'iterations': self.breakeven.iterations,
                'search_history': [
                    {'value': h.value, 'error': h.error, 'metrics': h.metric_values}
                    for h in self.breakeven.search_history
                ],
                'final_result': self.breakeven.final_result,
                'error_message': self.breakeven.error_message,
            }
        if self.sweep_results:
            result['sweep_results'] = self.sweep_results
        if self.compare_sweep_results:
            result['compare_sweep_results'] = self.compare_sweep_results
        return result


class DeclarativeAnalysisEngine:
    """
    Engine for running declarative analyses.

    Supports three analysis types:
    - find_breakeven: Binary search to find parameter value matching reference
    - compare: Compare multiple scenarios
    - sweep: Run breakeven analysis across parameter sweep
    """

    def __init__(self):
        self._builder: Optional[ScenarioBuilder] = None
        self._model: Optional[OverssubModel] = None
        self._config: Optional[AnalysisConfig] = None

    def run(self, config: AnalysisConfig) -> AnalysisResult:
        """
        Run analysis from config.

        Args:
            config: AnalysisConfig instance

        Returns:
            AnalysisResult with all computed results
        """
        self._config = config
        # Save original cost spec for sweep operations that may need it
        self._original_cost = copy.deepcopy(config.cost)

        # Resolve ratio-based costs if needed (except for sweeps over ratio params)
        analysis_type = config.analysis.type
        if analysis_type == 'sweep' and self._is_ratio_sweep_param(config.analysis.sweep_parameter):
            # Don't resolve yet - sweep will handle resolution per iteration
            self._setup_builder_for_ratio_sweep()
        elif config.cost.is_ratio_based():
            self._resolve_ratio_costs()
        else:
            self._setup_builder()

        if analysis_type == 'find_breakeven':
            return self._run_find_breakeven()
        elif analysis_type == 'compare':
            return self._run_compare()
        elif analysis_type == 'sweep':
            return self._run_sweep()
        elif analysis_type == 'compare_sweep':
            return self._run_compare_sweep()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    def run_from_file(self, path: Union[str, Path]) -> AnalysisResult:
        """Load config from JSON and run analysis."""
        config = AnalysisConfig.from_json(Path(path))
        return self.run(config)

    def _setup_builder(self):
        """Set up scenario builder from config."""
        cfg = self._config
        proc = cfg.processor
        cost = cfg.cost

        # Get processor specs (supporting arbitrary names)
        smt_spec = proc.get("smt") if "smt" in proc.processors else None
        nosmt_spec = proc.get("nosmt") if "nosmt" in proc.processors else None

        # Build ProcessorDefaults from the processor specs
        # For backward compatibility, we use smt/nosmt if available
        proc_defaults = ProcessorDefaults(
            smt_physical_cores=smt_spec.physical_cores if smt_spec else 48,
            smt_threads_per_core=smt_spec.threads_per_core if smt_spec else 2,
            smt_power_idle_w=smt_spec.power_idle_w if smt_spec else 100.0,
            smt_power_max_w=smt_spec.power_max_w if smt_spec else 400.0,
            smt_core_overhead=smt_spec.core_overhead if smt_spec else 0,
            nosmt_physical_cores=nosmt_spec.physical_cores if nosmt_spec else 48,
            nosmt_threads_per_core=nosmt_spec.threads_per_core if nosmt_spec else 1,
            nosmt_power_idle_w=nosmt_spec.power_idle_w if nosmt_spec else 90.0,
            nosmt_power_max_w=nosmt_spec.power_max_w if nosmt_spec else 340.0,
            nosmt_core_overhead=nosmt_spec.core_overhead if nosmt_spec else 0,
        )

        cost_defaults = CostDefaults(
            embodied_carbon_kg=cost.embodied_carbon_kg,
            server_cost_usd=cost.server_cost_usd,
            carbon_intensity_g_kwh=cost.carbon_intensity_g_kwh,
            electricity_cost_usd_kwh=cost.electricity_cost_usd_kwh,
            lifetime_years=cost.lifetime_years,
        )

        power_fn = cfg.power_curve.to_callable()
        self._builder = ScenarioBuilder(proc_defaults, cost_defaults, power_fn)

        # Store processor config for direct access
        self._processor_config = proc

        workload = self._builder.build_workload_params(
            cfg.workload.total_vcpus,
            cfg.workload.avg_util,
        )
        cost_params = self._builder.build_cost_params()
        self._model = OverssubModel(workload, cost_params)

    def _resolve_ratio_costs(self):
        """Resolve ratio-based cost specification to raw parameters.

        This method:
        1. Sets up a temporary builder with placeholder costs
        2. Evaluates the reference scenario to get server count and energy
        3. Uses CostResolver to derive the raw parameters
        4. Replaces config.cost with the resolved CostSpec
        5. Calls _setup_builder() with the resolved costs
        """
        cfg = self._config
        cost = cfg.cost

        cost.validate_ratio_mode()

        ref_scenario_name = cost.reference_scenario
        if ref_scenario_name not in cfg.scenarios:
            raise ValueError(
                f"reference_scenario '{ref_scenario_name}' not found in scenarios. "
                f"Available: {list(cfg.scenarios.keys())}"
            )

        # Set up temporary builder with placeholder costs to evaluate reference
        # We use raw defaults just to get server count and power consumption
        self._setup_builder_with_placeholder_costs()

        # Evaluate reference scenario to get energy and server count
        ref_scenario_cfg = cfg.scenarios[ref_scenario_name]
        ref_params = self._build_scenario_params(ref_scenario_cfg)
        ref_result = self._model.evaluate_scenario(ref_params)

        # Calculate energy consumption
        energy_kwh = (
            ref_result.num_servers *
            ref_result.power_per_server_w *
            cost.lifetime_years * 8760 / 1000
        )

        ref_scenario_result = ReferenceScenarioResult(
            num_servers=ref_result.num_servers,
            energy_kwh=energy_kwh,
            embodied_carbon_kg=ref_result.embodied_carbon_kg,
            embodied_cost_usd=ref_result.embodied_cost_usd,
        )

        # Resolve ratio-based costs to raw parameters
        resolver = CostResolver()
        resolved_cost = resolver.resolve(cost, ref_scenario_result)

        # Replace config's cost with resolved version
        self._config.cost = resolved_cost

        # Re-setup builder with resolved costs
        self._setup_builder()

    def _setup_builder_with_placeholder_costs(self):
        """Set up builder with placeholder costs for reference scenario evaluation.

        Uses the embodied values from config but placeholder operational values,
        since we only need server count and power for ratio resolution.
        """
        cfg = self._config
        proc = cfg.processor
        cost = cfg.cost

        # Get processor specs (supporting arbitrary names)
        smt_spec = proc.get("smt") if "smt" in proc.processors else None
        nosmt_spec = proc.get("nosmt") if "nosmt" in proc.processors else None

        # Build ProcessorDefaults from the processor specs
        proc_defaults = ProcessorDefaults(
            smt_physical_cores=smt_spec.physical_cores if smt_spec else 48,
            smt_threads_per_core=smt_spec.threads_per_core if smt_spec else 2,
            smt_power_idle_w=smt_spec.power_idle_w if smt_spec else 100.0,
            smt_power_max_w=smt_spec.power_max_w if smt_spec else 400.0,
            smt_core_overhead=smt_spec.core_overhead if smt_spec else 0,
            nosmt_physical_cores=nosmt_spec.physical_cores if nosmt_spec else 48,
            nosmt_threads_per_core=nosmt_spec.threads_per_core if nosmt_spec else 1,
            nosmt_power_idle_w=nosmt_spec.power_idle_w if nosmt_spec else 90.0,
            nosmt_power_max_w=nosmt_spec.power_max_w if nosmt_spec else 340.0,
            nosmt_core_overhead=nosmt_spec.core_overhead if nosmt_spec else 0,
        )

        # Use placeholder operational values - we only need server count and power
        cost_defaults = CostDefaults(
            embodied_carbon_kg=cost.embodied_carbon_kg,
            server_cost_usd=cost.server_cost_usd,
            carbon_intensity_g_kwh=1.0,  # Placeholder
            electricity_cost_usd_kwh=0.01,  # Placeholder
            lifetime_years=cost.lifetime_years,
        )

        power_fn = cfg.power_curve.to_callable()
        self._builder = ScenarioBuilder(proc_defaults, cost_defaults, power_fn)

        # Store processor config for direct access
        self._processor_config = proc

        workload = self._builder.build_workload_params(
            cfg.workload.total_vcpus,
            cfg.workload.avg_util,
        )
        cost_params = self._builder.build_cost_params()
        self._model = OverssubModel(workload, cost_params)

    def _setup_builder_for_ratio_sweep(self):
        """Set up builder with placeholder costs for ratio parameter sweeps.

        Similar to _setup_builder_with_placeholder_costs, but used when
        we're sweeping over ratio parameters and need the initial setup
        before iterating.
        """
        self._setup_builder_with_placeholder_costs()

    def _build_scenario_spec(self, name: str, scenario_cfg: ScenarioConfig) -> ScenarioSpec:
        """Build a ScenarioSpec from config."""
        proc_name = scenario_cfg.processor
        overrides = scenario_cfg.overrides or {}

        # Get processor spec and determine if SMT based on threads_per_core
        proc_spec = self._processor_config.get(proc_name)
        is_smt = proc_spec.threads_per_core > 1

        return self._builder.build_scenario(
            name=name,
            smt=is_smt,
            oversub_ratio=scenario_cfg.oversub_ratio,
            util_overhead=scenario_cfg.util_overhead,
            vcpu_demand_multiplier=scenario_cfg.vcpu_demand_multiplier,
            **overrides,
        )

    def _build_scenario_params(self, scenario_cfg: ScenarioConfig) -> ScenarioParams:
        """Build ScenarioParams from config."""
        proc_name = scenario_cfg.processor
        overrides = scenario_cfg.overrides or {}

        # Get processor spec from config
        proc_spec = self._processor_config.get(proc_name)

        # Build ProcessorConfig from spec with overrides
        # Use per-processor power curve if specified, otherwise fall back to global
        if proc_spec.power_curve is not None:
            power_fn = proc_spec.power_curve.to_callable()
        else:
            power_fn = self._config.power_curve.to_callable()

        physical_cores = overrides.get('physical_cores', proc_spec.physical_cores)
        threads_per_core = overrides.get('threads_per_core', proc_spec.threads_per_core)
        power_idle = overrides.get('power_idle_w', proc_spec.power_idle_w)
        power_max = overrides.get('power_max_w', proc_spec.power_max_w)
        core_overhead = overrides.get('core_overhead', proc_spec.core_overhead)

        # Build power components and composite curve if power_breakdown is present
        # Components without a power_curve use the global power curve; if no global, default to polynomial
        power_components = None
        if proc_spec.power_breakdown:
            global_curve = self._config.power_curve
            power_components = {
                name: comp.to_power_component_curve(default_curve_spec=global_curve)
                for name, comp in proc_spec.power_breakdown.items()
            }
            power_curve = build_composite_power_curve(power_components)
        else:
            power_curve = PowerCurve(
                p_idle=power_idle,
                p_max=power_max,
                curve_fn=power_fn,
            )

        processor = ProcessorConfig(
            physical_cores=physical_cores,
            threads_per_core=threads_per_core,
            power_curve=power_curve,
            core_overhead=core_overhead,
            power_components=power_components,
        )

        return ScenarioParams(
            processor=processor,
            oversub_ratio=scenario_cfg.oversub_ratio,
            util_overhead=scenario_cfg.util_overhead,
            vcpu_demand_multiplier=scenario_cfg.vcpu_demand_multiplier,
        )

    def _evaluate_scenario(self, name: str) -> Tuple[ScenarioParams, ScenarioResult]:
        """Evaluate a scenario by name and return params and result.

        Resolves embodied carbon and server cost using priority chain:
        1. Processor-level structured (embodied_carbon: {per_core, per_server})
        2. Processor-level flat (embodied_carbon_kg)
        3. Global cost structured (cost.embodied_carbon: {...})
        4. Global cost flat (cost.embodied_carbon_kg) - already in model defaults
        """
        scenario_cfg = self._config.scenarios[name]
        params = self._build_scenario_params(scenario_cfg)

        proc_name = scenario_cfg.processor
        proc_spec = self._processor_config.get(proc_name)
        phys = params.processor.physical_cores
        tpc = params.processor.threads_per_core
        cost_overrides = {}

        # Resolve embodied carbon: structured > flat > global structured > global flat
        carbon_breakdown = None
        if proc_spec.embodied_carbon is not None:
            cost_overrides['embodied_carbon_kg'] = proc_spec.embodied_carbon.resolve_total(phys, tpc)
            carbon_breakdown = proc_spec.embodied_carbon.to_component_breakdown(phys, tpc)
        elif proc_spec.embodied_carbon_kg is not None:
            cost_overrides['embodied_carbon_kg'] = proc_spec.embodied_carbon_kg
        elif self._config.cost.embodied_carbon is not None:
            cost_overrides['embodied_carbon_kg'] = self._config.cost.embodied_carbon.resolve_total(phys, tpc)
            carbon_breakdown = self._config.cost.embodied_carbon.to_component_breakdown(phys, tpc)

        # Resolve server cost: structured > flat > global structured > global flat
        cost_breakdown = None
        if proc_spec.server_cost is not None:
            cost_overrides['server_cost_usd'] = proc_spec.server_cost.resolve_total(phys, tpc)
            cost_breakdown = proc_spec.server_cost.to_component_breakdown(phys, tpc)
        elif proc_spec.server_cost_usd is not None:
            cost_overrides['server_cost_usd'] = proc_spec.server_cost_usd
        elif self._config.cost.server_cost is not None:
            cost_overrides['server_cost_usd'] = self._config.cost.server_cost.resolve_total(phys, tpc)
            cost_breakdown = self._config.cost.server_cost.to_component_breakdown(phys, tpc)

        # Attach breakdown metadata if available
        if carbon_breakdown is not None:
            cost_overrides['carbon_breakdown'] = carbon_breakdown
        if cost_breakdown is not None:
            cost_overrides['cost_breakdown'] = cost_breakdown

        result = self._model.evaluate_scenario(params, cost_overrides if cost_overrides else None)
        return params, result

    def _run_find_breakeven(self) -> AnalysisResult:
        """Run breakeven finding analysis."""
        analysis = self._config.analysis

        # Evaluate baseline and reference
        baseline_params, baseline_result = self._evaluate_scenario(analysis.baseline)
        ref_params, reference_result = self._evaluate_scenario(analysis.reference)
        target_params, _ = self._evaluate_scenario(analysis.target)

        # Build match condition
        match_metric = analysis.match_metric
        if isinstance(match_metric, str):
            condition = SimpleCondition(
                metric=match_metric,
                match_type=MatchType.EQUAL,
            )
        elif isinstance(match_metric, dict):
            condition = CompoundCondition.from_dict(match_metric)
        else:
            condition = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL)

        # Run breakeven search
        finder = GeneralizedBreakevenFinder(self._model)
        bounds = tuple(analysis.search_bounds or [1.0, 10.0])
        breakeven_result = finder.find_breakeven(
            base_scenario=target_params,
            reference_result=reference_result,
            vary_parameter=analysis.vary_parameter,
            match_condition=condition,
            search_bounds=bounds,
        )

        # Collect results
        scenario_results = {
            analysis.baseline: asdict(baseline_result),
            analysis.reference: asdict(reference_result),
        }

        if breakeven_result.final_result:
            scenario_results[analysis.target] = breakeven_result.final_result

        # Compute comparisons
        comparisons = self._compute_comparisons(
            scenario_results,
            baseline_name=analysis.baseline,
        )

        # Build summary
        summary = self._build_breakeven_summary(
            analysis, breakeven_result, baseline_result, reference_result
        )

        return AnalysisResult(
            config=self._config,
            analysis_type='find_breakeven',
            scenario_results=scenario_results,
            comparisons=comparisons,
            breakeven=breakeven_result,
            summary=summary,
        )

    def _run_compare(self) -> AnalysisResult:
        """Run scenario comparison analysis."""
        analysis = self._config.analysis
        scenarios_to_compare = analysis.scenarios or list(self._config.scenarios.keys())
        baseline_name = analysis.baseline or scenarios_to_compare[0]

        scenario_results = {}
        for name in scenarios_to_compare:
            _, result = self._evaluate_scenario(name)
            scenario_results[name] = asdict(result)

        comparisons = self._compute_comparisons(scenario_results, baseline_name)
        summary = self._build_compare_summary(scenario_results, comparisons, baseline_name)

        return AnalysisResult(
            config=self._config,
            analysis_type='compare',
            scenario_results=scenario_results,
            comparisons=comparisons,
            summary=summary,
        )

    def _run_sweep(self) -> AnalysisResult:
        """Run sweep analysis (breakeven at multiple parameter values)."""
        analysis = self._config.analysis
        sweep_param = analysis.sweep_parameter
        sweep_values = analysis.sweep_values

        if not sweep_param or not sweep_values:
            raise ValueError("Sweep analysis requires sweep_parameter and sweep_values")

        param_path = ParameterPath(sweep_param)
        sweep_results = []

        for value in sweep_values:
            # Reset cost spec for each iteration when sweeping ratio params
            if self._is_ratio_sweep_param(sweep_param):
                self._config.cost = copy.deepcopy(self._original_cost)

            # Modify workload or cost based on sweep parameter
            requires_ratio_resolution = self._apply_sweep_value(param_path, value)

            # Rebuild with new values, resolving ratios if needed
            if requires_ratio_resolution and self._config.cost.is_ratio_based():
                self._resolve_ratio_costs()
            else:
                self._setup_builder()

            # Run breakeven for this value
            _, baseline_result = self._evaluate_scenario(analysis.baseline)
            _, reference_result = self._evaluate_scenario(analysis.reference)
            target_params, _ = self._evaluate_scenario(analysis.target)

            # Build match condition
            match_metric = analysis.match_metric
            if isinstance(match_metric, str):
                condition = SimpleCondition(metric=match_metric, match_type=MatchType.EQUAL)
            else:
                condition = CompoundCondition.from_dict(match_metric)

            finder = GeneralizedBreakevenFinder(self._model)
            bounds = tuple(analysis.search_bounds or [1.0, 10.0])
            be_result = finder.find_breakeven(
                base_scenario=target_params,
                reference_result=reference_result,
                vary_parameter=analysis.vary_parameter,
                match_condition=condition,
                search_bounds=bounds,
            )

            sweep_results.append({
                'parameter_value': value,
                'breakeven_value': be_result.breakeven_value,
                'achieved': be_result.achieved,
                'baseline': asdict(baseline_result),
                'reference': asdict(reference_result),
            })

        summary = self._build_sweep_summary(sweep_param, sweep_results)

        return AnalysisResult(
            config=self._config,
            analysis_type='sweep',
            scenario_results={},
            comparisons={},
            sweep_results=sweep_results,
            summary=summary,
        )

    def _run_compare_sweep(self) -> AnalysisResult:
        """Run comparison sweep: compare scenarios while sweeping a parameter.

        This analysis type computes % difference from baseline at each sweep value,
        without searching for breakeven. Useful for sensitivity analysis showing
        how savings change across parameter ranges.

        Supports multiple sweep scenarios for multi-line plots via sweep_scenarios.
        """
        analysis = self._config.analysis
        sweep_param = analysis.sweep_parameter
        sweep_values = analysis.sweep_values
        baseline_name = analysis.baseline

        # Support both single scenario (sweep_scenario) and multiple (sweep_scenarios)
        if analysis.sweep_scenarios:
            sweep_scenario_list = analysis.sweep_scenarios
        elif analysis.sweep_scenario or analysis.target:
            sweep_scenario_list = [analysis.sweep_scenario or analysis.target]
        else:
            raise ValueError("compare_sweep requires sweep_scenario, sweep_scenarios, or target")

        if not sweep_param or not sweep_values:
            raise ValueError("compare_sweep requires sweep_parameter and sweep_values")
        if not baseline_name:
            raise ValueError("compare_sweep requires baseline scenario")

        # Results organized by scenario for multi-line support
        compare_sweep_results = []

        # Store original scenario configs
        original_configs = {
            name: copy.deepcopy(self._config.scenarios[name])
            for name in sweep_scenario_list
        }

        for value in sweep_values:
            # Reset all scenario configs and apply sweep value
            for scenario_name in sweep_scenario_list:
                self._config.scenarios[scenario_name] = copy.deepcopy(original_configs[scenario_name])
                self._apply_scenario_sweep_value(scenario_name, sweep_param, value)

            # Rebuild model
            self._setup_builder()

            # Evaluate baseline once
            _, baseline_result = self._evaluate_scenario(baseline_name)
            baseline_dict = asdict(baseline_result)

            # Evaluate each sweep scenario
            scenario_results = {}
            for scenario_name in sweep_scenario_list:
                _, sweep_result = self._evaluate_scenario(scenario_name)
                sweep_dict = asdict(sweep_result)

                # Compute % differences
                carbon_diff_pct = self._pct_diff(
                    sweep_dict.get('total_carbon_kg', 0),
                    baseline_dict.get('total_carbon_kg', 1),
                )
                tco_diff_pct = self._pct_diff(
                    sweep_dict.get('total_cost_usd', 0),
                    baseline_dict.get('total_cost_usd', 1),
                )
                server_diff_pct = self._pct_diff(
                    sweep_dict.get('num_servers', 0),
                    baseline_dict.get('num_servers', 1),
                )

                scenario_results[scenario_name] = {
                    'result': sweep_dict,
                    'carbon_diff_pct': carbon_diff_pct,
                    'tco_diff_pct': tco_diff_pct,
                    'server_diff_pct': server_diff_pct,
                    'carbon_diff_abs': sweep_dict.get('total_carbon_kg', 0) - baseline_dict.get('total_carbon_kg', 0),
                    'tco_diff_abs': sweep_dict.get('total_cost_usd', 0) - baseline_dict.get('total_cost_usd', 0),
                    'server_diff_abs': sweep_dict.get('num_servers', 0) - baseline_dict.get('num_servers', 0),
                }

            result_entry = {
                'parameter_value': value,
                'baseline': baseline_dict,
                'scenarios': scenario_results,
            }

            # For backward compatibility with single-scenario case
            if len(sweep_scenario_list) == 1:
                single_name = sweep_scenario_list[0]
                result_entry['sweep_scenario'] = scenario_results[single_name]['result']
                result_entry['carbon_diff_pct'] = scenario_results[single_name]['carbon_diff_pct']
                result_entry['tco_diff_pct'] = scenario_results[single_name]['tco_diff_pct']
                result_entry['server_diff_pct'] = scenario_results[single_name]['server_diff_pct']
                result_entry['carbon_diff_abs'] = scenario_results[single_name]['carbon_diff_abs']
                result_entry['tco_diff_abs'] = scenario_results[single_name]['tco_diff_abs']
                result_entry['server_diff_abs'] = scenario_results[single_name]['server_diff_abs']

            compare_sweep_results.append(result_entry)

        # Restore original scenario configs
        for scenario_name, cfg in original_configs.items():
            self._config.scenarios[scenario_name] = cfg

        summary = self._build_compare_sweep_summary(
            sweep_param, sweep_scenario_list, baseline_name, compare_sweep_results
        )

        return AnalysisResult(
            config=self._config,
            analysis_type='compare_sweep',
            scenario_results={},
            comparisons={},
            compare_sweep_results=compare_sweep_results,
            summary=summary,
        )

    def _apply_scenario_sweep_value(
        self,
        scenario_name: str,
        param: str,
        value: float,
    ) -> None:
        """Apply sweep value to a specific scenario's parameters.

        Args:
            scenario_name: Name of the scenario to modify
            param: Parameter name (e.g., 'vcpu_demand_multiplier', 'oversub_ratio')
            value: Value to set
        """
        scenario_cfg = self._config.scenarios[scenario_name]

        # Handle scenario-level parameters
        if param == 'vcpu_demand_multiplier':
            scenario_cfg.vcpu_demand_multiplier = value
        elif param == 'oversub_ratio':
            scenario_cfg.oversub_ratio = value
        elif param == 'util_overhead':
            scenario_cfg.util_overhead = value
        else:
            # Try to apply as a workload or cost parameter
            param_path = ParameterPath(param)
            self._apply_sweep_value(param_path, value)

    def _build_compare_sweep_summary(
        self,
        sweep_param: str,
        sweep_scenarios: List[str],
        baseline_name: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Build summary for compare_sweep analysis."""
        is_multi = len(sweep_scenarios) > 1
        labels = self._config.analysis.labels or {}
        param_label = self._config.analysis.sweep_parameter_label or sweep_param

        def get_label(name: str) -> str:
            return labels.get(name, name)

        lines = [
            f"# Compare Sweep Analysis: {self._config.name}",
            "",
            f"## Configuration",
            f"- Baseline: {get_label(baseline_name)}",
        ]

        if is_multi:
            scenario_labels = [get_label(s) for s in sweep_scenarios]
            lines.append(f"- Sweep Scenarios: {', '.join(scenario_labels)}")
        else:
            lines.append(f"- Sweep Scenario: {get_label(sweep_scenarios[0])}")

        lines.extend([
            f"- Sweep Parameter: {param_label}",
            "",
        ])

        # For multi-scenario, create a table per scenario
        for scenario_name in sweep_scenarios:
            scenario_label = get_label(scenario_name)
            if is_multi:
                lines.append(f"## Results for {scenario_label}: % Change vs Baseline")
            else:
                lines.append("## Results: % Change vs Baseline")
            lines.append("")
            lines.append("| {:<15} | {:>12} | {:>12} | {:>12} |".format(
                param_label[:15], "Carbon %", "TCO %", "Servers %"
            ))
            lines.append("|{:-<17}|{:->14}|{:->14}|{:->14}|".format("", "", "", ""))

            for r in results:
                value = r['parameter_value']
                if is_multi:
                    scenario_data = r['scenarios'].get(scenario_name, {})
                    carbon_pct = scenario_data.get('carbon_diff_pct', 0)
                    tco_pct = scenario_data.get('tco_diff_pct', 0)
                    server_pct = scenario_data.get('server_diff_pct', 0)
                else:
                    carbon_pct = r['carbon_diff_pct']
                    tco_pct = r['tco_diff_pct']
                    server_pct = r['server_diff_pct']
                lines.append(
                    "| {:<15.3f} | {:>+11.1f}% | {:>+11.1f}% | {:>+11.1f}% |".format(
                        value, carbon_pct, tco_pct, server_pct
                    )
                )
            lines.append("")

        # Find and report breakeven points
        lines.append("## Breakeven Points (where % change crosses 0)")
        for scenario_name in sweep_scenarios:
            scenario_label = get_label(scenario_name)
            breakeven = self._find_breakeven_in_results(results, scenario_name, 'carbon_diff_pct', is_multi)
            if breakeven is not None:
                lines.append(f"- {scenario_label} Carbon breakeven: {param_label} = {breakeven:.3f}")
            else:
                lines.append(f"- {scenario_label} Carbon breakeven: not found in range")
        lines.append("")

        lines.extend([
            "## Interpretation",
            "- Negative % = savings vs baseline (good)",
            "- Positive % = increase vs baseline (bad)",
        ])

        return "\n".join(lines)

    def _find_breakeven_in_results(
        self,
        results: List[Dict[str, Any]],
        scenario_name: str,
        metric: str,
        is_multi: bool,
    ) -> Optional[float]:
        """Find the breakeven point (where metric crosses 0) using linear interpolation."""
        if len(results) < 2:
            return None

        for i in range(len(results) - 1):
            if is_multi:
                val1 = results[i]['scenarios'].get(scenario_name, {}).get(metric, 0)
                val2 = results[i + 1]['scenarios'].get(scenario_name, {}).get(metric, 0)
            else:
                val1 = results[i].get(metric, 0)
                val2 = results[i + 1].get(metric, 0)

            x1 = results[i]['parameter_value']
            x2 = results[i + 1]['parameter_value']

            # Check if crosses zero
            if (val1 <= 0 <= val2) or (val2 <= 0 <= val1):
                # Linear interpolation to find x where y=0
                if val2 == val1:
                    return x1
                t = -val1 / (val2 - val1)
                return x1 + t * (x2 - x1)

        return None

    def _is_ratio_sweep_param(self, param: str) -> bool:
        """Check if parameter is a ratio-based cost parameter."""
        ratio_params = {
            'cost.operational_carbon_fraction',
            'cost.operational_cost_fraction',
            'cost.total_carbon_kg',
            'cost.total_cost_usd',
        }
        return param in ratio_params

    def _apply_sweep_value(self, param_path: ParameterPath, value: float) -> bool:
        """Apply sweep parameter value to config.

        Args:
            param_path: The parameter path to modify
            value: The value to set

        Returns:
            True if the parameter requires ratio cost re-resolution
        """
        # Handle different parameter targets
        first_part = param_path.parts[0]
        rest = '.'.join(param_path.parts[1:]) if len(param_path.parts) > 1 else None

        requires_ratio_resolution = False

        if first_part == 'workload':
            if rest == 'avg_util':
                self._config.workload.avg_util = value
            elif rest == 'total_vcpus':
                self._config.workload.total_vcpus = int(value)
        elif first_part == 'cost':
            if rest == 'embodied_carbon_kg':
                self._config.cost.embodied_carbon_kg = value
            elif rest == 'carbon_intensity_g_kwh':
                self._config.cost.carbon_intensity_g_kwh = value
            elif rest == 'lifetime_years':
                self._config.cost.lifetime_years = value
            elif rest == 'operational_carbon_fraction':
                self._config.cost.operational_carbon_fraction = value
                self._config.cost.mode = CostMode.RATIO_BASED
                requires_ratio_resolution = True
            elif rest == 'operational_cost_fraction':
                self._config.cost.operational_cost_fraction = value
                self._config.cost.mode = CostMode.RATIO_BASED
                requires_ratio_resolution = True
            elif rest == 'server_cost_usd':
                self._config.cost.server_cost_usd = value
            elif rest == 'electricity_cost_usd_kwh':
                self._config.cost.electricity_cost_usd_kwh = value
            elif rest == 'total_carbon_kg':
                self._config.cost.total_carbon_kg = value
                self._config.cost.mode = CostMode.RATIO_BASED
                requires_ratio_resolution = True
            elif rest == 'total_cost_usd':
                self._config.cost.total_cost_usd = value
                self._config.cost.mode = CostMode.RATIO_BASED
                requires_ratio_resolution = True
        elif first_part in ('avg_util', 'total_vcpus'):
            # Direct workload params
            if first_part == 'avg_util':
                self._config.workload.avg_util = value
            else:
                self._config.workload.total_vcpus = int(value)

        return requires_ratio_resolution

    def _compute_comparisons(
        self,
        results: Dict[str, Dict[str, Any]],
        baseline_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute comparison metrics vs baseline."""
        baseline = results.get(baseline_name, {})
        if not baseline:
            return {}

        comparisons = {}
        for name, result in results.items():
            if name == baseline_name:
                continue

            comparisons[name] = {
                'carbon_diff_pct': self._pct_diff(
                    result.get('total_carbon_kg', 0),
                    baseline.get('total_carbon_kg', 1),
                ),
                'tco_diff_pct': self._pct_diff(
                    result.get('total_cost_usd', 0),
                    baseline.get('total_cost_usd', 1),
                ),
                'server_diff_pct': self._pct_diff(
                    result.get('num_servers', 0),
                    baseline.get('num_servers', 1),
                ),
                'carbon_diff_abs': result.get('total_carbon_kg', 0) - baseline.get('total_carbon_kg', 0),
                'tco_diff_abs': result.get('total_cost_usd', 0) - baseline.get('total_cost_usd', 0),
                'server_diff_abs': result.get('num_servers', 0) - baseline.get('num_servers', 0),
            }

        return comparisons

    def _pct_diff(self, value: float, baseline: float) -> float:
        if baseline == 0:
            return 0.0
        return ((value - baseline) / baseline) * 100

    def _build_breakeven_summary(
        self,
        analysis: AnalysisSpec,
        breakeven: BreakevenResult,
        baseline: ScenarioResult,
        reference: ScenarioResult,
    ) -> str:
        """Build human-readable summary for breakeven analysis."""
        lines = [
            f"# Breakeven Analysis: {self._config.name}",
            "",
            f"## Configuration",
            f"- Baseline: {analysis.baseline}",
            f"- Reference: {analysis.reference}",
            f"- Target: {analysis.target}",
            f"- Vary parameter: {analysis.vary_parameter}",
            f"- Match metric: {analysis.match_metric}",
            "",
            f"## Results",
        ]

        if breakeven.achieved:
            lines.append(f"**Breakeven value: {breakeven.breakeven_value:.4f}**")
            lines.append(f"- Found in {breakeven.iterations} iterations")
        else:
            lines.append(f"**Breakeven not achieved**")
            if breakeven.error_message:
                lines.append(f"- Reason: {breakeven.error_message}")

        lines.extend([
            "",
            f"## Scenario Comparison",
            f"- Baseline carbon: {baseline.total_carbon_kg:,.0f} kg CO2e",
            f"- Reference carbon: {reference.total_carbon_kg:,.0f} kg CO2e",
            f"- Baseline TCO: ${baseline.total_cost_usd:,.0f}",
            f"- Reference TCO: ${reference.total_cost_usd:,.0f}",
        ])

        if breakeven.final_result:
            final = breakeven.final_result
            lines.extend([
                "",
                f"## Target at Breakeven",
                f"- Carbon: {final['total_carbon_kg']:,.0f} kg CO2e",
                f"- TCO: ${final['total_cost_usd']:,.0f}",
                f"- Servers: {final['num_servers']}",
            ])

        return "\n".join(lines)

    def _build_compare_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        comparisons: Dict[str, Dict[str, float]],
        baseline_name: str,
    ) -> str:
        """Build summary for comparison analysis."""
        lines = [
            f"# Scenario Comparison: {self._config.name}",
            "",
            f"## Baseline: {baseline_name}",
        ]

        baseline = results.get(baseline_name, {})
        if baseline:
            lines.extend([
                f"- Carbon: {baseline.get('total_carbon_kg', 0):,.0f} kg CO2e",
                f"- TCO: ${baseline.get('total_cost_usd', 0):,.0f}",
                f"- Servers: {baseline.get('num_servers', 0)}",
            ])

        lines.append("")
        lines.append("## Comparisons vs Baseline")

        for name, comp in comparisons.items():
            lines.extend([
                "",
                f"### {name}",
                f"- Carbon: {comp['carbon_diff_pct']:+.1f}%",
                f"- TCO: {comp['tco_diff_pct']:+.1f}%",
                f"- Servers: {comp['server_diff_pct']:+.1f}%",
            ])

        return "\n".join(lines)

    def _build_sweep_summary(
        self,
        sweep_param: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Build summary for sweep analysis."""
        lines = [
            f"# Sweep Analysis: {self._config.name}",
            "",
            f"## Sweep Parameter: {sweep_param}",
            "",
            "| Value | Breakeven | Achieved |",
            "|-------|-----------|----------|",
        ]

        for r in results:
            value = r['parameter_value']
            be_val = r['breakeven_value']
            achieved = "Yes" if r['achieved'] else "No"
            be_str = f"{be_val:.4f}" if be_val is not None else "N/A"
            lines.append(f"| {value} | {be_str} | {achieved} |")

        return "\n".join(lines)


def is_valid_analysis_config(path: Path) -> bool:
    """
    Check if a file is a valid declarative analysis config.

    A valid config must have 'name', 'scenarios', and 'analysis' keys.
    """
    try:
        with open(path, 'r') as f:
            if _HAS_JSON5:
                data = json5.load(f)
            else:
                data = json.load(f)
        # Check for required keys
        return all(key in data for key in ('name', 'scenarios', 'analysis'))
    except (json.JSONDecodeError, KeyError, OSError):
        return False
    except Exception:
        # json5 may raise different exceptions
        return False


def run_analysis(config_path: Union[str, Path]) -> AnalysisResult:
    """Convenience function to run analysis from config file."""
    engine = DeclarativeAnalysisEngine()
    return engine.run_from_file(config_path)


@dataclass
class BatchResult:
    """Result of running multiple analyses."""
    results: Dict[str, AnalysisResult]  # path -> result
    errors: Dict[str, str]  # path -> error message
    skipped: List[str]  # paths skipped (not valid configs)

    @property
    def summary(self) -> str:
        """Build summary of batch run."""
        lines = [
            f"# Batch Analysis Results",
            f"",
            f"- Successful: {len(self.results)}",
            f"- Errors: {len(self.errors)}",
            f"- Skipped: {len(self.skipped)}",
            f"",
        ]

        if self.results:
            lines.append("## Successful Analyses")
            for path, result in self.results.items():
                lines.append(f"- {path}: {result.config.name}")
            lines.append("")

        if self.errors:
            lines.append("## Errors")
            for path, error in self.errors.items():
                lines.append(f"- {path}: {error}")
            lines.append("")

        if self.skipped:
            lines.append("## Skipped (not valid analysis configs)")
            for path in self.skipped:
                lines.append(f"- {path}")

        return "\n".join(lines)


def run_analysis_batch(
    directory: Union[str, Path],
    pattern: str = "*.json*",
) -> BatchResult:
    """
    Run all valid analysis configs in a directory.

    Args:
        directory: Directory containing config files
        pattern: Glob pattern for config files (default: "*.json*" matches .json and .jsonc)

    Returns:
        BatchResult with results, errors, and skipped files
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    results: Dict[str, AnalysisResult] = {}
    errors: Dict[str, str] = {}
    skipped: List[str] = []

    # Find all matching files
    config_files = sorted(directory.glob(pattern))

    for config_path in config_files:
        if not config_path.is_file():
            continue

        path_str = str(config_path)

        # Check if it's a valid analysis config
        if not is_valid_analysis_config(config_path):
            skipped.append(path_str)
            continue

        # Try to run the analysis
        try:
            result = run_analysis(config_path)
            results[path_str] = result
        except Exception as e:
            errors[path_str] = str(e)

    return BatchResult(results=results, errors=errors, skipped=skipped)


# CLI entry point
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m smt_oversub_model.declarative <config.json | directory>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_dir():
        # Run all configs in directory
        batch_result = run_analysis_batch(input_path)
        print(batch_result.summary)

        # Print individual summaries for successful runs
        if batch_result.results:
            print("\n" + "=" * 60 + "\n")
            for path, result in batch_result.results.items():
                print(f"## {path}\n")
                print(result.summary)
                print("\n" + "-" * 40 + "\n")

                # Save results if output_dir specified
                if result.config.output_dir:
                    from .output import OutputWriter
                    writer = OutputWriter(result.config.output_dir)
                    writer.write(result)
                    print(f"Results saved to: {result.config.output_dir}\n")
    else:
        # Single file
        result = run_analysis(input_path)

        # Print summary
        print(result.summary)

        # Save results if output_dir specified
        if result.config.output_dir:
            from .output import OutputWriter
            writer = OutputWriter(result.config.output_dir)
            writer.write(result)
            print(f"\nResults saved to: {result.config.output_dir}")
