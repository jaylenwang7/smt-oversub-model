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

from .model import (
    OverssubModel, PowerCurve, ProcessorConfig,
    ScenarioParams, WorkloadParams, CostParams, ScenarioResult,
)
from .analysis import ScenarioSpec, ScenarioBuilder, ProcessorDefaults, CostDefaults
from .config import make_power_curve_fn


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
    type: str  # "find_breakeven", "compare", "sweep"
    baseline: Optional[str] = None
    reference: Optional[str] = None
    target: Optional[str] = None
    scenarios: Optional[List[str]] = None  # For "compare" type
    vary_parameter: Optional[str] = None
    match_metric: Optional[Union[str, Dict[str, str]]] = None
    search_bounds: Optional[List[float]] = None
    sweep_parameter: Optional[str] = None  # For "sweep" type
    sweep_values: Optional[List[float]] = None

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
        return d


@dataclass
class ProcessorSpec:
    """Processor configuration."""
    physical_cores: int = 48
    power_idle_w: float = 100.0
    power_max_w: float = 400.0
    core_overhead: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessorSpec':
        return cls(
            physical_cores=data.get('physical_cores', 48),
            power_idle_w=data.get('power_idle_w', 100.0),
            power_max_w=data.get('power_max_w', 400.0),
            core_overhead=data.get('core_overhead', 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NoSmtProcessorSpec:
    """Non-SMT processor configuration."""
    physical_cores: int = 48
    power_ratio: float = 0.85  # Relative to SMT
    idle_ratio: float = 0.9
    core_overhead: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoSmtProcessorSpec':
        return cls(
            physical_cores=data.get('physical_cores', 48),
            power_ratio=data.get('power_ratio', 0.85),
            idle_ratio=data.get('idle_ratio', 0.9),
            core_overhead=data.get('core_overhead', 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessorConfigSpec:
    """Combined processor specs."""
    smt: ProcessorSpec = field(default_factory=ProcessorSpec)
    nosmt: NoSmtProcessorSpec = field(default_factory=NoSmtProcessorSpec)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessorConfigSpec':
        return cls(
            smt=ProcessorSpec.from_dict(data.get('smt', {})),
            nosmt=NoSmtProcessorSpec.from_dict(data.get('nosmt', {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'smt': self.smt.to_dict(),
            'nosmt': self.nosmt.to_dict(),
        }


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
    """Cost configuration."""
    embodied_carbon_kg: float = 1000.0
    server_cost_usd: float = 10000.0
    carbon_intensity_g_kwh: float = 400.0
    electricity_cost_usd_kwh: float = 0.10
    lifetime_years: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostSpec':
        return cls(
            embodied_carbon_kg=data.get('embodied_carbon_kg', 1000.0),
            server_cost_usd=data.get('server_cost_usd', 10000.0),
            carbon_intensity_g_kwh=data.get('carbon_intensity_g_kwh', 400.0),
            electricity_cost_usd_kwh=data.get('electricity_cost_usd_kwh', 0.10),
            lifetime_years=data.get('lifetime_years', 5.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        scenarios = {}
        for name, spec in data.get('scenarios', {}).items():
            scenarios[name] = ScenarioConfig.from_dict(spec)

        return cls(
            name=data.get('name', 'unnamed'),
            description=data.get('description', ''),
            scenarios=scenarios,
            analysis=AnalysisSpec.from_dict(data.get('analysis', {})),
            processor=ProcessorConfigSpec.from_dict(data.get('processor', {})),
            workload=WorkloadSpec.from_dict(data.get('workload', {})),
            cost=CostSpec.from_dict(data.get('cost', {})),
            power_curve=PowerCurveSpec.from_dict(data.get('power_curve', {})),
            output_dir=data.get('output_dir'),
        )

    @classmethod
    def from_json(cls, path: Path) -> 'AnalysisConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

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
        self._setup_builder()

        analysis_type = config.analysis.type

        if analysis_type == 'find_breakeven':
            return self._run_find_breakeven()
        elif analysis_type == 'compare':
            return self._run_compare()
        elif analysis_type == 'sweep':
            return self._run_sweep()
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

        proc_defaults = ProcessorDefaults(
            smt_physical_cores=proc.smt.physical_cores,
            smt_power_idle_w=proc.smt.power_idle_w,
            smt_power_max_w=proc.smt.power_max_w,
            smt_core_overhead=proc.smt.core_overhead,
            nosmt_physical_cores=proc.nosmt.physical_cores,
            nosmt_power_ratio=proc.nosmt.power_ratio,
            nosmt_idle_ratio=proc.nosmt.idle_ratio,
            nosmt_core_overhead=proc.nosmt.core_overhead,
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

        workload = self._builder.build_workload_params(
            cfg.workload.total_vcpus,
            cfg.workload.avg_util,
        )
        cost_params = self._builder.build_cost_params()
        self._model = OverssubModel(workload, cost_params)

    def _build_scenario_spec(self, name: str, scenario_cfg: ScenarioConfig) -> ScenarioSpec:
        """Build a ScenarioSpec from config."""
        is_smt = scenario_cfg.processor.lower() == 'smt'
        overrides = scenario_cfg.overrides or {}

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
        is_smt = scenario_cfg.processor.lower() == 'smt'
        overrides = scenario_cfg.overrides or {}

        if is_smt:
            processor = self._builder.build_smt_processor(**overrides)
        else:
            processor = self._builder.build_nosmt_processor(**overrides)

        return ScenarioParams(
            processor=processor,
            oversub_ratio=scenario_cfg.oversub_ratio,
            util_overhead=scenario_cfg.util_overhead,
            vcpu_demand_multiplier=scenario_cfg.vcpu_demand_multiplier,
        )

    def _evaluate_scenario(self, name: str) -> Tuple[ScenarioParams, ScenarioResult]:
        """Evaluate a scenario by name and return params and result."""
        scenario_cfg = self._config.scenarios[name]
        params = self._build_scenario_params(scenario_cfg)
        result = self._model.evaluate_scenario(params)
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
            # Modify workload or cost based on sweep parameter
            self._apply_sweep_value(param_path, value)
            self._setup_builder()  # Rebuild with new values

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

    def _apply_sweep_value(self, param_path: ParameterPath, value: float):
        """Apply sweep parameter value to config."""
        # Handle different parameter targets
        first_part = param_path.parts[0]
        rest = '.'.join(param_path.parts[1:]) if len(param_path.parts) > 1 else None

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
        elif first_part in ('avg_util', 'total_vcpus'):
            # Direct workload params
            if first_part == 'avg_util':
                self._config.workload.avg_util = value
            else:
                self._config.workload.total_vcpus = int(value)

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


def run_analysis(config_path: Union[str, Path]) -> AnalysisResult:
    """Convenience function to run analysis from config file."""
    engine = DeclarativeAnalysisEngine()
    return engine.run_from_file(config_path)


# CLI entry point
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m smt_oversub_model.declarative <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    result = run_analysis(config_path)

    # Print summary
    print(result.summary)

    # Save results if output_dir specified
    if result.config.output_dir:
        from .output import OutputWriter
        writer = OutputWriter(result.config.output_dir)
        writer.write(result)
        print(f"\nResults saved to: {result.config.output_dir}")
