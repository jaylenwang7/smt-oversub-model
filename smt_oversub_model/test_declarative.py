"""
Unit tests for the declarative analysis framework.

Run with: pytest smt_oversub_model/test_declarative.py -v
"""

import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import asdict

from .declarative import (
    ParameterPath,
    SimpleCondition,
    CompoundCondition,
    MatchType,
    GeneralizedBreakevenFinder,
    AnalysisConfig,
    ScenarioConfig,
    ResourceScalingConfig,
    ResourceConstraintSpec,
    ResourceConstraintsConfig,
    DeclarativeAnalysisEngine,
    run_analysis,
    is_valid_analysis_config,
    PowerComponentSpec,
    PowerCurveSpec,
    ProcessorSpec,
    EmbodiedComponentSpec,
)
from .model import (
    PowerCurve, ProcessorConfig, ScenarioParams,
    WorkloadParams, CostParams, OverssubModel, ScenarioResult,
    ComponentBreakdown, EmbodiedBreakdown,
    ResourceConstraintDetail, ResourceConstraintResult,
)


class TestParameterPath:
    """Tests for ParameterPath class."""

    def test_get_direct_dict_key(self):
        """Get a direct key from a dict."""
        obj = {'oversub_ratio': 1.5, 'util_overhead': 0.05}
        path = ParameterPath('oversub_ratio')
        assert path.get(obj) == 1.5

    def test_get_nested_dict_key(self):
        """Get a nested key from a dict."""
        obj = {'processor': {'physical_cores': 48, 'power_curve': {'p_max': 400}}}
        path = ParameterPath('processor.power_curve.p_max')
        assert path.get(obj) == 400

    def test_get_from_dataclass(self):
        """Get attribute from a dataclass."""
        power = PowerCurve(p_idle=100, p_max=300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power)

        path = ParameterPath('physical_cores')
        assert path.get(proc) == 64

        path = ParameterPath('power_curve.p_max')
        assert path.get(proc) == 300

    def test_set_direct_dict_key(self):
        """Set a direct key in a dict."""
        obj = {'oversub_ratio': 1.0}
        path = ParameterPath('oversub_ratio')
        new_obj = path.set(obj, 2.0)

        # Original unchanged
        assert obj['oversub_ratio'] == 1.0
        # New value set
        assert new_obj['oversub_ratio'] == 2.0

    def test_set_nested_dict_key(self):
        """Set a nested key in a dict."""
        obj = {'processor': {'physical_cores': 48}}
        path = ParameterPath('processor.physical_cores')
        new_obj = path.set(obj, 64)

        assert new_obj['processor']['physical_cores'] == 64

    def test_set_dataclass_field(self):
        """Set a field on a dataclass (returns new instance)."""
        power = PowerCurve(p_idle=100, p_max=300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power)
        scenario = ScenarioParams(processor=proc, oversub_ratio=1.0)

        path = ParameterPath('oversub_ratio')
        new_scenario = path.set(scenario, 2.0)

        # Original unchanged
        assert scenario.oversub_ratio == 1.0
        # New value set
        assert new_scenario.oversub_ratio == 2.0

    def test_set_nested_dataclass_field(self):
        """Set a nested field on dataclasses."""
        power = PowerCurve(p_idle=100, p_max=300)

        path = ParameterPath('p_max')
        new_power = path.set(power, 500)

        assert power.p_max == 300
        assert new_power.p_max == 500


class TestSimpleCondition:
    """Tests for SimpleCondition class."""

    def test_equal_condition_satisfied(self):
        """EQUAL condition satisfied when within tolerance."""
        cond = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL, tolerance=0.01)

        target = {'total_carbon_kg': 1000}
        reference = {'total_carbon_kg': 1005}  # 0.5% diff

        assert cond.evaluate(target, reference) is True

    def test_equal_condition_not_satisfied(self):
        """EQUAL condition not satisfied when outside tolerance."""
        cond = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL, tolerance=0.01)

        target = {'total_carbon_kg': 1000}
        reference = {'total_carbon_kg': 1100}  # 10% diff

        assert cond.evaluate(target, reference) is False

    def test_within_percent_condition(self):
        """WITHIN_PERCENT condition."""
        cond = SimpleCondition(
            metric='tco',
            match_type=MatchType.WITHIN_PERCENT,
            percent=5.0
        )

        target = {'total_cost_usd': 1040}
        reference = {'total_cost_usd': 1000}  # 4% diff

        assert cond.evaluate(target, reference) is True

        target = {'total_cost_usd': 1100}  # 10% diff
        assert cond.evaluate(target, reference) is False

    def test_less_or_equal_condition(self):
        """LESS_OR_EQUAL condition."""
        cond = SimpleCondition(metric='carbon', match_type=MatchType.LESS_OR_EQUAL)

        target = {'total_carbon_kg': 900}
        reference = {'total_carbon_kg': 1000}

        assert cond.evaluate(target, reference) is True

        target = {'total_carbon_kg': 1100}
        assert cond.evaluate(target, reference) is False

    def test_get_error_positive(self):
        """Get error when target > reference."""
        cond = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL)

        target = {'total_carbon_kg': 1100}
        reference = {'total_carbon_kg': 1000}

        error = cond.get_error(target, reference)
        assert error == pytest.approx(0.1, rel=0.01)

    def test_get_error_negative(self):
        """Get error when target < reference."""
        cond = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL)

        target = {'total_carbon_kg': 900}
        reference = {'total_carbon_kg': 1000}

        error = cond.get_error(target, reference)
        assert error == pytest.approx(-0.1, rel=0.01)

    def test_from_string_match(self):
        """Parse 'match' condition string."""
        cond = SimpleCondition.from_string('match')
        assert cond.match_type == MatchType.EQUAL

    def test_from_string_within_percent(self):
        """Parse 'within_5%' condition string."""
        cond = SimpleCondition.from_string('within_5%')
        assert cond.match_type == MatchType.WITHIN_PERCENT
        assert cond.percent == 5.0

    def test_from_string_comparison(self):
        """Parse '<=' and '>=' condition strings."""
        cond = SimpleCondition.from_string('<=')
        assert cond.match_type == MatchType.LESS_OR_EQUAL

        cond = SimpleCondition.from_string('>=')
        assert cond.match_type == MatchType.GREATER_OR_EQUAL


class TestCompoundCondition:
    """Tests for CompoundCondition class."""

    def test_from_dict(self):
        """Create compound condition from dict."""
        spec = {'carbon': 'match', 'tco': 'within_5%'}
        compound = CompoundCondition.from_dict(spec)

        assert 'carbon' in compound.conditions
        assert 'tco' in compound.conditions
        assert compound.conditions['carbon'].match_type == MatchType.EQUAL
        assert compound.conditions['tco'].match_type == MatchType.WITHIN_PERCENT

    def test_all_conditions_must_pass(self):
        """All conditions in compound must be satisfied."""
        spec = {'carbon': 'match', 'tco': 'within_5%'}
        compound = CompoundCondition.from_dict(spec)

        # Both satisfied
        target = {'total_carbon_kg': 1000, 'total_cost_usd': 1020}
        reference = {'total_carbon_kg': 1005, 'total_cost_usd': 1000}
        assert compound.evaluate(target, reference) is True

        # Carbon fails
        target = {'total_carbon_kg': 1200, 'total_cost_usd': 1020}
        assert compound.evaluate(target, reference) is False

    def test_get_primary_error(self):
        """Primary error from EQUAL condition."""
        spec = {'carbon': 'match', 'tco': 'within_10%'}
        compound = CompoundCondition.from_dict(spec)

        target = {'total_carbon_kg': 1100, 'total_cost_usd': 1050}
        reference = {'total_carbon_kg': 1000, 'total_cost_usd': 1000}

        # Should return carbon error (the EQUAL condition)
        error = compound.get_primary_error(target, reference)
        assert error == pytest.approx(0.1, rel=0.01)


class TestGeneralizedBreakevenFinder:
    """Tests for GeneralizedBreakevenFinder class."""

    @pytest.fixture
    def model_setup(self):
        """Create model for breakeven tests."""
        workload = WorkloadParams(total_vcpus=10000, avg_util=0.3)
        cost = CostParams(
            embodied_carbon_kg=1000,
            server_cost_usd=15000,
            carbon_intensity_g_kwh=400,
            electricity_cost_usd_kwh=0.10,
            lifetime_hours=4 * 8760,
        )
        return OverssubModel(workload, cost)

    @pytest.fixture
    def nosmt_processor(self):
        """Create non-SMT processor."""
        power = PowerCurve(85, 255)  # 85% of SMT
        return ProcessorConfig(48, 1, power)

    @pytest.fixture
    def smt_processor(self):
        """Create SMT processor."""
        power = PowerCurve(100, 300)
        return ProcessorConfig(64, 2, power)

    def test_find_breakeven_oversub_ratio(self, model_setup, nosmt_processor, smt_processor):
        """Find breakeven oversub ratio (regression test)."""
        model = model_setup

        # Reference: SMT with oversub
        smt_scenario = ScenarioParams(smt_processor, oversub_ratio=1.3, util_overhead=0.05)
        smt_result = model.evaluate_scenario(smt_scenario)

        # Target: non-SMT, vary oversub_ratio
        nosmt_base = ScenarioParams(nosmt_processor, oversub_ratio=1.0)

        condition = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL)
        finder = GeneralizedBreakevenFinder(model)

        result = finder.find_breakeven(
            base_scenario=nosmt_base,
            reference_result=smt_result,
            vary_parameter='oversub_ratio',
            match_condition=condition,
            search_bounds=(1.0, 5.0),
        )

        assert result.achieved is True
        assert result.breakeven_value is not None
        assert result.breakeven_value > 1.0

    def test_find_breakeven_vcpu_multiplier(self, model_setup, nosmt_processor, smt_processor):
        """Find breakeven vCPU demand multiplier (new capability)."""
        model = model_setup

        # Reference: SMT with oversub
        smt_scenario = ScenarioParams(smt_processor, oversub_ratio=1.1, util_overhead=0.05)
        smt_result = model.evaluate_scenario(smt_scenario)

        # Target: non-SMT with fixed oversub, vary vcpu_demand_multiplier
        nosmt_base = ScenarioParams(
            nosmt_processor,
            oversub_ratio=1.4,
            vcpu_demand_multiplier=1.0
        )

        condition = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL)
        finder = GeneralizedBreakevenFinder(model)

        result = finder.find_breakeven(
            base_scenario=nosmt_base,
            reference_result=smt_result,
            vary_parameter='vcpu_demand_multiplier',
            match_condition=condition,
            search_bounds=(0.5, 1.0),
        )

        # Should find some breakeven value
        assert result.iterations > 0
        assert len(result.search_history) > 0

    def test_no_solution_in_bounds(self, model_setup, nosmt_processor, smt_processor):
        """Return appropriate result when no solution exists in bounds."""
        model = model_setup

        # Reference with very low carbon
        smt_scenario = ScenarioParams(smt_processor, oversub_ratio=3.0)
        smt_result = model.evaluate_scenario(smt_scenario)

        # Target that can't achieve the same
        nosmt_base = ScenarioParams(nosmt_processor, oversub_ratio=1.0)

        condition = SimpleCondition(metric='carbon', match_type=MatchType.EQUAL)
        finder = GeneralizedBreakevenFinder(model)

        result = finder.find_breakeven(
            base_scenario=nosmt_base,
            reference_result=smt_result,
            vary_parameter='oversub_ratio',
            match_condition=condition,
            search_bounds=(1.0, 1.1),  # Very narrow bounds
        )

        # May or may not achieve depending on scenario
        assert result.iterations >= 2  # At least checked bounds


class TestDeclarativeAnalysisEngine:
    """Tests for DeclarativeAnalysisEngine class."""

    def test_run_compare_analysis(self):
        """Run a basic compare analysis."""
        from .declarative import AnalysisSpec

        config = AnalysisConfig(
            name='test_compare',
            scenarios={
                'baseline': ScenarioConfig(processor='smt', oversub_ratio=1.0),
                'oversub': ScenarioConfig(processor='smt', oversub_ratio=1.5),
            },
            analysis=AnalysisSpec(
                type='compare',
                baseline='baseline',
                scenarios=['baseline', 'oversub'],
            ),
        )

        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        assert result.analysis_type == 'compare'
        assert 'baseline' in result.scenario_results
        assert 'oversub' in result.scenario_results
        assert 'oversub' in result.comparisons

    def test_run_find_breakeven_analysis(self):
        """Run a find_breakeven analysis."""
        from .declarative import AnalysisSpec

        config = AnalysisConfig(
            name='test_breakeven',
            scenarios={
                'baseline': ScenarioConfig(processor='smt', oversub_ratio=1.0),
                'smt_oversub': ScenarioConfig(processor='smt', oversub_ratio=1.3, util_overhead=0.05),
                'nosmt': ScenarioConfig(processor='nosmt', oversub_ratio=1.0),
            },
            analysis=AnalysisSpec(
                type='find_breakeven',
                baseline='baseline',
                reference='smt_oversub',
                target='nosmt',
                vary_parameter='oversub_ratio',
                match_metric='carbon',
                search_bounds=[1.0, 5.0],
            ),
        )

        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        assert result.analysis_type == 'find_breakeven'
        assert result.breakeven is not None
        assert result.breakeven.iterations > 0

    def test_run_from_json_file(self):
        """Run analysis from a JSON config file."""
        config_dict = {
            'name': 'test_json',
            'scenarios': {
                'baseline': {'processor': 'smt', 'oversub_ratio': 1.0},
                'oversub': {'processor': 'smt', 'oversub_ratio': 1.5},
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'baseline',
                'scenarios': ['baseline', 'oversub'],
            },
            'workload': {'total_vcpus': 5000, 'avg_util': 0.4},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            engine = DeclarativeAnalysisEngine()
            result = engine.run_from_file(config_path)

            assert result.analysis_type == 'compare'
            assert len(result.scenario_results) == 2
        finally:
            Path(config_path).unlink()

    def test_compound_match_condition(self):
        """Test analysis with compound match condition."""
        from .declarative import AnalysisSpec

        config = AnalysisConfig(
            name='test_compound',
            scenarios={
                'baseline': ScenarioConfig(processor='smt', oversub_ratio=1.0),
                'reference': ScenarioConfig(processor='smt', oversub_ratio=1.2),
                'target': ScenarioConfig(processor='nosmt', oversub_ratio=1.0),
            },
            analysis=AnalysisSpec(
                type='find_breakeven',
                baseline='baseline',
                reference='reference',
                target='target',
                vary_parameter='oversub_ratio',
                match_metric={'carbon': 'match', 'tco': 'within_10%'},
                search_bounds=[1.0, 5.0],
            ),
        )

        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        assert result.breakeven is not None

    def test_find_breakeven_uses_processor_cost_overrides(self):
        """Breakeven search must use per-processor cost overrides, not model defaults.

        Regression test: the GeneralizedBreakevenFinder previously called
        model.evaluate_scenario() without cost_overrides, so it used defaults
        (1000 kg, $10000) instead of the processor's actual structured costs.
        """
        config_dict = {
            'name': 'test_breakeven_cost_overrides',
            'scenarios': {
                'baseline': {'processor': 'smt', 'oversub_ratio': 1.0},
                'reference': {'processor': 'smt', 'oversub_ratio': 1.0},
                'target': {'processor': 'nosmt', 'oversub_ratio': 1.0,
                           'vcpu_demand_multiplier': 1.0},
            },
            'analysis': {
                'type': 'find_breakeven',
                'baseline': 'baseline',
                'reference': 'reference',
                'target': 'target',
                'vary_parameter': 'vcpu_demand_multiplier',
                'match_metric': 'carbon',
                'search_bounds': [0.3, 1.0],
            },
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'thread_overhead': 0,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    # Structured costs that differ from defaults (1000/$10000)
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0},
                        'per_server': {'chassis': 200.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 40.0},
                        'per_server': {'chassis': 3000.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'thread_overhead': 0,
                    'power_idle_w': 90,
                    'power_max_w': 340,
                    # Per-server: 5*48 + 200 = 440 kg, 40*48 + 3000 = $4920
                    # Very different from defaults of 1000/$10000
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0},
                        'per_server': {'chassis': 200.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 40.0},
                        'per_server': {'chassis': 3000.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
            'cost': {
                'carbon_intensity_g_kwh': 400,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5,
            },
        }

        config = AnalysisConfig.from_dict(config_dict)
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        assert result.breakeven is not None
        assert result.breakeven.achieved is True

        # Verify the target at breakeven uses the correct per-server costs
        target_data = result.scenario_results.get('target', {})
        num_servers = target_data['num_servers']
        # nosmt per-server embodied: 5.0 * 48 + 200 = 440 kg (NOT the default 1000)
        expected_embodied_per_server = 5.0 * 48 + 200.0  # = 440
        actual_embodied_per_server = target_data['embodied_carbon_kg'] / num_servers
        assert abs(actual_embodied_per_server - expected_embodied_per_server) < 1.0, \
            f"Expected ~{expected_embodied_per_server} kg/server, got {actual_embodied_per_server:.1f} " \
            f"(breakeven finder may be using default costs instead of processor overrides)"


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_from_dict(self):
        """Create config from dict."""
        data = {
            'name': 'test',
            'scenarios': {
                'baseline': {'processor': 'smt', 'oversub_ratio': 1.0},
            },
            'analysis': {'type': 'compare'},
            'workload': {'total_vcpus': 8000},
        }

        config = AnalysisConfig.from_dict(data)

        assert config.name == 'test'
        assert 'baseline' in config.scenarios
        assert config.workload.total_vcpus == 8000

    def test_to_dict_roundtrip(self):
        """Config survives to_dict/from_dict roundtrip."""
        from .declarative import AnalysisSpec

        original = AnalysisConfig(
            name='roundtrip_test',
            scenarios={
                'a': ScenarioConfig(processor='smt', oversub_ratio=1.5),
            },
            analysis=AnalysisSpec(type='compare'),
        )

        as_dict = original.to_dict()
        restored = AnalysisConfig.from_dict(as_dict)

        assert restored.name == original.name
        assert 'a' in restored.scenarios
        assert restored.scenarios['a'].oversub_ratio == 1.5

    def test_from_json_file(self):
        """Load config from JSON file."""
        config_dict = {
            'name': 'json_test',
            'scenarios': {
                'base': {'processor': 'nosmt'},
            },
            'analysis': {'type': 'compare'},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name

        try:
            config = AnalysisConfig.from_json(Path(config_path))
            assert config.name == 'json_test'
            assert 'base' in config.scenarios
        finally:
            Path(config_path).unlink()


class TestIntegration:
    """Integration tests with example config files."""

    def test_vcpu_demand_breakeven_config(self):
        """Test the vcpu_demand_breakeven example config."""
        config_path = Path(__file__).parent.parent / 'configs' / 'vcpu_demand_breakeven.json'

        if not config_path.exists():
            pytest.skip("Example config not found")

        config = AnalysisConfig.from_json(config_path)

        assert config.name == 'vcpu_demand_breakeven'
        assert config.analysis.type == 'find_breakeven'
        assert config.analysis.vary_parameter == 'vcpu_demand_multiplier'
        assert 'baseline' in config.scenarios
        assert 'smt_oversub' in config.scenarios
        assert 'nosmt_oversub' in config.scenarios

        # Actually run the analysis
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        assert result.analysis_type == 'find_breakeven'
        assert result.breakeven is not None
        # The search should complete (may or may not achieve breakeven)
        assert result.breakeven.iterations > 0


class TestEmbodiedComponentSpec:
    """Tests for EmbodiedComponentSpec and structured embodied carbon/cost."""

    def test_from_dict(self):
        """EmbodiedComponentSpec.from_dict should parse per_core and per_server."""
        from .declarative import EmbodiedComponentSpec

        data = {
            'per_thread': {'cpu_die': 10.0, 'dram': 2.0},
            'per_server': {'chassis': 100.0},
        }
        spec = EmbodiedComponentSpec.from_dict(data)
        assert spec.per_thread == {'cpu_die': 10.0, 'dram': 2.0}
        assert spec.per_server == {'chassis': 100.0}

    def test_resolve_total(self):
        """resolve_total should compute flat per-server value."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_thread={'cpu_die': 10.0},
            per_server={'chassis': 20.0},
        )
        # 48 cores * 2 threads = 96 hw threads
        # 10 * 96 + 20 = 980
        assert spec.resolve_total(48, 2) == 980.0

    def test_resolve_total_nosmt(self):
        """resolve_total with threads_per_core=1."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_thread={'cpu_die': 10.0},
            per_server={'chassis': 50.0},
        )
        # 52 cores * 1 thread = 52 hw threads
        # 10 * 52 + 50 = 570
        assert spec.resolve_total(52, 1) == 570.0

    def test_to_component_breakdown(self):
        """to_component_breakdown should produce a resolved ComponentBreakdown."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_thread={'cpu_die': 10.0},
            per_server={'chassis': 20.0},
        )
        bd = spec.to_component_breakdown(48, 2)
        assert bd.physical_cores == 48
        assert bd.threads_per_core == 2
        assert bd.total_per_server == 980.0

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_thread={'cpu_die': 10.0},
            per_server={'chassis': 20.0},
        )
        d = spec.to_dict()
        assert d == {'per_thread': {'cpu_die': 10.0}, 'per_server': {'chassis': 20.0}}

    def test_empty_spec(self):
        """Empty spec should resolve to 0."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec()
        assert spec.resolve_total(48, 2) == 0.0


class TestProcessorSpecStructured:
    """Tests for ProcessorSpec with structured embodied_carbon/server_cost."""

    def test_from_dict_with_structured_carbon(self):
        """ProcessorSpec should parse structured embodied_carbon."""
        from .declarative import ProcessorSpec

        data = {
            'physical_cores': 48,
            'threads_per_core': 2,
            'power_idle_w': 90.0,
            'power_max_w': 600.0,
            'embodied_carbon': {
                'per_thread': {'cpu_die': 10.0},
                'per_server': {'chassis': 20.0},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.embodied_carbon is not None
        assert spec.embodied_carbon.per_thread == {'cpu_die': 10.0}
        assert spec.embodied_carbon_kg is None  # flat not set

    def test_from_dict_with_flat_carbon(self):
        """ProcessorSpec should still work with flat embodied_carbon_kg."""
        from .declarative import ProcessorSpec

        data = {
            'physical_cores': 48,
            'threads_per_core': 2,
            'power_idle_w': 90.0,
            'power_max_w': 600.0,
            'embodied_carbon_kg': 980.0,
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.embodied_carbon is None
        assert spec.embodied_carbon_kg == 980.0

    def test_to_dict_structured_over_flat(self):
        """to_dict should prefer structured format over flat when both set."""
        from .declarative import ProcessorSpec, EmbodiedComponentSpec

        spec = ProcessorSpec(
            physical_cores=48,
            threads_per_core=2,
            embodied_carbon_kg=980.0,  # flat
            embodied_carbon=EmbodiedComponentSpec(
                per_thread={'cpu_die': 10.0},
                per_server={'chassis': 20.0},
            ),
        )
        d = spec.to_dict()
        assert 'embodied_carbon' in d
        assert 'embodied_carbon_kg' not in d

    def test_from_dict_with_structured_cost(self):
        """ProcessorSpec should parse structured server_cost."""
        from .declarative import ProcessorSpec

        data = {
            'physical_cores': 48,
            'threads_per_core': 2,
            'power_idle_w': 90.0,
            'power_max_w': 600.0,
            'server_cost': {
                'per_thread': {'cpu': 73.0},
                'per_server': {'base': 32.0},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.server_cost is not None
        assert spec.server_cost.per_thread == {'cpu': 73.0}
        assert spec.server_cost_usd is None


class TestPriorityChain:
    """Tests for structured cost resolution priority chain."""

    def _make_config(self, proc_kwargs=None, cost_kwargs=None):
        """Helper to make a minimal AnalysisConfig."""
        from .declarative import (
            AnalysisConfig, ScenarioConfig, AnalysisSpec,
            ProcessorConfigSpec, ProcessorSpec, WorkloadSpec, CostSpec,
            EmbodiedComponentSpec,
        )

        proc_kw = {
            'physical_cores': 48,
            'threads_per_core': 2,
            'power_idle_w': 100.0,
            'power_max_w': 400.0,
        }
        if proc_kwargs:
            proc_kw.update(proc_kwargs)

        cost_kw = {
            'embodied_carbon_kg': 1000.0,
            'server_cost_usd': 10000.0,
            'carbon_intensity_g_kwh': 400.0,
            'electricity_cost_usd_kwh': 0.10,
            'lifetime_years': 5.0,
        }
        if cost_kwargs:
            cost_kw.update(cost_kwargs)

        return AnalysisConfig(
            name='test_priority',
            scenarios={
                'baseline': ScenarioConfig(processor='smt', oversub_ratio=1.0),
            },
            analysis=AnalysisSpec(type='compare', baseline='baseline'),
            processor=ProcessorConfigSpec(processors={
                'smt': ProcessorSpec(**proc_kw),
            }),
            workload=WorkloadSpec(total_vcpus=10000, avg_util=0.3),
            cost=CostSpec(**cost_kw),
        )

    def test_processor_structured_takes_priority(self):
        """Processor-level structured should override everything."""
        from .declarative import DeclarativeAnalysisEngine, EmbodiedComponentSpec

        config = self._make_config(
            proc_kwargs={
                'embodied_carbon': EmbodiedComponentSpec(
                    per_thread={'cpu_die': 5.0},
                    per_server={'chassis': 10.0},
                ),
                'embodied_carbon_kg': 9999.0,  # should be ignored
            },
        )
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        baseline = result.scenario_results['baseline']
        # 5 * 96 + 10 = 490 per server
        expected_per_server = 490.0
        assert baseline['embodied_carbon_kg'] == baseline['num_servers'] * expected_per_server

    def test_processor_flat_takes_priority_over_global(self):
        """Processor-level flat should override global cost."""
        from .declarative import DeclarativeAnalysisEngine

        config = self._make_config(
            proc_kwargs={'embodied_carbon_kg': 500.0},
            cost_kwargs={'embodied_carbon_kg': 9999.0},
        )
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        baseline = result.scenario_results['baseline']
        assert baseline['embodied_carbon_kg'] == baseline['num_servers'] * 500.0

    def test_global_structured_used_when_no_processor_override(self):
        """Global cost structured should be used when processor has no override."""
        from .declarative import DeclarativeAnalysisEngine, EmbodiedComponentSpec, CostSpec

        config = self._make_config(
            cost_kwargs={
                'embodied_carbon': EmbodiedComponentSpec(
                    per_thread={'cpu_die': 8.0},
                    per_server={'chassis': 32.0},
                ),
            },
        )
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        baseline = result.scenario_results['baseline']
        # 8 * 96 + 32 = 800 per server
        expected_per_server = 800.0
        assert baseline['embodied_carbon_kg'] == baseline['num_servers'] * expected_per_server

    def test_global_flat_is_default(self):
        """Global flat cost should be the default when nothing overrides it."""
        from .declarative import DeclarativeAnalysisEngine

        config = self._make_config()  # no processor overrides, no structured
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        baseline = result.scenario_results['baseline']
        # Default: 1000 per server
        assert baseline['embodied_carbon_kg'] == baseline['num_servers'] * 1000.0

    def test_breakdown_attached_when_structured(self):
        """Result should have embodied_breakdown when structured format is used."""
        from .declarative import DeclarativeAnalysisEngine, EmbodiedComponentSpec

        config = self._make_config(
            proc_kwargs={
                'embodied_carbon': EmbodiedComponentSpec(
                    per_thread={'cpu_die': 10.0},
                    per_server={'chassis': 20.0},
                ),
            },
        )
        engine = DeclarativeAnalysisEngine()
        result = engine.run(config)

        baseline = result.scenario_results['baseline']
        assert baseline.get('embodied_breakdown') is not None
        assert baseline['embodied_breakdown']['carbon'] is not None


class TestStructuredEndToEnd:
    """End-to-end tests with structured embodied carbon configs."""

    def test_structured_config_from_json(self, tmp_path):
        """Test a full analysis config with structured embodied carbon from JSON."""
        config_data = {
            'name': 'structured_test',
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 100.0,
                    'power_max_w': 400.0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 10.0},
                        'per_server': {'chassis': 20.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 73.0},
                        'per_server': {'base': 32.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 52,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 10.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 72.0},
                        'per_server': {'base': 16.0},
                    },
                },
            },
            'scenarios': {
                'baseline': {'processor': 'smt', 'oversub_ratio': 1.0},
                'nosmt_r1': {'processor': 'nosmt', 'oversub_ratio': 1.0},
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'baseline',
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
            'cost': {
                'embodied_carbon_kg': 9999.0,  # should be overridden by processor
                'server_cost_usd': 9999.0,
                'carbon_intensity_g_kwh': 400.0,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5.0,
            },
        }

        config_file = tmp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'compare'

        # SMT: 10*96+20 = 980 per server
        smt = result.scenario_results['baseline']
        assert smt['embodied_carbon_kg'] == smt['num_servers'] * 980.0

        # noSMT: 10*52+50 = 570 per server
        nosmt = result.scenario_results['nosmt_r1']
        assert nosmt['embodied_carbon_kg'] == nosmt['num_servers'] * 570.0

        # SMT cost: 73*96+32 = 7040 per server
        assert smt['embodied_cost_usd'] == smt['num_servers'] * 7040.0

        # noSMT cost: 72*52+16 = 3760 per server
        assert nosmt['embodied_cost_usd'] == nosmt['num_servers'] * 3760.0


class TestPowerComponentSpec:
    """Tests for PowerComponentSpec config parsing."""

    def test_from_dict_basic(self):
        """Parse a basic power component spec."""
        data = {'idle_w': 23, 'max_w': 153}
        spec = PowerComponentSpec.from_dict(data)
        assert spec.idle_w == 23
        assert spec.max_w == 153
        assert spec.power_curve is None

    def test_from_dict_with_power_curve(self):
        """Parse a power component spec with power curve."""
        data = {'idle_w': 23, 'max_w': 153, 'power_curve': {'type': 'specpower'}}
        spec = PowerComponentSpec.from_dict(data)
        assert spec.idle_w == 23
        assert spec.max_w == 153
        assert spec.power_curve is not None
        assert spec.power_curve.type == 'specpower'

    def test_to_dict_roundtrip(self):
        """to_dict should produce equivalent dict."""
        data = {'idle_w': 23, 'max_w': 153, 'power_curve': {'type': 'specpower'}}
        spec = PowerComponentSpec.from_dict(data)
        result = spec.to_dict()
        assert result['idle_w'] == 23
        assert result['max_w'] == 153
        assert result['power_curve'] == {'type': 'specpower'}

    def test_to_power_component_curve(self):
        """Convert spec to model-level PowerComponentCurve."""
        data = {'idle_w': 23, 'max_w': 153, 'power_curve': {'type': 'specpower'}}
        spec = PowerComponentSpec.from_dict(data)
        curve = spec.to_power_component_curve()
        assert curve.idle_w == 23
        assert curve.max_w == 153
        assert curve.power_at_util(0.0) == 23.0
        assert curve.power_at_util(1.0) == 153.0

    def test_to_power_component_curve_uses_default_when_no_component_curve(self):
        """Component without power_curve uses default_curve_spec (global), else polynomial."""
        spec = PowerComponentSpec.from_dict({'idle_w': 0, 'max_w': 100})
        assert spec.power_curve is None

        # With default_curve_spec=linear, power at 0.5 util is 50
        curve_linear = spec.to_power_component_curve(
            default_curve_spec=PowerCurveSpec(type='linear')
        )
        assert curve_linear.power_at_util(0.5) == 50.0

        # With no default, falls back to polynomial (not linear)
        curve_poly = spec.to_power_component_curve()
        power_at_half = curve_poly.power_at_util(0.5)
        assert power_at_half != 50.0  # polynomial shape differs from linear

        # With default_curve_spec=polynomial, same as no default
        curve_poly2 = spec.to_power_component_curve(
            default_curve_spec=PowerCurveSpec(type='polynomial')
        )
        assert curve_poly2.power_at_util(0.5) == power_at_half


class TestProcessorSpecPowerBreakdown:
    """Tests for ProcessorSpec with power_breakdown."""

    def test_from_dict_with_power_breakdown(self):
        """Parse processor spec with power_breakdown."""
        data = {
            'physical_cores': 80,
            'threads_per_core': 2,
            'power_idle_w': 150,
            'power_max_w': 440,
            'power_breakdown': {
                'cpu': {'idle_w': 23, 'max_w': 153, 'power_curve': {'type': 'specpower'}},
                'memory': {'idle_w': 56, 'max_w': 74},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.power_breakdown is not None
        assert len(spec.power_breakdown) == 2
        assert 'cpu' in spec.power_breakdown
        assert spec.power_breakdown['cpu'].idle_w == 23
        assert spec.power_breakdown['memory'].max_w == 74

    def test_to_dict_with_power_breakdown(self):
        """to_dict should include power_breakdown."""
        data = {
            'physical_cores': 80,
            'threads_per_core': 2,
            'power_idle_w': 150,
            'power_max_w': 440,
            'power_breakdown': {
                'cpu': {'idle_w': 23, 'max_w': 153},
                'memory': {'idle_w': 56, 'max_w': 74},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        result = spec.to_dict()
        assert 'power_breakdown' in result
        assert result['power_breakdown']['cpu'] == {'idle_w': 23, 'max_w': 153}
        assert result['power_breakdown']['memory'] == {'idle_w': 56, 'max_w': 74}

    def test_from_dict_without_power_breakdown(self):
        """Processor without power_breakdown should have None."""
        data = {'physical_cores': 48, 'threads_per_core': 1, 'power_idle_w': 100, 'power_max_w': 300}
        spec = ProcessorSpec.from_dict(data)
        assert spec.power_breakdown is None


class TestPowerBreakdownEndToEnd:
    """End-to-end tests for power breakdown through declarative analysis."""

    def test_compare_with_power_breakdown(self, tmp_path):
        """Run a compare analysis with power_breakdown in processor config."""
        config_data = {
            'name': 'power_breakdown_test',
            'scenarios': {
                'baseline': {'processor': 'smt', 'oversub_ratio': 1.0},
                'nosmt': {'processor': 'nosmt', 'oversub_ratio': 1.0},
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'baseline',
            },
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 150,
                    'power_max_w': 440,
                    'power_breakdown': {
                        'cpu': {'idle_w': 23, 'max_w': 153, 'power_curve': {'type': 'specpower'}},
                        'memory': {'idle_w': 56, 'max_w': 74},
                        'ssd': {'idle_w': 25, 'max_w': 50},
                    },
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 100,
                    'power_max_w': 300,
                },
            },
            'workload': {'total_vcpus': 1000, 'avg_util': 0.3},
        }

        config_file = tmp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'compare'

        # SMT scenario should have power_breakdown
        smt = result.scenario_results['baseline']
        assert smt.get('power_breakdown') is not None
        assert 'cpu' in smt['power_breakdown']['component_power_w']
        assert 'memory' in smt['power_breakdown']['component_power_w']
        assert 'ssd' in smt['power_breakdown']['component_power_w']

        # nosmt scenario should NOT have power_breakdown (no power_breakdown in config)
        nosmt = result.scenario_results['nosmt']
        assert nosmt.get('power_breakdown') is None

    def test_power_breakdown_total_matches_power_per_server(self, tmp_path):
        """Power breakdown total should match power_per_server_w when all components use linear."""
        config_data = {
            'name': 'power_total_test',
            'scenarios': {
                'test': {'processor': 'proc', 'oversub_ratio': 1.0},
            },
            'analysis': {
                'type': 'compare',
            },
            'processor': {
                'proc': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'power_breakdown': {
                        'cpu': {'idle_w': 50, 'max_w': 200},
                        'memory': {'idle_w': 75, 'max_w': 100},
                        'other': {'idle_w': 25, 'max_w': 50},
                    },
                },
            },
            'workload': {'total_vcpus': 1000, 'avg_util': 0.3},
        }

        config_file = tmp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        test = result.scenario_results['test']
        breakdown = test['power_breakdown']

        # Power per server should equal breakdown total
        assert abs(test['power_per_server_w'] - breakdown['total_power_w']) < 0.01

        # And should equal sum of components
        comp_sum = sum(breakdown['component_power_w'].values())
        assert abs(breakdown['total_power_w'] - comp_sum) < 0.01


class TestResourceScaling:
    """Tests for ResourceScalingConfig and resource scaling in _evaluate_scenario."""

    def test_from_dict_scale_with_vcpus(self):
        """Parse scale_with_vcpus from config."""
        data = {'scale_with_vcpus': ['memory', 'ssd']}
        cfg = ResourceScalingConfig.from_dict(data)
        assert cfg.scale_with_vcpus == ['memory', 'ssd']
        assert cfg.scale_power is True
        assert cfg.per_vcpu_carbon == {}
        assert cfg.per_vcpu_cost == {}

    def test_from_dict_custom_per_vcpu(self):
        """Parse custom per-vCPU values."""
        data = {
            'per_vcpu': {
                'carbon': {'extra_memory': 2.0},
                'cost': {'extra_memory': 30.0},
            },
        }
        cfg = ResourceScalingConfig.from_dict(data)
        assert cfg.per_vcpu_carbon == {'extra_memory': 2.0}
        assert cfg.per_vcpu_cost == {'extra_memory': 30.0}

    def test_from_dict_scale_power_false(self):
        """Parse scale_power: false."""
        data = {'scale_with_vcpus': ['memory'], 'scale_power': False}
        cfg = ResourceScalingConfig.from_dict(data)
        assert cfg.scale_power is False

    def test_to_dict_round_trip(self):
        """from_dict -> to_dict preserves data."""
        data = {
            'scale_with_vcpus': ['memory', 'ssd'],
            'per_vcpu': {'carbon': {'extra': 1.0}},
            'scale_power': False,
        }
        cfg = ResourceScalingConfig.from_dict(data)
        d = cfg.to_dict()
        assert d['scale_with_vcpus'] == ['memory', 'ssd']
        assert d['per_vcpu']['carbon'] == {'extra': 1.0}
        assert d['scale_power'] is False

    def test_scenario_config_with_resource_scaling(self):
        """ScenarioConfig parses resource_scaling field."""
        data = {
            'processor': 'nosmt',
            'oversub_ratio': 2.0,
            'resource_scaling': {
                'scale_with_vcpus': ['memory', 'ssd'],
            },
        }
        cfg = ScenarioConfig.from_dict(data)
        assert cfg.resource_scaling is not None
        assert cfg.resource_scaling.scale_with_vcpus == ['memory', 'ssd']

    def test_scenario_config_to_dict_with_resource_scaling(self):
        """ScenarioConfig.to_dict includes resource_scaling."""
        cfg = ScenarioConfig(
            processor='nosmt',
            oversub_ratio=2.0,
            resource_scaling=ResourceScalingConfig(scale_with_vcpus=['memory']),
        )
        d = cfg.to_dict()
        assert 'resource_scaling' in d
        assert d['resource_scaling']['scale_with_vcpus'] == ['memory']

    def test_scale_factor_clamped_at_1(self):
        """At R=1.0, scale_factor should be 1.0 (no downscaling)."""
        # Genoa nosmt: 80 cores, 1 tpc, thread_overhead=8
        # available_pcpus = 72, at R=1.0: vcpus = 72
        # hw_threads = 80, scale_factor = max(1.0, 72/80) = 1.0
        config_data = {
            'name': 'test_scale_clamp',
            'scenarios': {
                'nosmt_r1': {
                    'processor': 'nosmt',
                    'oversub_ratio': 1.0,
                    'resource_scaling': {'scale_with_vcpus': ['memory', 'ssd']},
                },
                'nosmt_baseline': {
                    'processor': 'nosmt',
                    'oversub_ratio': 1.0,
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'nosmt_baseline',
                'scenarios': ['nosmt_baseline', 'nosmt_r1'],
            },
            'processor': {
                'nosmt': {
                    'physical_cores': 80,
                    'threads_per_core': 1,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'thread_overhead': 8,
                    'embodied_carbon': {
                        'per_thread': {'memory': 4.43, 'ssd': 3.86},
                        'per_server': {'cpu': 34.2},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 33.0, 'ssd': 10.23},
                        'per_server': {'cpu': 1487.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
        }
        engine = DeclarativeAnalysisEngine()
        config = AnalysisConfig.from_dict(config_data)
        result = engine.run(config)
        # Both should produce same embodied carbon since scale_factor <= 1
        baseline = result.scenario_results['nosmt_baseline']
        scaled = result.scenario_results['nosmt_r1']
        assert abs(baseline['embodied_carbon_kg'] - scaled['embodied_carbon_kg']) < 0.01

    def test_borrowed_components_move_to_per_vcpu(self):
        """scale_with_vcpus moves components from per_core to per_vcpu (no double-counting)."""
        config_data = {
            'name': 'test_borrow',
            'scenarios': {
                'unscaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                },
                'scaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {'scale_with_vcpus': ['memory']},
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'unscaled',
                'scenarios': ['unscaled', 'scaled'],
            },
            'processor': {
                'nosmt': {
                    'physical_cores': 80,
                    'threads_per_core': 1,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'thread_overhead': 8,
                    'embodied_carbon': {
                        'per_thread': {'memory': 4.43, 'ssd': 3.86},
                        'per_server': {'cpu': 34.2},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
        }
        engine = DeclarativeAnalysisEngine()
        config = AnalysisConfig.from_dict(config_data)
        result = engine.run(config)
        scaled = result.scenario_results['scaled']
        bd = scaled['embodied_breakdown']['carbon']
        # memory should be in per_vcpu, not per_core
        assert 'memory' not in bd['per_thread']
        assert 'memory' in bd['per_vcpu']
        # ssd should remain in per_core
        assert 'ssd' in bd['per_thread']

    def test_end_to_end_genoa_nosmt_r2(self):
        """End-to-end math verification for Genoa nosmt at R=2.0 with memory/SSD scaling."""
        # Genoa nosmt: 80 cores, 1 tpc, thread_overhead=8
        # available_pcpus = 72, at R=2.0: vcpus_per_server = 144
        # hw_threads = 80, scale_factor = 144/80 = 1.8
        config_data = {
            'name': 'test_genoa_r2',
            'scenarios': {
                'unscaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                },
                'scaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {'scale_with_vcpus': ['memory', 'ssd']},
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'unscaled',
                'scenarios': ['unscaled', 'scaled'],
            },
            'processor': {
                'nosmt': {
                    'physical_cores': 80,
                    'threads_per_core': 1,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'thread_overhead': 8,
                    'embodied_carbon': {
                        'per_thread': {'memory': 4.43, 'ssd': 3.86},
                        'per_server': {'cpu': 34.2},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 33.0, 'ssd': 10.23},
                        'per_server': {'cpu': 1487.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
        }
        engine = DeclarativeAnalysisEngine()
        config = AnalysisConfig.from_dict(config_data)
        result = engine.run(config)

        unscaled = result.scenario_results['unscaled']
        scaled = result.scenario_results['scaled']

        # Both should have same number of servers (same oversub ratio, same processor)
        assert unscaled['num_servers'] == scaled['num_servers']
        num_servers = unscaled['num_servers']

        # Verify unscaled embodied carbon per server:
        # per_core: (4.43 + 3.86) * 80 = 663.2
        # per_server: 34.2
        # total: 697.4
        unscaled_per_server = (4.43 + 3.86) * 80 + 34.2
        assert abs(unscaled['embodied_carbon_kg'] - unscaled_per_server * num_servers) < 1.0

        # Verify scaled embodied carbon per server:
        # memory and ssd moved to per_vcpu, multiplied by 144 instead of 80
        # per_vcpu: (4.43 + 3.86) * 144 = 1193.76
        # per_server: 34.2
        # total: 1227.96
        scaled_per_server = (4.43 + 3.86) * 144 + 34.2
        assert abs(scaled['embodied_carbon_kg'] - scaled_per_server * num_servers) < 1.0

        # Scaled should be significantly higher
        assert scaled['embodied_carbon_kg'] > unscaled['embodied_carbon_kg']

    def test_power_scaling_correctness(self):
        """Power components scale by correct factor with resource scaling."""
        config_data = {
            'name': 'test_power_scale',
            'scenarios': {
                'unscaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                },
                'scaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {'scale_with_vcpus': ['memory']},
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'unscaled',
                'scenarios': ['unscaled', 'scaled'],
            },
            'processor': {
                'nosmt': {
                    'physical_cores': 80,
                    'threads_per_core': 1,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'thread_overhead': 8,
                    'power_breakdown': {
                        'cpu': {'idle_w': 94, 'max_w': 315},
                        'memory': {'idle_w': 20, 'max_w': 66},
                        'chassis': {'idle_w': 60, 'max_w': 116},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
        }
        engine = DeclarativeAnalysisEngine()
        config = AnalysisConfig.from_dict(config_data)
        result = engine.run(config)

        unscaled = result.scenario_results['unscaled']
        scaled = result.scenario_results['scaled']

        # scale_factor = 144/80 = 1.8
        # Scaled power should be higher because memory power is scaled
        assert scaled['power_per_server_w'] > unscaled['power_per_server_w']

        # Check power breakdown: memory should be scaled by 1.8
        unscaled_mem = unscaled['power_breakdown']['component_power_w']['memory']
        scaled_mem = scaled['power_breakdown']['component_power_w']['memory']
        # Scale factor = 1.8
        assert abs(scaled_mem / unscaled_mem - 1.8) < 0.01

        # CPU should be unchanged
        unscaled_cpu = unscaled['power_breakdown']['component_power_w']['cpu']
        scaled_cpu = scaled['power_breakdown']['component_power_w']['cpu']
        assert abs(unscaled_cpu - scaled_cpu) < 0.01

    def test_scale_power_false_disables_power_scaling(self):
        """scale_power: false prevents power component scaling."""
        config_data = {
            'name': 'test_no_power_scale',
            'scenarios': {
                'unscaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                },
                'scaled_no_power': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {
                        'scale_with_vcpus': ['memory'],
                        'scale_power': False,
                    },
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'unscaled',
                'scenarios': ['unscaled', 'scaled_no_power'],
            },
            'processor': {
                'nosmt': {
                    'physical_cores': 80,
                    'threads_per_core': 1,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'thread_overhead': 8,
                    'power_breakdown': {
                        'cpu': {'idle_w': 94, 'max_w': 315},
                        'memory': {'idle_w': 20, 'max_w': 66},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
        }
        engine = DeclarativeAnalysisEngine()
        config = AnalysisConfig.from_dict(config_data)
        result = engine.run(config)

        unscaled = result.scenario_results['unscaled']
        scaled = result.scenario_results['scaled_no_power']

        # Power should be the same since scale_power is false
        assert abs(unscaled['power_per_server_w'] - scaled['power_per_server_w']) < 0.01

    def test_custom_per_vcpu_additive(self):
        """Custom per_vcpu components are additive on top of borrowed."""
        config_data = {
            'name': 'test_custom_per_vcpu',
            'scenarios': {
                'scaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {
                        'scale_with_vcpus': ['memory'],
                        'per_vcpu': {
                            'carbon': {'extra_cooling': 1.0},
                        },
                    },
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'scaled',
                'scenarios': ['scaled'],
            },
            'processor': {
                'nosmt': {
                    'physical_cores': 80,
                    'threads_per_core': 1,
                    'power_idle_w': 200,
                    'power_max_w': 500,
                    'thread_overhead': 8,
                    'embodied_carbon': {
                        'per_thread': {'memory': 4.0},
                        'per_server': {'cpu': 30.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
        }
        engine = DeclarativeAnalysisEngine()
        config = AnalysisConfig.from_dict(config_data)
        result = engine.run(config)

        scaled = result.scenario_results['scaled']
        bd = scaled['embodied_breakdown']['carbon']
        # memory should be in per_vcpu (moved from per_core)
        assert 'memory' in bd['per_vcpu']
        # extra_cooling should also be in per_vcpu
        assert 'extra_cooling' in bd['per_vcpu']
        assert abs(bd['per_vcpu']['extra_cooling'] - 1.0) < 0.01

    def test_integration_full_declarative_engine(self, tmp_path):
        """Integration test: full config dict through DeclarativeAnalysisEngine."""
        config_data = {
            'name': 'resource_scaling_integration',
            'scenarios': {
                'baseline': {'processor': 'smt', 'oversub_ratio': 1.0},
                'nosmt_unscaled': {'processor': 'nosmt', 'oversub_ratio': 2.0},
                'nosmt_scaled': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {
                        'scale_with_vcpus': ['memory', 'ssd'],
                    },
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'baseline',
                'scenarios': ['baseline', 'nosmt_unscaled', 'nosmt_scaled'],
            },
            'processor': {
                'smt': {
                    'physical_cores': 48, 'threads_per_core': 2,
                    'power_idle_w': 100, 'power_max_w': 400,
                    'thread_overhead': 4,
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0, 'ssd': 3.0},
                        'per_server': {'cpu': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 30.0, 'ssd': 10.0},
                        'per_server': {'cpu': 1500.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 48, 'threads_per_core': 1,
                    'power_idle_w': 90, 'power_max_w': 340,
                    'thread_overhead': 4,
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0, 'ssd': 3.0},
                        'per_server': {'cpu': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 30.0, 'ssd': 10.0},
                        'per_server': {'cpu': 1500.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
            'cost': {
                'carbon_intensity_g_kwh': 400,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5,
            },
        }

        config_file = tmp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)

        # All three scenarios should exist
        assert 'baseline' in result.scenario_results
        assert 'nosmt_unscaled' in result.scenario_results
        assert 'nosmt_scaled' in result.scenario_results

        unscaled = result.scenario_results['nosmt_unscaled']
        scaled = result.scenario_results['nosmt_scaled']

        # Scaled should have higher embodied carbon (memory/ssd scale with vCPUs)
        assert scaled['embodied_carbon_kg'] > unscaled['embodied_carbon_kg']
        # Scaled should have higher embodied cost
        assert scaled['embodied_cost_usd'] > unscaled['embodied_cost_usd']
        # Both should have same server count
        assert scaled['num_servers'] == unscaled['num_servers']


class TestBreakevenCurve:
    """Tests for breakeven_curve analysis type."""

    def _make_compare_sweep_config(self, avg_util, tmp_path, suffix=""):
        """Helper to create a compare_sweep config file that has a breakeven."""
        config_data = {
            'name': f'util_{int(avg_util*100)}pct{suffix}',
            'scenarios': {
                'smt_baseline': {
                    'processor': 'smt',
                    'oversub_ratio': 1.0,
                },
                'nosmt_target': {
                    'processor': 'nosmt',
                    'oversub_ratio': 1.5,
                    'vcpu_demand_multiplier': 1.0,
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': avg_util},
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 100.0,
                    'power_max_w': 400.0,
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                },
            },
            'cost': {
                'carbon_intensity_g_kwh': 400,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5,
            },
            'analysis': {
                'type': 'compare_sweep',
                'baseline': 'smt_baseline',
                'sweep_scenario': 'nosmt_target',
                'sweep_parameter': 'vcpu_demand_multiplier',
                'sweep_values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'show_breakeven_marker': True,
            },
        }
        filename = f'util_{int(avg_util*100)}{suffix}.json'
        config_file = tmp_path / filename
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        return config_file

    def test_basic_breakeven_curve(self, tmp_path):
        """Test basic breakeven_curve execution with multiple sub-configs."""
        # Create sub-configs at different utilization levels
        cfg_10 = self._make_compare_sweep_config(0.1, tmp_path)
        cfg_20 = self._make_compare_sweep_config(0.2, tmp_path)
        cfg_30 = self._make_compare_sweep_config(0.3, tmp_path)

        # Create breakeven_curve config
        curve_config = {
            'name': 'test_breakeven_curve',
            'analysis': {
                'type': 'breakeven_curve',
                'series': [
                    {
                        'label': 'Test Series',
                        'configs': [str(cfg_10), str(cfg_20), str(cfg_30)],
                    }
                ],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'breakeven_metric': 'carbon',
                'x_label': 'Utilization (%)',
                'y_label': 'Breakeven Multiplier',
            },
        }
        curve_file = tmp_path / 'curve.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)

        assert result.analysis_type == 'breakeven_curve'
        assert result.breakeven_curve_results is not None
        assert len(result.breakeven_curve_results) == 1

        series = result.breakeven_curve_results[0]
        assert series['label'] == 'Test Series'
        assert len(series['points']) == 3

        # Check x-values are display-multiplied
        assert series['points'][0]['x_value'] == pytest.approx(10.0)
        assert series['points'][1]['x_value'] == pytest.approx(20.0)
        assert series['points'][2]['x_value'] == pytest.approx(30.0)

        # Check raw values preserved
        assert series['points'][0]['x_raw'] == pytest.approx(0.1)

    def test_x_display_multiplier(self, tmp_path):
        """Test that x_display_multiplier correctly scales displayed values."""
        cfg = self._make_compare_sweep_config(0.2, tmp_path)

        # With multiplier=1 (no scaling)
        curve_config = {
            'name': 'test_no_multiplier',
            'analysis': {
                'type': 'breakeven_curve',
                'series': [{'label': 'S', 'configs': [str(cfg)]}],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 1.0,
                'breakeven_metric': 'carbon',
            },
        }
        curve_file = tmp_path / 'curve_nomult.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)
        pt = result.breakeven_curve_results[0]['points'][0]
        assert pt['x_value'] == pytest.approx(0.2)
        assert pt['x_raw'] == pytest.approx(0.2)

    def test_multiple_series(self, tmp_path):
        """Test breakeven_curve with multiple series."""
        cfg_a = self._make_compare_sweep_config(0.2, tmp_path, suffix="_a")
        cfg_b = self._make_compare_sweep_config(0.2, tmp_path, suffix="_b")

        curve_config = {
            'name': 'test_multi_series',
            'analysis': {
                'type': 'breakeven_curve',
                'series': [
                    {'label': 'Series A', 'configs': [str(cfg_a)]},
                    {'label': 'Series B', 'configs': [str(cfg_b)]},
                ],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'breakeven_metric': 'carbon',
            },
        }
        curve_file = tmp_path / 'curve_multi.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)
        assert len(result.breakeven_curve_results) == 2
        assert result.breakeven_curve_results[0]['label'] == 'Series A'
        assert result.breakeven_curve_results[1]['label'] == 'Series B'

    def test_validation_without_scenarios(self, tmp_path):
        """Test that breakeven_curve configs are valid without 'scenarios' key."""
        config_data = {
            'name': 'test_valid',
            'analysis': {
                'type': 'breakeven_curve',
                'series': [],
                'x_parameter': 'workload.avg_util',
            },
        }
        config_file = tmp_path / 'valid_no_scenarios.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        assert is_valid_analysis_config(config_file) is True

    def test_non_breakeven_curve_requires_scenarios(self, tmp_path):
        """Test that non-breakeven_curve configs still require 'scenarios'."""
        config_data = {
            'name': 'test_invalid',
            'analysis': {'type': 'compare'},
        }
        config_file = tmp_path / 'invalid_no_scenarios.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        assert is_valid_analysis_config(config_file) is False

    def test_to_dict_roundtrip(self):
        """Test that breakeven_curve AnalysisSpec serializes correctly."""
        from .declarative import AnalysisSpec

        spec = AnalysisSpec(
            type='breakeven_curve',
            series=[{'label': 'A', 'configs': ['a.json']}],
            x_parameter='workload.avg_util',
            x_display_multiplier=100,
            breakeven_metric='carbon',
            x_label='Util (%)',
            y_label='Breakeven',
        )
        d = spec.to_dict()
        assert d['type'] == 'breakeven_curve'
        assert d['series'] == [{'label': 'A', 'configs': ['a.json']}]
        assert d['x_parameter'] == 'workload.avg_util'
        assert d['x_display_multiplier'] == 100
        assert d['breakeven_metric'] == 'carbon'
        assert d['x_label'] == 'Util (%)'
        assert d['y_label'] == 'Breakeven'

        # Roundtrip
        spec2 = AnalysisSpec.from_dict(d)
        assert spec2.type == 'breakeven_curve'
        assert spec2.series == spec.series
        assert spec2.x_parameter == spec.x_parameter
        assert spec2.x_display_multiplier == spec.x_display_multiplier

    def test_summary_output(self, tmp_path):
        """Test that summary text is generated."""
        cfg = self._make_compare_sweep_config(0.2, tmp_path)

        curve_config = {
            'name': 'test_summary',
            'analysis': {
                'type': 'breakeven_curve',
                'series': [{'label': 'My Series', 'configs': [str(cfg)]}],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'breakeven_metric': 'carbon',
                'x_label': 'Util (%)',
                'y_label': 'Breakeven',
            },
        }
        curve_file = tmp_path / 'curve_summary.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)
        assert 'Breakeven Curve Analysis' in result.summary
        assert 'My Series' in result.summary
        assert '20.0' in result.summary


class TestSavingsCurve:
    """Tests for savings_curve analysis type."""

    def _make_compare_sweep_config(self, avg_util, tmp_path, suffix=""):
        """Helper to create a compare_sweep config file."""
        config_data = {
            'name': f'util_{int(avg_util*100)}pct{suffix}',
            'scenarios': {
                'smt_baseline': {
                    'processor': 'smt',
                    'oversub_ratio': 1.0,
                },
                'nosmt_target': {
                    'processor': 'nosmt',
                    'oversub_ratio': 1.5,
                    'vcpu_demand_multiplier': 1.0,
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': avg_util},
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 100.0,
                    'power_max_w': 400.0,
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                },
            },
            'cost': {
                'carbon_intensity_g_kwh': 400,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5,
            },
            'analysis': {
                'type': 'compare_sweep',
                'baseline': 'smt_baseline',
                'sweep_scenario': 'nosmt_target',
                'sweep_parameter': 'vcpu_demand_multiplier',
                'sweep_values': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'show_breakeven_marker': True,
            },
        }
        filename = f'util_{int(avg_util*100)}{suffix}.json'
        config_file = tmp_path / filename
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        return config_file

    def test_basic_savings_curve(self, tmp_path):
        """Test basic savings_curve execution."""
        cfg_10 = self._make_compare_sweep_config(0.1, tmp_path)
        cfg_20 = self._make_compare_sweep_config(0.2, tmp_path)
        cfg_30 = self._make_compare_sweep_config(0.3, tmp_path)

        curve_config = {
            'name': 'test_savings_curve',
            'analysis': {
                'type': 'savings_curve',
                'configs': [str(cfg_10), str(cfg_20), str(cfg_30)],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'marker_values': [0.7, 0.8, 0.9],
                'marker_labels': ['low', 'mid', 'high'],
                'metrics': ['carbon', 'tco'],
                'x_label': 'Utilization (%)',
                'y_label': 'Savings vs Baseline (%)',
            },
        }
        curve_file = tmp_path / 'savings_curve.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)

        assert result.analysis_type == 'savings_curve'
        assert result.savings_curve_results is not None
        assert len(result.savings_curve_results) == 3  # one per marker value

        # Check structure of first marker series
        series = result.savings_curve_results[0]
        assert series['marker_value'] == 0.7
        assert 'low' in series['label']
        assert len(series['points']) == 3  # one per config

        # Check x-values are display-multiplied
        assert series['points'][0]['x_value'] == pytest.approx(10.0)
        assert series['points'][1]['x_value'] == pytest.approx(20.0)
        assert series['points'][2]['x_value'] == pytest.approx(30.0)

        # Check raw values preserved
        assert series['points'][0]['x_raw'] == pytest.approx(0.1)

        # Each point should have both metrics
        for pt in series['points']:
            assert 'carbon_diff_pct' in pt
            assert 'tco_diff_pct' in pt
            assert pt['carbon_diff_pct'] is not None
            assert pt['tco_diff_pct'] is not None

    def test_savings_curve_summary(self, tmp_path):
        """Test that savings_curve produces a summary."""
        cfg = self._make_compare_sweep_config(0.2, tmp_path)

        curve_config = {
            'name': 'test_summary',
            'analysis': {
                'type': 'savings_curve',
                'configs': [str(cfg)],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'marker_values': [0.7],
                'marker_labels': ['mid'],
                'metrics': ['carbon'],
                'x_label': 'Utilization (%)',
            },
        }
        curve_file = tmp_path / 'savings_summary.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)
        assert 'Savings Curve Analysis' in result.summary
        assert 'mid' in result.summary
        assert '20.0' in result.summary

    def test_savings_curve_to_dict(self, tmp_path):
        """Test that savings_curve results serialize correctly."""
        cfg = self._make_compare_sweep_config(0.2, tmp_path)

        curve_config = {
            'name': 'test_to_dict',
            'analysis': {
                'type': 'savings_curve',
                'configs': [str(cfg)],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'marker_values': [0.7, 0.9],
                'marker_labels': ['low', 'high'],
                'metrics': ['carbon', 'tco'],
            },
        }
        curve_file = tmp_path / 'savings_dict.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)
        d = result.to_dict()
        assert 'savings_curve_results' in d
        assert len(d['savings_curve_results']) == 2

    def test_savings_curve_interpolation(self, tmp_path):
        """Test interpolation for marker values between sweep points."""
        cfg = self._make_compare_sweep_config(0.2, tmp_path)

        # Use marker value 0.75 which is between sweep values 0.7 and 0.8
        curve_config = {
            'name': 'test_interp',
            'analysis': {
                'type': 'savings_curve',
                'configs': [str(cfg)],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'marker_values': [0.75],
                'marker_labels': ['interp'],
                'metrics': ['carbon'],
            },
        }
        curve_file = tmp_path / 'savings_interp.json'
        with open(curve_file, 'w') as f:
            json.dump(curve_config, f)

        result = run_analysis(curve_file)
        pt = result.savings_curve_results[0]['points'][0]
        assert pt['carbon_diff_pct'] is not None

    def test_is_valid_analysis_config(self, tmp_path):
        """Test that savings_curve configs pass validation without scenarios."""
        from smt_oversub_model.declarative import is_valid_analysis_config
        config_data = {
            'name': 'test_valid',
            'analysis': {
                'type': 'savings_curve',
                'configs': ['some/path.json'],
                'x_parameter': 'workload.avg_util',
                'marker_values': [0.7],
            },
        }
        config_file = tmp_path / 'valid.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        assert is_valid_analysis_config(config_file) is True


class TestMaxVmsPerServerDeclarative:
    """Tests for max_vms_per_server and avg_vm_size_vcpus in declarative configs."""

    def test_scenario_config_roundtrip(self):
        """ScenarioConfig serializes/deserializes new fields."""
        data = {
            'processor': 'nosmt',
            'oversub_ratio': 2.0,
            'max_vms_per_server': 50,
            'avg_vm_size_vcpus': 4.0,
        }
        cfg = ScenarioConfig.from_dict(data)
        assert cfg.max_vms_per_server == 50
        assert cfg.avg_vm_size_vcpus == 4.0

        d = cfg.to_dict()
        assert d['max_vms_per_server'] == 50
        assert d['avg_vm_size_vcpus'] == 4.0

    def test_scenario_config_defaults(self):
        """ScenarioConfig defaults to None for new fields."""
        cfg = ScenarioConfig.from_dict({'processor': 'smt'})
        assert cfg.max_vms_per_server is None
        assert cfg.avg_vm_size_vcpus is None

        d = cfg.to_dict()
        assert 'max_vms_per_server' not in d
        assert 'avg_vm_size_vcpus' not in d

    def test_integration_vm_cap_increases_servers(self, tmp_path):
        """Declarative config with max_vms_per_server increases server count."""
        import math
        config = {
            'name': 'vm_cap_test',
            'processor': {
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    'embodied_carbon': {
                        'per_thread': {'cpu': 5.0},
                        'per_server': {'chassis': 200.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 1000.0},
                    },
                },
            },
            'workload': {'total_vcpus': 1000, 'avg_util': 0.3, 'avg_vm_size_vcpus': 1.0},
            'scenarios': {
                'uncapped': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                },
                'capped': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'max_vms_per_server': 30,
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'uncapped',
            },
        }
        config_file = tmp_path / 'vm_cap.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        uncapped = result.scenario_results['uncapped']
        capped = result.scenario_results['capped']

        # Natural capacity = 48 * 2.0 = 96 vcpus/server
        # Capped: 30 VMs * 1 vcpu = 30 vcpus/server
        assert uncapped['num_servers'] == math.ceil(1000 / 96)  # 11
        assert capped['num_servers'] == math.ceil(1000 / 30)    # 34
        assert capped['num_servers'] > uncapped['num_servers']

        # Verify structured costs are used (not defaults)
        # Per server: cpu=5*48 + chassis=200 = 440 kg
        expected_carbon_per_server = 5.0 * 48 + 200.0
        assert abs(uncapped['embodied_carbon_kg'] / uncapped['num_servers'] - expected_carbon_per_server) < 0.01

    def test_integration_vm_cap_with_avg_vm_size(self, tmp_path):
        """Declarative config with avg_vm_size_vcpus at workload level."""
        import math
        config = {
            'name': 'vm_size_test',
            'processor': {
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                },
            },
            'workload': {
                'total_vcpus': 1000,
                'avg_util': 0.3,
                'avg_vm_size_vcpus': 4.0,
            },
            'scenarios': {
                'capped': {
                    'processor': 'nosmt',
                    'oversub_ratio': 2.0,
                    'max_vms_per_server': 20,
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'capped',
            },
        }
        config_file = tmp_path / 'vm_size.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        capped = result.scenario_results['capped']
        # VM cap: 20 VMs * 4 vcpus = 80 vcpus/server (< 96 natural)
        assert capped['num_servers'] == math.ceil(1000 / 80)  # 13


class TestComponentBreakdownPerServerComponents:
    """Tests for ComponentBreakdown.per_server_components property."""

    def test_per_core_only(self):
        """Per-core components resolve correctly."""
        bd = ComponentBreakdown(
            per_thread={'memory': 4.8, 'ssd': 75.0},
            per_server={},
            physical_cores=80,
            threads_per_core=2,
        )
        comps = bd.per_server_components
        assert comps['memory'] == pytest.approx(4.8 * 160, abs=0.1)
        assert comps['ssd'] == pytest.approx(75.0 * 160, abs=0.1)

    def test_per_server_only(self):
        """Per-server components resolve correctly."""
        bd = ComponentBreakdown(
            per_thread={},
            per_server={'cpu': 34.2, 'nic': 115.0},
            physical_cores=80,
            threads_per_core=1,
        )
        comps = bd.per_server_components
        assert comps['cpu'] == 34.2
        assert comps['nic'] == 115.0

    def test_per_vcpu_only(self):
        """Per-vCPU components resolve correctly with vcpus_per_server."""
        bd = ComponentBreakdown(
            per_thread={},
            per_server={},
            per_vcpu={'memory': 4.8},
            physical_cores=80,
            threads_per_core=1,
            vcpus_per_server=144,
        )
        comps = bd.per_server_components
        assert comps['memory'] == pytest.approx(4.8 * 144, abs=0.1)

    def test_per_vcpu_fallback_hw_threads(self):
        """Per-vCPU falls back to hw_threads when vcpus_per_server is 0."""
        bd = ComponentBreakdown(
            per_thread={},
            per_server={},
            per_vcpu={'memory': 4.8},
            physical_cores=80,
            threads_per_core=2,
            vcpus_per_server=0,
        )
        comps = bd.per_server_components
        assert comps['memory'] == pytest.approx(4.8 * 160, abs=0.1)

    def test_mixed_components(self):
        """Mix of per_core, per_server, and per_vcpu resolves correctly."""
        bd = ComponentBreakdown(
            per_thread={'ssd': 75.0},
            per_server={'cpu': 34.2},
            per_vcpu={'memory': 4.8},
            physical_cores=80,
            threads_per_core=1,
            vcpus_per_server=144,
        )
        comps = bd.per_server_components
        assert comps['ssd'] == pytest.approx(75.0 * 80, abs=0.1)
        assert comps['cpu'] == 34.2
        assert comps['memory'] == pytest.approx(4.8 * 144, abs=0.1)


class TestCapacityParsing:
    """Tests for capacity field in ProcessorSpec."""

    def test_capacity_from_dict(self):
        """ProcessorSpec with capacity field parses correctly."""
        data = {
            'physical_cores': 80,
            'threads_per_core': 2,
            'power_idle_w': 100,
            'power_max_w': 400,
            'capacity': {
                'per_thread': {'memory': 4.8, 'ssd': 75.0},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.capacity is not None
        assert spec.capacity.per_thread['memory'] == 4.8
        assert spec.capacity.per_thread['ssd'] == 75.0

    def test_capacity_to_dict(self):
        """ProcessorSpec with capacity serializes correctly."""
        spec = ProcessorSpec(
            physical_cores=80,
            threads_per_core=2,
            capacity=EmbodiedComponentSpec(
                per_thread={'memory': 4.8, 'ssd': 75.0},
            ),
        )
        d = spec.to_dict()
        assert 'capacity' in d
        assert d['capacity']['per_thread']['memory'] == 4.8

    def test_capacity_none_by_default(self):
        """ProcessorSpec without capacity has None."""
        spec = ProcessorSpec(physical_cores=48, threads_per_core=1)
        assert spec.capacity is None

    def test_capacity_not_in_dict_when_none(self):
        """ProcessorSpec without capacity omits it from dict."""
        spec = ProcessorSpec(physical_cores=48, threads_per_core=1)
        d = spec.to_dict()
        assert 'capacity' not in d

    def test_capacity_inline_processor_detection(self):
        """capacity field detected as inline processor."""
        from .declarative import ProcessorConfigSpec
        assert ProcessorConfigSpec._is_inline_processor({
            'capacity': {'per_thread': {'memory': 4.8}},
        })


class TestCapacityResolution:
    """Tests for capacity resolution in _resolve_scenario_cost_overrides."""

    def test_capacity_in_cost_overrides(self, tmp_path):
        """Capacity breakdown appears in cost_overrides after resolution."""
        config = {
            'name': 'capacity_test',
            'scenarios': {
                'test': {'processor': 'proc', 'oversub_ratio': 1.0},
            },
            'processor': {
                'proc': {
                    'physical_cores': 10,
                    'threads_per_core': 2,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0},
                        'per_server': {'cpu': 30.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 40.0},
                        'per_server': {'cpu': 1000.0},
                    },
                    'capacity': {
                        'per_thread': {'memory': 4.0, 'ssd': 100.0},
                    },
                },
            },
            'analysis': {'type': 'compare'},
            'workload': {'total_vcpus': 1000, 'avg_util': 0.3},
        }
        config_file = tmp_path / 'cap.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        test_data = result.scenario_results['test']
        bd = test_data.get('embodied_breakdown')
        assert bd is not None
        capacity = bd.get('capacity')
        assert capacity is not None
        # 10 cores * 2 tpc = 20 hw threads
        # per_core memory: 4.0 * 20 = 80 per server
        assert capacity['per_thread']['memory'] == 4.0

    def test_capacity_with_resource_scaling(self, tmp_path):
        """Capacity components move from per_core to per_vcpu with resource scaling."""
        config = {
            'name': 'capacity_scaling_test',
            'scenarios': {
                'scaled': {
                    'processor': 'proc',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {
                        'scale_with_vcpus': ['memory', 'ssd'],
                    },
                },
            },
            'processor': {
                'proc': {
                    'physical_cores': 10,
                    'threads_per_core': 1,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0},
                        'per_server': {'cpu': 30.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 40.0},
                        'per_server': {'cpu': 1000.0},
                    },
                    'capacity': {
                        'per_thread': {'memory': 4.0, 'ssd': 100.0},
                    },
                },
            },
            'analysis': {'type': 'compare'},
            'workload': {'total_vcpus': 1000, 'avg_util': 0.3},
        }
        config_file = tmp_path / 'cap_scaled.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        test_data = result.scenario_results['scaled']
        bd = test_data.get('embodied_breakdown')
        assert bd is not None
        capacity = bd.get('capacity')
        assert capacity is not None
        # 10 cores * 1 tpc = 10 hw threads, available_pcpus = 10
        # R=2.0 -> vcpus_per_server = max(10, 10*2.0) = 20
        # memory and ssd moved from per_core to per_vcpu
        assert 'memory' not in capacity.get('per_thread', {})
        assert 'ssd' not in capacity.get('per_thread', {})
        assert capacity['per_vcpu']['memory'] == 4.0
        assert capacity['per_vcpu']['ssd'] == 100.0
        assert capacity['vcpus_per_server'] == 20.0


class TestPerServerComparisonAnalysis:
    """Tests for per_server_comparison analysis type."""

    def test_basic_per_server_comparison(self, tmp_path):
        """Full per_server_comparison analysis produces correct results."""
        config = {
            'name': 'per_server_test',
            'scenarios': {
                'smt_a': {
                    'processor': 'smt',
                    'oversub_ratio': 2.0,
                    'resource_scaling': {'scale_with_vcpus': ['memory', 'ssd']},
                },
                'nosmt_a': {
                    'processor': 'nosmt',
                    'oversub_ratio': 3.0,
                    'resource_scaling': {'scale_with_vcpus': ['memory', 'ssd']},
                },
            },
            'processor': {
                'smt': {
                    'physical_cores': 10,
                    'threads_per_core': 2,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0, 'ssd': 3.0},
                        'per_server': {'cpu': 30.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 40.0, 'ssd': 10.0},
                        'per_server': {'cpu': 1000.0},
                    },
                    'capacity': {
                        'per_thread': {'memory': 4.0, 'ssd': 100.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 10,
                    'threads_per_core': 1,
                    'power_idle_w': 90,
                    'power_max_w': 340,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'memory': 5.0, 'ssd': 3.0},
                        'per_server': {'cpu': 30.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 40.0, 'ssd': 10.0},
                        'per_server': {'cpu': 1000.0},
                    },
                    'capacity': {
                        'per_thread': {'memory': 4.0, 'ssd': 100.0},
                    },
                },
            },
            'analysis': {
                'type': 'per_server_comparison',
                'groups': [
                    {'label': 'Group A', 'scenarios': ['smt_a', 'nosmt_a']},
                ],
                'metrics': ['capacity.memory', 'capacity.ssd',
                           'embodied_carbon.memory', 'embodied_carbon.ssd'],
                'metric_labels': {
                    'capacity.memory': 'Memory (GB)',
                    'capacity.ssd': 'SSD (GB)',
                    'embodied_carbon.memory': 'Memory Carbon (kg)',
                    'embodied_carbon.ssd': 'SSD Carbon (kg)',
                },
                'labels': {
                    'smt_a': 'SMT',
                    'nosmt_a': 'No-SMT',
                },
            },
            'workload': {'total_vcpus': 1000, 'avg_util': 0.3},
        }
        config_file = tmp_path / 'per_server.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'per_server_comparison'
        assert result.per_server_comparison_results is not None
        assert len(result.per_server_comparison_results) == 1

        group = result.per_server_comparison_results[0]
        assert group['label'] == 'Group A'
        assert 'smt_a' in group['scenarios']
        assert 'nosmt_a' in group['scenarios']
        assert group['scenarios']['smt_a']['label'] == 'SMT'
        assert group['scenarios']['nosmt_a']['label'] == 'No-SMT'

        # SMT: 10 cores * 2 tpc = 20 hw_threads, available_pcpus = 20
        # R=2.0 -> vcpus_per_server = max(20, 20*2.0) = 40
        # capacity memory scaled: 4.0/thread, moved to per_vcpu -> 4.0 * 40 = 160
        smt_metrics = group['scenarios']['smt_a']['metrics']
        assert smt_metrics['capacity.memory'] == pytest.approx(160.0, abs=0.1)
        assert smt_metrics['capacity.ssd'] == pytest.approx(4000.0, abs=0.1)

        # NoSMT: 10 cores * 1 tpc = 10 hw_threads, available_pcpus = 10
        # R=3.0 -> vcpus_per_server = max(10, 10*3.0) = 30
        # capacity memory: 4.0 * 30 = 120
        nosmt_metrics = group['scenarios']['nosmt_a']['metrics']
        assert nosmt_metrics['capacity.memory'] == pytest.approx(120.0, abs=0.1)
        assert nosmt_metrics['capacity.ssd'] == pytest.approx(3000.0, abs=0.1)

    def test_per_server_comparison_structured_costs(self, tmp_path):
        """Per-server comparison uses processor-resolved structured costs (not defaults)."""
        config = {
            'name': 'per_server_cost_test',
            'scenarios': {
                'proc_a': {
                    'processor': 'custom',
                    'oversub_ratio': 1.0,
                },
            },
            'processor': {
                'custom': {
                    'physical_cores': 10,
                    'threads_per_core': 2,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'memory': 7.0},
                        'per_server': {'cpu': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'memory': 25.0},
                        'per_server': {'cpu': 800.0},
                    },
                    'capacity': {
                        'per_thread': {'memory': 8.0},
                        'per_server': {'rack_units': 1.0},
                    },
                },
            },
            'analysis': {
                'type': 'per_server_comparison',
                'groups': [
                    {'label': 'Test', 'scenarios': ['proc_a']},
                ],
                'metrics': ['capacity.memory', 'capacity.rack_units',
                           'embodied_carbon.memory', 'embodied_carbon.cpu'],
                'labels': {'proc_a': 'Custom'},
            },
            'workload': {'total_vcpus': 100, 'avg_util': 0.3},
        }
        config_file = tmp_path / 'per_server_cost.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'per_server_comparison'

        group = result.per_server_comparison_results[0]
        metrics = group['scenarios']['proc_a']['metrics']

        # 10 cores * 2 tpc = 20 hw_threads
        # capacity memory: 8.0 * 20 = 160 per server
        assert metrics['capacity.memory'] == pytest.approx(160.0, abs=0.1)
        # capacity rack_units: 1.0 per server (flat)
        assert metrics['capacity.rack_units'] == pytest.approx(1.0, abs=0.1)
        # embodied_carbon memory: 7.0 * 20 = 140 per server
        assert metrics['embodied_carbon.memory'] == pytest.approx(140.0, abs=0.1)
        # embodied_carbon cpu: 50.0 per server (flat)
        assert metrics['embodied_carbon.cpu'] == pytest.approx(50.0, abs=0.1)

        # Verify structured costs were used (not defaults)
        proc_a = result.scenario_results['proc_a']
        # embodied_carbon_kg should be 7.0 * 20 + 50.0 = 190, NOT default 1000
        import math
        expected_carbon = 7.0 * 20 + 50.0  # 190 per server
        num_servers = proc_a['num_servers']
        assert proc_a['embodied_carbon_kg'] == pytest.approx(expected_carbon * num_servers, abs=1)

    def test_per_server_comparison_to_dict(self, tmp_path):
        """per_server_comparison_results included in to_dict output."""
        config = {
            'name': 'dict_test',
            'scenarios': {
                's1': {'processor': 'p1', 'oversub_ratio': 1.0},
            },
            'processor': {
                'p1': {
                    'physical_cores': 10,
                    'threads_per_core': 1,
                    'power_idle_w': 100,
                    'power_max_w': 400,
                    'capacity': {
                        'per_thread': {'memory': 4.0},
                    },
                },
            },
            'analysis': {
                'type': 'per_server_comparison',
                'groups': [{'label': 'G1', 'scenarios': ['s1']}],
                'metrics': ['capacity.memory'],
                'labels': {'s1': 'S1'},
            },
            'workload': {'total_vcpus': 100, 'avg_util': 0.3},
        }
        config_file = tmp_path / 'dict.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        result = run_analysis(config_file)
        d = result.to_dict()
        assert 'per_server_comparison_results' in d
        assert len(d['per_server_comparison_results']) == 1


class TestResourceConstraints:
    """Tests for resource-constrained packing."""

    # --- Unit tests ---

    def test_resource_constraint_spec_max_vcpus_per_core(self):
        """capacity_per_thread with demand_per_vcpu computes max vCPUs correctly."""
        spec = ResourceConstraintSpec(capacity_per_thread=2.0, demand_per_vcpu=4.0)
        # 48 threads * 2.0 GB per thread = 96 GB total, / 4.0 GB per vCPU = 24
        assert spec.max_vcpus(48) == 24.0

    def test_resource_constraint_spec_max_vcpus_per_server(self):
        """capacity_per_server with demand_per_vcpu computes max vCPUs correctly."""
        spec = ResourceConstraintSpec(capacity_per_server=2000, demand_per_vcpu=20.0)
        # 2000 / 20 = 100 regardless of hw_threads
        assert spec.max_vcpus(48) == 100.0
        assert spec.max_vcpus(96) == 100.0

    def test_resource_constraint_spec_zero_demand(self):
        """Zero demand means infinite capacity."""
        spec = ResourceConstraintSpec(capacity_per_thread=2.0, demand_per_vcpu=0.0)
        assert spec.max_vcpus(48) == float('inf')

    def test_resource_constraint_spec_no_capacity(self):
        """No capacity specified means infinite."""
        spec = ResourceConstraintSpec(demand_per_vcpu=4.0)
        assert spec.max_vcpus(48) == float('inf')

    def test_resource_constraint_spec_from_dict_to_dict(self):
        """from_dict -> to_dict preserves data."""
        data = {'capacity_per_thread': 2.0, 'demand_per_vcpu': 4.0}
        spec = ResourceConstraintSpec.from_dict(data)
        d = spec.to_dict()
        assert d == data
        assert ResourceConstraintSpec.from_dict(d).max_vcpus(48) == spec.max_vcpus(48)

    def test_resource_constraints_config_from_dict_to_dict(self):
        """ResourceConstraintsConfig roundtrips through dict."""
        data = {
            'memory_gb': {'capacity_per_thread': 2.0, 'demand_per_vcpu': 4.0},
            'ssd_gb': {'capacity_per_server': 2000, 'demand_per_vcpu': 20.0},
        }
        cfg = ResourceConstraintsConfig.from_dict(data)
        assert len(cfg.constraints) == 2
        assert 'memory_gb' in cfg.constraints
        d = cfg.to_dict()
        assert d['memory_gb']['capacity_per_thread'] == 2.0
        assert d['ssd_gb']['capacity_per_server'] == 2000

    def test_scenario_config_parses_resource_constraints(self):
        """ScenarioConfig parses resource_constraints field."""
        data = {
            'processor': 'nosmt',
            'oversub_ratio': 2.0,
            'resource_constraints': {
                'memory_gb': {'capacity_per_thread': 2.0, 'demand_per_vcpu': 4.0},
            },
        }
        cfg = ScenarioConfig.from_dict(data)
        assert cfg.resource_constraints is not None
        assert 'memory_gb' in cfg.resource_constraints.constraints

    def test_scenario_config_to_dict_includes_resource_constraints(self):
        """ScenarioConfig.to_dict includes resource_constraints."""
        cfg = ScenarioConfig(
            processor='nosmt',
            oversub_ratio=2.0,
            resource_constraints=ResourceConstraintsConfig(
                constraints={'memory_gb': ResourceConstraintSpec(capacity_per_thread=2.0, demand_per_vcpu=4.0)}
            ),
        )
        d = cfg.to_dict()
        assert 'resource_constraints' in d
        assert d['resource_constraints']['memory_gb']['capacity_per_thread'] == 2.0

    # --- Mutual exclusion ---

    def test_mutual_exclusion_with_resource_scaling(self):
        """resource_scaling + resource_constraints raises ValueError."""
        data = {
            'processor': 'nosmt',
            'oversub_ratio': 2.0,
            'resource_scaling': {'scale_with_vcpus': ['memory']},
            'resource_constraints': {
                'memory_gb': {'capacity_per_thread': 2.0, 'demand_per_vcpu': 4.0},
            },
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            ScenarioConfig.from_dict(data)

    # --- Integration tests with structured costs ---

    def _make_config_with_constraints(self, constraints, oversub_ratio=2.0):
        """Helper to build a config with resource constraints and structured costs."""
        return {
            'name': 'test_constraints',
            'scenarios': {
                'baseline': {
                    'processor': 'smt',
                    'oversub_ratio': 1.0,
                },
                'constrained': {
                    'processor': 'nosmt',
                    'oversub_ratio': oversub_ratio,
                    'resource_constraints': constraints,
                },
            },
            'analysis': {
                'type': 'compare',
                'baseline': 'baseline',
            },
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 100.0,
                    'power_max_w': 400.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
            'cost': {
                'carbon_intensity_g_kwh': 400.0,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5.0,
            },
        }

    def test_memory_bottleneck(self, tmp_path):
        """Memory constraint limits effective R below requested R."""
        constraints = {
            'memory_gb': {
                'capacity_per_thread': 2.0,   # 48 * 2.0 = 96 GB per server
                'demand_per_vcpu': 4.0,     # max 24 vCPUs from memory
            },
        }
        config_data = self._make_config_with_constraints(constraints, oversub_ratio=2.0)
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        constrained = result.scenario_results['constrained']

        cr = constrained.get('resource_constraint_result')
        assert cr is not None
        assert cr['was_constrained'] is True
        assert cr['bottleneck_resource'] == 'memory_gb'
        # nosmt: 48 cores, 1 tpc, 48 hw_threads, available_pcpus=48
        # core_limit = 48 * 2.0 = 96
        # memory max_vcpus = 96 / 4.0 = 24
        # effective_vcpus = 24, effective_R = 24 / 48 = 0.5
        assert cr['effective_oversub_ratio'] == pytest.approx(0.5, abs=0.01)
        assert cr['effective_vcpus_per_server'] == pytest.approx(24.0, abs=0.1)
        assert cr['requested_oversub_ratio'] == pytest.approx(2.0)

        # Verify structured costs are correct (not defaults)
        # nosmt: 48 cores * 1 tpc = 48 hw_threads
        # embodied_carbon = 5.0 * 48 + 50.0 = 290 per server
        per_server_carbon = 5.0 * 48 + 50.0  # 290
        num_servers = constrained['num_servers']
        assert constrained['embodied_carbon_kg'] == pytest.approx(num_servers * per_server_carbon, rel=0.01)

    def test_no_binding_constraint(self, tmp_path):
        """Generous capacity means cores are the bottleneck."""
        constraints = {
            'memory_gb': {
                'capacity_per_thread': 100.0,   # 48 * 100 = 4800 GB
                'demand_per_vcpu': 4.0,       # max 1200 vCPUs
            },
            'ssd_gb': {
                'capacity_per_server': 100000,  # 100 TB
                'demand_per_vcpu': 20.0,        # max 5000 vCPUs
            },
        }
        config_data = self._make_config_with_constraints(constraints, oversub_ratio=2.0)
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        constrained = result.scenario_results['constrained']

        cr = constrained.get('resource_constraint_result')
        assert cr is not None
        assert cr['was_constrained'] is False
        assert cr['bottleneck_resource'] == 'cores'
        # effective R should equal requested R
        assert cr['effective_oversub_ratio'] == pytest.approx(2.0, abs=0.01)

    def test_multiple_constraints_tightest_wins(self, tmp_path):
        """With multiple constraints, the tightest resource wins."""
        constraints = {
            'memory_gb': {
                'capacity_per_thread': 2.0,     # 48 * 2 = 96 / 4 = 24 max vCPUs
                'demand_per_vcpu': 4.0,
            },
            'ssd_gb': {
                'capacity_per_server': 200,   # 200 / 20 = 10 max vCPUs (tightest!)
                'demand_per_vcpu': 20.0,
            },
        }
        config_data = self._make_config_with_constraints(constraints, oversub_ratio=3.0)
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        constrained = result.scenario_results['constrained']

        cr = constrained.get('resource_constraint_result')
        assert cr is not None
        assert cr['bottleneck_resource'] == 'ssd_gb'
        assert cr['effective_vcpus_per_server'] == pytest.approx(10.0, abs=0.1)
        # effective_R = 10 / 48 ~ 0.208
        assert cr['effective_oversub_ratio'] == pytest.approx(10.0 / 48.0, abs=0.01)

        # Verify resource details
        details = cr['resource_details']
        assert details['cores']['is_bottleneck'] is False
        assert details['memory_gb']['is_bottleneck'] is False
        assert details['ssd_gb']['is_bottleneck'] is True

        # ssd: 10 used / 10 max = 100% utilization
        assert details['ssd_gb']['utilization_pct'] == pytest.approx(100.0, abs=0.1)
        # memory: 10 used / 24 max = 41.7% utilization
        assert details['memory_gb']['utilization_pct'] == pytest.approx(10.0 / 24.0 * 100, abs=0.1)

    def test_compare_sweep_with_constraints(self, tmp_path):
        """Sweep oversub_ratio with constraints: carbon/TCO flatten after binding."""
        config_data = {
            'name': 'test_constraint_sweep',
            'scenarios': {
                'baseline': {
                    'processor': 'smt',
                    'oversub_ratio': 1.0,
                },
                'constrained': {
                    'processor': 'nosmt',
                    'oversub_ratio': 1.0,
                    'resource_constraints': {
                        'memory_gb': {
                            'capacity_per_thread': 2.0,   # 48 * 2 = 96 GB / 4 = 24 max vCPUs
                            'demand_per_vcpu': 4.0,
                        },
                    },
                },
            },
            'analysis': {
                'type': 'compare_sweep',
                'baseline': 'baseline',
                'sweep_scenario': 'constrained',
                'sweep_parameter': 'oversub_ratio',
                'sweep_values': [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
            },
            'processor': {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 100.0,
                    'power_max_w': 400.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
            'cost': {
                'carbon_intensity_g_kwh': 400.0,
                'electricity_cost_usd_kwh': 0.10,
                'lifetime_years': 5.0,
            },
        }
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'compare_sweep'
        sweep_results = result.compare_sweep_results
        assert len(sweep_results) == 6

        # At R=0.5, constraint shouldn't bind (core limit = 24, memory limit = 24)
        r05 = sweep_results[0]  # R=0.5
        r05_data = r05['scenarios']['constrained']
        # core limit at R=0.5 = 48*0.5 = 24, memory = 24, cores is bottleneck (or tied)
        assert r05_data.get('effective_oversub_ratio') is not None

        # At R=3.0 and R=4.0, effective R should be same (memory limits at 24 vCPUs)
        r30 = sweep_results[4]  # R=3.0
        r40 = sweep_results[5]  # R=4.0
        r30_data = r30['scenarios']['constrained']
        r40_data = r40['scenarios']['constrained']
        assert r30_data.get('was_constrained') is True
        assert r40_data.get('was_constrained') is True
        # Effective R should be same since memory limits both
        assert r30_data['effective_oversub_ratio'] == pytest.approx(
            r40_data['effective_oversub_ratio'], abs=0.01)
        # Carbon/TCO should also flatten
        assert r30_data['carbon_diff_pct'] == pytest.approx(
            r40_data['carbon_diff_pct'], abs=0.1)

        # Verify summary contains constraint columns
        assert 'Eff. R' in result.summary
        assert 'Bottleneck' in result.summary

    def test_structured_cost_verification(self, tmp_path):
        """Ensure per-server carbon matches structured calculation, not defaults."""
        constraints = {
            'memory_gb': {
                'capacity_per_thread': 2.0,
                'demand_per_vcpu': 4.0,
            },
        }
        config_data = self._make_config_with_constraints(constraints, oversub_ratio=2.0)
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)

        # Check baseline (smt) structured costs
        baseline = result.scenario_results['baseline']
        # smt: 48 cores * 2 tpc = 96 hw_threads
        # embodied = 5.0 * 96 + 50.0 = 530 per server
        smt_per_server = 5.0 * 96 + 50.0  # 530
        baseline_servers = baseline['num_servers']
        assert baseline['embodied_carbon_kg'] == pytest.approx(
            baseline_servers * smt_per_server, rel=0.01)
        # NOT the default 1000 kg
        assert baseline['embodied_carbon_kg'] != pytest.approx(
            baseline_servers * 1000, rel=0.1)

        # Check constrained (nosmt) structured costs
        constrained = result.scenario_results['constrained']
        nosmt_per_server = 5.0 * 48 + 50.0  # 290
        constrained_servers = constrained['num_servers']
        assert constrained['embodied_carbon_kg'] == pytest.approx(
            constrained_servers * nosmt_per_server, rel=0.01)
        assert constrained['embodied_carbon_kg'] != pytest.approx(
            constrained_servers * 1000, rel=0.1)

    def test_asdict_serialization(self):
        """ResourceConstraintResult survives asdict() roundtrip."""
        cr = ResourceConstraintResult(
            requested_oversub_ratio=2.0,
            effective_oversub_ratio=0.5,
            effective_vcpus_per_server=24.0,
            bottleneck_resource='memory_gb',
            resource_details={
                'cores': ResourceConstraintDetail(
                    max_vcpus=96.0, utilization_pct=25.0, stranded_pct=75.0, is_bottleneck=False,
                ),
                'memory_gb': ResourceConstraintDetail(
                    max_vcpus=24.0, utilization_pct=100.0, stranded_pct=0.0, is_bottleneck=True,
                ),
            },
            was_constrained=True,
        )
        d = asdict(cr)
        assert d['requested_oversub_ratio'] == 2.0
        assert d['effective_oversub_ratio'] == 0.5
        assert d['bottleneck_resource'] == 'memory_gb'
        assert d['resource_details']['cores']['max_vcpus'] == 96.0
        assert d['resource_details']['memory_gb']['is_bottleneck'] is True
        assert d['was_constrained'] is True

    def test_baseline_has_no_constraint_result(self, tmp_path):
        """Baseline without constraints has no resource_constraint_result."""
        constraints = {
            'memory_gb': {'capacity_per_thread': 2.0, 'demand_per_vcpu': 4.0},
        }
        config_data = self._make_config_with_constraints(constraints, oversub_ratio=2.0)
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        baseline = result.scenario_results['baseline']
        assert baseline.get('resource_constraint_result') is None


class TestResourcePacking:
    """Tests for resource_packing analysis type."""

    def _make_sweep_config(self, scenarios, sweep_values=None, resources=None,
                           include_cores=True, processors=None, labels=None):
        """Build a resource_packing sweep config."""
        if sweep_values is None:
            sweep_values = [1.0, 2.0]
        if processors is None:
            processors = {
                'smt': {
                    'physical_cores': 48,
                    'threads_per_core': 2,
                    'power_idle_w': 100.0,
                    'power_max_w': 400.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                },
            }

        analysis = {
            'type': 'resource_packing',
            'scenarios': list(scenarios.keys()),
            'sweep_parameter': 'oversub_ratio',
            'sweep_values': sweep_values,
            'include_cores': include_cores,
        }
        if resources:
            analysis['resources'] = resources
        if labels:
            analysis['labels'] = labels

        return {
            'name': 'test_resource_packing',
            'scenarios': scenarios,
            'analysis': analysis,
            'processor': processors,
            'workload': {'total_vcpus': 10000, 'avg_util': 0.3},
            'cost': {
                'carbon_intensity_g_kwh': 175,
                'electricity_cost_usd_kwh': 0.28,
                'lifetime_years': 6,
            },
        }

    def test_sweep_cores_only(self, tmp_path):
        """Direct sweep mode with only cores resource (include_cores=True, no named resources).

        Core capacity = available_pcpus (fixed hardware, constant regardless of R).
        Core demand = effective_vcpus / R (oversub discounts demand).
        """
        scenarios = {
            'nosmt_plain': {
                'processor': 'nosmt',
                'oversub_ratio': 1.0,
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[1.0, 2.0], include_cores=True
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'resource_packing'
        assert result.resource_packing_results is not None
        assert len(result.resource_packing_results) == 2

        # nosmt: 48 cores, 1 tpc, 0 overhead → 48 available_pcpus
        # At R=1.0: capacity = 48 (fixed hw), demand = 48*1/1 = 48, util = 100%
        pt1 = result.resource_packing_results[0]
        assert pt1['x_value'] == 1.0
        s1 = pt1['scenarios']['nosmt_plain']
        assert 'cores' in s1['resources']
        assert s1['resources']['cores']['capacity'] == pytest.approx(48.0)
        assert s1['resources']['cores']['demand'] == pytest.approx(48.0)

        # At R=2.0: capacity = 48 (same hw!), demand = 96/2 = 48, util = 100%
        pt2 = result.resource_packing_results[1]
        s2 = pt2['scenarios']['nosmt_plain']
        assert s2['resources']['cores']['capacity'] == pytest.approx(48.0)
        assert s2['resources']['cores']['demand'] == pytest.approx(48.0)

    def test_sweep_constrained_vs_scaled(self, tmp_path):
        """Direct sweep with constrained and scaled scenarios."""
        scenarios = {
            'nosmt_scaled': {
                'processor': 'nosmt',
                'oversub_ratio': 1.0,
                'resource_scaling': {
                    'scale_with_vcpus': ['memory'],
                },
            },
            'nosmt_constrained': {
                'processor': 'nosmt',
                'oversub_ratio': 1.0,
                'resource_constraints': {
                    'memory_gb': {
                        'capacity_per_thread': 4.0,
                        'demand_per_vcpu': 2.0,
                    },
                },
            },
        }
        resources = {
            'memory_gb': {
                'demand_per_vcpu': 2.0,
                'capacity_per_thread': 4.0,
                'label': 'Memory (GB)',
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[1.0, 2.0, 3.0], resources=resources,
            labels={
                'nosmt_scaled': 'Scaled',
                'nosmt_constrained': 'Constrained',
            }
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'resource_packing'
        assert len(result.resource_packing_results) == 3

        # At R=2.0:
        # nosmt: 48 cores, 1 tpc, 0 overhead → 48 hw_threads, 48 available_pcpus
        # Scaled: vcpus_per_server = max(48, 48*2) = 96
        #   memory capacity = 4.0 * 96 = 384 (scaled), demand = 2.0 * 96 = 192
        # Constrained: memory capacity = 4.0 * 48 = 192, max_vcpus = 192/2.0 = 96
        #   core_limit = 48*2 = 96, effective_vcpus = min(96, 96) = 96
        #   memory demand = 2.0 * 96 = 192
        pt2 = result.resource_packing_results[1]
        assert pt2['x_value'] == 2.0

        scaled = pt2['scenarios']['Scaled']
        assert scaled['resources']['memory_gb']['capacity'] == pytest.approx(384.0)
        assert scaled['resources']['memory_gb']['demand'] == pytest.approx(192.0)

        constrained = pt2['scenarios']['Constrained']
        assert constrained['resources']['memory_gb']['capacity'] == pytest.approx(192.0)
        assert constrained['resources']['memory_gb']['demand'] == pytest.approx(192.0)

        # At R=3.0: constrained should hit memory bottleneck
        # core_limit = 48*3 = 144, memory max_vcpus = 192/2.0 = 96
        # effective_vcpus = min(144, 96) = 96 → memory is bottleneck
        # Core capacity = 48 (fixed hw), demand = 96/3 = 32 (stranded!)
        pt3 = result.resource_packing_results[2]
        constrained_3 = pt3['scenarios']['Constrained']
        assert constrained_3['was_constrained'] is True
        assert constrained_3['bottleneck_resource'] == 'memory_gb'
        assert constrained_3['resources']['cores']['capacity'] == pytest.approx(48.0)
        assert constrained_3['resources']['cores']['demand'] == pytest.approx(32.0)
        assert constrained_3['resources']['cores']['stranded_pct'] == pytest.approx(100.0 - 32.0/48.0 * 100.0)

    def test_sweep_structured_costs(self, tmp_path):
        """Verify structured per-processor costs are used (not defaults)."""
        # Use processors with distinctive embodied_carbon values
        processors = {
            'custom': {
                'physical_cores': 32,
                'threads_per_core': 2,
                'power_idle_w': 80.0,
                'power_max_w': 300.0,
                'thread_overhead': 4,
                'embodied_carbon': {
                    'per_thread': {'cpu_die': 7.0},
                    'per_server': {'chassis': 100.0},
                },
                'server_cost': {
                    'per_thread': {'cpu': 60.0},
                    'per_server': {'base': 800.0},
                },
            },
        }
        scenarios = {
            'test_scenario': {
                'processor': 'custom',
                'oversub_ratio': 1.0,
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[1.0], processors=processors
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'resource_packing'

        # Verify packing data is present and correct
        pt = result.resource_packing_results[0]
        s = pt['scenarios']['test_scenario']
        # 32 cores * 2 tpc = 64 hw_threads, 64 - 4 = 60 available_pcpus
        assert s['hw_threads'] == 64
        assert s['available_pcpus'] == 60
        assert s['resources']['cores']['capacity'] == pytest.approx(60.0)

    def test_config_derived_mode(self, tmp_path):
        """Config-derived mode: load sub-configs and extract packing data."""
        # Create two sub-configs with different workload.avg_util
        for util, R in [(0.1, 2.0), (0.3, 1.0)]:
            sub_config = {
                'name': f'sub_{int(util*100)}',
                'scenarios': {
                    'test_scenario': {
                        'processor': 'proc_a',
                        'oversub_ratio': R,
                    },
                },
                'workload': {'total_vcpus': 10000, 'avg_util': util},
                'analysis': {
                    'type': 'compare_sweep',
                    'baseline': 'test_scenario',
                    'sweep_scenario': 'test_scenario',
                    'sweep_parameter': 'oversub_ratio',
                    'sweep_values': [1.0],
                },
                'processor': {
                    'proc_a': {
                        'physical_cores': 48,
                        'threads_per_core': 1,
                        'power_idle_w': 90.0,
                        'power_max_w': 340.0,
                        'thread_overhead': 0,
                        'embodied_carbon': {
                            'per_thread': {'cpu_die': 5.0},
                            'per_server': {'chassis': 50.0},
                        },
                        'server_cost': {
                            'per_thread': {'cpu': 50.0},
                            'per_server': {'base': 500.0},
                        },
                    },
                },
                'cost': {
                    'carbon_intensity_g_kwh': 175,
                    'electricity_cost_usd_kwh': 0.28,
                    'lifetime_years': 6,
                },
            }
            sub_file = tmp_path / f'sub_{int(util*100)}.json'
            with open(sub_file, 'w') as f:
                json.dump(sub_config, f)

        # Create the resource_packing config referencing sub-configs
        main_config = {
            'name': 'test_config_derived',
            'analysis': {
                'type': 'resource_packing',
                'config_sets': [
                    {
                        'label': 'Test Scenario',
                        'configs': [
                            str(tmp_path / 'sub_10.json'),
                            str(tmp_path / 'sub_30.json'),
                        ],
                        'scenario': 'test_scenario',
                    },
                ],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'x_label': 'Utilization (%)',
                'include_cores': True,
            },
        }
        config_file = tmp_path / 'main.json'
        with open(config_file, 'w') as f:
            json.dump(main_config, f)

        result = run_analysis(config_file)
        assert result.analysis_type == 'resource_packing'
        assert len(result.resource_packing_results) == 2

        # First point: 10% util, R=2.0
        # Core capacity = available_pcpus = 48 (constant, fixed hardware)
        # Core demand = core_limit / R = 96 / 2.0 = 48
        pt1 = result.resource_packing_results[0]
        assert pt1['x_value'] == pytest.approx(10.0)
        s1 = pt1['scenarios']['Test Scenario']
        assert s1['oversub_ratio'] == pytest.approx(2.0)
        assert s1['resources']['cores']['capacity'] == pytest.approx(48.0)
        assert s1['resources']['cores']['demand'] == pytest.approx(48.0)

        # Second point: 30% util, R=1.0
        # Core capacity = 48 (same hw!), demand = 48/1 = 48
        pt2 = result.resource_packing_results[1]
        assert pt2['x_value'] == pytest.approx(30.0)
        s2 = pt2['scenarios']['Test Scenario']
        assert s2['oversub_ratio'] == pytest.approx(1.0)
        assert s2['resources']['cores']['capacity'] == pytest.approx(48.0)
        assert s2['resources']['cores']['demand'] == pytest.approx(48.0)

    def test_serialization(self, tmp_path):
        """resource_packing_results included in to_dict() serialization."""
        scenarios = {
            'nosmt_plain': {
                'processor': 'nosmt',
                'oversub_ratio': 1.0,
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[1.0]
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        d = result.to_dict()
        assert 'resource_packing_results' in d
        assert len(d['resource_packing_results']) == 1
        assert 'scenarios' in d['resource_packing_results'][0]

    def test_include_cores_false(self, tmp_path):
        """When include_cores=False, cores resource is not included."""
        scenarios = {
            'nosmt_constrained': {
                'processor': 'nosmt',
                'oversub_ratio': 2.0,
                'resource_constraints': {
                    'memory_gb': {
                        'capacity_per_thread': 4.0,
                        'demand_per_vcpu': 2.0,
                    },
                },
            },
        }
        resources = {
            'memory_gb': {
                'demand_per_vcpu': 2.0,
                'capacity_per_thread': 4.0,
                'label': 'Memory (GB)',
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[2.0], resources=resources,
            include_cores=False
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        pt = result.resource_packing_results[0]
        s = pt['scenarios']['nosmt_constrained']
        assert 'cores' not in s['resources']
        assert 'memory_gb' in s['resources']

    def test_cores_capacity_constant_across_oversub(self, tmp_path):
        """Core capacity is fixed hardware (available_pcpus), constant regardless of R.

        Core demand = effective_vcpus / R. For unconstrained, demand = available_pcpus
        (since effective_vcpus = available_pcpus * R, and demand = (avail * R) / R = avail).
        """
        scenarios = {
            'nosmt_plain': {
                'processor': 'nosmt',
                'oversub_ratio': 1.0,
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[1.0, 2.0, 5.0], include_cores=True
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        # nosmt: 48 cores, 1 tpc, 0 overhead → 48 available_pcpus
        for pt in result.resource_packing_results:
            s = pt['scenarios']['nosmt_plain']
            # Capacity always = 48 (fixed hardware)
            assert s['resources']['cores']['capacity'] == pytest.approx(48.0)
            # Demand always = 48 (effective_vcpus / R = avail * R / R = avail)
            assert s['resources']['cores']['demand'] == pytest.approx(48.0)
            assert s['resources']['cores']['utilization_pct'] == pytest.approx(100.0)
            assert s['resources']['cores']['stranded_pct'] == pytest.approx(0.0)

    def test_cores_stranding_from_resource_constraint(self, tmp_path):
        """When a non-core resource limits packing, cores show stranding."""
        scenarios = {
            'constrained': {
                'processor': 'nosmt',
                'oversub_ratio': 1.0,
                'resource_constraints': {
                    'memory_gb': {
                        'capacity_per_thread': 4.0,  # 4.0 * 48 = 192 GB
                        'demand_per_vcpu': 2.0,       # 192/2 = 96 max vCPUs
                    },
                },
            },
        }
        resources = {
            'memory_gb': {
                'demand_per_vcpu': 2.0,
                'capacity_per_thread': 4.0,
                'label': 'Memory (GB)',
            },
        }
        # At R=4.0: core_limit = 48*4 = 192, memory max_vcpus = 96
        # effective_vcpus = min(192, 96) = 96, memory is bottleneck
        # Core capacity = 48, demand = 96/4 = 24, util = 50%
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[4.0], resources=resources
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        pt = result.resource_packing_results[0]
        s = pt['scenarios']['constrained']

        assert s['was_constrained'] is True
        assert s['bottleneck_resource'] == 'memory_gb'
        assert s['resources']['cores']['capacity'] == pytest.approx(48.0)
        assert s['resources']['cores']['demand'] == pytest.approx(24.0)  # 96/4
        assert s['resources']['cores']['utilization_pct'] == pytest.approx(50.0)
        assert s['resources']['cores']['stranded_pct'] == pytest.approx(50.0)

    def test_proc_capacity_fallback(self, tmp_path):
        """When resource_defs lack capacity_per_thread, falls back to processor capacity."""
        processors = {
            'with_capacity': {
                'physical_cores': 48,
                'threads_per_core': 1,
                'power_idle_w': 90.0,
                'power_max_w': 340.0,
                'thread_overhead': 0,
                'embodied_carbon': {
                    'per_thread': {'cpu_die': 5.0},
                    'per_server': {'chassis': 50.0},
                },
                'server_cost': {
                    'per_thread': {'cpu': 50.0},
                    'per_server': {'base': 500.0},
                },
                # Capacity field with per_thread breakdown
                'capacity': {
                    'per_thread': {
                        'memory': 4.8,  # matches memory_gb via name stripping
                        'ssd': 75.0,    # matches ssd_gb via name stripping
                    },
                },
            },
        }
        scenarios = {
            'scaled': {
                'processor': 'with_capacity',
                'oversub_ratio': 1.0,
                'resource_scaling': {
                    'scale_with_vcpus': ['memory_gb', 'ssd_gb'],
                },
            },
        }
        # resource_defs intentionally OMIT capacity_per_thread to test fallback
        resources = {
            'memory_gb': {
                'demand_per_vcpu': 4.0,
                'label': 'Memory (GB)',
            },
            'ssd_gb': {
                'demand_per_vcpu': 50.0,
                'label': 'SSD (GB)',
            },
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[2.0], resources=resources,
            processors=processors
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        pt = result.resource_packing_results[0]
        s = pt['scenarios']['scaled']

        # At R=2.0: vcpus_per_server = max(48, 48*2) = 96
        # Memory: capacity = 4.8 * 96 = 460.8, demand = 4.0 * 96 = 384
        # Stranded = 1 - 384/460.8 = 16.7%
        assert s['resources']['memory_gb']['capacity'] == pytest.approx(460.8)
        assert s['resources']['memory_gb']['demand'] == pytest.approx(384.0)
        assert s['resources']['memory_gb']['stranded_pct'] == pytest.approx(
            100.0 - 384.0 / 460.8 * 100.0, abs=0.1
        )

        # SSD: capacity = 75.0 * 96 = 7200, demand = 50.0 * 96 = 4800
        # Stranded = 1 - 4800/7200 = 33.3%
        assert s['resources']['ssd_gb']['capacity'] == pytest.approx(7200.0)
        assert s['resources']['ssd_gb']['demand'] == pytest.approx(4800.0)
        assert s['resources']['ssd_gb']['stranded_pct'] == pytest.approx(
            100.0 - 4800.0 / 7200.0 * 100.0, abs=0.1
        )

    def test_bottleneck_resource_detection(self, tmp_path):
        """Bottleneck resource is detected for all scenario types."""
        processors = {
            'proc': {
                'physical_cores': 48,
                'threads_per_core': 1,
                'power_idle_w': 90.0,
                'power_max_w': 340.0,
                'thread_overhead': 0,
                'embodied_carbon': {
                    'per_thread': {'cpu_die': 5.0},
                    'per_server': {'chassis': 50.0},
                },
                'server_cost': {
                    'per_thread': {'cpu': 50.0},
                    'per_server': {'base': 500.0},
                },
                'capacity': {
                    'per_thread': {'memory': 4.8, 'ssd': 75.0},
                },
            },
        }
        scenarios = {
            'scaled': {
                'processor': 'proc',
                'oversub_ratio': 1.0,
                'resource_scaling': {
                    'scale_with_vcpus': ['memory_gb', 'ssd_gb'],
                },
            },
            'constrained': {
                'processor': 'proc',
                'oversub_ratio': 1.0,
                'resource_constraints': {
                    'memory_gb': {
                        'capacity_per_thread': 4.8,
                        'demand_per_vcpu': 4.0,
                    },
                    'ssd_gb': {
                        'capacity_per_thread': 75.0,
                        'demand_per_vcpu': 50.0,
                    },
                },
            },
        }
        resources = {
            'memory_gb': {'demand_per_vcpu': 4.0, 'label': 'Memory'},
            'ssd_gb': {'demand_per_vcpu': 50.0, 'label': 'SSD'},
        }
        config_data = self._make_sweep_config(
            scenarios, sweep_values=[3.0], resources=resources,
            processors=processors
        )
        config_file = tmp_path / 'test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_analysis(config_file)
        pt = result.resource_packing_results[0]

        # Scaled at R=3.0: cores 100%, memory 83%, ssd 67% → bottleneck = cores
        scaled = pt['scenarios']['scaled']
        assert scaled['bottleneck_resource'] == 'cores'

        # Constrained at R=3.0: core_limit=144, memory max=57.6, ssd max=72
        # effective_vcpus = 57.6 → memory is bottleneck
        constrained = pt['scenarios']['constrained']
        assert constrained['bottleneck_resource'] == 'memory_gb'
        assert constrained['was_constrained'] is True

    def test_scenario_overrides_in_config_derived(self, tmp_path):
        """Config-derived mode with scenario_overrides: override resource_scaling to constraints."""
        # Create a sub-config with resource_scaling
        sub_config = {
            'name': 'sub',
            'scenarios': {
                'scenario_a': {
                    'processor': 'proc',
                    'oversub_ratio': 3.0,
                    'resource_scaling': {
                        'scale_with_vcpus': ['memory_gb'],
                    },
                },
            },
            'workload': {'total_vcpus': 10000, 'avg_util': 0.2},
            'analysis': {
                'type': 'compare_sweep',
                'baseline': 'scenario_a',
                'sweep_scenario': 'scenario_a',
                'sweep_parameter': 'oversub_ratio',
                'sweep_values': [1.0],
            },
            'processor': {
                'proc': {
                    'physical_cores': 48,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                    'thread_overhead': 0,
                    'embodied_carbon': {
                        'per_thread': {'cpu_die': 5.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_thread': {'cpu': 50.0},
                        'per_server': {'base': 500.0},
                    },
                    'capacity': {
                        'per_thread': {'memory': 4.8},
                    },
                },
            },
            'cost': {
                'carbon_intensity_g_kwh': 175,
                'electricity_cost_usd_kwh': 0.28,
                'lifetime_years': 6,
            },
        }
        sub_file = tmp_path / 'sub.json'
        with open(sub_file, 'w') as f:
            json.dump(sub_config, f)

        # Create resource_packing config with scenario_overrides
        main_config = {
            'name': 'test_overrides',
            'analysis': {
                'type': 'resource_packing',
                'config_sets': [
                    {
                        'label': 'Scaled',
                        'configs': [str(sub_file)],
                        'scenario': 'scenario_a',
                    },
                    {
                        'label': 'Constrained (override)',
                        'configs': [str(sub_file)],
                        'scenario': 'scenario_a',
                        'scenario_overrides': {
                            'resource_scaling': None,
                            'resource_constraints': {
                                'memory_gb': {
                                    'capacity_per_thread': 4.8,
                                    'demand_per_vcpu': 4.0,
                                },
                            },
                        },
                    },
                ],
                'x_parameter': 'workload.avg_util',
                'x_display_multiplier': 100,
                'resources': {
                    'memory_gb': {
                        'demand_per_vcpu': 4.0,
                        'label': 'Memory (GB)',
                    },
                },
                'include_cores': True,
            },
        }
        config_file = tmp_path / 'main.json'
        with open(config_file, 'w') as f:
            json.dump(main_config, f)

        result = run_analysis(config_file)
        assert len(result.resource_packing_results) == 1
        pt = result.resource_packing_results[0]

        # Scaled: uses original resource_scaling from sub-config
        scaled = pt['scenarios']['Scaled']
        assert scaled['was_constrained'] is False
        # vcpus_per_server = max(48, 48*3) = 144
        # Memory capacity = 4.8 * 144 = 691.2 (scaled with vcpus)
        assert scaled['resources']['memory_gb']['capacity'] == pytest.approx(691.2)

        # Constrained (override): resource_scaling removed, resource_constraints added
        constrained = pt['scenarios']['Constrained (override)']
        # memory: 4.8 * 48 = 230.4 GB, max_vcpus = 230.4/4.0 = 57.6
        # core_limit = 48 * 3 = 144, effective_vcpus = min(144, 57.6) = 57.6
        assert constrained['was_constrained'] is True
        assert constrained['bottleneck_resource'] == 'memory_gb'
        assert constrained['resources']['memory_gb']['capacity'] == pytest.approx(230.4)
