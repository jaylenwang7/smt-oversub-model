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
    DeclarativeAnalysisEngine,
    run_analysis,
    PowerComponentSpec,
    ProcessorSpec,
)
from .model import (
    PowerCurve, ProcessorConfig, ScenarioParams,
    WorkloadParams, CostParams, OverssubModel, ScenarioResult,
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
            'per_core': {'cpu_die': 10.0, 'dram': 2.0},
            'per_server': {'chassis': 100.0},
        }
        spec = EmbodiedComponentSpec.from_dict(data)
        assert spec.per_core == {'cpu_die': 10.0, 'dram': 2.0}
        assert spec.per_server == {'chassis': 100.0}

    def test_resolve_total(self):
        """resolve_total should compute flat per-server value."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_core={'cpu_die': 10.0},
            per_server={'chassis': 20.0},
        )
        # 48 cores * 2 threads = 96 hw threads
        # 10 * 96 + 20 = 980
        assert spec.resolve_total(48, 2) == 980.0

    def test_resolve_total_nosmt(self):
        """resolve_total with threads_per_core=1."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_core={'cpu_die': 10.0},
            per_server={'chassis': 50.0},
        )
        # 52 cores * 1 thread = 52 hw threads
        # 10 * 52 + 50 = 570
        assert spec.resolve_total(52, 1) == 570.0

    def test_to_component_breakdown(self):
        """to_component_breakdown should produce a resolved ComponentBreakdown."""
        from .declarative import EmbodiedComponentSpec

        spec = EmbodiedComponentSpec(
            per_core={'cpu_die': 10.0},
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
            per_core={'cpu_die': 10.0},
            per_server={'chassis': 20.0},
        )
        d = spec.to_dict()
        assert d == {'per_core': {'cpu_die': 10.0}, 'per_server': {'chassis': 20.0}}

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
                'per_core': {'cpu_die': 10.0},
                'per_server': {'chassis': 20.0},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.embodied_carbon is not None
        assert spec.embodied_carbon.per_core == {'cpu_die': 10.0}
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
                per_core={'cpu_die': 10.0},
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
                'per_core': {'cpu': 73.0},
                'per_server': {'base': 32.0},
            },
        }
        spec = ProcessorSpec.from_dict(data)
        assert spec.server_cost is not None
        assert spec.server_cost.per_core == {'cpu': 73.0}
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
                    per_core={'cpu_die': 5.0},
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
                    per_core={'cpu_die': 8.0},
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
                    per_core={'cpu_die': 10.0},
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
                        'per_core': {'cpu_die': 10.0},
                        'per_server': {'chassis': 20.0},
                    },
                    'server_cost': {
                        'per_core': {'cpu': 73.0},
                        'per_server': {'base': 32.0},
                    },
                },
                'nosmt': {
                    'physical_cores': 52,
                    'threads_per_core': 1,
                    'power_idle_w': 90.0,
                    'power_max_w': 340.0,
                    'embodied_carbon': {
                        'per_core': {'cpu_die': 10.0},
                        'per_server': {'chassis': 50.0},
                    },
                    'server_cost': {
                        'per_core': {'cpu': 72.0},
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
