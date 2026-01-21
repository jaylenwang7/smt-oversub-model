"""
Unit tests for SMT vs Non-SMT oversubscription model.

Run with: pytest test_model.py -v
"""

import pytest
import math
from .model import (
    PowerCurve, ProcessorConfig, ScenarioParams, 
    WorkloadParams, CostParams, OverssubModel, ScenarioResult
)
from .sweep import ParameterSweeper, create_default_sweeper


class TestPowerCurve:
    """Tests for PowerCurve class."""
    
    def test_power_at_zero_util(self):
        """Power at 0% util should equal idle power."""
        curve = PowerCurve(p_idle=100, p_max=300)
        assert curve.power_at_util(0.0) == 100.0
    
    def test_power_at_full_util(self):
        """Power at 100% util should equal max power."""
        curve = PowerCurve(p_idle=100, p_max=300)
        assert curve.power_at_util(1.0) == 300.0
    
    def test_power_at_half_util_linear(self):
        """Linear curve at 50% util should be midpoint."""
        curve = PowerCurve(p_idle=100, p_max=300)
        assert curve.power_at_util(0.5) == 200.0
    
    def test_power_clamps_util(self):
        """Utilization should be clamped to [0, 1]."""
        curve = PowerCurve(p_idle=100, p_max=300)
        assert curve.power_at_util(-0.5) == 100.0
        assert curve.power_at_util(1.5) == 300.0
    
    def test_nonlinear_curve(self):
        """Non-linear curve should apply function correctly."""
        # Square curve: f(u) = u^2
        curve = PowerCurve(p_idle=100, p_max=300, curve_fn=lambda u: u**2)
        # At 50%: 100 + (300-100) * 0.25 = 150
        assert curve.power_at_util(0.5) == 150.0
    
    def test_scaled_curve(self):
        """Scaled curve should multiply idle and max correctly."""
        curve = PowerCurve(p_idle=100, p_max=300)
        scaled = curve.scaled(idle_ratio=0.8, max_ratio=0.9)
        assert scaled.p_idle == 80.0
        assert scaled.p_max == 270.0


class TestProcessorConfig:
    """Tests for ProcessorConfig class."""
    
    def test_smt_pcpu_count(self):
        """SMT processor should have 2x physical cores as pCPUs."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power)
        assert proc.pcpus == 128
    
    def test_nosmt_pcpu_count(self):
        """Non-SMT processor should have same pCPUs as physical cores."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=48, threads_per_core=1, power_curve=power)
        assert proc.pcpus == 48


class TestOverssubModel:
    """Tests for OverssubModel class."""
    
    @pytest.fixture
    def simple_setup(self):
        """Create a simple test setup."""
        power = PowerCurve(p_idle=100, p_max=300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power)
        workload = WorkloadParams(total_vcpus=1000, avg_util=0.3)
        cost = CostParams(
            embodied_carbon_kg=1000,
            server_cost_usd=15000,
            carbon_intensity_g_kwh=400,
            electricity_cost_usd_kwh=0.10,
            lifetime_hours=4 * 8760,
        )
        return proc, workload, cost
    
    def test_server_count_no_oversub(self, simple_setup):
        """Server count without oversub should match vCPU/pCPU."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        scenario = ScenarioParams(proc, oversub_ratio=1.0)
        result = model.evaluate_scenario(scenario)
        
        # 1000 vCPUs / 128 pCPUs = 7.8125 -> 8 servers
        assert result.num_servers == 8
    
    def test_server_count_with_oversub(self, simple_setup):
        """Server count with oversub should decrease."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        scenario = ScenarioParams(proc, oversub_ratio=2.0)
        result = model.evaluate_scenario(scenario)
        
        # 1000 vCPUs / (128 * 2) = 3.9 -> 4 servers
        assert result.num_servers == 4
    
    def test_utilization_increases_with_oversub(self, simple_setup):
        """Average utilization should increase with oversubscription."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        
        no_oversub = ScenarioParams(proc, oversub_ratio=1.0)
        with_oversub = ScenarioParams(proc, oversub_ratio=2.0)
        
        r1 = model.evaluate_scenario(no_oversub)
        r2 = model.evaluate_scenario(with_oversub)
        
        assert r2.avg_util_per_server > r1.avg_util_per_server
    
    def test_util_overhead_applied(self, simple_setup):
        """Utilization overhead should be added to effective util."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        
        no_overhead = ScenarioParams(proc, oversub_ratio=1.0, util_overhead=0.0)
        with_overhead = ScenarioParams(proc, oversub_ratio=1.0, util_overhead=0.1)
        
        r1 = model.evaluate_scenario(no_overhead)
        r2 = model.evaluate_scenario(with_overhead)
        
        assert r2.effective_util_per_server == pytest.approx(
            r1.effective_util_per_server + 0.1, rel=0.01
        )
    
    def test_effective_util_capped_at_one(self, simple_setup):
        """Effective utilization should not exceed 1.0."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        
        # High overhead that would push util over 1.0
        scenario = ScenarioParams(proc, oversub_ratio=1.0, util_overhead=0.9)
        result = model.evaluate_scenario(scenario)
        
        assert result.effective_util_per_server <= 1.0
    
    def test_embodied_carbon_scales_with_servers(self, simple_setup):
        """Embodied carbon should scale linearly with server count."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        
        no_oversub = ScenarioParams(proc, oversub_ratio=1.0)
        with_oversub = ScenarioParams(proc, oversub_ratio=2.0)
        
        r1 = model.evaluate_scenario(no_oversub)
        r2 = model.evaluate_scenario(with_oversub)
        
        assert r1.embodied_carbon_kg == r1.num_servers * cost.embodied_carbon_kg
        assert r2.embodied_carbon_kg == r2.num_servers * cost.embodied_carbon_kg
    
    def test_operational_carbon_calculation(self, simple_setup):
        """Operational carbon should be calculated correctly."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        
        scenario = ScenarioParams(proc, oversub_ratio=1.0)
        result = model.evaluate_scenario(scenario)
        
        # Manual calculation
        expected_energy_kwh = (
            result.num_servers * result.power_per_server_w * 
            cost.lifetime_hours / 1000
        )
        expected_carbon = expected_energy_kwh * cost.carbon_intensity_g_kwh / 1000
        
        assert result.operational_carbon_kg == pytest.approx(expected_carbon, rel=0.01)
    
    def test_total_carbon_is_sum(self, simple_setup):
        """Total carbon should be embodied + operational."""
        proc, workload, cost = simple_setup
        model = OverssubModel(workload, cost)
        
        scenario = ScenarioParams(proc, oversub_ratio=1.0)
        result = model.evaluate_scenario(scenario)
        
        assert result.total_carbon_kg == pytest.approx(
            result.embodied_carbon_kg + result.operational_carbon_kg, rel=0.01
        )


class TestBreakevenSearch:
    """Tests for breakeven oversubscription search."""
    
    @pytest.fixture
    def model_setup(self):
        """Create model for breakeven tests."""
        smt_power = PowerCurve(100, 300)
        nosmt_power = PowerCurve(85, 255)  # 85% of SMT
        
        smt_proc = ProcessorConfig(64, 2, smt_power)
        nosmt_proc = ProcessorConfig(48, 1, nosmt_power)
        
        workload = WorkloadParams(10000, 0.3)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)
        
        return smt_proc, nosmt_proc, workload, cost
    
    def test_breakeven_exists(self, model_setup):
        """Breakeven should exist when non-SMT can achieve enough oversub."""
        smt_proc, nosmt_proc, workload, cost = model_setup
        model = OverssubModel(workload, cost)
        
        # SMT with moderate oversub
        smt_scenario = ScenarioParams(smt_proc, 1.3, 0.05)
        smt_result = model.evaluate_scenario(smt_scenario)
        
        breakeven = model.find_breakeven_oversub(
            smt_result, nosmt_proc, 0.0, metric='carbon'
        )
        
        assert breakeven is not None
        assert breakeven > 1.0
    
    def test_breakeven_achieves_target(self, model_setup):
        """At breakeven, carbon should match target.

        Note: Due to discrete server counts (ceiling function), the binary search
        can only approximate the target. We use 2% tolerance to account for step
        changes in carbon when server count changes by 1.
        """
        smt_proc, nosmt_proc, workload, cost = model_setup
        model = OverssubModel(workload, cost)

        smt_scenario = ScenarioParams(smt_proc, 1.3, 0.05)
        smt_result = model.evaluate_scenario(smt_scenario)

        breakeven = model.find_breakeven_oversub(
            smt_result, nosmt_proc, 0.0, metric='carbon'
        )

        if breakeven:
            nosmt_scenario = ScenarioParams(nosmt_proc, breakeven, 0.0)
            nosmt_result = model.evaluate_scenario(nosmt_scenario)

            assert nosmt_result.total_carbon_kg == pytest.approx(
                smt_result.total_carbon_kg, rel=0.02
            )


class TestParameterSweeper:
    """Tests for parameter sweep functionality."""
    
    def test_default_sweeper_creates(self):
        """Default sweeper should initialize without error."""
        sweeper = create_default_sweeper()
        assert sweeper is not None
    
    def test_compute_breakeven_returns_dict(self):
        """compute_breakeven should return dict with expected keys."""
        sweeper = create_default_sweeper()
        result = sweeper.compute_breakeven()
        
        assert 'breakeven_oversub_carbon' in result
        assert 'breakeven_oversub_tco' in result
        assert 'baseline' in result
        assert 'smt_oversub' in result
    
    def test_sweep_parameter_returns_results(self):
        """Sweeping a parameter should return results for each value."""
        sweeper = create_default_sweeper()
        values = [200, 400, 600]
        
        result = sweeper.sweep_parameter('carbon_intensity_g_kwh', values)
        
        assert len(result.param_values) == 3
        assert len(result.breakeven_oversub_carbon) == 3
    
    def test_sweep_restores_original_value(self):
        """After sweep, original parameter value should be restored."""
        sweeper = create_default_sweeper(carbon_intensity_g_kwh=400)
        
        sweeper.sweep_parameter('carbon_intensity_g_kwh', [200, 600, 800])
        
        assert sweeper.carbon_intensity_g_kwh == 400
    
    def test_higher_carbon_intensity_favors_embodied(self):
        """Higher carbon intensity should increase importance of embodied savings."""
        sweeper = create_default_sweeper()
        
        # At low carbon intensity, operational carbon is lower
        sweeper.carbon_intensity_g_kwh = 100
        low_result = sweeper.compute_breakeven()
        
        sweeper.carbon_intensity_g_kwh = 800
        high_result = sweeper.compute_breakeven()
        
        # Higher intensity means operational is larger fraction
        # So embodied savings matter relatively less
        # Breakeven oversub should change accordingly
        assert low_result['breakeven_oversub_carbon'] != high_result['breakeven_oversub_carbon']


class TestCoreOverhead:
    """Tests for core overhead functionality."""

    def test_available_pcpus_no_overhead(self):
        """With no overhead, available_pcpus equals pcpus."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power)
        assert proc.available_pcpus == proc.pcpus == 128

    def test_available_pcpus_with_overhead(self):
        """With overhead, available_pcpus is reduced."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power, core_overhead=8)
        assert proc.pcpus == 128
        assert proc.available_pcpus == 120

    def test_available_pcpus_nosmt_with_overhead(self):
        """Non-SMT processor with overhead."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=48, threads_per_core=1, power_curve=power, core_overhead=4)
        assert proc.pcpus == 48
        assert proc.available_pcpus == 44

    def test_available_pcpus_clamps_to_zero(self):
        """available_pcpus should not go negative."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=8, threads_per_core=1, power_curve=power, core_overhead=100)
        assert proc.available_pcpus == 0

    def test_server_count_with_overhead(self):
        """Server count should increase when cores are reserved for host."""
        power = PowerCurve(100, 300)
        # 64 physical * 2 threads = 128 pCPUs, minus 8 overhead = 120 available
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power, core_overhead=8)
        workload = WorkloadParams(total_vcpus=1000, avg_util=0.3)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)

        model = OverssubModel(workload, cost)
        scenario = ScenarioParams(proc, oversub_ratio=1.0)
        result = model.evaluate_scenario(scenario)

        # 1000 vCPUs / 120 available pCPUs = 8.33 -> 9 servers
        assert result.num_servers == 9

    def test_server_count_without_overhead_for_comparison(self):
        """Baseline: same config without overhead needs fewer servers."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(physical_cores=64, threads_per_core=2, power_curve=power, core_overhead=0)
        workload = WorkloadParams(total_vcpus=1000, avg_util=0.3)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)

        model = OverssubModel(workload, cost)
        scenario = ScenarioParams(proc, oversub_ratio=1.0)
        result = model.evaluate_scenario(scenario)

        # 1000 vCPUs / 128 pCPUs = 7.8125 -> 8 servers
        assert result.num_servers == 8

    def test_overhead_affects_utilization_calculation(self):
        """Utilization should be calculated based on available pCPUs."""
        power = PowerCurve(100, 300)
        proc_no_overhead = ProcessorConfig(64, 2, power, core_overhead=0)
        proc_with_overhead = ProcessorConfig(64, 2, power, core_overhead=8)

        # Use exact vCPU count to avoid rounding differences
        workload = WorkloadParams(total_vcpus=120, avg_util=0.5)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)
        model = OverssubModel(workload, cost)

        # With overhead: 120 vCPUs / 120 available = 1 server
        scenario_with = ScenarioParams(proc_with_overhead, 1.0)
        result_with = model.evaluate_scenario(scenario_with)

        # Without overhead: 120 vCPUs / 128 available = 1 server
        scenario_without = ScenarioParams(proc_no_overhead, 1.0)
        result_without = model.evaluate_scenario(scenario_without)

        # Both need 1 server, but utilization differs
        assert result_with.num_servers == 1
        assert result_without.num_servers == 1

        # With overhead: 120 * 0.5 / 120 = 0.5
        # Without overhead: 120 * 0.5 / 128 = 0.46875
        assert result_with.avg_util_per_server > result_without.avg_util_per_server


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_exact_fit_no_rounding(self):
        """When vCPUs exactly fit, should not over-provision."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(32, 2, power)  # 64 pCPUs
        workload = WorkloadParams(total_vcpus=64, avg_util=0.5)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)
        
        model = OverssubModel(workload, cost)
        scenario = ScenarioParams(proc, 1.0)
        result = model.evaluate_scenario(scenario)
        
        assert result.num_servers == 1
    
    def test_one_vcpu_over_needs_new_server(self):
        """One vCPU over capacity should add another server."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(32, 2, power)  # 64 pCPUs
        workload = WorkloadParams(total_vcpus=65, avg_util=0.5)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)
        
        model = OverssubModel(workload, cost)
        scenario = ScenarioParams(proc, 1.0)
        result = model.evaluate_scenario(scenario)
        
        assert result.num_servers == 2
    
    def test_very_high_oversub_reduces_servers(self):
        """Very high oversub should significantly reduce servers."""
        power = PowerCurve(100, 300)
        proc = ProcessorConfig(64, 2, power)
        workload = WorkloadParams(total_vcpus=10000, avg_util=0.1)
        cost = CostParams(1000, 15000, 400, 0.10, 4 * 8760)
        
        model = OverssubModel(workload, cost)
        
        low_oversub = ScenarioParams(proc, 1.0)
        high_oversub = ScenarioParams(proc, 5.0)
        
        r1 = model.evaluate_scenario(low_oversub)
        r2 = model.evaluate_scenario(high_oversub)
        
        assert r2.num_servers < r1.num_servers / 3