"""
Parameter sweep utilities for SMT vs Non-SMT oversubscription model.

Provides functions for sweeping parameters and analyzing sensitivity.
"""

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Callable, Any
import numpy as np

from .model import (
    OverssubModel, PowerCurve, ProcessorConfig, 
    ScenarioParams, WorkloadParams, CostParams, ScenarioResult
)


@dataclass
class SweepResult:
    """Result of a parameter sweep."""
    param_name: str
    param_values: List[float]
    breakeven_oversub_carbon: List[Optional[float]]
    breakeven_oversub_tco: List[Optional[float]]
    smt_oversub_carbon: List[float]
    smt_oversub_tco: List[float]
    baseline_carbon: List[float]
    baseline_tco: List[float]


class ParameterSweeper:
    """
    Utility class for sweeping parameters and finding sensitivity.
    """
    
    def __init__(
        self,
        # Base processor configs
        smt_physical_cores: int = 64,
        nosmt_physical_cores: int = 48,  # Parameterized
        smt_core_overhead: int = 0,  # pCPUs reserved for host on SMT servers
        nosmt_core_overhead: int = 0,  # pCPUs reserved for host on non-SMT servers

        # Power curves
        smt_p_idle: float = 100.0,
        smt_p_max: float = 300.0,
        nosmt_power_ratio: float = 0.85,  # Non-SMT P_max as fraction of SMT
        nosmt_idle_ratio: float = 0.85,   # Non-SMT P_idle as fraction of SMT

        # Oversubscription
        smt_oversub_ratio: float = 1.3,
        smt_util_overhead: float = 0.05,  # 5% util overhead from SMT contention
        nosmt_util_overhead: float = 0.0,

        # Workload
        total_vcpus: int = 10000,
        avg_util: float = 0.3,

        # Cost params
        embodied_carbon_kg: float = 1000.0,
        server_cost_usd: float = 15000.0,
        carbon_intensity_g_kwh: float = 400.0,
        electricity_cost_usd_kwh: float = 0.10,
        lifetime_years: float = 4.0,

        # Power curve shape (default: slightly sublinear)
        power_curve_fn: Callable[[float], float] = None,
    ):
        self.smt_physical_cores = smt_physical_cores
        self.nosmt_physical_cores = nosmt_physical_cores
        self.smt_core_overhead = smt_core_overhead
        self.nosmt_core_overhead = nosmt_core_overhead
        self.smt_p_idle = smt_p_idle
        self.smt_p_max = smt_p_max
        self.nosmt_power_ratio = nosmt_power_ratio
        self.nosmt_idle_ratio = nosmt_idle_ratio
        self.smt_oversub_ratio = smt_oversub_ratio
        self.smt_util_overhead = smt_util_overhead
        self.nosmt_util_overhead = nosmt_util_overhead
        self.total_vcpus = total_vcpus
        self.avg_util = avg_util
        self.embodied_carbon_kg = embodied_carbon_kg
        self.server_cost_usd = server_cost_usd
        self.carbon_intensity_g_kwh = carbon_intensity_g_kwh
        self.electricity_cost_usd_kwh = electricity_cost_usd_kwh
        self.lifetime_hours = lifetime_years * 8760
        
        # Default: slightly sublinear power curve (SPECpower-like)
        if power_curve_fn is None:
            power_curve_fn = lambda u: u ** 0.9
        self.power_curve_fn = power_curve_fn
    
    def _build_configs(self) -> Tuple[ProcessorConfig, ProcessorConfig, WorkloadParams, CostParams]:
        """Build processor, workload, and cost configs from current params."""
        smt_power = PowerCurve(self.smt_p_idle, self.smt_p_max, self.power_curve_fn)
        nosmt_power = PowerCurve(
            self.smt_p_idle * self.nosmt_idle_ratio,
            self.smt_p_max * self.nosmt_power_ratio,
            self.power_curve_fn
        )
        
        smt_proc = ProcessorConfig(
            physical_cores=self.smt_physical_cores,
            threads_per_core=2,
            power_curve=smt_power,
            core_overhead=self.smt_core_overhead,
        )
        nosmt_proc = ProcessorConfig(
            physical_cores=self.nosmt_physical_cores,
            threads_per_core=1,
            power_curve=nosmt_power,
            core_overhead=self.nosmt_core_overhead,
        )
        
        workload = WorkloadParams(
            total_vcpus=self.total_vcpus,
            avg_util=self.avg_util,
        )
        
        cost = CostParams(
            embodied_carbon_kg=self.embodied_carbon_kg,
            server_cost_usd=self.server_cost_usd,
            carbon_intensity_g_kwh=self.carbon_intensity_g_kwh,
            electricity_cost_usd_kwh=self.electricity_cost_usd_kwh,
            lifetime_hours=self.lifetime_hours,
        )
        
        return smt_proc, nosmt_proc, workload, cost
    
    def compute_breakeven(self) -> dict:
        """
        Compute breakeven oversubscription for non-SMT given current params.
        
        Returns dict with breakeven ratios and scenario comparisons.
        """
        smt_proc, nosmt_proc, workload, cost = self._build_configs()
        model = OverssubModel(workload, cost)
        
        # Baseline: SMT, no oversub
        baseline = ScenarioParams(smt_proc, oversub_ratio=1.0, util_overhead=0.0)
        baseline_result = model.evaluate_scenario(baseline)
        
        # SMT with oversub
        smt_oversub = ScenarioParams(
            smt_proc, 
            oversub_ratio=self.smt_oversub_ratio,
            util_overhead=self.smt_util_overhead
        )
        smt_result = model.evaluate_scenario(smt_oversub)
        
        # Find breakeven for non-SMT
        breakeven_carbon = model.find_breakeven_oversub(
            smt_result, nosmt_proc, self.nosmt_util_overhead, metric='carbon'
        )
        breakeven_tco = model.find_breakeven_oversub(
            smt_result, nosmt_proc, self.nosmt_util_overhead, metric='tco'
        )
        
        # Evaluate non-SMT at breakeven (if achievable)
        nosmt_carbon_result = None
        nosmt_tco_result = None
        if breakeven_carbon:
            nosmt_scenario = ScenarioParams(
                nosmt_proc, breakeven_carbon, self.nosmt_util_overhead
            )
            nosmt_carbon_result = model.evaluate_scenario(nosmt_scenario)
        if breakeven_tco:
            nosmt_scenario = ScenarioParams(
                nosmt_proc, breakeven_tco, self.nosmt_util_overhead
            )
            nosmt_tco_result = model.evaluate_scenario(nosmt_scenario)
        
        return {
            'breakeven_oversub_carbon': breakeven_carbon,
            'breakeven_oversub_tco': breakeven_tco,
            'baseline': baseline_result,
            'smt_oversub': smt_result,
            'nosmt_at_carbon_breakeven': nosmt_carbon_result,
            'nosmt_at_tco_breakeven': nosmt_tco_result,
            'smt_carbon_savings_vs_baseline_pct': (
                (1 - smt_result.total_carbon_kg / baseline_result.total_carbon_kg) * 100
            ),
            'smt_tco_savings_vs_baseline_pct': (
                (1 - smt_result.total_cost_usd / baseline_result.total_cost_usd) * 100
            ),
        }
    
    def sweep_parameter(
        self,
        param_name: str,
        values: List[float],
    ) -> SweepResult:
        """
        Sweep a parameter and compute breakeven at each value.
        
        param_name must be an attribute of ParameterSweeper.
        """
        breakeven_carbon = []
        breakeven_tco = []
        smt_carbon = []
        smt_tco = []
        baseline_carbon = []
        baseline_tco = []
        
        original_value = getattr(self, param_name)
        
        for val in values:
            setattr(self, param_name, val)
            result = self.compute_breakeven()
            
            breakeven_carbon.append(result['breakeven_oversub_carbon'])
            breakeven_tco.append(result['breakeven_oversub_tco'])
            smt_carbon.append(result['smt_oversub'].total_carbon_kg)
            smt_tco.append(result['smt_oversub'].total_cost_usd)
            baseline_carbon.append(result['baseline'].total_carbon_kg)
            baseline_tco.append(result['baseline'].total_cost_usd)
        
        # Restore original value
        setattr(self, param_name, original_value)
        
        return SweepResult(
            param_name=param_name,
            param_values=values,
            breakeven_oversub_carbon=breakeven_carbon,
            breakeven_oversub_tco=breakeven_tco,
            smt_oversub_carbon=smt_carbon,
            smt_oversub_tco=smt_tco,
            baseline_carbon=baseline_carbon,
            baseline_tco=baseline_tco,
        )
    
    def sweep_nosmt_oversub(
        self,
        oversub_values: List[float],
    ) -> dict:
        """
        Sweep non-SMT oversubscription ratio and compare to SMT.
        
        Returns carbon and TCO for each oversub value, plus comparison to SMT.
        """
        smt_proc, nosmt_proc, workload, cost = self._build_configs()
        model = OverssubModel(workload, cost)
        
        baseline = ScenarioParams(smt_proc, 1.0, 0.0)
        smt_oversub = ScenarioParams(smt_proc, self.smt_oversub_ratio, self.smt_util_overhead)
        
        baseline_result = model.evaluate_scenario(baseline)
        smt_result = model.evaluate_scenario(smt_oversub)
        
        nosmt_results = []
        for oversub in oversub_values:
            scenario = ScenarioParams(nosmt_proc, oversub, self.nosmt_util_overhead)
            nosmt_results.append(model.evaluate_scenario(scenario))
        
        return {
            'oversub_values': oversub_values,
            'baseline': baseline_result,
            'smt_oversub': smt_result,
            'nosmt_results': nosmt_results,
            'nosmt_carbon': [r.total_carbon_kg for r in nosmt_results],
            'nosmt_tco': [r.total_cost_usd for r in nosmt_results],
            'nosmt_carbon_vs_smt_pct': [
                (1 - r.total_carbon_kg / smt_result.total_carbon_kg) * 100 
                for r in nosmt_results
            ],
            'nosmt_tco_vs_smt_pct': [
                (1 - r.total_cost_usd / smt_result.total_cost_usd) * 100 
                for r in nosmt_results
            ],
        }


def create_default_sweeper(**kwargs) -> ParameterSweeper:
    """Create a ParameterSweeper with default values, overriding with kwargs."""
    return ParameterSweeper(**kwargs)