"""
SMT vs Non-SMT Oversubscription Carbon/TCO Tradeoff Model

This module provides a first-order framework for modeling the tradeoff between:
- SMT-enabled processors with constrained oversubscription
- Non-SMT processors with potentially higher oversubscription

The goal is to find the breakeven oversubscription ratio for non-SMT that matches
the carbon/TCO savings of SMT with oversubscription (relative to SMT baseline).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import math


@dataclass
class PowerCurve:
    """
    Models server power as a function of utilization.
    
    P(u) = p_idle + (p_max - p_idle) * curve_fn(u)
    
    where curve_fn maps [0,1] -> [0,1] (linear by default).
    """
    p_idle: float  # Watts at 0% utilization
    p_max: float   # Watts at 100% utilization
    curve_fn: Callable[[float], float] = field(default_factory=lambda: lambda u: u)
    
    def power_at_util(self, util: float) -> float:
        """Return power (Watts) at given utilization [0, 1]."""
        util = max(0.0, min(1.0, util))  # Clamp to [0, 1]
        return self.p_idle + (self.p_max - self.p_idle) * self.curve_fn(util)
    
    def scaled(self, idle_ratio: float, max_ratio: float) -> 'PowerCurve':
        """Return a new PowerCurve with scaled idle and max power."""
        return PowerCurve(
            p_idle=self.p_idle * idle_ratio,
            p_max=self.p_max * max_ratio,
            curve_fn=self.curve_fn
        )


@dataclass
class ProcessorConfig:
    """Configuration for a processor type (SMT or non-SMT)."""
    physical_cores: int
    threads_per_core: int  # 2 for SMT, 1 for non-SMT
    power_curve: PowerCurve
    
    @property
    def pcpus(self) -> int:
        """Total logical cores (pCPUs) per processor."""
        return self.physical_cores * self.threads_per_core


@dataclass
class ScenarioParams:
    """Parameters defining a deployment scenario."""
    processor: ProcessorConfig
    oversub_ratio: float  # vCPU:pCPU ratio (1.0 = no oversub)
    util_overhead: float = 0.0  # Additive overhead to effective utilization


@dataclass 
class WorkloadParams:
    """Parameters defining the VM workload to serve."""
    total_vcpus: int  # Total vCPU demand
    avg_util: float   # Average utilization of VMs [0, 1]


@dataclass
class CostParams:
    """Parameters for carbon and TCO calculations."""
    # Embodied
    embodied_carbon_kg: float  # kg CO2e per server
    server_cost_usd: float     # $ per server
    
    # Operational
    carbon_intensity_g_kwh: float  # g CO2/kWh
    electricity_cost_usd_kwh: float  # $/kWh
    
    # Time horizon
    lifetime_hours: float  # Server lifetime for amortization
    
    # Optional: servers turned off consume zero power
    allow_server_poweroff: bool = False


@dataclass
class ScenarioResult:
    """Results from evaluating a scenario."""
    num_servers: int
    avg_util_per_server: float
    effective_util_per_server: float  # After overhead
    power_per_server_w: float
    
    # Carbon (kg CO2e over lifetime)
    embodied_carbon_kg: float
    operational_carbon_kg: float
    total_carbon_kg: float
    
    # TCO (USD over lifetime)
    embodied_cost_usd: float
    operational_cost_usd: float
    total_cost_usd: float
    
    # Derived metrics
    carbon_per_vcpu_kg: float
    cost_per_vcpu_usd: float


class OverssubModel:
    """
    Core model for evaluating SMT vs non-SMT oversubscription tradeoffs.
    """
    
    def __init__(self, workload: WorkloadParams, cost: CostParams):
        self.workload = workload
        self.cost = cost
    
    def evaluate_scenario(self, scenario: ScenarioParams) -> ScenarioResult:
        """Evaluate carbon and TCO for a given scenario."""
        proc = scenario.processor
        
        # Calculate server count needed
        vcpu_capacity_per_server = proc.pcpus * scenario.oversub_ratio
        num_servers = math.ceil(self.workload.total_vcpus / vcpu_capacity_per_server)
        
        # Calculate utilization per server
        # Total "work" in pCPU-equivalents = total_vcpus * avg_util
        total_work = self.workload.total_vcpus * self.workload.avg_util
        total_pcpu_capacity = num_servers * proc.pcpus
        avg_util = total_work / total_pcpu_capacity
        
        # Apply utilization overhead
        effective_util = min(1.0, avg_util + scenario.util_overhead)
        
        # Calculate power
        power_per_server = proc.power_curve.power_at_util(effective_util)
        
        # Embodied carbon/cost (amortized over lifetime, but we report total)
        embodied_carbon = num_servers * self.cost.embodied_carbon_kg
        embodied_cost = num_servers * self.cost.server_cost_usd
        
        # Operational carbon/cost
        total_energy_kwh = (num_servers * power_per_server * 
                           self.cost.lifetime_hours / 1000)
        operational_carbon = total_energy_kwh * self.cost.carbon_intensity_g_kwh / 1000
        operational_cost = total_energy_kwh * self.cost.electricity_cost_usd_kwh
        
        total_carbon = embodied_carbon + operational_carbon
        total_cost = embodied_cost + operational_cost
        
        return ScenarioResult(
            num_servers=num_servers,
            avg_util_per_server=avg_util,
            effective_util_per_server=effective_util,
            power_per_server_w=power_per_server,
            embodied_carbon_kg=embodied_carbon,
            operational_carbon_kg=operational_carbon,
            total_carbon_kg=total_carbon,
            embodied_cost_usd=embodied_cost,
            operational_cost_usd=operational_cost,
            total_cost_usd=total_cost,
            carbon_per_vcpu_kg=total_carbon / self.workload.total_vcpus,
            cost_per_vcpu_usd=total_cost / self.workload.total_vcpus,
        )
    
    def find_breakeven_oversub(
        self,
        target_result: ScenarioResult,
        nosmt_processor: ProcessorConfig,
        nosmt_util_overhead: float = 0.0,
        metric: str = 'carbon',  # 'carbon' or 'tco'
        tolerance: float = 0.001,
        max_oversub: float = 10.0,
    ) -> Optional[float]:
        """
        Find the oversubscription ratio for non-SMT that achieves the same
        carbon or TCO as the target result.
        
        Uses binary search to find the breakeven point.
        
        Returns None if breakeven is not achievable within max_oversub.
        """
        target_value = (target_result.total_carbon_kg if metric == 'carbon' 
                       else target_result.total_cost_usd)
        
        def get_metric(oversub: float) -> float:
            scenario = ScenarioParams(
                processor=nosmt_processor,
                oversub_ratio=oversub,
                util_overhead=nosmt_util_overhead,
            )
            result = self.evaluate_scenario(scenario)
            return (result.total_carbon_kg if metric == 'carbon' 
                   else result.total_cost_usd)
        
        # Check if breakeven is achievable
        min_value = get_metric(max_oversub)
        if min_value > target_value:
            return None  # Can't reach target even at max oversub
        
        # Binary search
        low, high = 1.0, max_oversub
        while high - low > tolerance:
            mid = (low + high) / 2
            mid_value = get_metric(mid)
            if mid_value > target_value:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2
    
    def compare_scenarios(
        self,
        baseline: ScenarioParams,
        smt_oversub: ScenarioParams,
        nosmt_oversub: ScenarioParams,
    ) -> dict:
        """
        Compare three scenarios and compute relative savings.
        
        Returns dict with results and relative metrics.
        """
        r_base = self.evaluate_scenario(baseline)
        r_smt = self.evaluate_scenario(smt_oversub)
        r_nosmt = self.evaluate_scenario(nosmt_oversub)
        
        return {
            'baseline': r_base,
            'smt_oversub': r_smt,
            'nosmt_oversub': r_nosmt,
            'smt_vs_baseline': {
                'carbon_reduction_pct': (1 - r_smt.total_carbon_kg / r_base.total_carbon_kg) * 100,
                'tco_reduction_pct': (1 - r_smt.total_cost_usd / r_base.total_cost_usd) * 100,
                'server_reduction_pct': (1 - r_smt.num_servers / r_base.num_servers) * 100,
            },
            'nosmt_vs_baseline': {
                'carbon_reduction_pct': (1 - r_nosmt.total_carbon_kg / r_base.total_carbon_kg) * 100,
                'tco_reduction_pct': (1 - r_nosmt.total_cost_usd / r_base.total_cost_usd) * 100,
                'server_reduction_pct': (1 - r_nosmt.num_servers / r_base.num_servers) * 100,
            },
            'nosmt_vs_smt': {
                'carbon_reduction_pct': (1 - r_nosmt.total_carbon_kg / r_smt.total_carbon_kg) * 100,
                'tco_reduction_pct': (1 - r_nosmt.total_cost_usd / r_smt.total_cost_usd) * 100,
                'server_reduction_pct': (1 - r_nosmt.num_servers / r_smt.num_servers) * 100,
            },
        }