"""
SMT vs Non-SMT Oversubscription Carbon/TCO Tradeoff Model

This module provides a first-order framework for modeling the tradeoff between:
- SMT-enabled processors with constrained oversubscription
- Non-SMT processors with potentially higher oversubscription

The goal is to find the breakeven oversubscription ratio for non-SMT that matches
the carbon/TCO savings of SMT with oversubscription (relative to SMT baseline).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import math


def _polynomial_power_raw(cpu_util_pct: float, freq_mhz: float) -> float:
    """
    Raw polynomial power model (internal use).

    Based on empirical SPECpower-like curve fitting.

    Args:
        cpu_util_pct: CPU utilization percentage (0-100)
        freq_mhz: CPU frequency in MHz

    Returns:
        Power consumption in watts (raw, unnormalized)
    """
    # This function estimates CPU/system power draw (in watts) as a weighted combination
    # of constant, linear, quadratic, cubic, and cross terms in utilization and frequency,
    # fit to resemble actual server rack power curves (e.g., those in SPECpower measurements).
    #
    # In words:
    #   "Start from a fixed base power, add terms that scale linearly, quadratically, and cubicly
    #    with CPU utilization (%), frequency, and their combinations. Adjust for nonlinear interactions
    #    between utilization and frequency, all fitted to empirical power data."
    #
    # The equation computed (with cpu = utilization percent [0-100], freq = MHz) is:
    #
    #    power = (
    #        225
    #        + 2.14 * cpu
    #        - 0.0166 * freq
    #        - 0.0248 * cpu**2
    #        + 0.000784 * cpu * freq
    #        + 7.31e-08 * freq**2
    #        + 0.000136 * cpu**3
    #        - 1.22e-05 * cpu**2 * freq
    #        + 4.08e-07 * cpu * freq**2
    #        + 8e-10 * freq**3
    #    )
    #
    # where:
    #   - cpu: CPU utilization percentage (0-100)
    #   - freq: CPU frequency in MHz
    # The output is not yet normalized; it is the modeled absolute platform power estimate in watts.
    cpu = cpu_util_pct
    freq = freq_mhz

    power = (225 + 2.14*cpu - 0.0166*freq - 0.0248*cpu*cpu +
            0.000784*cpu*freq + 7.31e-08*freq*freq + 0.000136*cpu*cpu*cpu -
            1.22e-05*cpu*cpu*freq + 4.08e-07*cpu*freq*freq + 8e-10*freq*freq*freq)

    return max(power, 0)


def polynomial_power_curve_fn(freq_mhz: float = 3500.0) -> Callable[[float], float]:
    """
    Create a normalized power curve function based on polynomial model.

    The polynomial model captures nonlinear power-utilization behavior
    observed in real server hardware (similar to SPECpower curves).

    The returned function maps utilization [0,1] -> [0,1] where:
    - 0 corresponds to idle power
    - 1 corresponds to max power
    - Values in between follow the polynomial shape

    Args:
        freq_mhz: CPU frequency in MHz (default 3500 for typical max freq)

    Returns:
        A curve_fn suitable for use with PowerCurve

    Example:
        curve_fn = polynomial_power_curve_fn(freq_mhz=3500)
        power_curve = PowerCurve(p_idle=100, p_max=500, curve_fn=curve_fn)
    """
    # Compute raw power at idle and max for normalization
    power_at_idle = _polynomial_power_raw(0, freq_mhz)
    power_at_max = _polynomial_power_raw(100, freq_mhz)
    power_range = power_at_max - power_at_idle

    def curve_fn(util: float) -> float:
        """Normalized polynomial curve: maps [0,1] -> [0,1]."""
        if power_range <= 0:
            return util  # Fallback to linear if curve is degenerate
        cpu_pct = util * 100
        raw_power = _polynomial_power_raw(cpu_pct, freq_mhz)
        return (raw_power - power_at_idle) / power_range

    return curve_fn


# Pre-built curve functions for convenience
SPECPOWER_CURVE_FN = lambda u: u ** 0.9  # Classic SPECpower approximation
POLYNOMIAL_CURVE_FN = polynomial_power_curve_fn()  # Default polynomial at 3500 MHz


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
class PowerComponentCurve:
    """Power model for a single server component (CPU, memory, SSD, etc.).

    Each component has its own idle/max power and optional curve shape.
    Default curve is linear.
    """
    idle_w: float
    max_w: float
    curve_fn: Callable[[float], float] = field(default_factory=lambda: lambda u: u)

    def power_at_util(self, util: float) -> float:
        """Return power (Watts) at given utilization [0, 1]."""
        util = max(0.0, min(1.0, util))
        return self.idle_w + (self.max_w - self.idle_w) * self.curve_fn(util)


@dataclass
class PowerBreakdown:
    """Result-side per-component power breakdown at a specific utilization.

    Stores the computed watts for each component at the scenario's effective
    utilization, plus the total.
    """
    component_power_w: Dict[str, float]
    total_power_w: float


def build_composite_power_curve(
    components: Dict[str, PowerComponentCurve],
) -> PowerCurve:
    """Create a composite PowerCurve from per-component power models.

    The composite curve sums individual component powers at each utilization.
    The returned PowerCurve has p_idle = sum of component idles,
    p_max = sum of component maxes, and a curve_fn that correctly maps
    the normalized utilization to the sum of individual component behaviors.

    Args:
        components: Dict of component name -> PowerComponentCurve

    Returns:
        A PowerCurve whose power_at_util(u) equals the sum of all
        component power_at_util(u) values.
    """
    total_idle = sum(c.idle_w for c in components.values())
    total_max = sum(c.max_w for c in components.values())
    power_range = total_max - total_idle

    # Capture components dict in closure
    comps = dict(components)

    def composite_curve_fn(util: float) -> float:
        if power_range <= 0:
            return util
        total_power = sum(c.power_at_util(util) for c in comps.values())
        return (total_power - total_idle) / power_range

    return PowerCurve(
        p_idle=total_idle,
        p_max=total_max,
        curve_fn=composite_curve_fn,
    )


@dataclass
class ProcessorConfig:
    """Configuration for a processor type (SMT or non-SMT)."""
    physical_cores: int
    threads_per_core: int  # 2 for SMT, 1 for non-SMT
    power_curve: PowerCurve
    thread_overhead: int = 0  # HW threads reserved for host (not oversubscribable)
    power_components: Optional[Dict[str, PowerComponentCurve]] = None

    @property
    def pcpus(self) -> int:
        """Total logical cores (pCPUs) per processor."""
        return self.physical_cores * self.threads_per_core

    @property
    def available_pcpus(self) -> int:
        """pCPUs available for VMs (total minus overhead)."""
        return max(0, self.pcpus - self.thread_overhead)


@dataclass
class ScenarioParams:
    """Parameters defining a deployment scenario."""
    processor: ProcessorConfig
    oversub_ratio: float  # vCPU:pCPU ratio (1.0 = no oversub)
    util_overhead: float = 0.0  # Additive overhead to effective utilization
    vcpu_demand_multiplier: float = 1.0  # Multiplier for vCPU demand (e.g., 0.7 = 30% less demand)
    max_vms_per_server: Optional[int] = None  # Optional cap on VMs per server
    avg_vm_size_vcpus: Optional[float] = None  # Per-scenario override for avg VM size


@dataclass
class WorkloadParams:
    """Parameters defining the VM workload to serve."""
    total_vcpus: int  # Total vCPU demand
    avg_util: float   # Average utilization of VMs [0, 1]
    avg_vm_size_vcpus: float = 4.0  # Average vCPUs per VM (for VM cap conversion)


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
class ComponentBreakdown:
    """Breakdown of per-thread, per-server, and per-vCPU cost/carbon components.

    Per-thread components scale with total HW threads (physical_cores * threads_per_core).
    Per-server components are flat per server.
    Per-vCPU components scale with vcpus_per_server (for resource scaling with oversubscription).
    """
    per_thread: Dict[str, float] = field(default_factory=dict)
    per_server: Dict[str, float] = field(default_factory=dict)
    per_vcpu: Dict[str, float] = field(default_factory=dict)
    physical_cores: int = 0
    threads_per_core: int = 1
    vcpus_per_server: float = 0  # 0 means "not set, fall back to hw_threads for per_vcpu"

    @property
    def total_hw_threads(self) -> int:
        return self.physical_cores * self.threads_per_core

    @property
    def per_thread_total_per_server(self) -> float:
        return sum(self.per_thread.values()) * self.total_hw_threads

    @property
    def per_server_total(self) -> float:
        return sum(self.per_server.values())

    @property
    def per_vcpu_total_per_server(self) -> float:
        multiplier = self.vcpus_per_server if self.vcpus_per_server > 0 else self.total_hw_threads
        return sum(self.per_vcpu.values()) * multiplier

    @property
    def total_per_server(self) -> float:
        return self.per_thread_total_per_server + self.per_server_total + self.per_vcpu_total_per_server

    @property
    def per_server_components(self) -> Dict[str, float]:
        """Return flat dict of component_name -> per-server contribution.

        Resolves per_thread, per_vcpu, and per_server multipliers into a single
        per-server value for each component.
        """
        result = {}
        for name, val in self.per_thread.items():
            result[name] = val * self.total_hw_threads
        for name, val in self.per_server.items():
            result[name] = result.get(name, 0) + val
        vcpu_mult = self.vcpus_per_server if self.vcpus_per_server > 0 else self.total_hw_threads
        for name, val in self.per_vcpu.items():
            result[name] = result.get(name, 0) + val * vcpu_mult
        return result

    def resolve(self, physical_cores: int, threads_per_core: int, vcpus_per_server: float = 0) -> 'ComponentBreakdown':
        """Return a new breakdown with core counts set for computing totals."""
        return ComponentBreakdown(
            per_thread=dict(self.per_thread),
            per_server=dict(self.per_server),
            per_vcpu=dict(self.per_vcpu),
            physical_cores=physical_cores,
            threads_per_core=threads_per_core,
            vcpus_per_server=vcpus_per_server,
        )


@dataclass
class EmbodiedBreakdown:
    """Fleet-level embodied carbon and cost breakdown."""
    carbon: Optional[ComponentBreakdown] = None
    cost: Optional[ComponentBreakdown] = None
    num_servers: int = 0
    capacity: Optional[ComponentBreakdown] = None

    @property
    def carbon_fleet_components(self) -> Dict[str, float]:
        """Component name -> fleet total carbon (kg)."""
        if not self.carbon:
            return {}
        result = {}
        for name, val in self.carbon.per_thread.items():
            result[f"per_thread.{name}"] = val * self.carbon.total_hw_threads * self.num_servers
        for name, val in self.carbon.per_server.items():
            result[f"per_server.{name}"] = val * self.num_servers
        vcpu_mult = self.carbon.vcpus_per_server if self.carbon.vcpus_per_server > 0 else self.carbon.total_hw_threads
        for name, val in self.carbon.per_vcpu.items():
            result[f"per_vcpu.{name}"] = val * vcpu_mult * self.num_servers
        return result

    @property
    def cost_fleet_components(self) -> Dict[str, float]:
        """Component name -> fleet total cost (USD)."""
        if not self.cost:
            return {}
        result = {}
        for name, val in self.cost.per_thread.items():
            result[f"per_thread.{name}"] = val * self.cost.total_hw_threads * self.num_servers
        for name, val in self.cost.per_server.items():
            result[f"per_server.{name}"] = val * self.num_servers
        vcpu_mult = self.cost.vcpus_per_server if self.cost.vcpus_per_server > 0 else self.cost.total_hw_threads
        for name, val in self.cost.per_vcpu.items():
            result[f"per_vcpu.{name}"] = val * vcpu_mult * self.num_servers
        return result


@dataclass
class ResourceConstraintDetail:
    """Per-resource constraint analysis detail."""
    max_vcpus: float
    utilization_pct: float    # 0-100, how much of this resource is used
    stranded_pct: float       # 0-100, unused capacity
    is_bottleneck: bool


@dataclass
class ResourceConstraintResult:
    """Result of resource-constrained packing analysis."""
    requested_oversub_ratio: float
    effective_oversub_ratio: float
    effective_vcpus_per_server: float
    bottleneck_resource: str
    resource_details: Dict[str, ResourceConstraintDetail]
    was_constrained: bool     # True if a non-core resource limited packing


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

    # Optional breakdowns
    embodied_breakdown: Optional[EmbodiedBreakdown] = None
    power_breakdown: Optional[PowerBreakdown] = None
    resource_constraint_result: Optional[ResourceConstraintResult] = None


class OverssubModel:
    """
    Core model for evaluating SMT vs non-SMT oversubscription tradeoffs.
    """
    
    def __init__(self, workload: WorkloadParams, cost: CostParams):
        self.workload = workload
        self.cost = cost
    
    def evaluate_scenario(
        self,
        scenario: ScenarioParams,
        cost_overrides: Optional[dict] = None,
    ) -> ScenarioResult:
        """Evaluate carbon and TCO for a given scenario.

        Args:
            scenario: The scenario parameters to evaluate
            cost_overrides: Optional dict with 'embodied_carbon_kg' and/or
                'server_cost_usd' to override the model's default cost params.
                Useful for per-processor cost modeling.

        Returns:
            ScenarioResult with carbon and TCO metrics
        """
        proc = scenario.processor

        # Apply vCPU demand multiplier (e.g., 0.7 means 30% less vCPU demand)
        effective_vcpus = self.workload.total_vcpus * scenario.vcpu_demand_multiplier

        # Calculate server count needed (using available pCPUs, excluding host overhead)
        vcpu_capacity_per_server = proc.available_pcpus * scenario.oversub_ratio

        # Apply VM cap if configured
        if scenario.max_vms_per_server is not None:
            avg_vm_vcpus = scenario.avg_vm_size_vcpus or self.workload.avg_vm_size_vcpus
            max_vcpus_from_vm_cap = scenario.max_vms_per_server * avg_vm_vcpus
            vcpu_capacity_per_server = min(vcpu_capacity_per_server, max_vcpus_from_vm_cap)

        num_servers = math.ceil(effective_vcpus / vcpu_capacity_per_server)

        # Calculate utilization per server
        # Total "work" in pCPU-equivalents = effective_vcpus * avg_util
        # Use available_pcpus for capacity since overhead cores run host workload
        total_work = effective_vcpus * self.workload.avg_util
        total_pcpu_capacity = num_servers * proc.available_pcpus
        avg_util = total_work / total_pcpu_capacity
        
        # Apply utilization overhead
        effective_util = min(1.0, avg_util + scenario.util_overhead)
        
        # Calculate power (with optional per-component breakdown)
        power_per_server = proc.power_curve.power_at_util(effective_util)
        power_breakdown = None
        if proc.power_components:
            component_power = {
                name: comp.power_at_util(effective_util)
                for name, comp in proc.power_components.items()
            }
            power_breakdown = PowerBreakdown(
                component_power_w=component_power,
                total_power_w=sum(component_power.values()),
            )

        # Get cost values, allowing per-processor overrides
        embodied_carbon_kg = self.cost.embodied_carbon_kg
        server_cost_usd = self.cost.server_cost_usd
        embodied_breakdown = None
        resource_constraint_result = None
        if cost_overrides:
            if 'embodied_carbon_kg' in cost_overrides:
                embodied_carbon_kg = cost_overrides['embodied_carbon_kg']
            if 'server_cost_usd' in cost_overrides:
                server_cost_usd = cost_overrides['server_cost_usd']
            if 'resource_constraint_result' in cost_overrides:
                resource_constraint_result = cost_overrides['resource_constraint_result']
            # Build breakdown if component data is provided
            carbon_bd = cost_overrides.get('carbon_breakdown')
            cost_bd = cost_overrides.get('cost_breakdown')
            capacity_bd = cost_overrides.get('capacity_breakdown')
            if carbon_bd or cost_bd or capacity_bd:
                embodied_breakdown = EmbodiedBreakdown(
                    carbon=carbon_bd,
                    cost=cost_bd,
                    num_servers=num_servers,
                    capacity=capacity_bd,
                )

        # Embodied carbon/cost (amortized over lifetime, but we report total)
        embodied_carbon = num_servers * embodied_carbon_kg
        embodied_cost = num_servers * server_cost_usd

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
            embodied_breakdown=embodied_breakdown,
            power_breakdown=power_breakdown,
            resource_constraint_result=resource_constraint_result,
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