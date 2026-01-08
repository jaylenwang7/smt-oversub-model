"""
SMT vs Non-SMT Oversubscription Carbon/TCO Tradeoff Model

A framework for modeling the tradeoff between SMT-enabled processors with
constrained oversubscription versus non-SMT processors with potentially
higher oversubscription.

Example usage:
    from smt_oversub_model import ParameterSweeper, create_default_sweeper
    
    sweeper = create_default_sweeper(
        smt_oversub_ratio=1.3,
        nosmt_physical_cores=48,
    )
    result = sweeper.compute_breakeven()
    print(f"Breakeven oversub for non-SMT: {result['breakeven_oversub_carbon']:.2f}")
"""

from .model import (
    PowerCurve,
    ProcessorConfig,
    ScenarioParams,
    WorkloadParams,
    CostParams,
    ScenarioResult,
    OverssubModel,
)

from .sweep import (
    ParameterSweeper,
    SweepResult,
    create_default_sweeper,
)

__all__ = [
    'PowerCurve',
    'ProcessorConfig', 
    'ScenarioParams',
    'WorkloadParams',
    'CostParams',
    'ScenarioResult',
    'OverssubModel',
    'ParameterSweeper',
    'SweepResult',
    'create_default_sweeper',
]

__version__ = '0.1.0'