"""
SMT vs Non-SMT Oversubscription Carbon/TCO Tradeoff Model

A framework for modeling the tradeoff between SMT-enabled processors with
constrained oversubscription versus non-SMT processors with potentially
higher oversubscription.

Example usage (programmatic):
    from smt_oversub_model import ParameterSweeper, create_default_sweeper

    sweeper = create_default_sweeper(
        smt_oversub_ratio=1.3,
        nosmt_physical_cores=48,
    )
    result = sweeper.compute_breakeven()
    print(f"Breakeven oversub for non-SMT: {result['breakeven_oversub_carbon']:.2f}")

Example usage (JSON config):
    from smt_oversub_model import load_config, Runner, save_result

    config = load_config("configs/my_experiment.json")
    runner = Runner(config)
    result = runner.run()
    save_result(result, "results/my_experiment.json")

CLI usage:
    python -m smt_oversub_model.cli configs/my_experiment.json
"""

from .model import (
    PowerCurve,
    ProcessorConfig,
    ScenarioParams,
    WorkloadParams,
    CostParams,
    ScenarioResult,
    OverssubModel,
    polynomial_power_curve_fn,
    SPECPOWER_CURVE_FN,
    POLYNOMIAL_CURVE_FN,
)

from .sweep import (
    ParameterSweeper,
    SweepResult,
    create_default_sweeper,
)

from .config import (
    ExperimentConfig,
    PowerCurveSpec,
    ProcessorSpec,
    NoSmtProcessorSpec,
    WorkloadSpec,
    CostSpec,
    OversubSpec,
    SweepSpec,
    load_config,
    save_config,
    validate_config,
)

from .runner import (
    Runner,
    RunResult,
    save_result,
    load_result,
)

# Analysis utilities
from .analysis import (
    ScenarioBuilder,
    ScenarioSpec,
    ProcessorDefaults,
    CostDefaults,
    evaluate_scenarios,
    compare_scenarios,
    compare_smt_vs_nosmt,
    compare_oversub_ratios,
)

# Plotting (optional, requires matplotlib)
try:
    from .plot import (
        plot_scenario_comparison,
        plot_sweep_breakeven,
        plot_scenarios,
        plot_result,
    )
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    plot_scenario_comparison = None
    plot_sweep_breakeven = None
    plot_scenarios = None
    plot_result = None

__all__ = [
    # Core model
    'PowerCurve',
    'ProcessorConfig',
    'ScenarioParams',
    'WorkloadParams',
    'CostParams',
    'ScenarioResult',
    'OverssubModel',
    'polynomial_power_curve_fn',
    'SPECPOWER_CURVE_FN',
    'POLYNOMIAL_CURVE_FN',
    # Sweep utilities
    'ParameterSweeper',
    'SweepResult',
    'create_default_sweeper',
    # Config
    'ExperimentConfig',
    'PowerCurveSpec',
    'ProcessorSpec',
    'NoSmtProcessorSpec',
    'WorkloadSpec',
    'CostSpec',
    'OversubSpec',
    'SweepSpec',
    'load_config',
    'save_config',
    'validate_config',
    # Runner
    'Runner',
    'RunResult',
    'save_result',
    'load_result',
    # Analysis utilities
    'ScenarioBuilder',
    'ScenarioSpec',
    'ProcessorDefaults',
    'CostDefaults',
    'evaluate_scenarios',
    'compare_scenarios',
    'compare_smt_vs_nosmt',
    'compare_oversub_ratios',
    # Plotting (optional)
    'plot_scenario_comparison',
    'plot_sweep_breakeven',
    'plot_scenarios',
    'plot_result',
]

__version__ = '0.1.0'