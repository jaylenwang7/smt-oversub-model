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
    PowerComponentCurve,
    PowerBreakdown,
    build_composite_power_curve,
    ProcessorConfig,
    ScenarioParams,
    WorkloadParams,
    CostParams,
    ComponentBreakdown,
    EmbodiedBreakdown,
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
    ProcessorConfigSpec,
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

# Declarative analysis framework
from .declarative import (
    ParameterPath,
    MatchType,
    SimpleCondition,
    CompoundCondition,
    GeneralizedBreakevenFinder,
    BreakevenResult,
    EmbodiedComponentSpec,
    PowerComponentSpec,
    ResourceScalingConfig,
    ScenarioConfig,
    AnalysisSpec,
    AnalysisConfig,
    AnalysisResult,
    DeclarativeAnalysisEngine,
    run_analysis,
    BatchResult,
    run_analysis_batch,
    is_valid_analysis_config,
)

# Output utilities
from .output import (
    OutputWriter,
    save_result as save_declarative_result,
    load_result as load_declarative_result,
)

# Formatter utilities
from .formatter import (
    colorize,
    supports_color,
)

# Plotting (optional, requires matplotlib)
try:
    from .plot import (
        PlotStyle,
        plot_scenario_comparison,
        plot_sweep_breakeven,
        plot_scenarios,
        plot_result,
        plot_breakeven_search,
        plot_analysis_result,
        plot_sweep_analysis,
        plot_breakeven_curve,
        plot_per_server_comparison,
    )
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    PlotStyle = None
    plot_scenario_comparison = None
    plot_sweep_breakeven = None
    plot_scenarios = None
    plot_result = None
    plot_breakeven_search = None
    plot_analysis_result = None
    plot_sweep_analysis = None
    plot_breakeven_curve = None
    plot_per_server_comparison = None

__all__ = [
    # Core model
    'PowerCurve',
    'PowerComponentCurve',
    'PowerBreakdown',
    'build_composite_power_curve',
    'ProcessorConfig',
    'ScenarioParams',
    'WorkloadParams',
    'CostParams',
    'ComponentBreakdown',
    'EmbodiedBreakdown',
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
    'ProcessorConfigSpec',
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
    # Declarative analysis framework
    'ParameterPath',
    'MatchType',
    'SimpleCondition',
    'CompoundCondition',
    'GeneralizedBreakevenFinder',
    'BreakevenResult',
    'EmbodiedComponentSpec',
    'PowerComponentSpec',
    'ResourceScalingConfig',
    'ScenarioConfig',
    'AnalysisSpec',
    'AnalysisConfig',
    'AnalysisResult',
    'DeclarativeAnalysisEngine',
    'run_analysis',
    'BatchResult',
    'run_analysis_batch',
    'is_valid_analysis_config',
    # Output utilities
    'OutputWriter',
    'save_declarative_result',
    'load_declarative_result',
    # Formatter utilities
    'colorize',
    'supports_color',
    # Plotting (optional)
    'PlotStyle',
    'plot_scenario_comparison',
    'plot_sweep_breakeven',
    'plot_scenarios',
    'plot_result',
    'plot_breakeven_search',
    'plot_analysis_result',
    'plot_sweep_analysis',
    'plot_breakeven_curve',
    'plot_per_server_comparison',
]

__version__ = '0.1.0'