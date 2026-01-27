# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note**: This repo is for personal use/experimentation.

**Environment**: Use `python` (not `python3`) for running Python commands.

## Build and Test Commands

```bash
# Install package in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"    # pytest
pip install -e ".[plot]"   # matplotlib, numpy
pip install -e ".[all]"    # everything

# Run all tests
pytest smt_oversub_model/test_model.py -v

# Run a single test
pytest smt_oversub_model/test_model.py::TestPowerCurve::test_power_at_zero_util -v
```

## Architecture

This is a pure Python library (no external dependencies for core functionality) that models carbon and TCO tradeoffs between SMT and non-SMT processor configurations in cloud environments.

### Core Classes (`model.py`)

- **`PowerCurve`**: Models server power consumption as a function of utilization. Default is linear but supports custom curves (e.g., `u^0.9` for SPECpower-like behavior).

- **`ProcessorConfig`**: Defines processor characteristics - physical cores, threads per core (2 for SMT, 1 for non-SMT), and associated power curve.

- **`OverssubModel`**: Main model class. Given workload and cost parameters, evaluates scenarios and finds breakeven oversubscription ratios using binary search.

### Sweep Utilities (`sweep.py`)

- **`ParameterSweeper`**: High-level API that bundles all configuration parameters. Use `compute_breakeven()` to find the non-SMT oversubscription ratio that matches SMT+oversub savings.

- **`create_default_sweeper()`**: Factory function for creating a sweeper with sensible defaults, accepting keyword overrides.

### Analysis Utilities (`analysis.py`)

Flexible API for comparing arbitrary scenarios without the breakeven-finding workflow:

- **`compare_smt_vs_nosmt()`**: Quick comparison of SMT vs non-SMT at the same oversubscription ratio.
- **`compare_oversub_ratios()`**: Compare different R values for the same processor type.
- **`ScenarioBuilder`**: Build custom scenarios with processor/cost overrides.
- **`compare_scenarios()`**: General-purpose scenario comparison.

### Plotting (`plot.py`)

- **`plot_scenarios()`**: Flexible plotting for arbitrary scenario lists. Shows TCO and Carbon stacked bars with baseline diff annotations.
- **`plot_scenario_comparison()`**: Plot results from Runner for breakeven analysis.
- **`plot_sweep_breakeven()`**: Plot breakeven curves across parameter sweeps.
- **`plot_breakeven_search()`**: Visualize breakeven search convergence.
- **`plot_analysis_result()`**: Auto-detect analysis type and plot appropriate visualization.

### Declarative Analysis Framework (`declarative.py`)

Generalized, config-driven analysis where breakeven can be found for **any** numeric parameter.

- **`ParameterPath`**: Resolve dot-notation paths like `processor.physical_cores` for nested parameter access.
- **`SimpleCondition`/`CompoundCondition`**: Define match conditions (e.g., `"carbon": "match"`, `"tco": "within_5%"`).
- **`GeneralizedBreakevenFinder`**: Binary search to find any parameter value matching reference.
- **`AnalysisConfig`**: JSON config schema for declarative analyses.
- **`DeclarativeAnalysisEngine`**: Run analyses from config objects or JSON files.

### Output Utilities (`output.py`)

- **`OutputWriter`**: Write structured results to directory (results.json, config.json, summary.md, plots/).

### Key Calculations

Server count: `ceil(total_vcpus / (pcpus_per_server * oversub_ratio))`

Effective utilization: `min(1.0, (total_vcpus * avg_util) / (num_servers * pcpus) + util_overhead)`

Total carbon: Embodied (servers * kg_per_server) + Operational (energy_kwh * carbon_intensity)

### Three Scenarios Compared (Breakeven Analysis)

1. **Baseline**: SMT with no oversubscription (R=1.0)
2. **SMT + Oversub**: SMT with achievable oversubscription (default R=1.3) plus utilization overhead
3. **Non-SMT + Oversub**: Binary search finds the R that matches SMT+Oversub carbon/TCO

## Declarative Analysis API

The declarative framework allows config-driven analyses with support for finding breakeven on any parameter.

### Analysis Types

1. **`find_breakeven`**: Binary search to find parameter value matching reference
2. **`compare`**: Compare multiple scenarios
3. **`sweep`**: Run breakeven analysis across parameter sweep

### Example Config

```json
{
  "name": "vcpu_demand_breakeven",
  "scenarios": {
    "baseline": {"processor": "smt", "oversub_ratio": 1.0},
    "smt_oversub": {"processor": "smt", "oversub_ratio": 1.1, "util_overhead": 0.05},
    "nosmt_oversub": {"processor": "nosmt", "oversub_ratio": 1.4}
  },
  "analysis": {
    "type": "find_breakeven",
    "baseline": "baseline",
    "reference": "smt_oversub",
    "target": "nosmt_oversub",
    "vary_parameter": "vcpu_demand_multiplier",
    "match_metric": "carbon",
    "search_bounds": [0.5, 1.0]
  },
  "workload": {"total_vcpus": 10000, "avg_util": 0.3}
}
```

### CLI Usage

```bash
python -m smt_oversub_model.declarative configs/vcpu_demand_breakeven.json
```

### Programmatic Usage

```python
from smt_oversub_model import DeclarativeAnalysisEngine, run_analysis

# From file
result = run_analysis("configs/vcpu_demand_breakeven.json")

# Or with engine
engine = DeclarativeAnalysisEngine()
result = engine.run_from_file("configs/analysis.json")
print(result.summary)
```

### Supported Parameter Paths

- Direct: `oversub_ratio`, `util_overhead`, `vcpu_demand_multiplier`
- Nested: `processor.physical_cores`, `processor.power_curve.p_max`
- Workload: `workload.avg_util`, `workload.total_vcpus`
- Cost: `cost.embodied_carbon_kg`, `cost.carbon_intensity_g_kwh`

### Match Conditions

- Simple: `"carbon"` or `"tco"` (match exactly)
- Compound: `{"carbon": "match", "tco": "within_5%"}`
- Comparison: `"<="`, `">="`

## Example Notebooks

See `notebooks/` for Jupyter notebooks demonstrating various analyses:

- **`smt_vs_nosmt_comparison.ipynb`**: Compare SMT vs non-SMT at the same oversubscription ratio
