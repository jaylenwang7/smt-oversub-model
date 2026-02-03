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

- **`ProcessorConfig`**: Defines processor characteristics - physical cores, threads per core (configurable: 1 = no SMT, 2+ = SMT enabled), and associated power curve.

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
4. **`compare_sweep`**: Compare scenarios while sweeping a parameter, showing % change vs baseline at each sweep value. Useful for sensitivity analysis (e.g., "how do savings change as vCPU discount varies?")

### Example Config

Processors are defined with arbitrary names and explicit `threads_per_core` (1 = no SMT, 2+ = SMT enabled):

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
  "processor": {
    "smt": {
      "physical_cores": 48,
      "threads_per_core": 2,
      "power_idle_w": 100.0,
      "power_max_w": 400.0,
      "core_overhead": 0
    },
    "nosmt": {
      "physical_cores": 48,
      "threads_per_core": 1,
      "power_idle_w": 90.0,
      "power_max_w": 340.0,
      "core_overhead": 0
    }
  },
  "workload": {"total_vcpus": 10000, "avg_util": 0.3}
}
```

You can define processors with any name (not limited to "smt"/"nosmt"):

```json
{
  "processor": {
    "standard": {"physical_cores": 32, "threads_per_core": 1, "power_idle_w": 80, "power_max_w": 300},
    "hyperthreaded": {"physical_cores": 32, "threads_per_core": 2, "power_idle_w": 90, "power_max_w": 350},
    "smt4": {"physical_cores": 16, "threads_per_core": 4, "power_idle_w": 100, "power_max_w": 400}
  }
}
```

### Processor Config Loading

Processor configurations support three loading modes: load all from file, inline definitions, or mixed mode.

**Mode 1: Load all from external file**

Load all processors from an external JSON file. Multiple analysis configs can share the same processor definitions.

```json
{
  "name": "my_analysis",
  "processor": "../shared/processors.jsonc",
  "scenarios": { ... }
}
```

Or use the `processor_file` key:
```json
{
  "name": "my_analysis",
  "processor_file": "../shared/processors.jsonc",
  "scenarios": { ... }
}
```

**Mode 2: Inline definitions**

Define processors directly in the config (original behavior):

```json
{
  "processor": {
    "smt": {"physical_cores": 48, "threads_per_core": 2, "power_idle_w": 100, "power_max_w": 400},
    "nosmt": {"physical_cores": 48, "threads_per_core": 1, "power_idle_w": 90, "power_max_w": 340}
  }
}
```

**Mode 3: Mixed mode (selective imports + custom definitions)**

Mix inline definitions with selective imports from external files. Each processor can be:
- An inline definition (object with processor fields)
- A string reference: `"file:processor_name"`
- An object reference: `{"file": "path", "name": "processor_name"}`

```json
{
  "processor": {
    // Import 'smt' from external file, rename to 'standard_smt'
    "standard_smt": "../shared/processors.jsonc:smt",
    
    // Object reference format (equivalent)
    "standard_nosmt": {"file": "../shared/processors.jsonc", "name": "nosmt"},
    
    // Custom inline definition
    "custom_lowpower": {
      "physical_cores": 32,
      "threads_per_core": 1,
      "power_idle_w": 40.0,
      "power_max_w": 200.0,
      "core_overhead": 2
    }
  }
}
```

This allows you to:
- Selectively import only the processors you need from a shared file
- Rename imported processors to different local names
- Mix shared processors with custom overrides
- Use multiple external files in the same config

**Shared processor file format** (`configs/shared/processors.jsonc`):
```json
{
  "smt": {
    "physical_cores": 48,
    "threads_per_core": 2,
    "power_idle_w": 50.0,
    "power_max_w": 300.0,
    "core_overhead": 4
  },
  "nosmt": {
    "physical_cores": 52,
    "threads_per_core": 1,
    "power_idle_w": 50.0,
    "power_max_w": 300.0,
    "core_overhead": 5
  }
}
```

**Path resolution:**
- Relative paths are resolved relative to the config file's directory
- Absolute paths work as-is
- Supports `.json` and `.jsonc` (with comments, requires `json5` package)

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
- Cost (raw): `cost.embodied_carbon_kg`, `cost.carbon_intensity_g_kwh`, `cost.server_cost_usd`, `cost.electricity_cost_usd_kwh`, `cost.lifetime_years`
- Cost (ratio): `cost.operational_carbon_fraction`, `cost.operational_cost_fraction`, `cost.total_carbon_kg`, `cost.total_cost_usd`

### Match Conditions

- Simple: `"carbon"` or `"tco"` (match exactly)
- Compound: `{"carbon": "match", "tco": "within_5%"}`
- Comparison: `"<="`, `">="`

### Ratio-Based Cost Specification

Instead of specifying raw cost parameters, you can specify operational/embodied ratios and let the system derive the raw parameters. This enables intuitive ratio-based analysis (e.g., "75% operational carbon").

**Two Modes:**
- `raw` (default): Direct specification of all cost parameters
- `ratio_based`: Specify ratios, system derives `carbon_intensity_g_kwh` and/or `electricity_cost_usd_kwh`

**Two Anchor Types:**
- **Embodied Anchor**: Specify per-server embodied values; system derives operational params
- **Total Anchor**: Specify total carbon/cost budget; system derives both embodied and operational

**Embodied Anchor Example** (75% operational carbon):
```json
{
  "cost": {
    "mode": "ratio_based",
    "reference_scenario": "baseline",
    "operational_carbon_fraction": 0.75,
    "operational_cost_fraction": 0.6,
    "embodied_carbon_kg": 2000.0,
    "server_cost_usd": 10000.0,
    "lifetime_years": 5.0
  }
}
```

**Total Anchor Example** (specify total budget):
```json
{
  "cost": {
    "mode": "ratio_based",
    "reference_scenario": "baseline",
    "operational_carbon_fraction": 0.75,
    "operational_cost_fraction": 0.6,
    "total_carbon_kg": 50000.0,
    "total_cost_usd": 500000.0,
    "lifetime_years": 5.0
  }
}
```

**Important Behavior**: Parameters are derived **once** from the reference scenario, then applied consistently to all scenarios. The reference scenario will achieve exactly the specified ratio; other scenarios may have different actual ratios due to different server counts and power consumption.

**Sweep Over Ratio**:
```json
{
  "analysis": {
    "type": "sweep",
    "sweep_parameter": "cost.operational_carbon_fraction",
    "sweep_values": [0.25, 0.5, 0.75, 0.9]
  }
}
```

### Compare Sweep Analysis

Compare a target scenario against a baseline while sweeping a parameter, showing % increase/decrease at each value:

```json
{
  "name": "nosmt_vcpu_discount_savings",
  "scenarios": {
    "smt_baseline": {"processor": "smt", "oversub_ratio": 1.0},
    "nosmt_no_oversub": {"processor": "nosmt", "oversub_ratio": 1.0, "vcpu_demand_multiplier": 1.0}
  },
  "analysis": {
    "type": "compare_sweep",
    "baseline": "smt_baseline",
    "sweep_scenario": "nosmt_no_oversub",
    "sweep_parameter": "vcpu_demand_multiplier",
    "sweep_values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "show_breakeven_marker": true
  }
}
```

**Multi-Scenario Comparison** (multiple lines on same plot):

```json
{
  "analysis": {
    "type": "compare_sweep",
    "baseline": "smt_baseline",
    "sweep_scenarios": ["nosmt_no_oversub", "nosmt_oversub_1_5", "nosmt_oversub_2_0"],
    "sweep_parameter": "vcpu_demand_multiplier",
    "sweep_values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "show_breakeven_marker": true
  }
}
```

Output includes:
- Table with % change vs baseline for carbon, TCO, and servers at each sweep value
- Plot showing the % change curves across the sweep range
- Breakeven points (where line crosses 0%) marked with diamond markers and labels
- Negative % = savings, Positive % = increase vs baseline

**Options:**
- `sweep_scenario`: Single scenario to sweep (backward compatible)
- `sweep_scenarios`: List of scenarios for multi-line comparison
- `show_breakeven_marker`: Show/hide breakeven point markers on plot (default: true)

**Math (Embodied Anchor)**:
```
Given: operational_carbon_fraction (f_op), embodied_carbon_kg, reference scenario
1. embodied_total = num_servers × embodied_carbon_kg
2. operational_total = embodied_total × f_op / (1 - f_op)
3. carbon_intensity_g_kwh = operational_total × 1000 / energy_kwh
```

**Math (Total Anchor)**:
```
Given: operational_carbon_fraction (f_op), total_carbon_kg, reference scenario
1. operational_total = total_carbon_kg × f_op
2. embodied_total = total_carbon_kg × (1 - f_op)
3. embodied_carbon_kg = embodied_total / num_servers
4. carbon_intensity_g_kwh = operational_total × 1000 / energy_kwh
```

## Example Notebooks

See `notebooks/` for Jupyter notebooks demonstrating various analyses:

- **`smt_vs_nosmt_comparison.ipynb`**: Compare SMT vs non-SMT at the same oversubscription ratio
