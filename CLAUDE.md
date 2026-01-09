# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Key Calculations

Server count: `ceil(total_vcpus / (pcpus_per_server * oversub_ratio))`

Effective utilization: `min(1.0, (total_vcpus * avg_util) / (num_servers * pcpus) + util_overhead)`

Total carbon: Embodied (servers * kg_per_server) + Operational (energy_kwh * carbon_intensity)

### Three Scenarios Compared

1. **Baseline**: SMT with no oversubscription (R=1.0)
2. **SMT + Oversub**: SMT with achievable oversubscription (default R=1.3) plus utilization overhead
3. **Non-SMT + Oversub**: Binary search finds the R that matches SMT+Oversub carbon/TCO
