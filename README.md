# SMT vs Non-SMT Oversubscription Tradeoff Model

A first-order framework for modeling the carbon and TCO tradeoffs between SMT-enabled and non-SMT processor configurations in cloud environments. Supports detailed per-component cost breakdowns, resource-constrained packing, heterogeneous fleet modeling, and config-driven declarative analyses.

**[Detailed Model Documentation](docs/MODELING.md)** — Mathematical formulas, assumptions, and worked examples.

## Problem Statement

SMT (Simultaneous Multi-Threading) doubles vCPU capacity per server but introduces scheduling constraints in virtualized environments:

1. **Anti-affinity constraints**: vCPUs from different VMs cannot share sibling SMT threads (security/performance isolation)
2. **Topology constraints**: Guest VM topology expectations create scheduling limitations
3. **Contention overhead**: Co-running threads on the same core increases effective utilization

These constraints limit achievable oversubscription with SMT. Non-SMT configurations avoid these constraints but have fewer logical CPUs per server, requiring more servers for the same workload — or enabling higher oversubscription to compensate.

## Questions This Model Can Answer

### Oversubscription & Breakeven

- **At what oversubscription ratio does non-SMT match SMT on carbon/TCO?** Given SMT with limited oversubscription, find the non-SMT oversub ratio that achieves equivalent carbon or cost.
- **What vCPU demand discount makes non-SMT viable?** If non-SMT processors are less performant per-thread, how much of a performance gap is acceptable before carbon/TCO parity is lost?
- **How do breakeven points shift with utilization?** At 10% vs 30% vs 50% avg utilization, does non-SMT become more or less attractive?

### Sensitivity & Savings Analysis

- **How do carbon savings vary as vCPU demand discount changes?** Sweep the performance multiplier to see where savings cross zero (breakeven).
- **How does grid carbon intensity affect the tradeoff?** In clean-grid regions, embodied carbon dominates; in carbon-heavy grids, operational efficiency matters more.
- **What's the impact of different power models?** Compare linear vs SPECpower-like vs polynomial power curves to understand model sensitivity.

### Resource Constraints & Packing

- **What's the effective oversubscription ratio when memory/SSD constrain packing?** Servers have finite memory and storage — at high oversub ratios, which resource becomes the bottleneck?
- **How much capacity is stranded?** When memory is the bottleneck, how much CPU and SSD capacity goes unused?
- **Purpose-built vs same-hardware deployment**: If non-SMT runs on existing SMT hardware (with extra DIMMs/SSDs already installed), how do savings differ from purpose-built non-SMT servers?

### Heterogeneous Fleets

- **Should we deploy all SMT, all non-SMT, or a mixed fleet?** Model composite scenarios where different workload segments are routed to different server pools.
- **What's the optimal split point?** Given a distribution of workload performance characteristics (vCPU discount), where should we draw the line between SMT and non-SMT pools?
- **How sensitive is the mixed fleet to the split point?** Sweep the allocation threshold to find the optimal balance.

### Per-Server & Fleet Comparisons

- **What does the per-server embodied carbon breakdown look like?** CPU die vs memory vs SSD vs chassis contributions, and how they differ between SMT and non-SMT.
- **Fleet-level TCO/carbon comparison across utilization levels**: Aggregate view of total carbon and cost for different configurations at multiple utilization points.

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[dev]"    # pytest
pip install -e ".[plot]"   # matplotlib, numpy
pip install -e ".[all]"    # everything
```

## Quick Start

### Declarative Analysis (Recommended)

The primary interface is config-driven JSON analysis files. Run from the CLI:

```bash
# Run a single analysis
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/savings_curve.jsonc

# Run all configs in a directory
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/
```

Or programmatically:

```python
from smt_oversub_model import DeclarativeAnalysisEngine, run_analysis

result = run_analysis("configs/oversub_analysis/genoa/linear/savings_curve.jsonc")
print(result.summary)
```

### Programmatic API

```python
from smt_oversub_model import create_default_sweeper

sweeper = create_default_sweeper(
    smt_physical_cores=64,
    nosmt_physical_cores=48,
    smt_oversub_ratio=1.3,
    nosmt_power_ratio=0.85,
    total_vcpus=10000,
    avg_util=0.30,
)

result = sweeper.compute_breakeven()
print(f"Carbon breakeven: {result['breakeven_oversub_carbon']:.2f}x oversub")
print(f"TCO breakeven: {result['breakeven_oversub_tco']:.2f}x oversub")
```

## Analysis Types

The declarative framework supports nine analysis types:

| Type | Description |
|------|-------------|
| `find_breakeven` | Binary search to find a parameter value where target matches reference on carbon/TCO |
| `compare` | Evaluate and compare multiple scenarios side-by-side |
| `sweep` | Run `find_breakeven` repeatedly across different values of a second parameter |
| `compare_sweep` | Sweep a parameter showing % change vs baseline at each value (sensitivity analysis) |
| `breakeven_curve` | Aggregate breakeven values from multiple sub-configs into a curve (e.g., breakeven vs utilization) |
| `savings_curve` | Aggregate compare_sweep results from multiple sub-configs into a multi-line savings plot |
| `per_server_comparison` | Compare per-server metrics (capacity, embodied carbon breakdown) across configurations |
| `resource_packing` | Visualize resource utilization, bottlenecks, and stranded capacity under constraints |
| `fleet_comparison` | Aggregate fleet-level TCO/carbon totals from multiple scenario sets |

### find_breakeven

Answers: *"What value of X makes the target match the reference?"*

Varies one parameter via binary search until two scenarios produce matching carbon or TCO.

```json
{
  "analysis": {
    "type": "find_breakeven",
    "baseline": "smt_baseline",
    "reference": "smt_oversub",
    "target": "nosmt_target",
    "vary_parameter": "oversub_ratio",
    "match_metric": "carbon",
    "search_bounds": [1.0, 5.0]
  }
}
```

### compare_sweep

Answers: *"How do savings change as parameter Y varies?"*

Sweeps a parameter across values, showing % change in carbon/TCO relative to baseline at each step. Supports single or multi-scenario comparison with breakeven markers.

```json
{
  "analysis": {
    "type": "compare_sweep",
    "baseline": "smt_baseline",
    "sweep_scenarios": ["nosmt_r1", "nosmt_r1_5", "nosmt_r2"],
    "sweep_parameter": "vcpu_demand_multiplier",
    "sweep_values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "show_breakeven_marker": true
  }
}
```

### sweep

Answers: *"How does the breakeven value change as Y varies?"*

Iterates over values of one parameter; at each, runs a full `find_breakeven` on another parameter.

### savings_curve / breakeven_curve

Meta-analysis types that aggregate results from multiple sub-config files (e.g., one per utilization level) into a single multi-line curve plot.

## Processor Configuration

Processors are defined with arbitrary names and explicit `threads_per_core` (1 = no SMT, 2 = SMT):

```json
{
  "processor": {
    "smt": {
      "physical_cores": 80,
      "threads_per_core": 2,
      "thread_overhead": 8,
      "power_breakdown": {
        "cpu": {"idle_w": 30, "max_w": 234, "power_curve": {"type": "polynomial"}},
        "memory": {"idle_w": 20, "max_w": 66, "power_curve": {"type": "linear"}},
        "ssd": {"idle_w": 6, "max_w": 12, "power_curve": {"type": "linear"}},
        "nic": {"idle_w": 10, "max_w": 15, "power_curve": {"type": "linear"}},
        "chassis": {"idle_w": 10, "max_w": 10, "power_curve": {"type": "linear"}}
      },
      "embodied_carbon": {
        "per_thread": {"cpu_die": 2.37, "memory": 4.43, "ssd": 1.25},
        "per_server": {"chassis": 100.0, "nic": 15.0}
      },
      "server_cost": {
        "per_thread": {"cpu": 73.0, "memory": 42.0, "ssd": 11.0},
        "per_server": {"chassis": 1500.0, "nic": 200.0}
      }
    }
  }
}
```

### Shared Processor Files

Processor definitions can be loaded from external files, enabling reuse across analyses:

```json
{
  "processor": "configs/shared/genoa_processors.jsonc"
}
```

Or selectively import with renaming:

```json
{
  "processor": {
    "smt": "configs/shared/genoa_processors.jsonc:genoa_smt",
    "nosmt": "configs/shared/genoa_processors.jsonc:genoa_nosmt"
  }
}
```

See `configs/shared/` for pre-defined processor families (Genoa 80-core, Bergamo 128-core, generic 48-core).

### Power Curves

Each processor (or component within a processor) can specify its own power curve:

- `"linear"`: P(u) = P_idle + (P_max - P_idle) * u
- `"specpower"`: P(u) = P_idle + (P_max - P_idle) * u^0.9
- `"power"`: Custom exponent (requires `"exponent"` field)
- `"polynomial"`: Frequency-dependent polynomial fit

When `power_breakdown` is specified, a composite power curve is automatically built from per-component curves.

## Resource Modeling

### Resource Scaling

When oversubscription packs more vCPUs than HW threads, resources like memory and SSD scale with vCPU count:

```json
{
  "nosmt_oversub": {
    "processor": "nosmt",
    "oversub_ratio": 2.0,
    "resource_scaling": {
      "scale_with_vcpus": ["memory", "ssd"],
      "scale_power": true
    }
  }
}
```

This models purpose-built servers where memory/SSD are provisioned per-vCPU, increasing both embodied carbon/cost and power consumption proportionally.

### Resource Constraints

The opposite model: servers have fixed capacities, and multiple resources independently limit packing:

```json
{
  "nosmt_constrained": {
    "processor": "nosmt",
    "oversub_ratio": 2.0,
    "resource_constraints": {
      "memory_gb": {"capacity_per_thread": 4.8, "demand_per_vcpu": 4.0},
      "ssd_gb": {"capacity_per_server": 6000, "demand_per_vcpu": 50}
    }
  }
}
```

This models running non-SMT on existing hardware where memory/SSD capacity is fixed. The model identifies the effective R, bottleneck resource, and stranded capacity.

## Composite Scenarios (Heterogeneous Fleets)

Model a mixed fleet where different workload segments route to different server pools:

```json
{
  "scenarios": {
    "smt_pool": {"processor": "smt", "oversub_ratio": 1.0},
    "nosmt_pool": {"processor": "nosmt", "oversub_ratio": 2.0, "vcpu_demand_multiplier": 0.65},
    "mixed_fleet": {
      "composite": {
        "smt_pool": {"allocation": "above_split"},
        "nosmt_pool": {"allocation": "below_split", "parameter_effects": {"vcpu_demand_multiplier": "weighted_average"}}
      },
      "split_trait": "vcpu_discount",
      "split_point": 0.75
    }
  }
}
```

Supports trait-based allocation with discrete or CDF distributions, auto-breakeven split point computation, and sweeping the split point for sensitivity analysis.

## Cost Specification

### Structured Breakdown (Per-Thread + Per-Server)

Embodied carbon and cost support structured breakdowns with per-thread components (scaling with HW thread count) and per-server components (flat):

```json
{
  "embodied_carbon": {
    "per_thread": {"cpu_die": 2.37, "memory": 4.43},
    "per_server": {"chassis": 100.0}
  }
}
```

### Ratio-Based Mode

Instead of raw parameters, specify operational/embodied ratios:

```json
{
  "cost": {
    "mode": "ratio_based",
    "reference_scenario": "baseline",
    "operational_carbon_fraction": 0.75,
    "embodied_carbon_kg": 2000.0,
    "lifetime_years": 5.0
  }
}
```

## Config Directory Structure

```
configs/
├── shared/                          # Reusable processor definitions
│   ├── genoa_processors.jsonc       # AMD EPYC Genoa (80-core)
│   ├── bergamo_processors.jsonc     # AMD EPYC Bergamo (128-core)
│   └── processors.jsonc             # Generic processors
├── oversub_analysis/
│   └── genoa/
│       ├── linear/                  # Linear power curve analyses
│       │   ├── savings_curve.jsonc                    # Savings across utilization levels
│       │   ├── breakeven_curve_comparison.jsonc        # Breakeven curves with/without scaling
│       │   ├── resource_packing_oversub_sweep.jsonc    # Scaled vs constrained packing
│       │   ├── per_server_capacity_comparison.jsonc    # Per-server breakdowns
│       │   ├── tco_carbon_constrained_vs_unconstrained.jsonc  # Fleet-level comparison
│       │   ├── smt_vs_nosmt_breakeven_carbon.jsonc    # Breakeven vCPU discount
│       │   ├── util_*_pct_linear*.jsonc               # Per-utilization compare_sweep
│       │   └── multi_cluster/                         # Heterogeneous fleet analyses
│       └── non-linear/              # SPECpower/polynomial curve analyses
├── nosmt_savings_sweep/             # vCPU discount sensitivity analyses
├── vcpu_demand_breakeven/           # vCPU demand breakeven analyses
└── util_oversub_comparison/         # Utilization vs oversub comparison
```

## Model Details

> **For detailed formulas, assumptions, and worked examples, see [docs/MODELING.md](docs/MODELING.md).**

### Core Calculation Pipeline

```
Server count = ceil(total_vcpus * vcpu_demand_multiplier / (available_pcpus * oversub_ratio))

Effective utilization = min(1.0, (total_vcpus * avg_util) / (num_servers * available_pcpus) + util_overhead)

Power per server = P_idle + (P_max - P_idle) * f(utilization)

Total carbon = Embodied (servers * carbon_per_server) + Operational (energy_kwh * carbon_intensity)
Total cost   = Embodied (servers * cost_per_server)   + Operational (energy_kwh * electricity_price)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `physical_cores` | Physical CPU cores per server |
| `threads_per_core` | 1 (no SMT) or 2 (SMT) |
| `thread_overhead` | pCPUs reserved for hypervisor/host |
| `oversub_ratio` | vCPU:pCPU ratio (1.0 = no oversub) |
| `util_overhead` | Additive utilization overhead (e.g., 0.05 for SMT contention) |
| `vcpu_demand_multiplier` | Scales total vCPU demand (e.g., 0.7 = 30% less demand) |
| `total_vcpus` | Total vCPU demand across all VMs |
| `avg_util` | Average VM utilization (0.0-1.0) |

## Running Tests

```bash
pytest smt_oversub_model/test_model.py smt_oversub_model/test_declarative.py -v
```

## Example Output

```
=== Scenario: smt_baseline (R=1.0) ===
  Servers: 139 | Util: 10.0% | Power/srv: 88.9W
  Carbon: 127,660 kg (embodied: 73,339 + operational: 54,321)
  TCO: $1,946,765 (capex: $1,388,861 + opex: $557,905)

=== Scenario: nosmt_oversub_scaled (R=5.0, scaled) ===
  Servers: 42 | Util: 33.3% | Power/srv: 154.4W
  Carbon: 62,581 kg (embodied: 33,542 + operational: 29,039)
  TCO: $824,649 (capex: $554,978 + opex: $269,671)

Carbon: -51.0% vs baseline | TCO: -57.6% vs baseline
```
