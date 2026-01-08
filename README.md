# SMT vs Non-SMT Oversubscription Tradeoff Model

A first-order framework for modeling the carbon and TCO tradeoffs between SMT-enabled processors with constrained oversubscription versus non-SMT processors with potentially higher oversubscription.

## Problem Statement

SMT constraints in cloud settings limit practical oversubscription:
1. vCPUs of different VMs cannot run on sibling SMT threads
2. Guest VM topology constraints create scheduling limitations
3. SMT co-running overhead increases effective utilization

**Question**: At what oversubscription improvement does non-SMT break even on carbon/TCO?

## Installation

```bash
pip install -e .
# Or just copy the smt_oversub_model/ directory
```

## Quick Start

```python
from smt_oversub_model import create_default_sweeper

# Configure with your assumptions
sweeper = create_default_sweeper(
    smt_physical_cores=64,       # 128 pCPUs with SMT
    nosmt_physical_cores=48,     # 48 pCPUs without SMT
    smt_oversub_ratio=1.3,       # 30% oversub achievable with SMT
    nosmt_power_ratio=0.85,      # Non-SMT uses 85% power
    total_vcpus=10000,
    avg_util=0.30,
)

# Find breakeven
result = sweeper.compute_breakeven()
print(f"Carbon breakeven: {result['breakeven_oversub_carbon']:.2f}x oversub")
print(f"TCO breakeven: {result['breakeven_oversub_tco']:.2f}x oversub")
```

## Three Scenarios Compared

1. **Baseline**: SMT processors, no oversubscription (R=1.0)
2. **SMT + Oversub**: SMT with achievable oversubscription (e.g., R=1.3)
3. **Non-SMT + Oversub**: Find breakeven R that matches SMT+Oversub savings

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `smt_physical_cores` | Physical cores per SMT chip | 64 |
| `nosmt_physical_cores` | Physical cores per non-SMT chip | 48 |
| `smt_oversub_ratio` | Achievable oversub with SMT | 1.3 |
| `smt_util_overhead` | Util overhead from SMT contention | 0.05 |
| `nosmt_power_ratio` | Non-SMT P_max as fraction of SMT | 0.85 |
| `total_vcpus` | Total vCPU demand | 10000 |
| `avg_util` | Average VM utilization | 0.30 |
| `embodied_carbon_kg` | Embodied carbon per server | 1000 |
| `carbon_intensity_g_kwh` | Grid carbon intensity | 400 |
| `lifetime_years` | Server lifetime | 4.0 |

## Sensitivity Analysis

```python
# Sweep any parameter
sweep = sweeper.sweep_parameter('carbon_intensity_g_kwh', [100, 200, 400, 600, 800])

# Compare non-SMT at different oversub ratios
comparison = sweeper.sweep_nosmt_oversub([1.0, 1.5, 2.0, 2.5, 3.0])
```

## Model Details

### Server Count
```
num_servers = ceil(total_vcpus / (pcpus_per_server × oversub_ratio))
```

### Utilization
```
avg_util = (total_vcpus × vm_avg_util) / (num_servers × pcpus)
effective_util = min(1.0, avg_util + util_overhead)
```

### Power
```
P(u) = P_idle + (P_max - P_idle) × f(u)
```
Default `f(u) = u^0.9` (slightly sublinear, SPECpower-like)

### Carbon
```
Total = Embodied + Operational
Embodied = num_servers × carbon_per_server
Operational = energy_kwh × carbon_intensity
```

## Running Tests

```bash
pytest smt_oversub_model/test_model.py -v
```

## Example Output

```
SMT oversub achieves vs baseline:
  Carbon savings: 18.5%
  TCO savings: 15.2%

To match SMT+oversub, non-SMT needs:
  Carbon breakeven: 2.15x oversub (115% oversub)
  TCO breakeven: 2.08x oversub (108% oversub)
```

## Extensions

The model can be extended for:
- Non-linear power curves (pass custom `power_curve_fn`)
- Multiple workload classes with different utilizations
- Time-varying carbon intensity
- Server power-off modeling