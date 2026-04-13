# 01: Naive SMT vs No-SMT Comparison (No Oversubscription)

## Question

> If you take an SMT-enabled server and simply disable SMT -- with no
> oversubscription in either case -- what is the raw TCO and carbon cost of
> halving the exposed logical processors?

This is the simplest possible comparison and establishes the **baseline penalty**
that no-SMT must overcome through other mechanisms (higher oversubscription
headroom, vCPU demand compression, etc.) to be competitive.

## Prerequisites

- Read the [spine document](SMT_VS_NOSMT_ANALYSIS.md) for terminology and
  processor specs.

## Key Assumptions

This analysis makes the most assumptions favorable to SMT:

1. **No oversubscription**: Both configurations run at R=1.0, meaning every vCPU
   gets a dedicated pCPU. This is the most conservative deployment model.
2. **Same physical server**: Both SMT and no-SMT use the same AMD Genoa 80-core
   chassis. Disabling SMT halves the HW threads (160 -> 80) and proportionally
   reduces memory/SSD (12 DIMMs -> 6, 6 SSDs -> 3).
3. **No scheduling constraints**: Since there is no oversubscription, VP
   constraints are irrelevant. This removes no-SMT's biggest advantage.
4. **No vCPU demand discount**: The same total vCPU demand (100,000) must be
   served regardless of SMT mode. This ignores the fact that no-SMT LPs deliver
   more performance per vCPU.

Under these assumptions, SMT should win easily -- and it does. The question is
**by how much**, which sets the bar for subsequent analyses.

## Experimental Inputs

### Power Proportionality

This analysis also compares two power curve models for no-SMT:

- **Polynomial (default)**: The standard SPECpower-like sublinear curve used for
  SMT. Assumes no-SMT has the same power-vs-utilization shape as SMT.
- **Linear CPU power**: The CPU component uses a linear power curve. This is
  motivated by experimental measurements showing that no-SMT power scales more
  linearly with utilization.

The experimental basis for the linear model comes from stress-ng load sweep
measurements on a c6620 server:

**Source**: [`smt_no_smt_power_proportionality.md`][exp-power] in the
experimental repo.

Key findings from that experiment:
- At matched realized host utilization, no-SMT uses 8-29W less CPU package power
- No-SMT power is well-described by a linear fit (R^2 = 0.998)
- SMT power follows a concave/saturating curve (linear R^2 = 0.896, needs
  quadratic R^2 = 0.999)
- SMT reaches near-peak package power by ~80% utilization; no-SMT keeps climbing
  linearly

| Regime | Linear fit | Linear R^2 | Quadratic fit | Quadratic R^2 |
|---|---|---|---|---|
| SMT | P(u) = 114.78 + 0.841u | 0.896 | P(u) = 96.61 + 1.934u - 0.01056u^2 | 0.999 |
| No-SMT | P(u) = 100.48 + 0.746u | 0.998 | P(u) = 98.44 + 0.860u - 0.00108u^2 | 0.999 |

**How this maps to the model**: The `genoa_nosmt` processor uses the default
polynomial power curve for all components. The `genoa_nosmt_linear` processor
overrides only the **CPU component** to use a linear curve (memory, SSD, NIC,
chassis remain polynomial). This is a partial application of the experimental
finding -- the full effect would also modify memory/SSD curves, but the CPU is the
dominant power component.

**Caveat**: The experimental measurements are from a c6620 (Intel Xeon, 28
cores), not a Genoa (AMD EPYC, 80 cores). The qualitative finding (no-SMT is more
linear) is likely architecture-general, but the exact coefficients differ. The
model uses Genoa-derived power breakdown values with only the curve *shape*
informed by the c6620 experiment.

[exp-power]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_power_proportionality.md

## Model Configuration

### Config file

[`configs/oversub_analysis/genoa/no_oversub_comparison.jsonc`](../../configs/oversub_analysis/genoa/no_oversub_comparison.jsonc)

### Scenarios

| Scenario | Processor | R | Description |
|---|---|---|---|
| `smt_no_oversub` | `genoa_smt` | 1.0 | SMT enabled, no oversub (baseline) |
| `nosmt_no_oversub` | `genoa_nosmt` | 1.0 | SMT disabled, polynomial power curve |
| `nosmt_linear_no_oversub` | `genoa_nosmt_linear` | 1.0 | SMT disabled, linear CPU power curve |

### Workload

- 100,000 total vCPUs
- 10% average utilization

### Cost parameters

- Carbon intensity: 175 g CO2/kWh
- Electricity cost: $0.28/kWh
- Server lifetime: 6 years

### How to run

```bash
python -m smt_oversub_model configs/oversub_analysis/genoa/no_oversub_comparison.jsonc
```

### Output location

```
results/oversub_analysis/genoa/no_oversub_comparison/
  summary.md          # High-level comparison
  comparison.txt      # Detailed breakdown with component-level data
  comparison.csv      # Machine-readable comparison
  results.json        # Full results
  config.json         # Expanded config (processor specs inlined)
  plots/
    comparison.png    # Stacked bar chart of TCO and carbon
  scenarios/
    smt_no_oversub.json
    nosmt_no_oversub.json
    nosmt_linear_no_oversub.json
```

## Results

### Server Count

| Scenario | Servers | vs Baseline |
|---|---|---|
| SMT (R=1.0) | 70 | -- |
| No-SMT polynomial (R=1.0) | 139 | +98.6% |
| No-SMT linear (R=1.0) | 139 | +98.6% |

No-SMT needs nearly **2x as many servers** because each server exposes only 72
available pCPUs (vs 144 for SMT). The server count is identical for both no-SMT
power curve variants since it depends only on pCPU capacity.

### Carbon

| Scenario | Embodied (kg) | Operational (kg) | Total (kg) | vs Baseline |
|---|---|---|---|---|
| SMT | 124,809 | 225,188 | 349,997 | -- |
| No-SMT poly | 155,649 | 389,220 | 544,870 | **+55.7%** |
| No-SMT linear | 155,649 | 359,497 | 515,147 | **+47.2%** |

### TCO

| Scenario | Embodied ($) | Operational ($) | Total ($) | vs Baseline |
|---|---|---|---|---|
| SMT | 800,856 | 360,301 | 1,161,157 | -- |
| No-SMT poly | 1,109,554 | 622,753 | 1,732,306 | **+49.2%** |
| No-SMT linear | 1,109,554 | 575,196 | 1,684,749 | **+45.1%** |

### Decomposition

Where does the penalty come from?

**Embodied carbon**: +24.7% for no-SMT. Even though no-SMT servers have lower
per-server embodied carbon (1,120 kg vs 1,783 kg, due to fewer DIMMs/SSDs), the
2x server count more than offsets this. The per-server fixed costs (CPU die, NIC,
chassis, rack share) are identical and are simply duplicated across twice as many
servers.

**Operational carbon**: +72.8% (polynomial) or +59.6% (linear). This is the
larger driver. Two factors compound:
1. 2x more servers, each consuming idle power even at low utilization
2. Higher per-server utilization does not occur (both are at R=1.0, so utilization
   is similar), but the 2x server count directly doubles total energy

**Power curve effect**: The linear no-SMT model saves ~30,000 kg CO2 (5.5 pp)
compared to polynomial no-SMT. At 10% utilization this is a modest effect because
utilization is low and both curves are similar near idle. The difference grows at
higher utilization where the polynomial curve's sublinearity becomes more
pronounced.

### Embodied Breakdown (Component Level)

From `comparison.txt`, the per-server embodied carbon breakdown shows where the
SMT advantage comes from:

| Component | SMT (per server) | No-SMT (per server) | Notes |
|---|---|---|---|
| Memory (per-thread) | 160 threads x 4.43 = 709 kg | 80 threads x 4.43 = 354 kg | Halved with HW threads |
| SSD (per-thread) | 160 threads x 3.86 = 618 kg | 80 threads x 3.86 = 309 kg | Halved with HW threads |
| CPU (per-server) | 34.2 kg | 34.2 kg | Same die |
| NIC (per-server) | 115.0 kg | 115.0 kg | Same NIC |
| Chassis (per-server) | 255.5 kg | 255.5 kg | Same chassis |
| Rack (per-server) | 51.9 kg | 51.9 kg | Same rack share |
| **Total per server** | **1,783 kg** | **1,120 kg** | No-SMT is 37% less per server |
| **Fleet total** | 70 x 1,783 = **124,809 kg** | 139 x 1,120 = **155,649 kg** | But 2x servers -> +24.7% |

The per-server fixed costs (CPU + NIC + chassis + rack = 457 kg) are identical
and represent 26% of SMT per-server carbon but 41% of no-SMT per-server carbon.
This fixed-cost dilution is a key structural advantage of SMT: more threads per
server means the fixed overhead is amortized over more vCPUs.

## Interpretation

Under the most favorable assumptions for SMT (no oversubscription, no vCPU demand
discount, no scheduling constraints), disabling SMT incurs:

- **+47% to +56% carbon penalty** (depending on power curve model)
- **+45% to +49% TCO penalty**
- **~2x server count**

This is the "bar" that no-SMT must clear through other mechanisms. The penalty is
large but not insurmountable -- subsequent analyses show that oversubscription
headroom ([02](02_scheduling_constraints_oversub.md)) and vCPU demand compression
([03](03_vcpu_demand_discount.md)) can close and in some cases reverse this gap.

**Key insight**: The penalty is driven more by operational carbon (2x servers
drawing idle power) than by embodied carbon (per-server carbon is lower for
no-SMT). This means any mechanism that reduces server count for no-SMT -- higher
oversubscription, demand compression, or both -- attacks the dominant cost driver.

## What This Does Not Address

1. **Oversubscription**: Both configurations run at R=1.0. In practice, VP
   constraints limit SMT oversubscription while no-SMT can oversubscribe more
   freely. See [02: Scheduling Constraints](02_scheduling_constraints_oversub.md).

2. **vCPU demand compression**: No-SMT LPs deliver more performance, so users may
   need fewer vCPUs. See [03: vCPU Demand Discount](03_vcpu_demand_discount.md).

3. **Resource constraints**: When running no-SMT on existing SMT hardware (without
   reducing DIMMs/SSDs), memory/SSD capacity may limit packing density.

4. **Heterogeneous fleets**: A mixed SMT + no-SMT fleet could route workloads to
   the most efficient pool based on their characteristics.
