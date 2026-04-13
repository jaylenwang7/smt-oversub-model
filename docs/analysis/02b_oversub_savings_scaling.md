# 02b: How Savings Scale with Oversubscription

## Question

> If oversubscription reduces server count, do carbon and TCO savings track
> proportionally? Or do real-world considerations erode the savings at higher R?

This is a side analysis that examines the **shape** of the savings curve as
oversubscription ratio increases. The main analysis progression
([01](01_naive_comparison.md) -> [02](02_scheduling_constraints_oversub.md) ->
[03](03_vcpu_demand_discount.md)) compares SMT vs no-SMT at specific R values
derived from experiments. This document steps back and asks a simpler structural
question: for a *given* configuration (no-SMT, fixed utilization), how do savings
accumulate as R increases, and what makes real savings deviate from the
theoretical ideal?

## Prerequisites

- [02: Scheduling Constraints](02_scheduling_constraints_oversub.md) for
  oversubscription ratios
- [02a: Resource Modeling](02a_resource_modeling.md) for the three resource
  modeling approaches (fixed, scaled, constrained)

## Motivation

When the model projects that no-SMT saves X% of servers at a given R, it is
tempting to assume carbon and TCO savings are proportional. A 50% server
reduction should give ~50% savings, right?

Not quite. Server count is not the only cost driver:

1. **Per-server costs are not eliminated uniformly**. Servers have per-thread
   components (memory, SSD) and per-server fixed components (CPU die, NIC,
   chassis, rack). Reducing server count by 50% eliminates 50% of all
   components -- but under scaled resources, the remaining servers are *more
   expensive* because they host more vCPUs and need more memory/SSD.

2. **Power does not scale linearly with server count.** Fewer servers means
   less total idle power (good), but each server runs at higher utilization
   (more energy per server). The net operational carbon depends on the power
   curve shape and the utilization level.

3. **Physical resource limits cap savings.** On existing hardware, memory or SSD
   capacity may prevent you from reaching the desired R, creating a savings
   ceiling.

This analysis quantifies these effects using three resource modeling approaches
on the same no-SMT configuration.

## Model Configuration

**Config**: [`configs/oversub_analysis/genoa/linear/nosmt_oversub_sweep_constrained_vs_scaled_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/nosmt_oversub_sweep_constrained_vs_scaled_linear.jsonc)

**Output**: `results/oversub_analysis/genoa/linear/nosmt_oversub_sweep_constrained_vs_scaled_linear/`

### Scenarios

All scenarios use the no-SMT linear processor (80 cores, 1 thread/core, linear
CPU power curve) at 30% utilization with 100,000 vCPUs. The baseline is no-SMT
at R=1.0 with no resource adjustments.

| Scenario | Resource model | Description |
|---|---|---|
| `nosmt_no_oversub` | -- | **Baseline**: no oversubscription (R=1.0) |
| `nosmt_oversub_unscaled` | Fixed | Memory/SSD unchanged regardless of R |
| `nosmt_oversub_scaled` | Scaled | Memory/SSD grow proportionally with vCPU count |
| `nosmt_oversub_constrained` | Constrained | SMT-provisioned hardware (2x DIMMs/SSDs), capacity-limited |

The constrained scenario uses `nosmt_smt_hw_linear` -- a no-SMT processor spec
with the memory and SSD of the full SMT configuration (12 DIMMs, 6 SSDs instead
of 6/3). This models disabling SMT on existing servers without reprovisioning.

### Sweep

Oversubscription ratio R is swept from 1.0 to 5.0 across all three scenarios.
The analysis includes an **ideal 1/R scaling reference line** showing where
savings would fall if they tracked perfectly with server reduction.

### How to run

```bash
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/nosmt_oversub_sweep_constrained_vs_scaled_linear.jsonc
```

## Results

### Fixed Resources (Upper Bound)

| R | Server % | Carbon % | TCO % |
|---|---|---|---|
| 1.0 | +0.0% | +0.0% | +0.0% |
| 1.5 | -33.3% | -24.1% | -28.2% |
| 2.0 | -50.0% | -37.1% | -42.9% |
| 3.0 | -66.7% | -51.9% | -58.5% |
| 5.0 | -80.0% | -70.1% | -74.5% |

Carbon and TCO savings track below the server reduction. At R=2.0, servers drop
50% but carbon drops only 37%. The gap comes from operational carbon: fewer
servers run at higher utilization, so per-server energy increases. The gap is
larger for carbon than TCO because TCO includes embodied cost (which tracks
server count more closely).

### Scaled Resources (Realistic)

| R | Server % | Carbon % | TCO % |
|---|---|---|---|
| 1.0 | +0.0% | +0.0% | +0.0% |
| 1.5 | -33.3% | -16.8% | -20.2% |
| 2.0 | -50.0% | -23.8% | -28.6% |
| 3.0 | -66.7% | -31.8% | -37.6% |
| 5.0 | -80.0% | -45.1% | -48.6% |

Savings are significantly lower than the fixed-resource case. At R=2.0, carbon
savings are -23.8% vs -37.1% (a 13 pp difference). At R=5.0, the gap widens to
25 pp on carbon.

The erosion comes from memory and SSD scaling. Each remaining server needs more
DIMMs and SSDs (higher embodied carbon, higher power draw). The per-server fixed
components (CPU die, NIC, chassis) are eliminated with each removed server, but
the per-vCPU components (memory, SSD) are just redistributed to the surviving
servers.

### Constrained (Existing Hardware, Hard Ceiling)

| R | Server % | Carbon % | TCO % | Eff. R | Bottleneck |
|---|---|---|---|---|---|
| 1.0 | +0.0% | +28.1% | +32.8% | 1.00 | cores |
| 1.5 | -33.3% | -3.4% | -5.3% | 1.50 | cores |
| 2.0 | -50.0% | -20.5% | -25.1% | 2.00 | cores |
| 2.5 | -60.0% | -31.8% | -37.5% | 2.50 | cores |
| 3.0 | -62.5% | -34.9% | -40.7% | 2.67 | memory |
| 3.5-5.0 | -62.5% | -34.9% | -40.7% | 2.67 | memory |

At R=1.0, the constrained scenario is *worse* than baseline (+28% carbon)
because the server has the full SMT complement of DIMMs and SSDs (higher
embodied carbon and power) but only half the HW threads. This overhead makes
sense -- it represents the cost of unused hardware capacity on an existing
SMT-provisioned server.

The savings grow with R until **memory becomes the bottleneck at R=2.67**. Beyond
that, the effective oversubscription is capped: 80 HW threads x 9.6 GB/thread =
768 GB total memory, at 4 GB/vCPU demand = 192 max vCPUs, with 72 available
pCPUs -> effective R = 192/72 = 2.67. Adding more requested oversubscription
does nothing.

The maximum savings plateau at -34.9% carbon and -40.7% TCO -- a hard ceiling
set by the physical hardware.

### What the Three Curves Show Together

```
Savings at R=2.0:
  Fixed:        -37.1% carbon,  -42.9% TCO
  Scaled:       -23.8% carbon,  -28.6% TCO
  Constrained:  -20.5% carbon,  -25.1% TCO

Savings at R=5.0:
  Fixed:        -70.1% carbon,  -74.5% TCO
  Scaled:       -45.1% carbon,  -48.6% TCO
  Constrained:  -34.9% carbon,  -40.7% TCO  (capped at R=2.67)
```

1. **The gap between fixed and scaled quantifies the resource cost of
   oversubscription.** This is "real money" -- additional DIMMs and SSDs that
   must be purchased and powered. At R=2.0 it costs ~13 pp of carbon savings;
   at R=5.0 it costs ~25 pp.

2. **The constrained curve shows the transition-planning ceiling.** If deploying
   no-SMT on existing SMT hardware (without reprovisioning), savings are capped
   at ~35% carbon / ~41% TCO regardless of how aggressively you oversubscribe
   CPU.

3. **All three curves are sublinear relative to server reduction.** Even the
   fixed-resource case (no resource cost growth) shows carbon savings below the
   ideal 1/R line, because operational carbon (energy) does not scale linearly
   with server count -- fewer servers run hotter.

### Ideal 1/R Scaling vs Reality

The ideal savings line represents `-(1 - 1/R) x 100%`: at R=2.0, ideal savings
are -50%; at R=5.0, -80%. This is the theoretical maximum if all costs scaled
linearly with server count.

The plots include this ideal line as a reference. All three resource modeling
approaches fall below it, with the gap reflecting:

- **Fixed**: Only operational carbon drag (higher utilization per server)
- **Scaled**: Operational drag + embodied cost redistribution (more DIMMs/SSDs
  per server)
- **Constrained**: All of the above + capacity ceiling

## Interpretation

### Why This Matters for the SMT vs No-SMT Analysis

The main analysis ([02](02_scheduling_constraints_oversub.md),
[03](03_vcpu_demand_discount.md)) compares no-SMT to SMT at specific R values.
The savings numbers in those documents use a particular resource modeling choice
(typically scaled for projected savings). This side analysis shows what those
numbers would look like under different assumptions:

- **Fixed resources overstate savings** by 5-25 pp depending on R. Results using
  fixed resources should be read as optimistic upper bounds.
- **Scaled resources are the appropriate baseline** for projecting real
  deployment costs where memory and SSD are provisioned per-VM.
- **Constrained resources apply to transition scenarios** where existing hardware
  is reused.

### Diminishing Returns

Even under the most optimistic (fixed) model, the savings curve is concave:
going from R=1.0 to R=2.0 captures 37% carbon savings, but going from R=2.0 to
R=5.0 adds only 33 pp more. Under scaled resources, the concavity is steeper:
the first doubling captures 24% and the next 2.5x adds only 21 pp. Each
incremental unit of oversubscription delivers less marginal savings.

This has practical implications: the analysis in [02](02_scheduling_constraints_oversub.md)
shows that experimentally safe R values for no-SMT range from ~1.6 (30% util) to
~5.0 (10% util). The diminishing returns mean that the exact R value matters
less at the high end -- the difference between R=3.0 and R=5.0 is only ~13 pp
carbon under scaled resources.

### The Resource Constraint as a Reality Check

The constrained scenario serves as a reality check for transition planning. If
an operator wants to test no-SMT by disabling SMT on existing servers:

- Savings are available up to R~2.67 (the memory ceiling)
- Maximum savings are ~35% carbon / ~41% TCO vs no-oversub no-SMT
- Beyond the memory ceiling, SSD has significant stranded capacity (memory
  becomes the sole bottleneck)
- Reprovisioning servers with right-sized memory would unlock the "scaled" curve

## What This Does Not Address

1. **SMT comparison**: This analysis is purely intra-no-SMT (no-SMT at various R
   values vs no-SMT at R=1.0). The SMT vs no-SMT comparison at specific R values
   is in [02](02_scheduling_constraints_oversub.md) and
   [03](03_vcpu_demand_discount.md).

2. **Utilization sensitivity**: All results are at 30% average utilization. The
   shape of the savings curve varies with utilization (higher utilization means
   more operational carbon drag, widening the gap between server reduction and
   carbon/TCO savings).

3. **vCPU demand discount interaction**: The sweep varies only R, holding
   `vcpu_demand_multiplier` at 1.0. In the full analysis, demand compression
   effectively shifts where on the R curve you operate.
