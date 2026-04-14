# 03a: Constrained Savings (Same Hardware, SMT Disabled)

## Question

> If instead of purpose-building no-SMT servers, you disable SMT on existing
> SMT-provisioned hardware, how do the vCPU demand discount savings from
> [03](03_vcpu_demand_discount.md) change?

Document [03](03_vcpu_demand_discount.md) projects no-SMT savings using
**scaled resources** -- purpose-built servers where memory and SSD are sized to
the vCPU count. This document repeats that analysis with **resource constraints**:
both SMT and no-SMT run on the same physical hardware (12 DIMMs, 6 SSDs), and
memory/SSD capacity limits how many vCPUs each server can host.

This is the more conservative deployment scenario: an operator disables SMT on
existing servers without reprovisioning, and the physical resource capacities may
cap oversubscription below what CPU alone could support.

## Prerequisites

- [02a: Resource Modeling](02a_resource_modeling.md) for the distinction between
  scaled and constrained resource approaches
- [03: vCPU Demand Discount](03_vcpu_demand_discount.md) for the unconstrained
  savings results this document compares against

## Key Assumptions

### Same Physical Hardware

Both SMT and no-SMT scenarios use identical physical servers: 80-core Genoa with
12 DIMMs (768 GB) and 6 SSDs (12 TB). Disabling SMT halves the HW threads
(160 -> 80) but does not change the installed memory or storage. This means:

- **No-SMT servers have more resource headroom per thread**: 9.6 GB memory per
  HW thread vs 4.8 GB under SMT (same total, half the threads).
- **No-SMT servers have higher per-server embodied carbon/cost**: The full 12
  DIMMs and 6 SSDs are present regardless of SMT mode, so per-server carbon is
  ~1,783 kg (same as SMT) rather than the ~1,120 kg of a purpose-built no-SMT
  server with 6 DIMMs / 3 SSDs.
- **Both configurations are subject to the same resource constraints**: Memory
  caps at 768 GB / demand_per_vcpu max vCPUs; SSD caps at 12,000 GB /
  demand_per_vcpu max vCPUs.

### Resource Demand Ratios

The resource constraints assume a fixed per-vCPU demand:

| Resource | Capacity (total) | Demand per vCPU | Max vCPUs per server |
|---|---|---|---|
| Memory | 768 GB (12 x 64 GB DIMMs) | 4.0 GB | 192 |
| SSD | 12,000 GB (6 x 2 TB SSDs) | 50 GB | 240 |

The **memory:vCPU ratio of 4 GB** is consistent with general-purpose cloud
instance sizing (e.g., AWS m-type instances allocate ~4 GB per vCPU). The
**SSD:vCPU ratio of 50 GB** reflects local storage for general workloads.

**TODO**: These demand ratios are modeling assumptions, not experimentally
derived. A production analysis should validate them against actual fleet VM size
distributions. The memory ratio in particular is critical -- lower demand (e.g.,
2 GB/vCPU for compute-optimized instances) would raise the memory ceiling and
allow more oversubscription; higher demand (e.g., 8 GB/vCPU for memory-optimized
instances) would lower it.

Memory is the binding constraint in all cases where constraints bite: 192 max
vCPUs (memory) < 240 max vCPUs (SSD). SSD has 25% stranded capacity at the
memory bottleneck.

### How Constraints Interact with Oversubscription

The effective oversubscription ratio is the minimum of the core limit and all
resource limits:

| Config | Util | Requested R | Core limit | Memory limit | Effective R | Constrained? |
|---|---|---|---|---|---|---|
| SMT 10% | 10% | 3.0 | 144 x 3.0 = 432 | 192 | **192/144 = 1.33** | Yes (memory) |
| No-SMT 10% | 10% | 5.0 | 72 x 5.0 = 360 | 192 | **192/72 = 2.67** | Yes (memory) |
| SMT 20% | 20% | 1.33 | 144 x 1.33 = 192 | 192 | **1.33** | No (barely) |
| No-SMT 20% | 20% | 2.34 | 72 x 2.34 = 169 | 192 | **2.34** | No |
| SMT 30% | 30% | 1.0 | 144 x 1.0 = 144 | 192 | **1.0** | No |
| No-SMT 30% | 30% | 1.67 | 72 x 1.67 = 120 | 192 | **1.67** | No |

At **10% utilization**, memory constrains *both* configurations heavily. SMT
drops from R=3.0 to effective R=1.33 (a 56% reduction in oversubscription); no-SMT
drops from R=5.0 to effective R=2.67 (a 47% reduction). This asymmetry -- memory
hitting SMT proportionally harder -- is the key dynamic at low utilization.

At **20% and 30% utilization**, neither configuration is memory-constrained
because the requested R values are modest enough that the core limit is below the
memory ceiling. However, the no-SMT servers still carry the full embodied cost
of 12 DIMMs / 6 SSDs, which makes them more expensive per-server than purpose-built
no-SMT servers.

## Model Configuration

### Per-Utilization Sweep Configs

Each config compares SMT vs no-SMT on the same hardware, sweeping
`vcpu_demand_multiplier`:

| Utilization | SMT R | No-SMT R | Config | Output |
|---|---|---|---|---|
| 10% | 3.0 | 5.0 | [`util_10_pct_linear_resource_constraints.jsonc`][cfg-10] | `results/.../util_10pct_linear_resource_constraints/` |
| 20% | 1.33 | 2.34 | [`util_20_pct_linear_resource_constraints.jsonc`][cfg-20] | `results/.../util_20pct_linear_resource_constraints/` |
| 30% | 1.0 | 1.67 | [`util_30_pct_linear_resource_constraints.jsonc`][cfg-30] | `results/.../util_30pct_linear_resource_constraints/` |

[cfg-10]: ../../configs/oversub_analysis/genoa/linear/util_10_pct_linear_resource_constraints.jsonc
[cfg-20]: ../../configs/oversub_analysis/genoa/linear/util_20_pct_linear_resource_constraints.jsonc
[cfg-30]: ../../configs/oversub_analysis/genoa/linear/util_30_pct_linear_resource_constraints.jsonc

### Savings Curve (Aggregate)

Aggregates the geomean-discount savings from each utilization point, comparing
constrained vs unconstrained (scaled):

**Config**: [`configs/oversub_analysis/genoa/linear/savings_curve_constrained_vs_unconstrained.jsonc`](../../configs/oversub_analysis/genoa/linear/savings_curve_constrained_vs_unconstrained.jsonc)

**Output**: `results/oversub_analysis/genoa/linear/savings_curve_constrained_vs_unconstrained/`

### How to run

```bash
# Per-utilization sweeps
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_10_pct_linear_resource_constraints.jsonc
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_20_pct_linear_resource_constraints.jsonc
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_30_pct_linear_resource_constraints.jsonc

# Savings curve comparison
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/savings_curve_constrained_vs_unconstrained.jsonc
```

## Results

### Carbon Breakeven: Constrained vs Scaled

The breakeven `vcpu_demand_multiplier` is the value at which no-SMT matches SMT
on carbon:

| Utilization | Breakeven (scaled) | Breakeven (constrained) | Shift |
|---|---|---|---|
| 10% | 0.847 | 0.899 | +0.052 (easier) |
| 20% | 0.844 | 0.799 | -0.045 (harder) |
| 30% | 0.859 | 0.769 | -0.090 (harder) |

**10% utilization**: Breakeven is *easier* under constraints (0.899 vs 0.847).
This is counterintuitive until you consider that memory constrains SMT more
severely than no-SMT at this utilization. SMT's requested R=3.0 gets capped to
effective R=1.33, destroying most of its oversubscription advantage. No-SMT's
R=5.0 is capped to R=2.67, which still leaves it with a 2:1 effective R
advantage over SMT.

**20% and 30% utilization**: Breakeven is *harder* under constraints (lower
multiplier needed). Neither configuration is memory-constrained at these R
values, but the no-SMT server carries the full cost of 12 DIMMs / 6 SSDs.
The higher per-server embodied carbon (~1,783 kg vs ~1,120 kg for purpose-built)
penalizes no-SMT and requires a larger demand discount to break even.

### Savings at Geomean Discount (multiplier=0.75)

| Utilization | Scaled Carbon | Constrained Carbon | Scaled TCO | Constrained TCO |
|---|---|---|---|---|
| 10% | **-11.5%** | **-16.5%** | **-16.3%** | **-20.8%** |
| 20% | **-11.1%** | **-6.1%** | **-15.6%** | **-10.0%** |
| 30% | **-12.7%** | **-2.5%** | **-17.2%** | **-6.0%** |

**10% utilization**: Constrained savings are *larger* than scaled (-16.5% vs
-11.5% carbon). Memory constraining SMT is the dominant effect here -- it
eliminates SMT's ability to oversubscribe beyond R=1.33, which levels the playing
field.

**20% utilization**: Constrained savings are about half the scaled savings (-6.1%
vs -11.1% carbon). No-SMT still saves, but the extra embodied cost of carrying
full SMT hardware erodes the benefit.

**30% utilization**: Constrained savings are marginal (-2.5% carbon, -6.0% TCO).
At this utilization, the oversubscription ratios are modest and no-SMT's higher
per-server cost nearly offsets the server count reduction.

### Savings at Other Discount Levels

**At "low" discount (multiplier=0.65, ~1.54x perf ratio):**

| Utilization | Scaled Carbon | Constrained Carbon | Scaled TCO | Constrained TCO |
|---|---|---|---|---|
| 10% | -23.3% | **-27.6%** | -27.5% | **-31.3%** |
| 20% | -23.0% | **-18.7%** | -27.0% | **-22.1%** |
| 30% | -24.3% | **-15.4%** | -28.2% | **-18.5%** |

**At "high" discount (multiplier=0.85, ~1.18x perf ratio):**

| Utilization | Scaled Carbon | Constrained Carbon | Scaled TCO | Constrained TCO |
|---|---|---|---|---|
| 10% | +0.4% | **-5.4%** | -5.1% | **-10.2%** |
| 20% | +0.7% | **+6.4%** | -4.5% | **+1.9%** |
| 30% | -1.1% | **+10.5%** | -6.2% | **+6.6%** |

At the high (weak) discount level, the constrained model shows no-SMT losing at
20% and 30% utilization (+6.4% and +10.5% carbon). The extra per-server embodied
cost of carrying full SMT hardware overwhelms the modest demand compression.

### The Cross-Over at Low Utilization

The most notable result is the role reversal at 10% utilization: constraints
**help** no-SMT rather than hurting it. This happens because:

1. At 10% utilization, both SMT and no-SMT can theoretically oversubscribe
   aggressively (R=3.0 and R=5.0 respectively).
2. Memory caps both to much lower effective R values (1.33 and 2.67).
3. But the *relative* impact is asymmetric: SMT loses 56% of its requested
   oversubscription while no-SMT loses 47%.
4. The result: no-SMT goes from a 2:1 effective-R advantage (5.0/3.0 = 1.67x
   unconstrained) to a 2:1 advantage (2.67/1.33 = 2.01x constrained) -- the
   constraint actually *widens* no-SMT's relative advantage.

This effect is specific to the 4 GB/vCPU memory demand ratio. A lower ratio
would raise the memory ceiling and reduce the constraint effect; a higher ratio
would lower it further and potentially constrain even the 20% and 30% configs.

## Interpretation

### Practical Implications

The constrained results are directly relevant to **transition planning**: an
operator who wants to evaluate no-SMT by disabling SMT on existing servers
(before committing to purpose-built hardware) can expect:

- **At low utilization (10%)**: Larger savings than the purpose-built projection,
  because memory constraints equalize the playing field. This is a favorable
  result for testing -- the easiest deployment scenario produces the best
  numbers.
- **At moderate utilization (20-30%)**: Smaller savings than purpose-built, with
  the gap widening as utilization increases. The extra per-server embodied cost
  of carrying unused resources erodes the benefit.
- **At geomean discount across all utilization levels**: Savings range from -2.5%
  to -16.5% on carbon and -6.0% to -20.8% on TCO. No-SMT saves everywhere, but
  the magnitude varies significantly.

### When to Purpose-Build vs Reuse Hardware

The savings curve comparison shows that purpose-built servers (scaled) deliver
**more consistent** savings across utilization levels (roughly -11% to -13%
carbon at geomean discount), while constrained servers show high variance
(-2.5% to -16.5%). If the fleet operates at a mix of utilization levels, the
purpose-built approach has a more predictable ROI.

However, purpose-building requires upfront investment in right-sized hardware.
The constrained results show that **reusing existing hardware still delivers
savings** at the geomean discount level, making it a viable starting point
before committing to purpose-built no-SMT servers.

### Connection to [02b: Savings Scaling](02b_oversub_savings_scaling.md)

Document [02b](02b_oversub_savings_scaling.md) shows how intra-no-SMT savings
scale with R under constraints (the savings ceiling at R=2.67). This document
shows the inter-configuration effect: how constraints change the *relative*
comparison between SMT and no-SMT. The two effects interact -- constraints limit
absolute savings for both, but the relative impact depends on which configuration
is constrained more heavily.

### Per-Server Resource Packing: Why Constraints Are Asymmetric

The fleet-level results above are driven by per-server resource utilization. Two
resource packing analyses visualize this:

**Cross-configuration comparison** (SMT and no-SMT, scaled vs constrained, at
each utilization level):

- **Config**: [`configs/oversub_analysis/genoa/linear/resource_packing_constrained_vs_unconstrained.jsonc`](../../configs/oversub_analysis/genoa/linear/resource_packing_constrained_vs_unconstrained.jsonc)
- **Output**: `results/oversub_analysis/genoa/linear/resource_packing_constrained_vs_unconstrained/`

```bash
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/resource_packing_constrained_vs_unconstrained.jsonc
```

**At 10% utilization (where constraints help no-SMT):**

| Scenario | Req. R | Eff. R | vCPUs/server | Memory used | Memory capacity | Mem stranded | Bottleneck |
|---|---|---|---|---|---|---|---|
| SMT scaled | 3.0 | 3.0 | 432 | 1,728 GB | 2,074 GB | 16.7% | cores |
| SMT constrained | 3.0 | **1.33** | 192 | 768 GB | 768 GB | 0.0% | **memory** |
| NoSMT scaled | 5.0 | 5.0 | 360 | 1,440 GB | 1,728 GB | 16.7% | cores |
| NoSMT constrained | 5.0 | **2.67** | 192 | 768 GB | 768 GB | 0.0% | **memory** |

Both SMT and no-SMT hit the same 768 GB memory ceiling (same physical hardware).
But the impact is starkly different: SMT drops from 432 to 192 vCPUs/server
(-56%), while no-SMT drops from 360 to 192 (-47%). This means constrained SMT
and constrained no-SMT host the same number of vCPUs per server (192), which
eliminates SMT's LP-count advantage entirely. At that point, no-SMT wins on
power efficiency and demand discount.

**At 20% utilization (where constraints don't bind):**

| Scenario | Req. R | Eff. R | vCPUs/server | Memory used | Memory capacity | Mem stranded |
|---|---|---|---|---|---|---|
| SMT constrained | 1.33 | 1.33 | 192 | 766 GB | 768 GB | 0.2% |
| NoSMT constrained | 2.34 | 2.34 | 168 | 674 GB | 768 GB | 12.2% |

Neither is memory-constrained (core limit is below memory limit for both). But
no-SMT has 12% stranded memory on a server that costs 2x the per-thread embodied
carbon of a purpose-built no-SMT server (8.87 vs 4.43 kg/thread for memory).
This stranded-resource penalty explains the lower savings at 20% and 30%.

**At 30% utilization (where stranded resources are largest):**

| Scenario | Req. R | Eff. R | vCPUs/server | Memory used | Memory capacity | Mem stranded | SSD stranded |
|---|---|---|---|---|---|---|---|
| SMT constrained | 1.0 | 1.0 | 144 | 576 GB | 768 GB | 25.0% | 40.0% |
| NoSMT constrained | 1.67 | 1.67 | 120 | 481 GB | 768 GB | **37.4%** | **49.9%** |

At low R, the no-SMT constrained server has nearly 50% stranded SSD and 37%
stranded memory. It is paying the full embodied cost of 12 DIMMs and 6 SSDs
but using barely half of them.

**SMT oversubscription sweep** (how early does memory constrain SMT?):

- **Config**: [`configs/oversub_analysis/genoa/linear/resource_packing_oversub_sweep_smt.jsonc`](../../configs/oversub_analysis/genoa/linear/resource_packing_oversub_sweep_smt.jsonc)
- **Output**: `results/oversub_analysis/genoa/linear/resource_packing_oversub_sweep_smt/`

```bash
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/resource_packing_oversub_sweep_smt.jsonc
```

| R | SMT scaled vCPUs/server | SMT constrained vCPUs/server | Constrained Eff. R | Bottleneck |
|---|---|---|---|---|
| 1.0 | 144 | 144 | 1.0 | cores |
| 1.5 | 216 | **173** | **1.20** | **memory** |
| 2.0 | 288 | 173 | 1.20 | memory |
| 3.0 | 432 | 173 | 1.20 | memory |

SMT hits the memory ceiling at **R=1.20** -- just above no-oversubscription.
Any requested R beyond 1.2 is wasted because memory cannot support more vCPUs.
This is much more constraining than for no-SMT, which hits the ceiling at R=2.67.
The difference arises because SMT's memory capacity per pCPU is lower: 768 GB /
144 pCPUs = 5.33 GB/pCPU for SMT vs 768 GB / 72 pCPUs = 10.67 GB/pCPU for
no-SMT. Since each vCPU demands 4 GB, SMT can only oversubscribe CPU by 5.33/4.0
= 1.33x before memory runs out, while no-SMT can go to 10.67/4.0 = 2.67x.

This asymmetry is the structural reason why memory constraints favor no-SMT at
low utilization: SMT has more HW threads sharing the same memory pool, so it
exhausts memory sooner when trying to oversubscribe.

## What This Does Not Address

1. **Mixed hardware fleets**: An operator might keep some SMT servers and convert
   others. The composite scenario analysis explores heterogeneous fleet
   optimization.

2. **Partial reprovisioning**: Instead of full purpose-built or full reuse, an
   operator might add DIMMs to existing servers to raise the memory ceiling.
   The model does not currently support partial reprovisioning, but the
   effect can be approximated by adjusting `capacity_per_thread`.

3. **Time-varying utilization**: A server that runs at 10% utilization at night
   and 30% during the day would see different constraint effects at different
   times. The steady-state model averages over this.

4. **Application-specific resource ratios**: The 4 GB/vCPU memory ratio is a
   fleet average. CPU-intensive workloads with small memory footprints would be
   less constrained; memory-intensive workloads would be more constrained.
