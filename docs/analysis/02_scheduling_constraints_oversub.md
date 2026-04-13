# 02: Scheduling Constraints and Oversubscription

## Question

> If VP scheduling constraints limit how much SMT can oversubscribe, but no-SMT
> can oversubscribe more freely, does the higher no-SMT oversubscription headroom
> close the carbon/TCO gap from [01](01_naive_comparison.md)?

This is the first analysis that incorporates **experimentally measured** data. The
oversubscription ratios used here are not theoretical -- they come from measured
steal-time thresholds on real hardware running real applications.

## Prerequisites

- [01: Naive Comparison](01_naive_comparison.md) for the baseline penalty
- [Spine](SMT_VS_NOSMT_ANALYSIS.md) for terminology

## Key Assumptions

Relative to [01](01_naive_comparison.md), this analysis **relaxes** the
no-oversubscription assumption:

1. **Oversubscription is allowed**, with R values determined by experimental steal
   time measurements at a 1% steal threshold.
2. **VP constraints apply to SMT**: The SMT configuration uses core scheduling
   with topology-aware VP constraints (the cloud-realistic minimum).
3. **No constraints for no-SMT**: Since each LP is the sole thread on its physical
   core, core scheduling is unnecessary.
4. **Linear CPU power curve for no-SMT**: Based on the experimental finding from
   [01](01_naive_comparison.md) that no-SMT power scales more linearly.
5. **No vCPU demand discount yet**: Total vCPU demand is still the same for both
   configurations (relaxed in [03](03_vcpu_demand_discount.md)).

## Experimental Inputs

### Where the Oversubscription Ratios Come From

The oversubscription ratios are derived from the **iso-physical-core** experiment
documented in [`scheduling_constraints_smt_iso_physical_core.md`][exp-iso]. That
experiment measures the maximum safe VP/LP rate at a 1% steal threshold for
go-cpu under three regimes:

- **VP Constraints (8 LP / 2 VM)**: SMT with topology-aware core scheduling
- **VP Constraints (16 LP / 4 VM)**: SMT with same physical cores, full LP
  exposure
- **No-SMT (8 LP / 2 VM)**: No constraints needed

The operating point tables in that document give max safe VP/LP rates at fixed
per-VM utilization targets. These VP/LP rates become oversubscription ratios (R)
in the model.

**Important note on methodology**: The configs here use discrete tested points
from the iso-physical-core operating point tables. A later refinement
interpolates the maximum oversubscription rate from the steal-time curves, which
produces slightly different values. The discrete-point values are conservative
(they take the last tested point before crossing the 1% steal threshold rather
than interpolating the exact crossing). This means the R values used here may
slightly understate the true safe oversubscription, especially at boundaries.

The specific R values used, from the Go-CPU operating point tables at each
utilization level:

| Avg Utilization | SMT R (VP 16LP) | No-SMT R | Source |
|---|---|---|---|
| 10% | 2.34 | 5.0 | Go-CPU VP 16LP and No-SMT max VP/LP at 10% util/VM |
| 20% | 1.33 | 2.34 | Go-CPU VP 16LP and No-SMT max VP/LP at 20% util/VM |
| 30% | 1.14 | 1.6 | Go-CPU VP 16LP and No-SMT max VP/LP at 30% util/VM |

**Why VP 16LP for SMT**: This analysis uses the pool-adjusted SMT regime (16 LPs
from 8 physical cores, 4 VMs) rather than the iso-LP regime (8 LPs, 2 VMs). This
is the operationally fair comparison: on the same 8 physical cores, SMT exposes
16 LPs while no-SMT exposes 8 LPs. The 16LP SMT rates are higher than the 8LP
rates, giving SMT the benefit of its larger LP pool.

**Why go-cpu**: Go-cpu was chosen as a representative CPU-bound workload. Its
steal-time behavior sits near the cross-application median (mean no-SMT/VP ratio
of 2.16x vs the 13-app mean of 2.34x). Different applications would yield
different R values; see the cross-application table in
[`scheduling_constraints_smt_impact.md`][exp-sched] for the full spread.

**Why no util_overhead for SMT**: The oversubscription ratios already embed the
constraint effect -- they are the maximum safe R at the 1% steal threshold *with*
VP constraints active. Adding a separate `util_overhead` would double-count. The
steal-time cost of VP constraints is captured in the *lower R* that SMT achieves,
not as an additive overhead on utilization.

[exp-iso]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_iso_physical_core.md
[exp-sched]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_impact.md

### Cross-Application Context

The go-cpu numbers are one point in a broad distribution. For context, the
iso-physical-core experiment shows these cross-application ranges at 10% util/VM:

| Regime | Mean max VP/LP | Min | Max |
|---|---|---|---|
| VP 8LP (SMT) | 2.10x | 1.26x (Elasticsearch) | 2.64x (Go-MemChase) |
| VP 16LP (SMT) | 2.94x | 1.66x (Elasticsearch) | 3.90x (TensorFlow) |
| No-SMT | 4.92x | 3.00x (Elasticsearch) | 7.93x (PgBench) |

Go-cpu sits at VP 16LP = 3.32x and No-SMT = 5.58x, which is above the mean but
not an outlier. Using go-cpu therefore gives a moderately favorable picture for
no-SMT compared to the cross-application average.

## Model Configuration

### Config files

| Utilization | Config | Output |
|---|---|---|
| 10% | [`configs/oversub_analysis/genoa/linear/util_10_pct_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/util_10_pct_linear.jsonc) | `results/oversub_analysis/genoa/linear/util_10pct_linear/` |
| 20% | [`configs/oversub_analysis/genoa/linear/util_20_pct_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/util_20_pct_linear.jsonc) | `results/oversub_analysis/genoa/linear/util_20pct_linear/` |
| 30% | [`configs/oversub_analysis/genoa/linear/util_30_pct_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/util_30_pct_linear.jsonc) | `results/oversub_analysis/genoa/linear/util_30pct_linear/` |

### Analysis type

All three configs use `compare_sweep`:

- **Baseline**: SMT with VP-constraint-limited R
- **Sweep scenario**: No-SMT linear with no-constraint R
- **Sweep parameter**: `vcpu_demand_multiplier` (0.5 to 1.5)
- **X-axis markers**: 0.65 ("low"), 0.75 ("geomean"), 0.85 ("high") -- these
  correspond to vCPU demand discount reference points from the peak performance
  experiments (see [03](03_vcpu_demand_discount.md))

Although the sweep parameter is `vcpu_demand_multiplier`, the key finding *in this
document* is the result at **multiplier = 1.0** (no demand discount), which
isolates the pure oversubscription effect. The full sweep is included to set up
[03](03_vcpu_demand_discount.md).

### How to run

```bash
# Run all three utilization points
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_10_pct_linear.jsonc
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_20_pct_linear.jsonc
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_30_pct_linear.jsonc
```

### Output location

Each produces:
```
results/oversub_analysis/genoa/linear/util_{10,20,30}pct_linear/
  summary.md            # Sweep table with % changes
  results.json          # Full results
  config.json           # Expanded config
  plots/
    compare_sweep_carbon.png   # Carbon % change vs vcpu_demand_multiplier
    compare_sweep_TCO.png      # TCO % change vs vcpu_demand_multiplier
```

## Results

### At vcpu_demand_multiplier = 1.0 (no demand discount)

This is the pure oversubscription-only comparison: same total vCPU demand, but
different R values reflecting scheduling-constraint-limited oversub.

| Utilization | SMT R | No-SMT R | No-SMT Carbon vs SMT | No-SMT TCO vs SMT | No-SMT Servers vs SMT |
|---|---|---|---|---|---|
| 10% | 2.34 | 5.0 | **+48.0%** | **+45.9%** | +99.9% |
| 20% | 1.33 | 2.34 | **-1.5%** | **-9.1%** | +13.6% |
| 30% | 1.14 | 1.6 | **+15.5%** | **+9.9%** | +42.5% |

### Interpretation of the Three Utilization Points

**10% utilization**: No-SMT still loses badly (+48% carbon). Even though no-SMT
can oversubscribe at R=5.0 vs SMT's R=2.34, the 2:1 LP count disadvantage still
dominates. At such low utilization, the high R values don't produce enough server
consolidation to overcome the fundamental capacity gap.

**20% utilization**: **Crossover point.** No-SMT is essentially at carbon parity
(-1.5%) and wins on TCO (-9.1%). At this utilization, the oversubscription
advantage (R=2.34 vs R=1.33) is enough to nearly match server counts (+13.6%
more no-SMT servers, down from +100% in [01](01_naive_comparison.md)), and the
lower no-SMT power curve offsets the remaining server count difference.

**30% utilization**: No-SMT loses again (+15.5% carbon), but the gap is much
smaller than the +47-56% from [01](01_naive_comparison.md). At 30% utilization,
both configurations have modest oversubscription (R=1.14 vs R=1.6), so the LP
count disadvantage partially reasserts itself.

### Carbon Breakeven vCPU Demand Multiplier

The sweep results also report the `vcpu_demand_multiplier` at which no-SMT
breaks even on carbon. This previews the analysis in
[03](03_vcpu_demand_discount.md):

| Utilization | Carbon Breakeven Multiplier | Implied Discount |
|---|---|---|
| 10% | 0.676 | 32.4% |
| 20% | 1.016 | -1.6% (already at parity) |
| 30% | 0.866 | 13.4% |

At 20% utilization, no-SMT already breaks even without any demand discount. At
10% and 30%, a discount of 13-32% is needed. The experimental peak performance
data (geomean ~26.5% discount across 30 apps) suggests these breakeven points are
often achievable.

### The Utilization Dependence

The non-monotonic behavior across utilization levels comes from two competing
effects:

1. **Oversubscription headroom shrinks with utilization**: At 10%, both configs
   can oversubscribe aggressively (R=2.34 and 5.0). At 30%, headroom is much
   smaller (R=1.14 and 1.6). The *ratio* of no-SMT R to SMT R also varies:
   - 10%: 5.0/2.34 = 2.14x
   - 20%: 2.34/1.33 = 1.76x
   - 30%: 1.6/1.14 = 1.40x

2. **Operational carbon becomes more important at higher utilization**: At 10%,
   servers are mostly idle, so embodied carbon (driven by server count) dominates.
   At 30%, operational carbon grows and the power-curve advantage of no-SMT
   becomes more relevant.

The sweet spot for no-SMT is at moderate utilization (~20%) where the
oversubscription ratio advantage is substantial but not yet squeezed by
diminishing headroom.

## What This Does Not Address

1. **vCPU demand compression**: The results at multiplier=1.0 assume identical
   demand. In reality, no-SMT LPs deliver more performance, so users may need
   fewer vCPUs. This is the single biggest factor that could shift the 10% and
   30% utilization results in no-SMT's favor. See
   [03: vCPU Demand Discount](03_vcpu_demand_discount.md).

2. **Application diversity**: The R values come from go-cpu only. Different
   applications have different steal-time profiles and would yield different R
   values. A production fleet would see a mix.

3. **Resource constraints**: When running no-SMT at high R on existing SMT
   hardware, memory or SSD capacity may limit effective packing. See
   [02a: Resource Modeling](02a_resource_modeling.md) for the three resource
   modeling approaches and [02b: Oversubscription Savings Scaling](02b_oversub_savings_scaling.md)
   for how savings scale with R under each approach.

4. **Non-linear power curves for no-SMT**: These configs use `genoa_nosmt_linear`
   (linear CPU power). The `non-linear/` directory contains equivalent configs
   using the default polynomial curve; those show a slightly larger no-SMT
   penalty (since no-SMT loses its power-proportionality advantage).
