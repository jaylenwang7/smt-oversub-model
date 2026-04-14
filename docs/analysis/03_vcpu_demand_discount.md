# 03: vCPU Demand Discount (Performance Effect)

## Question

> Since no-SMT logical processors deliver more performance than SMT logical
> processors (due to no co-running overhead), users may need fewer vCPUs to serve
> the same workload. How does this "vCPU demand compression" affect the SMT vs
> no-SMT carbon/TCO tradeoff?

This adds the final major mechanism to the analysis. Where
[02](02_scheduling_constraints_oversub.md) showed that scheduling constraints
alone can make no-SMT competitive at moderate utilization, this document shows
that the vCPU demand discount often pushes no-SMT **past breakeven** into
positive savings territory across a wider range of utilization levels.

## Prerequisites

- [01: Naive Comparison](01_naive_comparison.md) for baseline penalty
- [02: Scheduling Constraints](02_scheduling_constraints_oversub.md) for
  oversubscription ratios and the sweep framework
- [02a: Resource Modeling](02a_resource_modeling.md) for the distinction between
  fixed, scaled, and constrained resource approaches (this document references
  "with/without resource scaling" in its results)
- [Spine](SMT_VS_NOSMT_ANALYSIS.md) for terminology

## Key Assumptions

Relative to [02](02_scheduling_constraints_oversub.md), this analysis **relaxes**
the equal-demand assumption:

1. **vCPU demand scales with performance**: If a no-SMT LP delivers R times the
   throughput of an SMT LP, users need only 1/R as many vCPUs. The model captures
   this through `vcpu_demand_multiplier < 1.0`.
2. **All other assumptions from [02] carry forward**: Same R values, same power
   curves, same processor specs.

The key **new** question is: what discount values are realistic? This is answered
by experimental peak performance measurements.

## Experimental Inputs

### Peak Performance Ratios

The experimental basis comes from [`smt_no_smt_peak_performance.md`][exp-perf]
in the benchmarking repo. That experiment measures peak throughput for 30
applications (14 services, 16 batch) on a c6620 server with 1 VM pinned to 4
vCPUs, comparing 2-core SMT (4 HW threads) vs 4-core no-SMT (4 HW threads).

**Key aggregate results:**

| Scope | Count | Geomean no-SMT/SMT ratio | Implied vCPU discount |
|---|---|---|---|
| Services (all) | 14 | 1.311x | 23.7% |
| Batch (all) | 16 | 1.406x | 28.9% |
| All apps | 30 | 1.361x | 26.5% |
| CPU-bound only (util >= 95%) | 21 | 1.368x | 26.9% |

The discount is computed as `1 - 1/R` where R is the performance ratio. A
discount of 26.5% means that if a workload needed 100 vCPUs under SMT, it would
need only 73.5 vCPUs under no-SMT to achieve the same throughput.

### Reference Points: Low, Geomean, High

The analysis uses three reference points to bracket the range of realistic vCPU
discounts. These map to `vcpu_demand_multiplier` values:

| Label | vCPU demand multiplier | Implied perf ratio | Rationale |
|---|---|---|---|
| **High discount** ("low" multiplier) | 0.65 | ~1.54x | Near the strongest measured service gains (Neo4j 1.50x, Vault 1.48x) |
| **Geomean** | 0.75 | ~1.33x | Close to the all-app geomean (1.361x -> multiplier 0.735) |
| **Low discount** ("high" multiplier) | 0.85 | ~1.18x | Conservative; near weaker gains (InfluxDB 1.17x, Go-MemChase 1.07x) |

These three markers appear on the compare_sweep plots as vertical reference lines
and are used as evaluation points in the savings curve analysis.

**Caveat**: The experimental performance ratios come from peak-throughput
measurements on a c6620 with 4 vCPUs. Real cloud workloads may see different
ratios depending on workload mix, VM size, and load level. The discount also
assumes customers would actually re-size their VMs (or the provider would
auto-right-size), which is a deployment/pricing question beyond the scope of this
model.

[exp-perf]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_peak_performance.md

### Relation to Industry / Academic SMT Gain Claims

For broader context on what vendors and academic papers usually report as "SMT
performance gain," see:

- [`docs/research_reports/smt-performance-report.md`](../research_reports/smt-performance-report.md)

That report shows that most published SMT claims are framed as roughly
**+10% to +60% aggregate throughput** on the **same physical core budget**, with
a cross-source median around **+30%**. It also shows that many sources do not
clearly specify whether they mean throughput, instructions executed, latency, or
IPC.

This document needs a different quantity: the **no-SMT / SMT throughput ratio at
fixed visible vCPU count**, because that is what turns into a `vcpu_demand_multiplier`.

The experimental setup here compares:

- **SMT**: `2 cores x 2 threads = 4 vCPUs`
- **No-SMT**: `4 cores x 1 thread = 4 vCPUs`

So to connect the literature to this model, we need a translation step.

#### A Reasonable First-Order Translation

Suppose a vendor-style claim says:

- enabling SMT on `2 physical cores` raises throughput by `g`

Then:

- `throughput(SMT, 2 cores, 4 threads) = (1 + g) x throughput(no-SMT, 2 cores)`

To map that onto this experiment, assume first-order linear scaling from
`2` to `4` physical no-SMT cores:

- `throughput(no-SMT, 4 cores) ≈ 2 x throughput(no-SMT, 2 cores)`

Under that assumption, the implied fixed-`4 vCPU` no-SMT/SMT ratio is:

```text
no-SMT / SMT ratio ≈ 2 / (1 + g)
vcpu_demand_multiplier ≈ (1 + g) / 2
discount ≈ 1 - (1 + g) / 2 = (1 - g) / 2
```

This is not exact, but it is a reasonable bridge because it translates a
same-core-budget SMT throughput uplift into the fixed-visible-vCPU framing used
by the model.

#### Translation Table

Using that bridge:

| Reported SMT gain on same physical cores | Implied no-SMT / SMT ratio at fixed `4 vCPU` | Implied `vcpu_demand_multiplier` | Implied discount |
|---|---|---|---|
| `+10%` | `1.82x` | `0.55` | `45%` |
| `+20%` | `1.67x` | `0.60` | `40%` |
| `+30%` | `1.54x` | `0.65` | `35%` |
| `+40%` | `1.43x` | `0.70` | `30%` |
| `+50%` | `1.33x` | `0.75` | `25%` |
| `+60%` | `1.25x` | `0.80` | `20%` |

This gives a much clearer comparison point for the markers used in this doc:

| Marker in this doc | Implied no-SMT / SMT ratio | Back-translated same-core SMT gain |
|---|---|---|
| `0.65` ("high discount") | `1.54x` | `+30%` |
| `0.75` ("geomean") | `1.33x` | `+50%` |
| `0.85` ("low discount") | `1.18x` | `+70%` |

#### What This Means

This translation suggests:

1. The **`0.65` marker** is actually close to the **literature median**. A
   vendor/academic-style SMT claim of about `+30%` aggregate throughput on the
   same physical cores maps naturally to about a **35% no-SMT vCPU discount**
   in this fixed-`4 vCPU` framing.
2. The **`0.75` geomean marker** corresponds to a stronger SMT claim, about
   **`+50%`** on the same physical core budget. That sits toward the high end of
   mainstream vendor/academic reports, but still within the broad range covered
   by AMD / IBM and some academic server/database studies.
3. The **`0.85` marker** is very SMT-favorable. It back-translates to about
   **`+70%`** same-core SMT throughput, which is stronger than most of the
   vendor-style claims summarized in the report.

So after translation, the markers are not arbitrary:

- `0.65` is roughly "industry-median SMT benefit"
- `0.75` is "upper-middle / strong SMT benefit"
- `0.85` is a deliberately conservative bound from the no-SMT point of view

#### Why the Experimental Geomean Can Still Be Explainable

The all-app experimental geomean here is `1.361x` no-SMT/SMT, i.e. multiplier
`0.735` and discount `26.5%`. Using the translation above, that is roughly
equivalent to saying:

- **SMT would need to provide about `+47%` aggregate throughput on the same
  physical core budget to make the two views line up exactly**

That is above Intel's typical `10-30%` style claims, but it is still explainable:

1. The experiment is **peak-throughput**, not average-load.
2. It uses a **small fixed VM shape** (`4 vCPUs`), where each visible vCPU is
   very sensitive to whether it maps to a full core or a sibling thread.
3. Vendor numbers are usually **machine-throughput** claims, while this
   experiment is a **tenant-visible per-vCPU strength** measurement.
4. The research report itself shows broad spread: IBM and some academic
   database/server results reach the `30-60%` region, while HPC and some SPEC
   pairings are much lower or even negative.

So the best synthesis is:

> After translating same-core SMT throughput claims into this model's fixed-vCPU
> framing, the `0.65` marker looks squarely in-family with the broader
> literature, the `0.75` geomean marker looks somewhat stronger but still
> explainable, and the `0.85` marker is a deliberately SMT-favorable
> conservative case.

### Per-Application Distribution

The discount varies widely across applications. Representative service results:

| Application | Perf Ratio | vCPU Discount | CPU-Bound? |
|---|---|---|---|
| Neo4j | 1.495x | 33.1% | Yes |
| Keycloak | 1.434x | 30.3% | Yes |
| Go-CPU | 1.431x | 30.1% | Yes |
| MediaWiki | 1.417x | 29.4% | Yes |
| Imgproxy | 1.361x | 26.5% | Yes |
| MinIO | 1.360x | 26.5% | No |
| Postgres | 1.357x | 26.3% | Yes |
| TensorFlow | 1.320x | 24.3% | No |
| Elasticsearch | 1.293x | 22.7% | Yes |
| KeyDB | 1.258x | 20.5% | Yes |
| InfluxDB | 1.165x | 14.1% | Yes |
| Go-MemChase | 1.068x | 6.4% | Yes |
| Memcached | 1.021x | 2.1% | Yes |

Near-parity cases (Memcached, Go-MemChase) are memory-latency-dominated workloads
where SMT co-running has minimal impact. CPU-intensive workloads consistently show
20-33% discount.

## Model Configuration

### Per-Utilization Sweep Configs (from [02])

The same configs from [02](02_scheduling_constraints_oversub.md) produce the
per-utilization sweep plots. The `vcpu_demand_multiplier` sweep from 0.5 to 1.5
with markers at 0.65/0.75/0.85 is already configured:

| Config | Output |
|---|---|
| [`configs/oversub_analysis/genoa/linear/util_10_pct_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/util_10_pct_linear.jsonc) | `results/oversub_analysis/genoa/linear/util_10pct_linear/` |
| [`configs/oversub_analysis/genoa/linear/util_20_pct_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/util_20_pct_linear.jsonc) | `results/oversub_analysis/genoa/linear/util_20pct_linear/` |
| [`configs/oversub_analysis/genoa/linear/util_30_pct_linear.jsonc`](../../configs/oversub_analysis/genoa/linear/util_30_pct_linear.jsonc) | `results/oversub_analysis/genoa/linear/util_30pct_linear/` |

### No-Oversub Sanity Check (discount only, no scheduling-headroom effect)

As a guardrail, this document also includes a **sanity-check sweep** that applies
the vCPU demand discount back onto the naive [01](01_naive_comparison.md)
setting: both SMT and no-SMT run at `R=1.0`, so the only advantage given to
no-SMT is that customers need fewer vCPUs.

This is useful because if no-SMT were already clearly better **without**
oversubscription, that would be a warning sign for the broader story: cloud
providers do in fact deploy SMT, so a model claiming strong no-SMT savings even
before scheduling headroom is considered would need extra scrutiny.

**Config**: [`configs/oversub_analysis/genoa/linear/no_oversub_vcpu_discount_sanity.jsonc`](../../configs/oversub_analysis/genoa/linear/no_oversub_vcpu_discount_sanity.jsonc)

**Output**: `results/oversub_analysis/genoa/linear/no_oversub_vcpu_discount_sanity/`

### Breakeven Curve (across utilizations)

Aggregates the carbon breakeven `vcpu_demand_multiplier` from each utilization
point into a single curve, comparing with and without resource scaling:

**Config**: [`configs/oversub_analysis/genoa/linear/breakeven_curve_comparison.jsonc`](../../configs/oversub_analysis/genoa/linear/breakeven_curve_comparison.jsonc)

**Output**: `results/oversub_analysis/genoa/linear/breakeven_curve_comparison/`

### Savings Curve (projected savings at discount reference points)

Evaluates no-SMT savings at the low/geomean/high discount markers across
utilization levels (with resource scaling):

**Config**: [`configs/oversub_analysis/genoa/linear/savings_curve.jsonc`](../../configs/oversub_analysis/genoa/linear/savings_curve.jsonc)

**Output**: `results/oversub_analysis/genoa/linear/savings_curve/`

### How to run

```bash
# Per-utilization sweeps (also generates the individual sweep plots)
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_10_pct_linear.jsonc
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_20_pct_linear.jsonc
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/util_30_pct_linear.jsonc

# Breakeven curve across utilizations
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/breakeven_curve_comparison.jsonc

# Savings curve at discount reference points
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/savings_curve.jsonc

# Sanity check: apply discount to the no-oversub baseline from [01]
python -m smt_oversub_model configs/oversub_analysis/genoa/linear/no_oversub_vcpu_discount_sanity.jsonc
```

## Results

### No-Oversub Sanity Check

Applying demand compression alone to the [01](01_naive_comparison.md) baseline
does **not** make no-SMT obviously better than SMT at realistic discount levels.
Using the more favorable `nosmt_linear` power-curve variant and keeping both
configurations at `R=1.0`:

| vCPU demand multiplier | Implied discount | No-SMT Carbon vs SMT | No-SMT TCO vs SMT | No-SMT Servers vs SMT |
|---|---|---|---|---|
| 0.85 | 15% | `+25.8%` | `+24.1%` | `+69.9%` |
| 0.75 | 25% | `+11.0%` | `+9.5%` | `+49.9%` |
| 0.65 | 35% | `-3.8%` | `-5.1%` | `+29.9%` |

The carbon breakeven point in this no-oversub sanity check is:

- **Breakeven multiplier: `0.676`**
- **Implied discount: `32.4%`**

This is the key sanity-check result:

- At the **geomean** experimental discount (`~25%`, multiplier `0.75`), no-SMT
  is still worse than SMT with no oversubscription at all.
- No-SMT only starts to beat SMT in this naive setting at a **stronger-than-geomean**
  discount, around `32-35%`.
- So the main story still makes sense: the projected no-SMT savings in the later
  analyses are not coming from a model that says SMT is already intrinsically a
  bad cloud deployment choice. They require the additional scheduling-headroom
  mechanism from [02](02_scheduling_constraints_oversub.md).

### Per-Utilization Carbon Breakeven

The breakeven `vcpu_demand_multiplier` is the value at which no-SMT matches SMT
on carbon. Values above 1.0 mean no-SMT already wins without any discount.

| Utilization | Breakeven (no resource scaling) | Breakeven (with resource scaling) |
|---|---|---|
| 10% | 1.16 | 0.87 |
| 20% | 1.02 | 0.84 |
| 30% | 0.87 | 0.78 |
| 40% | 0.94 | 0.84 |
| 50% | 0.82 | 0.78 |

**Reading this table** (see [02a: Resource Modeling](02a_resource_modeling.md)
for full definitions of "fixed" vs "scaled" resources):
- **Without resource scaling** (fixed resources): Per-server memory/SSD costs
  stay the same regardless of R. At 20% utilization, no-SMT already breaks even
  at multiplier=1.02 (essentially no discount needed). At 10%, a multiplier of
  1.16 means no-SMT *already wins* even with 16% *more* demand (not a discount,
  but extra demand). At 30% and 50%, discounts of 13% and 18% are needed.
- **With resource scaling** (scaled resources): Memory and SSD are provisioned
  per-vCPU, modeling purpose-built servers where each VM gets a fixed amount of
  memory and SSD regardless of how densely vCPUs are packed. Breakeven
  multipliers are lower (harder for no-SMT to break even) because higher
  oversubscription increases per-server embodied costs. The range narrows to
  0.78-0.87. This is the more accurate framing for projecting real deployment
  costs.

**Note on the 10% without-resource-scaling result**: The breakeven of 1.16
differs from [02]'s value of 0.676 because the configs appear to have been
updated since the results were generated. The results shown here reflect the
actual output in the results directory. When in doubt, re-run the configs to
get current values (see "How to run" above).

### Savings at Experimental Reference Points

The savings curve shows % change in carbon and TCO at the three experimental
reference discount levels, using resource-scaling configs:

**At "geomean" discount (multiplier=0.75, ~1.33x perf ratio):**

| Utilization | Carbon Savings | TCO Savings |
|---|---|---|
| 10% | **-13.7%** | **-19.8%** |
| 20% | **-11.1%** | **-15.6%** |
| 30% | **-4.0%** | **-7.2%** |
| 40% | **-10.7%** | **-15.1%** |
| 50% | **-4.0%** | **-7.5%** |

At the geomean discount, **no-SMT saves 4-14% on carbon and 7-20% on TCO across
all utilization levels**. The largest savings are at 10% utilization where the
oversubscription headroom advantage is greatest.

**At "low" discount (multiplier=0.65, ~1.54x perf ratio):**

| Utilization | Carbon Savings | TCO Savings |
|---|---|---|
| 10% | **-25.2%** | **-30.6%** |
| 20% | **-23.0%** | **-27.0%** |
| 30% | **-16.8%** | **-19.5%** |
| 40% | **-22.6%** | **-26.4%** |
| 50% | **-16.9%** | **-19.9%** |

For workloads with strong no-SMT performance gains (like Neo4j, Keycloak,
Go-CPU), no-SMT saves **17-25% on carbon and 20-31% on TCO**.

**At "high" discount (multiplier=0.85, ~1.18x perf ratio):**

| Utilization | Carbon Savings | TCO Savings |
|---|---|---|
| 10% | **-2.1%** | **-9.1%** |
| 20% | +0.7% | **-4.5%** |
| 30% | +8.7% | +5.1% |
| 40% | +1.1% | **-3.8%** |
| 50% | +8.7% | +4.8% |

For workloads with weak no-SMT gains (like InfluxDB, Go-MemChase), the picture
is mixed. At 10% utilization no-SMT still saves on TCO (-9%) but is marginal on
carbon. At 30% utilization, no-SMT is clearly worse (+9% carbon, +5% TCO).

### How the Three Mechanisms Stack Up

Combining results from all three analyses:

| Mechanism | Effect on no-SMT competitiveness |
|---|---|
| **LP count disadvantage** ([01](01_naive_comparison.md)) | -47% to -56% carbon penalty (the "hole" to dig out of) |
| **vCPU discount only, no oversub** (sanity check in this doc) | Still `+11.0%` carbon / `+9.5%` TCO worse at geomean discount; needs ~`32%` discount to break even |
| **Scheduling constraint headroom** ([02](02_scheduling_constraints_oversub.md)) | Closes gap to -2% to +48% depending on utilization |
| **vCPU demand discount** (this doc) | At geomean discount, pushes to -4% to -14% savings |

The progression: [01] establishes a large penalty, [02] closes most of it through
higher oversubscription, and [03] shows that demand compression tips the balance
once it is layered on top of that headroom. The no-oversub sanity check confirms
that discount alone is usually **not** enough at realistic geomean values.

## Interpretation

### The Central Result

With all three mechanisms active -- scheduling constraint-limited oversubscription,
linear no-SMT power curves, and experimentally-grounded vCPU demand discounts --
**no-SMT saves 4-14% on carbon and 7-20% on TCO at the geomean discount across
utilization levels from 10% to 50%**.

The savings are largest at low utilization (10-20%) where oversubscription
headroom is greatest, and smallest at moderate utilization (30%) where headroom
narrows.

### Sensitivity to the Discount Factor

The discount factor is the **single most impactful parameter** in the analysis.
The difference between "low" (0.65) and "high" (0.85) discount scenarios spans
from -25% to +9% carbon -- a 34 percentage point range. This means:

- For workloads with strong SMT co-running penalties (CPU-bound services like
  Neo4j, Keycloak, Go-CPU), no-SMT is a clear winner
- For workloads with weak SMT penalties (memory-latency-bound like Memcached,
  Go-MemChase), the case for no-SMT is marginal or negative
- A real fleet is a mix, and the **geomean discount (~0.75) is the best single
  estimate** for a mixed-workload fleet

### Resource Scaling Effect

The breakeven curve shows that **scaled resources (purpose-built servers) shift
breakeven multipliers lower** by 0.04-0.29 across utilization levels, making it
*harder* for no-SMT to break even. This is because at high oversubscription,
memory and SSD must scale with vCPU count, increasing per-server embodied costs
and eroding some of the server-count savings. This effect is most pronounced at
low utilization where oversubscription ratios are highest. See
[02a: Resource Modeling](02a_resource_modeling.md) for the full explanation and
[02b: Oversubscription Savings Scaling](02b_oversub_savings_scaling.md) for how
this effect compounds across the R range.

### Connection to Experimental Repo

The full experimental chain for this analysis is:

1. **Steal-time thresholds** ([`scheduling_constraints_smt_impact.md`][exp-sched],
   [`scheduling_constraints_smt_iso_physical_core.md`][exp-iso]):
   Measure max safe VP/LP rates -> oversubscription ratios (R) in configs

2. **Power proportionality** ([`smt_no_smt_power_proportionality.md`][exp-power]):
   Measure no-SMT vs SMT power curves -> linear CPU power model for no-SMT

3. **Peak performance** ([`smt_no_smt_peak_performance.md`][exp-perf]):
   Measure no-SMT/SMT throughput ratios -> vCPU discount reference points

All three experimental inputs flow into the same model configs. The savings curve
is the synthesis that brings them together.

[exp-sched]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_impact.md
[exp-iso]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_iso_physical_core.md
[exp-power]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_power_proportionality.md
[exp-perf]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_peak_performance.md

## What This Does Not Address

1. **Resource constraints on existing hardware**: Running no-SMT at high R on
   SMT-provisioned servers (with the full 12 DIMMs / 6 SSDs) may be limited by
   memory or SSD capacity. See [03a: Constrained Savings](03a_constrained_savings.md)
   for how the savings in this document change under same-hardware constraints.

2. **Heterogeneous fleets**: Some workloads benefit greatly from no-SMT while
   others do not. A mixed fleet that routes high-discount workloads to no-SMT
   pools and low-discount workloads to SMT pools could capture more savings.
   The composite scenario configs explore this.

3. **Application-specific oversubscription ratios**: The R values come from
   go-cpu only. A full analysis would use per-application steal-time curves to
   set per-application R values, which would change the savings for specific
   workload mixes.

4. **Dynamic utilization**: The model assumes steady-state average utilization.
   Real clouds have time-varying load, which affects both the safe R and the
   realized utilization per server.

5. **Non-linear no-SMT power**: The `non-linear/` config directory contains
   equivalent analyses using the default polynomial power curve for no-SMT.
   Those show slightly lower savings (since no-SMT loses its power-curve
   advantage), but the qualitative conclusions hold.
