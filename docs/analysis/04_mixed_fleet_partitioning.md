# 04: Mixed Fleet Partitioning

## Question

> Instead of converting the entire fleet to no-SMT (homogeneous switch), what if
> you partition the fleet into an SMT pool and a no-SMT pool, routing workloads
> to the pool that minimizes fleet-wide carbon and TCO? How much does this mixed
> fleet strategy save, and how sensitive is it to the partition boundary?

This analysis extends the homogeneous switch story from
[03](03_vcpu_demand_discount.md) and [03a](03a_constrained_savings.md) by
introducing a **heterogeneous fleet** option, and uses the updated
scheduling-input calibrations from [02c](02c_scheduling_input_basis_sensitivity.md)
instead of the legacy R values.

## Prerequisites

- [02c: Scheduling Input Basis Sensitivity](02c_scheduling_input_basis_sensitivity.md)
  for the iso-LP and iso-physical-core R values used here
- [03: vCPU Demand Discount](03_vcpu_demand_discount.md) for the concept of
  per-workload vCPU discount and the experimental performance ratios that
  parameterize it
- [03a: Constrained Savings](03a_constrained_savings.md) for the same-hardware
  resource constraint model
- [Spine](SMT_VS_NOSMT_ANALYSIS.md) for terminology

## Key Assumptions

### Inherited from prior layers

1. **Genoa 80-core baseline**: Same processor specs, per-component cost/carbon
   breakdowns, and power models as all prior analyses.
2. **VP-constrained scheduling**: SMT oversubscription is limited by VP
   core-scheduling constraints; no-SMT can oversubscribe more freely.
3. **Go-CPU calibration**: R values come from interpolated Go-CPU
   operating-point tables, not cross-application data.

### New in this analysis

4. **Workload discount heterogeneity**: The fleet's workloads are assumed to
   have a distribution of vCPU demand discounts, modeled as a uniform
   distribution from 0.50 to 1.00 in 0.05 steps (11 bins, each ~9.1% of
   demand). A value of 0.50 means a workload needs only 50% as many vCPUs
   under no-SMT (strong performance gain); 1.00 means no benefit. The
   fleet-average discount is 0.75, matching the geomean from
   [03](03_vcpu_demand_discount.md).

5. **Binary routing by breakeven threshold**: Each workload is assigned to the
   no-SMT pool or the SMT pool based on whether its vCPU discount is below or
   above a **split point**. Workloads below the split (strong no-SMT advantage)
   go to no-SMT; workloads above it (weak or no advantage) stay on SMT.

6. **Auto-computed split point**: The split point is set to the **carbon
   breakeven vCPU demand multiplier** -- the discount level at which a single
   workload evaluated on no-SMT matches the same workload on SMT for carbon.
   This is a per-workload breakeven, not a fleet optimization.

7. **Two scheduling-input bases**: All analyses run under both the iso-LP
   (8 LP / 2 VM baseline) and iso-physical-core (16 LP / 4 VM SMT with
   larger LP pool) calibrations from [02c](02c_scheduling_input_basis_sensitivity.md).

8. **Two resource models**: Each basis is evaluated under both purpose-built
   no-SMT servers (resource scaling) and same-hardware SMT-off (resource
   constraints), consistent with [02a](02a_resource_modeling.md).

### Simplifications and caveats

- **Uniform discount distribution**: In practice, the distribution of vCPU
  discounts across a real fleet is unknown. The uniform assumption is a
  modeling baseline. A right-skewed distribution (most workloads near 1.0)
  would reduce mixed fleet savings; a left-skewed one would increase them.

- **Binary pool assignment**: Workloads are assigned to exactly one pool.
  Real cloud schedulers might use softer routing or multi-tier pools.

- **No scheduling interaction between pools**: Each pool is evaluated
  independently. In practice, workload migration between pools adds
  operational complexity and could affect utilization distributions.

- **Single discount per workload**: Each workload is assumed to have a fixed
  discount. In reality, the discount depends on the specific application, VM
  size, and load pattern -- and may vary over time.

## Experimental Inputs

### Oversubscription Ratios

From [02c](02c_scheduling_input_basis_sensitivity.md), using the interpolated
Go-CPU operating-point tables:

| Basis | 10% util | 20% util | 30% util |
|---|---|---|---|
| **Iso-LP** | SMT `2.59`, No-SMT `5.58` | SMT `1.29`, No-SMT `2.79` | SMT `1.00` (raw `0.86`, capped), No-SMT `1.86` |
| **Iso-Physical-Core** | SMT `3.32`, No-SMT `5.58` | SMT `1.66`, No-SMT `2.79` | SMT `1.11`, No-SMT `1.86` |

### vCPU Demand Discount Distribution

From [03](03_vcpu_demand_discount.md), the experimental geomean across 30
applications is a 26.5% discount (multiplier 0.735). The model uses a uniform
distribution from 0.50 to 1.00, with fleet-average 0.75.

### Auto-Breakeven Split Points

The split point is the vCPU demand multiplier at which no-SMT matches SMT on
carbon for a single workload. These are computed via binary search at runtime
(`auto_breakeven`). The resolved values:

**Purpose-Built (Resource Scaling):**

| Basis | 10% util | 20% util | 30% util |
|---|---|---|---|
| Iso-LP | 0.868 | 0.885 | 0.887 |
| Iso-Physical-Core | 0.852 | 0.834 | 0.824 |

**Same Hardware (Resource Constraints):**

| Basis | 10% util | 20% util | 30% util |
|---|---|---|---|
| Iso-LP | 0.898 | 0.893 | 0.827 |
| Iso-Physical-Core | 0.898 | 0.871 | 0.770 |

**Reading these**: A split point of 0.868 means workloads with `vcpu_demand_multiplier < 0.868` (i.e., discount > 13.2%) go to no-SMT, and those above
stay on SMT. At the fleet-average discount of 0.75 (25% discount), the
average workload falls below all split points, meaning most workloads are
routed to no-SMT in the mixed fleet.

The iso-physical-core split points are lower than iso-LP because SMT is more
competitive with its larger LP pool, so a stronger discount is needed before
no-SMT wins on a per-workload basis.

## Methodology

### How the mixed fleet model works

The model uses the **composite scenario** framework (see CLAUDE.md). A
composite scenario consists of two sub-scenarios (pools), each receiving a
fraction of the fleet's total vCPU demand:

1. **No-SMT pool** (`below_split`): Receives all workloads whose vCPU discount
   trait is below the split point. The pool's `vcpu_demand_multiplier` is set
   to the **weighted average** of the trait values in this partition.

2. **SMT pool** (`above_split`): Receives the remaining workloads. Its
   `vcpu_demand_multiplier` is fixed at 1.0 (SMT vCPUs are the performance
   baseline).

Each pool is evaluated independently with its own processor type, R value, and
resource model. Fleet-level carbon and TCO are the sum across pools.

### Example: 20% util, iso-LP, purpose-built

With a split point at 0.885:
- **No-SMT pool** receives bins 0.50 through 0.85 (8 bins, ~72.7% of vCPU demand).
  Weighted average discount = 0.675 (i.e., 32.5% demand reduction).
- **SMT pool** receives bins 0.90 through 1.00 (3 bins, ~27.3% of vCPU demand).
  No demand reduction.

Each pool then determines its server count, power consumption, and carbon/cost
independently.

### Why the mixed fleet can outperform homogeneous no-SMT

The homogeneous no-SMT scenario applies the fleet-average discount (0.75) to
**all** workloads, including those with weak or no performance gain from
disabling SMT. This means:

1. **Workloads near multiplier 1.0** (weak discount) still get counted as
   needing only 0.75x vCPUs under no-SMT. In practice, these workloads do
   not actually benefit from no-SMT and should remain on SMT to keep their
   vCPU count at the true demand level.

2. The mixed fleet **avoids this mismatch** by keeping high-multiplier
   workloads on SMT (where they need the full vCPU count anyway) and only
   routing genuinely-benefiting workloads to no-SMT.

3. This trades off fewer no-SMT servers (because only a subset of demand goes
   there) against more accurate demand modeling -- the no-SMT pool uses the
   weighted average of the strong-discount workloads, which is lower than the
   fleet average.

The net effect is that the mixed fleet typically achieves **more carbon savings**
than homogeneous no-SMT (because of better demand matching) while sometimes
using **slightly more servers** (because the SMT pool still operates at a
lower R). But the carbon and cost advantages dominate because the no-SMT pool
runs at a more aggressive discount.

## Model Configuration

### Config Structure

All configs are generated by a single script:

[`tools/generate_mixed_fleet_analysis.py`](../../tools/generate_mixed_fleet_analysis.py)

```
configs/oversub_analysis/genoa/mixed_fleet/
  iso_lp/
    resource_scaling/
      util_{10,20,30}_pct_compare.jsonc     # 3-way bar chart
      util_{10,20,30}_pct_sweep.jsonc        # Split-point sensitivity
    resource_constraints/
      util_{10,20,30}_pct_compare.jsonc
      util_{10,20,30}_pct_sweep.jsonc
  iso_physical_core/
    resource_scaling/
      ...
    resource_constraints/
      ...
```

### Outputs

```
results/oversub_analysis/genoa/mixed_fleet/
  iso_lp/
    resource_scaling/
      util_{10,20,30}pct_compare/            # comparison.txt, plots/
      util_{10,20,30}pct_sweep/              # sweep tables, plots/
    resource_constraints/
      ...
  iso_physical_core/
    ...
  summary.csv                                # All compare results
  summary.json
  summary.md                                 # Readable summary table
  plots/
    {basis}_{mode}_carbon_summary.png        # Cross-utilization bar charts
    {basis}_{mode}_tco_summary.png
    {basis}_{mode}_server_breakdown.png     # Server type breakdown (stacked)
    basis_comparison_{mode}_carbon_summary.png  # Iso-LP vs Iso-Physical-Core
    basis_comparison_{mode}_tco_summary.png
```

### How to Run

```bash
# Generate configs, run all analyses, produce summaries and plots
MPLCONFIGDIR=/tmp/mpl python tools/generate_mixed_fleet_analysis.py

# Generate configs only (no execution)
python tools/generate_mixed_fleet_analysis.py --generate-only

# Run a single config manually
python -m smt_oversub_model configs/oversub_analysis/genoa/mixed_fleet/iso_lp/resource_scaling/util_10_pct_compare.jsonc
```

## Results

### 1. Purpose-Built No-SMT Servers (Resource Scaling)

This models deploying purpose-built no-SMT servers alongside SMT servers, where
memory and SSD scale with vCPU count (each vCPU gets a fixed allocation).

#### Iso-LP Basis

| Util % | SMT R | No-SMT R | Split Point | No-SMT Homo. Carbon | Mixed Fleet Carbon | No-SMT Homo. TCO | Mixed Fleet TCO |
|---|---|---|---|---|---|---|---|
| 10 | 2.59 | 5.58 | 0.868 | **-13.6%** | **-15.8%** | -19.9% | -19.9% |
| 20 | 1.29 | 2.79 | 0.885 | **-15.2%** | **-17.2%** | -20.8% | -20.9% |
| 30 | 1.00 | 1.86 | 0.887 | **-15.4%** | **-17.3%** | -20.5% | -20.6% |

The mixed fleet improves on homogeneous no-SMT by **1.9-2.2 pp on carbon**
across all utilization levels.

**Why TCO converges while carbon diverges**: Under iso-LP, the no-SMT R
advantage is large (e.g., 5.58 vs 2.59 at 10%), producing a high split point
(~0.87-0.89). This means ~73% of demand goes to no-SMT and only ~27% stays
on SMT. The mixed fleet uses slightly more total servers (e.g., 197 vs 187 at
10%) because it maintains a separate SMT pool, but its no-SMT pool runs at a
more aggressive discount (~0.67 vs fleet-average 0.75) and thus needs fewer
no-SMT servers than the homogeneous fleet.

The near-exact TCO cancellation stems from an asymmetry in cost composition:

- **Embodied cost is 62% of total cost** for the homogeneous no-SMT fleet.
  The mixed fleet's 10 extra servers add ~$127K in embodied cost.
- **Operational cost is 38% of total cost**. The mixed fleet's better demand
  matching saves ~$125K in electricity.
- **Net TCO: +$1,608** on a $6.6M base (effectively zero).

For carbon, the same physical tradeoff plays out differently because the
composition is inverted:

- **Embodied carbon is only 31% of total carbon**. The 10 extra servers add
  only ~20K kg embodied carbon.
- **Operational carbon is 69% of total carbon**. Better demand matching saves
  ~78K kg.
- **Net carbon: -58K kg** (a 2.6% improvement).

In short, buying servers is expensive but their manufacturing carbon is modest
relative to lifetime electricity carbon. The mixed fleet saves real electricity
through better demand matching, and that electricity saving is a bigger lever
for carbon (where operational dominates) than for cost (where embodied
dominates). Under iso-physical-core, the lower split point creates a more
balanced fleet where the demand-matching savings exceed the embodied penalty
for *both* metrics, opening a TCO gap as well.

The server breakdown plots (`iso_lp_resource_scaling_server_breakdown.png`)
visualize the fleet composition differences.

The iso-LP basis produces fairly uniform savings across utilization levels,
because the updated 8LP calibration makes the no-SMT R advantage strong enough
that the mixed fleet benefit is consistent.

#### Iso-Physical-Core Basis

| Util % | SMT R | No-SMT R | Split Point | No-SMT Homo. Carbon | Mixed Fleet Carbon | No-SMT Homo. TCO | Mixed Fleet TCO |
|---|---|---|---|---|---|---|---|
| 10 | 3.32 | 5.58 | 0.852 | -11.8% | **-14.6%** | -16.8% | **-17.8%** |
| 20 | 1.66 | 2.79 | 0.834 | -10.1% | **-14.0%** | -14.6% | **-16.5%** |
| 30 | 1.11 | 1.86 | 0.824 | -9.0% | **-13.5%** | -13.2% | **-15.8%** |

The iso-physical-core basis gives SMT a larger LP pool, reducing homogeneous
no-SMT savings. But the mixed fleet advantage **grows** to **2.8-4.5 pp on
carbon**. This is because:

1. The lower split point (0.824-0.852 vs 0.868-0.887) means only the
   strongest-benefit workloads go to no-SMT, getting a more aggressive
   weighted average discount.
2. The remaining workloads stay on SMT where they do not incur the penalty of
   being on no-SMT hardware with a discount they do not actually experience.

### 2. Same Hardware SMT-Off (Resource Constraints)

This models disabling SMT on existing SMT hardware, where servers retain their
full memory/SSD complement and packing is limited by whichever resource
(cores, memory, SSD) is the bottleneck.

#### Iso-LP Basis

| Util % | SMT R | No-SMT R | Split Point | No-SMT Homo. Carbon | Mixed Fleet Carbon | No-SMT Homo. TCO | Mixed Fleet TCO |
|---|---|---|---|---|---|---|---|
| 10 | 2.59 | 5.58 | 0.898 | **-16.5%** | **-17.9%** | -20.8% | -20.7% |
| 20 | 1.29 | 2.79 | 0.893 | -15.9% | **-17.7%** | -21.2% | -21.2% |
| 30 | 1.00 | 1.86 | 0.827 | -9.3% | **-13.6%** | -13.8% | **-16.1%** |

At 10-20% utilization, memory is the bottleneck for the same-hardware model,
compressing the differences between homogeneous and mixed. But at 30% util,
where cores become the bottleneck and R values are lower, the mixed fleet
provides a substantial **4.3 pp carbon improvement** over homogeneous no-SMT.

#### Iso-Physical-Core Basis

| Util % | SMT R | No-SMT R | Split Point | No-SMT Homo. Carbon | Mixed Fleet Carbon | No-SMT Homo. TCO | Mixed Fleet TCO |
|---|---|---|---|---|---|---|---|
| 10 | 3.32 | 5.58 | 0.898 | **-16.5%** | **-17.9%** | -20.8% | -20.7% |
| 20 | 1.66 | 2.79 | 0.871 | -13.9% | **-16.2%** | -19.0% | **-19.5%** |
| 30 | 1.11 | 1.86 | 0.770 | **-2.4%** | **-10.1%** | -5.9% | **-11.7%** |

This is the most dramatic case. At 30% utilization:

- **Homogeneous no-SMT** saves only **2.4%** on carbon (marginal)
- **Mixed fleet** saves **10.1%** on carbon -- a **7.7 pp improvement**

The mixed fleet is transformative here because at 30% util with the
iso-physical-core basis, SMT is quite competitive (R=1.11 vs no-SMT R=1.86).
The homogeneous switch applies the fleet-average discount to all workloads,
including those that barely benefit from no-SMT. The mixed fleet avoids this
by only routing the strong-discount workloads (below multiplier 0.770) to
no-SMT.

At 10% utilization, the same-hardware memory constraint binds for both bases,
producing identical results regardless of the scheduling-input basis.

### 3. Split-Point Sensitivity

The sweep configs vary the split point from 0.50 to 1.01 to show how the
mixed fleet savings change with different routing thresholds.

Key patterns:

- **Mixed fleet carbon savings increase** as the split point rises from 0.50
  toward a peak around 0.80-0.90, then decrease as the split point approaches
  1.01 (converging to the homogeneous no-SMT result).
- At split point **0.50**, all demand goes to SMT (no partition), yielding 0%
  savings.
- At split point **1.01**, all demand goes to no-SMT (homogeneous), yielding
  the same result as the no-SMT homogeneous scenario.
- The **auto-breakeven split point** sits near but not exactly at the peak
  because it is determined by per-workload carbon breakeven, not by
  optimizing the composite fleet objective.

The sweep plots are in each utilization-specific output directory, e.g.:

- `results/oversub_analysis/genoa/mixed_fleet/iso_physical_core/resource_constraints/util_30pct_sweep/plots/`

### 4. Summary: Mixed Fleet Advantage (Carbon, pp vs Homogeneous No-SMT)

| | **Purpose-Built** | | **Same HW** | |
|---|---|---|---|---|
| **Util %** | **Iso-LP** | **Iso-Phys-Core** | **Iso-LP** | **Iso-Phys-Core** |
| 10 | +2.2 pp | +2.8 pp | +1.4 pp | +1.4 pp |
| 20 | +2.0 pp | +3.9 pp | +1.8 pp | +2.3 pp |
| 30 | +1.9 pp | +4.5 pp | +4.3 pp | +7.7 pp |

### 5. Server Composition

The server breakdown plots (`{basis}_{mode}_server_breakdown.png`) show the
fleet composition for each scenario as stacked bars. Key observations:

- Under **iso-LP / purpose-built**, the mixed fleet uses *more total servers*
  than homogeneous no-SMT at low utilization (197 vs 187 at 10%) but still
  saves on carbon because the no-SMT pool runs at a more aggressive discount.
  At higher utilization (30%), total server counts converge (557 vs 561).

- Under **iso-physical-core / same-HW**, the split is more balanced. At 30%
  utilization the mixed fleet contains 255 no-SMT and 285 SMT servers (540
  total) versus 561 homogeneous no-SMT. The mixed fleet uses *fewer* total
  servers because the SMT pool's higher per-server capacity more than
  compensates for the demand kept on SMT.

- The **pool ratio** (no-SMT / total) in the mixed fleet reflects the split
  point: a high split point (~0.89) means most demand goes to no-SMT
  (62-66% of servers), while a low split point (~0.77) means a roughly even
  split (~47% no-SMT).

The mixed fleet advantage is **largest where homogeneous no-SMT is weakest**:
iso-physical-core at 30% utilization with same-hardware constraints. This is
precisely the case where SMT is most competitive and the fleet-average discount
is least representative of any individual workload.

## Interpretation

### Where the mixed fleet matters most

The mixed fleet strategy provides the most value when:

1. **SMT is relatively competitive** (iso-physical-core basis, moderate
   utilization). In these cases, the breakeven discount is low (~0.77-0.85),
   meaning only workloads with strong no-SMT gains should be routed there. A
   homogeneous switch applies the fleet-average discount uniformly, which
   wastes the headroom on workloads that do not benefit.

2. **The discount distribution is heterogeneous**. If all workloads had exactly
   the same discount, the mixed fleet would offer no advantage over
   homogeneous. The more spread the distribution, the more value in selective
   routing.

3. **Utilization is moderate to high** (20-30%). At very low utilization (10%),
   the oversubscription headroom advantage is so large that even a
   homogeneous switch is strongly favorable, leaving less room for the mixed
   fleet to improve.

### Where the mixed fleet matters less

At 10% utilization under the same-hardware model, memory is the bottleneck for
both pools, and the mixed fleet provides only a modest 1.4 pp improvement.
Similarly, under the iso-LP basis where no-SMT already saves 13-17% carbon
homogeneously, the mixed fleet adds only 1.9-2.2 pp.

### Practical implications

1. **The mixed fleet is not a replacement for the homogeneous switch** -- it is
   an incremental improvement that matters most at the margin. If a provider
   can deploy homogeneous no-SMT and capture 10-17% carbon savings, the mixed
   fleet adds 2-8 pp on top.

2. **Workload classification is the key enabler**. The mixed fleet requires
   knowing (or estimating) each workload's vCPU discount to route it
   correctly. This could come from:
   - Historical performance profiling
   - Application class heuristics (CPU-bound services tend to have higher
     discounts than memory-bound ones)
   - Online measurement and migration

3. **The split point is not highly sensitive** near the optimum. The sweep plots
   show broad plateaus of near-optimal savings across a range of split points
   (typically 0.70-0.95). This means the routing does not need to be precise
   to capture most of the benefit.

### Assumptions that could change these results

| Assumption | If changed to... | Effect on mixed fleet value |
|---|---|---|
| Uniform discount distribution | Right-skewed (most workloads near 1.0) | Increases value -- more workloads correctly stay on SMT |
| Uniform discount distribution | Left-skewed (most workloads near 0.5) | Decreases value -- homogeneous switch already captures most savings |
| Go-CPU calibration for R values | Cross-application R values | Changes the breakeven split point; likely varies by application |
| Fleet-average discount 0.75 | Higher (0.85, conservative) | Reduces homogeneous savings more than mixed fleet, increasing the gap |
| Fleet-average discount 0.75 | Lower (0.65, aggressive) | Increases homogeneous savings, reducing the mixed fleet advantage |
| Binary pool assignment | Continuous per-workload optimization | Further improves on mixed fleet (upper bound) |

### Connection to the analysis progression

This document sits after [03a](03a_constrained_savings.md) in the analysis
progression and addresses the question flagged in [03](03_vcpu_demand_discount.md)
Section "What This Does Not Address" item 2 ("heterogeneous fleets").

It also uses the updated scheduling-input calibrations from
[02c](02c_scheduling_input_basis_sensitivity.md), which means the R values
here differ from the legacy values used in [02](02_scheduling_constraints_oversub.md)
through [03a](03a_constrained_savings.md). Specifically:

- The iso-LP basis generally makes no-SMT more favorable than the legacy calibration
- The iso-physical-core basis partially recovers SMT's competitiveness
- Both bases are analyzed here to bracket the range

The prior multi-cluster analysis in
`results/oversub_analysis/genoa/linear/multi_cluster/` used the legacy R values
and a manually-created config set. This document supersedes that analysis with:
1. Updated R values from the iso-LP and iso-physical-core calibrations
2. A generator script for reproducibility
3. Systematic coverage of both resource models and both scheduling-input bases
4. Cross-utilization summary bar charts

## What This Does Not Address

1. **Optimal fleet partitioning**: The split point is set by per-workload
   breakeven, not by optimizing the composite fleet objective. A true fleet
   optimizer could find a different (potentially better) split point by
   jointly considering demand allocation and server counts.

2. **More than two pools**: A real fleet could have multiple tiers (e.g.,
   high-discount no-SMT, moderate-discount no-SMT at lower R, SMT).

3. **Dynamic routing and migration costs**: The model evaluates steady-state
   fleet composition without accounting for the operational cost of
   classifying workloads and migrating VMs between pools.

4. **Non-uniform utilization across pools**: Both pools use the same average
   utilization. In practice, the no-SMT pool might see different utilization
   patterns if the routed workloads have different load profiles.

5. **Cross-application R calibration**: The R values come from Go-CPU only.
   Different workload types may have different safe oversubscription limits.

[exp-sched]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_impact.md
[exp-iso]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_iso_physical_core.md
[exp-perf]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_peak_performance.md
