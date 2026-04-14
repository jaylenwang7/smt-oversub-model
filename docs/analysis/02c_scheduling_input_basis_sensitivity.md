# 02c: Scheduling Input Basis Sensitivity

## Question

> How much do the SMT vs no-SMT conclusions change if we (1) recalibrate the
> baseline 8LP inputs from the new interpolated Go-CPU operating-point tables,
> and then (2) give SMT the larger LP pool it would have on the same physical
> cores?

This is an **additive follow-up** to
[02](02_scheduling_constraints_oversub.md), not a replacement. The earlier
analysis remains in place. This document isolates how much of the downstream
model behavior is driven by the **choice of scheduling-input basis**.

## Prerequisites

- [02: Scheduling Constraints and Oversubscription](02_scheduling_constraints_oversub.md)
- [03: vCPU Demand Discount](03_vcpu_demand_discount.md)
- [03a: Constrained Savings](03a_constrained_savings.md)

## Why This Follow-Up Exists

The new experiment-side note
[`scheduling_constraints_smt_iso_physical_core.md`][exp-iso] adds an important
realism layer beyond
[`scheduling_constraints_smt_impact.md`][exp-sched]:

1. The original `8 LP / 2 VM` comparison is still the right **iso-LP** answer
   for the direct scheduling-cost question.
2. But the same physical 8 cores would expose **16 LPs with SMT on** and only
   **8 LPs with SMT off**.
3. So there are now two distinct modeling questions:
   - **Iso-LP**: hold LP pool size fixed and measure the direct VP-constraint penalty
   - **Iso-physical-core**: hold physical cores fixed and let SMT keep its larger LP pool

That follow-up also exposes a second issue: the new interpolated Go-CPU
operating-point rates differ materially from some of the older max-oversub rates
already used in this repo. So this document separates three input bases:

- **Legacy existing**: the currently checked-in repo calibration
- **Updated iso-LP**: new interpolated `8 LP / 2 VM` baseline rates
- **Iso-physical-core**: new interpolated `16 LP / 4 VM` SMT rates against the
  same no-SMT rates

## Experimental Inputs

### Source Documents

- [`scheduling_constraints_smt_impact.md`][exp-sched]
- [`scheduling_constraints_smt_iso_physical_core.md`][exp-iso]
- [`analysis/scheduling_constraints_smt/cross_app_iso_physical_core_threshold_1.0pct.csv`][exp-csv]

### Oversubscription Ratios Used Here

All **new** configs in this doc use the interpolated Go-CPU `max safe VP/LP`
rates from [`scheduling_constraints_smt_iso_physical_core.md`][exp-iso], with
one modeling guardrail:

> If an inferred safe rate is below `1.0`, the model caps it at `R=1.0`.
> A safe oversubscription ratio below `1` would imply under-subscription, and
> operationally you would simply stop at no oversubscription.

| Basis | 10% util | 20% util | 30% util | Notes |
|---|---|---|---|---|
| Legacy existing | SMT `2.34`, No-SMT `5.00` | SMT `1.33`, No-SMT `2.34` | SMT `1.14`, No-SMT `1.60` | Existing baseline / scaled configs already in this repo |
| Updated iso-LP | SMT `2.59`, No-SMT `5.58` | SMT `1.29`, No-SMT `2.79` | SMT `1.00` (raw `0.86`, capped), No-SMT `1.86` | `VP 8LP` vs `No-SMT 8LP` from new operating-point tables |
| Iso-physical-core | SMT `3.32`, No-SMT `5.58` | SMT `1.66`, No-SMT `2.79` | SMT `1.11`, No-SMT `1.86` | `VP 16LP` vs `No-SMT 8LP` from same tables |

Two important observations:

1. The updated `iso_lp` inputs are **not** just minor perturbations of the
   current repo values. At 10% and 20%, both SMT and no-SMT safe rates rise; at
   30%, the raw interpolated SMT rate falls below `1.0`, so the model uses
   `R=1.0` while no-SMT rises from `1.60` to `1.86`.
2. The `iso_physical_core` step changes **SMT only**. The no-SMT side stays the
   same as updated `iso_lp`; SMT gets the larger LP pool it would actually have
   on the same physical cores.

## Model Configuration

### New Config Structure

To keep this additive branch separate from the existing analysis tree, the new
configs live under:

```text
configs/oversub_analysis/genoa/scheduling_input_sensitivity/
  iso_lp/
    baseline/
    resource_scaling/
    resource_constraints/
  iso_physical_core/
    baseline/
    resource_scaling/
    resource_constraints/
  basis_comparison_*_savings_curve.jsonc
  basis_comparison_*_breakeven_curve.jsonc
```

This makes the input basis explicit instead of burying it in ad hoc names like
`new_oversub`.

### Generator / Runner

All configs in this branch are generated and run via:

[`tools/generate_scheduling_input_sensitivity.py`](../../tools/generate_scheduling_input_sensitivity.py)

Run everything:

```bash
MPLCONFIGDIR=/tmp/mpl python tools/generate_scheduling_input_sensitivity.py
```

Generate configs only:

```bash
python tools/generate_scheduling_input_sensitivity.py --generate-only
```

### Outputs

Root output directory:

`results/oversub_analysis/genoa/scheduling_input_sensitivity/`

The script also writes consolidated summaries:

- [`basis_summary.csv`](../../results/oversub_analysis/genoa/scheduling_input_sensitivity/basis_summary.csv)
- [`basis_delta_summary.csv`](../../results/oversub_analysis/genoa/scheduling_input_sensitivity/basis_delta_summary.csv)
- [`summary.md`](../../results/oversub_analysis/genoa/scheduling_input_sensitivity/summary.md)

## Results

### 1. Updating the 8LP Baseline Alone Already Changes the Story

Before giving SMT any extra LPs, just swapping the old repo inputs for the new
interpolated `iso_lp` inputs materially changes the answer.

#### Baseline Model (No Resource Scaling / Constraints), Carbon vs SMT

| Utilization | Legacy existing @ multiplier `1.0` | Updated iso-LP @ `1.0` | Iso-physical-core @ `1.0` | Legacy existing @ `0.75` | Updated iso-LP @ `0.75` | Iso-physical-core @ `0.75` |
|---|---|---|---|---|---|---|
| 10% | `+48.0%` | `-14.0%` | `+2.3%` | `+11.0%` | `-35.5%` | `-23.2%` |
| 20% | `-1.5%` | `-14.1%` | `+2.4%` | `-26.1%` | `-35.6%` | `-23.1%` |
| 30% | `+15.5%` | `-4.5%` | `+2.7%` | `-13.4%` | `-28.3%` | `-22.9%` |

The biggest shift is from **legacy existing -> updated iso-LP**:

- At `10%`, carbon moves by `-62.0 pp` at multiplier `1.0`
- At `20%`, carbon moves by `-12.6 pp`
- At `30%`, carbon moves by `-20.0 pp`

So even before the pool-adjusted SMT question, the **new 8LP baseline itself**
is materially more favorable to no-SMT than the current repo calibration.

At `10-20%`, the updated `iso_lp` baseline is still close to utilization-invariant
in the raw model. The new cap matters at `30%`: instead of carrying the raw
sub-`1.0` SMT rate into the model, the branch uses `R=1.0`, which softens the
no-SMT advantage there to `-4.5%` at multiplier `1.0` and `-28.3%` at `0.75`.

### 2. Giving SMT the Larger LP Pool Closes a Real but Incomplete Share of That Gap

Moving from updated `iso_lp` to `iso_physical_core` makes SMT more competitive by
raising only the SMT safe oversubscription rate.

In the raw baseline model, the pool-adjusted SMT input shifts carbon by:

- `+16.3 pp` at `10%`
- `+16.6 pp` at `20%`
- `+7.2 pp` at `30%`

That is enough to move the raw no-discount result from about **`-14%` no-SMT
savings** to about **`+2-3%` no-SMT penalty**, but it does **not** erase the
discounted no-SMT advantage:

- At multiplier `0.75`, carbon still favors no-SMT by about `-23%`
- The carbon breakeven multiplier falls from about `1.16` to `0.98` at `10-20%`,
  but only from `1.05` to `0.97` at `30%` because the `iso_lp` side is now
  floored at `R=1.0`

So the physical-core normalization gives back a meaningful chunk to SMT, but it
does not restore the much less favorable legacy baseline.

### 3. Resource Modeling Determines How Much of the SMT Recovery Survives

The input-basis choice matters differently once the later analysis layers are
added.

#### Purpose-Built No-SMT Servers (`resource_scaling`), Carbon vs SMT

| Utilization | Legacy existing @ `1.0` | Updated iso-LP @ `1.0` | Iso-physical-core @ `1.0` | Legacy existing @ `0.75` | Updated iso-LP @ `0.75` | Iso-physical-core @ `0.75` |
|---|---|---|---|---|---|---|
| 10% | `+14.9%` | `+15.2%` | `+17.4%` | `-13.7%` | `-13.6%` | `-11.8%` |
| 20% | `+18.5%` | `+13.0%` | `+19.8%` | `-11.1%` | `-15.2%` | `-10.1%` |
| 30% | `+27.9%` | `+12.7%` | `+21.2%` | `-4.0%` | `-15.4%` | `-9.0%` |

This mode is where the new `30%` calibration matters most:

- **Updated iso-LP vs legacy existing** moves 30%-util carbon by `-15.3 pp` at
  multiplier `1.0` and by `-11.4 pp` at `0.75`
- **Iso-physical-core vs updated iso-LP** gives back `+8.5 pp` at `1.0` and
  `+6.4 pp` at `0.75`

So for purpose-built no-SMT servers, the new 8LP calibration makes no-SMT look
much better at moderate utilization, but the pool-adjusted SMT correction
recovers a large share of that gain.

#### Same-Hardware SMT-Off (`resource_constraints`), Carbon vs SMT

| Utilization | Legacy existing @ `1.0` | Updated iso-LP @ `1.0` | Iso-physical-core @ `1.0` | Legacy existing @ `0.75` | Updated iso-LP @ `0.75` | Iso-physical-core @ `0.75` |
|---|---|---|---|---|---|---|
| 10% | `+11.3%` | `+11.3%` | `+11.3%` | `-16.5%` | `-16.5%` | `-16.5%` |
| 20% | `+25.1%` | `+12.1%` | `+14.8%` | `-6.1%` | `-15.9%` | `-13.9%` |
| 30% | `+30.1%` | `+20.9%` | `+30.0%` | `-2.5%` | `-9.3%` | `-2.4%` |

Two distinct effects show up here:

1. **At 10%**, all bases collapse to the same result because the fixed `768 GB`
   memory ceiling binds both SMT and no-SMT. The requested R values above that
   point no longer matter.
2. **At 30%**, the same-hardware projection is extremely sensitive to the input
   basis:
   - updated `iso_lp`: `+20.9%` at `1.0`, `-9.3%` at `0.75`
   - `iso_physical_core`: `+30.0%` at `1.0`, `-2.4%` at `0.75`

So in the same-hardware transition case, the pool-adjusted SMT correction
still gives back most of the `30%` no-SMT gain created by the updated 8LP
calibration, but the capped `R=1.0` floor makes that updated-8LP gain much
smaller than it would have been under a literal `R=0.86` interpretation.

## Interpretation

### Direct Answer to the Research Question

The new experiment-side realism layer changes the modeling story in **two**
separate steps:

1. **Recalibrating the 8LP baseline** with the new interpolated operating-point
   tables makes no-SMT materially more attractive than the current repo
   calibration suggests.
2. **Giving SMT its larger LP pool** on the same physical cores shifts the
   answer back toward SMT, but only partially in the raw model and only
   partially in the purpose-built model.

The strongest synthesized conclusion is:

> The scheduling-input basis is now a first-order model choice. Updated 8LP
> calibration alone can flip the raw baseline from a no-SMT penalty to a
> no-SMT savings case, while the iso-physical-core correction gives a real but
> incomplete share of that gain back to SMT.

### What Changes Downstream

- **Raw scheduling-only analysis**:
  the old baseline is no longer a good proxy for the updated 8LP story
- **Purpose-built no-SMT savings**:
  the pool-adjusted SMT correction matters most at `20-30%` utilization
- **Same-hardware SMT-off savings**:
  low-utilization results are insensitive because memory binds first, while
  moderate/high-utilization results are highly sensitive to the chosen basis

### Practical Read

If the goal is to answer "what is the direct cost of VP constraints at fixed LP
pool size?", the updated `iso_lp` configs are the right additive baseline.

If the goal is to answer "on the same physical machine, how much of the no-SMT
advantage survives once SMT keeps its extra LP exposure?", the `iso_physical_core`
configs are the right additive follow-up.

Those are **different questions**, and this repo now has separate config trees
for both.

## What This Does Not Address

1. **Cross-application calibration in the model**: this doc still uses Go-CPU as
   the calibration workload, just like [02](02_scheduling_constraints_oversub.md).
2. **A full re-threading of [03](03_vcpu_demand_discount.md) and
   [03a](03a_constrained_savings.md)**: those documents remain intact; this note
   overlays a sensitivity branch on top of them.
3. **Mixed-fleet optimization under the new basis sets**: the multi-cluster
   configs still use the earlier calibration and would need a separate follow-up
   if this input-basis question becomes part of the heterogeneous-fleet story.

[exp-sched]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_impact.md
[exp-iso]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_iso_physical_core.md
[exp-csv]: /Users/jaylenw/Code/atf-benchmarking/scripts/analysis/scheduling_constraints_smt/cross_app_iso_physical_core_threshold_1.0pct.csv
