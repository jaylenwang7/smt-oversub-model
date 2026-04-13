# SMT vs No-SMT TCO/Carbon Tradeoff Analysis

This is the **spine** document for the SMT vs no-SMT analysis conducted in this
modeling repo. It provides high-level context, common terminology, and navigation
to sub-documents that each address a progressively more nuanced question about the
tradeoff.

**Audience**: Anyone trying to understand the TCO and carbon implications of
enabling vs disabling SMT in a cloud server deployment, and how experimental
measurements from the companion benchmarking repo feed into the model.

---

## Table of Contents

1. [Context and Motivation](#context-and-motivation)
2. [Terminology](#terminology)
3. [Repository Map](#repository-map)
4. [Analysis Progression](#analysis-progression)
5. [Sub-Document Index](#sub-document-index)
6. [Experimental Data Sources](#experimental-data-sources)
7. [How to Add a New Analysis](#how-to-add-a-new-analysis)

---

## Context and Motivation

This analysis is part of a PhD research project investigating CPU oversubscription
in cloud environments. The broader research question is:

> If a cloud provider only has historical VM-level observations, how can it
> oversubscribe CPU safely while still capturing real TCO / carbon reductions?

A key sub-question -- the one this analysis spine addresses -- is:

> Should a cloud provider enable or disable SMT on its servers, and how does that
> choice interact with oversubscription headroom, power consumption, and customer
> vCPU demand to affect total fleet carbon and TCO?

**Framing**: We are primarily investigating the case of taking an existing
SMT-capable CPU (AMD Genoa 80-core) and choosing whether to **enable or disable
SMT** on it. This is not a comparison of fundamentally different chip designs; both
configurations use the same physical die, same process node, and same server
chassis. The "no-SMT" option simply disables the second hardware thread per core,
which halves the exposed logical processors but eliminates SMT co-running
overheads.

The analysis builds in layers of complexity. Each sub-document adds one new
consideration to the tradeoff, building on the findings of the previous layer.

### Two Repositories

The analysis spans two repositories:

| Repository | Role | Location |
|---|---|---|
| **smt-oversub-model** (this repo) | TCO/carbon fleet model: takes processor specs, workload parameters, and oversubscription ratios as inputs and produces server counts, carbon, and cost | `/Users/jaylenw/Code/smt-oversub-model` |
| **atf-benchmarking** | Experimental benchmarking framework: runs real applications on real hardware (CloudLab c6620) to measure steal time, power, and throughput under SMT and no-SMT configurations | `/Users/jaylenw/Code/atf-benchmarking/scripts/` |

The experimental repo produces the **measurements** (oversubscription headroom,
power curves, performance ratios) that parameterize the model in this repo. Where
a sub-document uses experimental data, it links to the specific experimental doc
and data files.

---

## Terminology

These terms are used consistently across all sub-documents.

### Hardware Concepts

| Term | Definition |
|---|---|
| **Physical core** | One CPU core on the die. Contains execution units, caches, etc. |
| **Hardware thread / LP (Logical Processor)** | One schedulable execution context. With SMT, each physical core exposes 2 LPs (sibling threads). Without SMT, each core exposes 1 LP. |
| **SMT (Simultaneous Multi-Threading)** | Hardware feature that exposes 2 (or more) hardware threads per physical core, allowing them to share execution resources. Intel's implementation is called Hyper-Threading. |
| **HW threads** | Total logical processors on a server: `physical_cores x threads_per_core`. E.g., 80 cores x 2 tpc = 160 HW threads (SMT), or 80 cores x 1 tpc = 80 HW threads (no-SMT). |

### Virtualization Concepts

| Term | Definition |
|---|---|
| **vCPU / VP (Virtual Processor)** | A virtual CPU assigned to a VM. The hypervisor schedules VPs onto LPs. |
| **Oversubscription ratio (R)** | The ratio of allocated vCPUs to available physical CPUs (pCPUs): `R = total_vCPUs / available_pCPUs_per_server`. R=1.0 means no oversubscription; R=2.0 means 2 vCPUs per pCPU. |
| **Steal time / CWT (CPU Wait Time)** | Time a VP is runnable but cannot execute because its LP is busy with another VP. The key QoS metric for oversubscription safety. |
| **pCPU** | A physical CPU available for VM scheduling. Equals HW threads minus `thread_overhead` (reserved for hypervisor). |
| **thread_overhead** | pCPUs reserved for host/hypervisor use, not available for VM scheduling (modeled as 10% of HW threads). |

### Scheduling Constraint Concepts

| Term | Definition |
|---|---|
| **Core scheduling** | Linux kernel feature that prevents tasks with different security "cookies" from co-executing on sibling SMT threads of the same physical core. |
| **VP constraints** | The most restrictive (and cloud-realistic) scheduling mode: each pair of sibling vCPUs within a VM gets its own cookie, preventing *any* cross-VP co-execution on the same physical core's SMT threads. |
| **VM constraints** | Less restrictive: all vCPUs of a VM share one cookie; only cross-VM co-execution is prevented. |
| **No constraints** | No core scheduling; any VP can co-execute with any other on sibling threads. Not realistic for multi-tenant cloud. |

### Model Concepts

| Term | Definition |
|---|---|
| **vCPU demand multiplier** | A scaling factor on total vCPU demand. Values < 1.0 represent "demand compression" -- users needing fewer vCPUs because each vCPU delivers more performance (relevant for no-SMT). |
| **vCPU discount** | `1 - (1/R)` where R is the no-SMT/SMT performance ratio. If no-SMT delivers 1.36x the throughput, users need 1/1.36 = 0.735x as many vCPUs, i.e., a 26.5% discount. Equivalently, `vcpu_demand_multiplier = 1 - discount`. |
| **Power proportionality** | How linearly server power scales with utilization. A perfectly proportional server uses 0W at idle and max at full load. Real servers have significant idle power. |
| **Embodied carbon** | CO2 emissions from manufacturing the server (CPU, memory, SSD, chassis, rack). Scales with server count. |
| **Operational carbon** | CO2 emissions from electricity consumed during the server's operational lifetime. Scales with power consumption. |
| **TCO (Total Cost of Ownership)** | Capital cost (server purchase) + operational cost (electricity) over the server lifetime. |

### Processor Configurations (Genoa Baseline)

All analyses in this spine use an AMD EPYC Genoa 80-core single-socket server
as the reference platform, with per-component cost and power data from the
`frugal-model` dataset.

| Parameter | SMT | No-SMT |
|---|---|---|
| Physical cores | 80 | 80 |
| Threads per core | 2 | 1 |
| HW threads | 160 | 80 |
| Thread overhead | 16 (10%) | 8 (10%) |
| Available pCPUs | 144 | 72 |
| DIMMs / SSDs | 12 / 6 | 6 / 3 |
| Idle power (W) | ~244 | ~215 |
| Max power (W) | ~763 | ~651 |
| Embodied carbon/server | ~1,783 kg | ~1,120 kg |
| Server cost | ~$11,441 | ~$7,982 |

Processor definitions: [`configs/shared/genoa_processors.jsonc`](../../configs/shared/genoa_processors.jsonc)

**Key modeling choice**: The no-SMT server is assumed to be the *same physical
chassis* with SMT disabled, but with memory and SSD scaled proportionally to HW
thread count (half the DIMMs and SSDs, since there are half as many threads to
serve). This means lower per-server embodied carbon and cost, but also half the
vCPU capacity per server.

---

## Repository Map

```
smt-oversub-model/
  configs/
    shared/genoa_processors.jsonc         # Processor definitions (all analyses)
    oversub_analysis/genoa/
      no_oversub_comparison.jsonc          # [01] Naive comparison
      linear/
        util_10_pct_linear.jsonc           # [02] 10% util, scheduling constraints
        util_20_pct_linear.jsonc           # [02] 20% util, scheduling constraints
        util_30_pct_linear.jsonc           # [02] 30% util, scheduling constraints
        nosmt_oversub_sweep_constrained_vs_scaled_linear.jsonc  # [02b] Savings scaling
        savings_curve.jsonc                # [03] Multi-util savings curves
        breakeven_curve_comparison.jsonc   # [03] Breakeven curves
        ...
  results/
    oversub_analysis/genoa/
      no_oversub_comparison/               # [01] Output
      linear/
        util_10pct_linear/                 # [02] Output
        util_20pct_linear/                 # [02] Output
        util_30pct_linear/                 # [02] Output
        nosmt_oversub_sweep_constrained_vs_scaled_linear/  # [02b] Output
        savings_curve/                     # [03] Output
        breakeven_curve_comparison/        # [03] Output
        ...
  docs/
    analysis/
      SMT_VS_NOSMT_ANALYSIS.md            # THIS FILE (spine)
      01_naive_comparison.md               # Sub-doc 1
      02_scheduling_constraints_oversub.md # Sub-doc 2
      02a_resource_modeling.md             # Sub-doc 2a (side: resource modeling)
      02b_oversub_savings_scaling.md       # Sub-doc 2b (side: savings scaling)
      03_vcpu_demand_discount.md           # Sub-doc 3
```

---

## Analysis Progression

The analysis builds in layers. Each layer adds one new consideration:

```
Layer 1: Naive Comparison (no oversubscription)
  Question: If you just disable SMT, what happens to fleet carbon/TCO?
  Answer:   No-SMT is ~48-56% worse (needs ~2x servers)
            |
            v
Layer 2: + Scheduling Constraints (oversubscription)
  Question: If VP constraints limit SMT oversub but no-SMT can oversub more,
            does higher no-SMT oversub close the gap?
  Answer:   Yes -- at some utilization points, no-SMT with higher R
            breaks even or beats SMT on carbon/TCO
            |
            +--- Side: Resource Modeling (02a)
            |    How should per-server memory/SSD costs be modeled
            |    when R > 1.0? Fixed, scaled, or constrained.
            |
            +--- Side: Savings Scaling (02b)
            |    Do carbon/TCO savings track proportionally with
            |    server reduction? No -- resource costs and power
            |    curves create sublinear returns.
            |
            v
Layer 3: + vCPU Demand Discount (performance effect)
  Question: Since no-SMT LPs are stronger, users may need fewer vCPUs.
            Does this demand compression change the answer?
  Answer:   It shifts breakeven significantly in no-SMT's favor.
            Experimental data (geomean ~26.5% discount) often puts
            realistic workloads past the breakeven point.
```

Each layer's sub-document is self-contained but references prior layers for
context. Read them in order for the full progression, or jump to a specific layer
if you already have the background.

---

## Sub-Document Index

| # | Document | Question | Key Inputs |
|---|---|---|---|
| 01 | [Naive Comparison](01_naive_comparison.md) | What is the raw cost of disabling SMT with no oversubscription? | Genoa processor specs, power curves |
| 02 | [Scheduling Constraints + Oversubscription](02_scheduling_constraints_oversub.md) | How do experimentally-measured oversubscription limits change the picture? | Steal-time thresholds from go-cpu iso-physical-core experiments |
| 02a | [Resource Modeling](02a_resource_modeling.md) | How should memory/SSD costs be modeled when oversubscription > 1.0? | Per-component cost structure from processor specs |
| 02b | [Oversubscription Savings Scaling](02b_oversub_savings_scaling.md) | Do carbon/TCO savings track proportionally with server reduction at higher R? | Oversub sweep across fixed/scaled/constrained resource models |
| 03 | [vCPU Demand Discount](03_vcpu_demand_discount.md) | How does no-SMT's higher per-vCPU performance shift the breakeven? | Peak performance ratios from 30-app benchmark suite |

---

## Experimental Data Sources

These are the key experimental documents from the benchmarking repo that feed
into the analyses here. Each is referenced by the sub-document that uses its data.

| Experimental Doc | What It Provides | Used By |
|---|---|---|
| [`scheduling_constraints_smt_impact.md`][exp-sched] | Steal-time thresholds under VP constraints vs no-SMT across 13 applications. Mean no-SMT advantage: +14.1 pp. | [02], [03] |
| [`scheduling_constraints_smt_iso_physical_core.md`][exp-iso] | Max safe VP/LP rates at fixed per-VM utilization targets (5%, 10%, 20%, 30%) for go-cpu under VP 8LP, VP 16LP, and no-SMT 8LP regimes. These VP/LP rates become the oversubscription ratios in the model configs. | [02] |
| [`smt_no_smt_power_proportionality.md`][exp-power] | Package-power vs host-CPU-utilization curves showing no-SMT is more linear. At matched utilization, no-SMT uses 8-29W less. | [01] |
| [`smt_no_smt_peak_performance.md`][exp-perf] | Peak throughput ratios (no-SMT/SMT) across 30 applications. Geomean 1.361x (all), 1.311x (services), 1.406x (batch). Implied vCPU discount: ~24-29%. | [03] |
| [`oversubscription_research_synthesis.md`][exp-synth] | Master synthesis of the broader oversubscription research. Claim status board and document map for all experimental threads. | Context |

[exp-sched]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_impact.md
[exp-iso]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/scheduling_constraints_smt_iso_physical_core.md
[exp-power]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_power_proportionality.md
[exp-perf]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/smt_no_smt_peak_performance.md
[exp-synth]: /Users/jaylenw/Code/atf-benchmarking/scripts/docs/oversubscription_research_synthesis.md

---

## How to Add a New Analysis

This spine is designed to grow. To add a new sub-document:

### Structure

1. **Create the sub-document** as `docs/analysis/NN_short_name.md` where NN is the
   next sequential number (or use a letter suffix like `02a` for a side question
   that branches from an existing layer).

2. **Use the standard template** (see below).

3. **Add an entry** to the [Sub-Document Index](#sub-document-index) table above.

4. **Update the [Analysis Progression](#analysis-progression)** diagram if the new
   doc adds a new layer of complexity. If it's a side question (e.g., "what if we
   change the power curve model?"), note it as a branch rather than a new layer.

### Sub-Document Template

Every sub-document should include these sections:

```markdown
# NN: Title

## Question
> One-sentence question this analysis answers.

## Prerequisites
Which prior sub-documents to read first.

## Key Assumptions
What this analysis assumes (and what it relaxes relative to prior layers).

## Experimental Inputs (if any)
Where the numbers come from, with links to experimental docs and data files.

## Model Configuration
- Config file(s): exact paths
- How to run: exact command(s)
- Output location: exact paths

## Results
Key findings with specific numbers.

## Interpretation
What this means for the SMT vs no-SMT tradeoff.

## What This Does Not Address
Explicit scope limitations that motivate the next layer.
```

### Language Conventions

- Use **"SMT"** and **"no-SMT"** (not "HT" or "non-SMT")
- Use **"VP"** and **"LP"** for virtual and logical processors
- Use **R** for oversubscription ratio (not "OSR" or "oversub")
- Use **"steal time"** (not "CWT") in prose, but note CWT as the formal term
- Reference configs and results by **relative path** from the repo root
- Always include the **exact command** to reproduce an analysis
- When referencing experimental data, link to the **experimental doc** (which
  contains its own reproducibility instructions), not directly to raw data files
