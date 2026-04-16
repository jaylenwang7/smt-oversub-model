# SMT vs No-SMT Fleet Modeling: Results, Methodology, and Assumptions

**Status**: Working draft for collaborator review (April 2026)

---

## 1. Executive Summary and Headline Numbers

This document presents the current results of a fleet-level carbon and TCO model comparing SMT-enabled and SMT-disabled server deployments in a cloud environment. The central question is:

> On the same physical CPU, does disabling SMT and leveraging the resulting higher safe oversubscription headroom produce net carbon and cost savings at fleet scale?

**Headline result (homogeneous no-SMT fleet, purpose-built servers):**

| Average VM utilization | Carbon savings vs SMT | TCO savings vs SMT |
|---|---|---|
| 10% | **~12%** | **~17%** |
| 20% | **~10%** | **~15%** |

These numbers assume:

- Iso-physical-core comparison (same 80-core CPU, SMT gets 160 LPs while no-SMT gets 80 LPs)
- Linear CPU power proportionality for no-SMT (experimentally motivated)
- Geomean vCPU demand discount of ~26.5% (from 30-application benchmark suite)
- Purpose-built no-SMT servers with memory/SSD scaled to vCPU capacity
- Oversubscription ratios derived from a 1% steal-time contention threshold measured on a single benchmark application (go-cpu)

The document builds up to these numbers layer by layer, starting from a ~50% no-SMT *penalty* when neither oversubscription nor demand compression is considered, and progressively introducing the mechanisms that reverse the picture.

**A note on the contention threshold**: All oversubscription ratios in this analysis are derived from a single contention threshold --- a maximum 1% CPU wait time (steal time) --- measured for one application (go-cpu) on one machine (c6620). This threshold implicitly sets the "maximum safe oversubscription ratio" at each utilization level. For instance, at 10% average VM utilization, no-SMT can safely oversubscribe at ~5.6x, while at 20% utilization the safe ratio drops to ~2.8x. The threshold is a first-order simplification: in practice, the safe oversubscription limit depends on application behavior, hardware configuration, and scheduling policy, and would need to be calibrated per deployment. Evaluating the sensitivity of the results to different threshold values is noted as future work in Section 14.

---

## 2. Document Context

This document serves three purposes:

1. **Results communication**: Present the current headline numbers from the SMT vs no-SMT fleet modeling so that collaborators can understand what the model projects and how sensitive those projections are.

2. **Methodology transparency**: Lay out every modeling assumption, data source, and calculation step so that collaborators can verify, sanity-check, and critique the approach.

3. **Progress checkpoint**: Record the current state of the modeling work as of April 2026, including what has been completed, what requires further validation, and what remains as future work.

### Guiding Research Question

This modeling work is part of a broader investigation into CPU oversubscription in cloud environments. The overarching question is:

> If a cloud provider has historical VM-level observations but does not know the applications running inside VMs, how can it safely oversubscribe CPU while capturing real TCO and carbon reductions?

A key sub-question --- the one addressed here --- is:

> Should a cloud provider enable or disable SMT on its servers, and how does that choice interact with oversubscription headroom, power consumption, and customer vCPU demand to affect total fleet carbon and TCO?

The experimental characterization work (conducted on real hardware with real applications) has established that SMT scheduling constraints limit oversubscription headroom, and that disabling SMT removes those constraints while also providing stronger per-vCPU performance. This document quantifies the fleet-level carbon and cost implications of those experimental findings.

---

## 3. Definitions and Recurring Terminology

The following terms are used consistently throughout this document.

### 3.1 Hardware Concepts

| Term | Definition |
|---|---|
| **Physical core** | One CPU core on the die, containing execution units, caches, and other microarchitectural resources. |
| **Hardware thread / LP (Logical Processor)** | One schedulable execution context exposed by the hardware. With SMT enabled, each physical core exposes 2 LPs (sibling threads sharing execution resources). With SMT disabled, each core exposes 1 LP. |
| **SMT (Simultaneous Multi-Threading)** | A hardware feature that exposes multiple hardware threads per physical core, allowing them to share execution resources. Intel's implementation is marketed as Hyper-Threading. AMD EPYC processors implement SMT-2 (two threads per core). |
| **HW threads** | Total logical processors on a server: `physical_cores x threads_per_core`. For example, an 80-core server has 160 HW threads with SMT enabled, or 80 HW threads with SMT disabled. |

### 3.2 Virtualization and Scheduling Concepts

| Term | Definition |
|---|---|
| **vCPU (Virtual CPU)** | A virtual processor assigned to a VM. The hypervisor schedules vCPUs onto LPs. From the VM's perspective, a vCPU is an independent processor. |
| **pCPU (Physical CPU)** | A logical processor available for VM scheduling, after subtracting those reserved for the hypervisor. `pCPUs = HW_threads - thread_overhead`. |
| **Thread overhead** | LPs reserved for host/hypervisor use, not available for VM scheduling. Modeled as 10% of HW threads (e.g., 16 of 160 for SMT, 8 of 80 for no-SMT). |
| **Oversubscription ratio (R)** | The ratio of allocated vCPUs to available pCPUs per server: `R = total_allocated_vCPUs / available_pCPUs`. R = 1.0 means no oversubscription (each vCPU maps to one pCPU); R = 2.0 means two vCPUs share each pCPU on average. |
| **Steal time / CPU wait time (CWT)** | The fraction of time a vCPU is runnable (wants to execute) but cannot because its LP is occupied by another vCPU. This is the primary quality-of-service metric for oversubscription safety. |
| **Contention threshold** | The maximum tolerable steal time, set at **1%** in this analysis. At a given average VM utilization, the contention threshold determines the maximum safe oversubscription ratio: the highest R at which steal time remains below 1%. See Section 5 for details. |
| **VP constraints / Core scheduling** | A Linux kernel security feature that prevents tasks with different security domains from co-executing on sibling SMT threads of the same physical core. In cloud deployments, this is the operationally realistic minimum: vCPUs from different VMs must not share a physical core's SMT threads simultaneously. This constraint limits SMT oversubscription by creating scheduling conflicts. It is irrelevant for no-SMT, where each LP is the sole thread on its core. |

### 3.3 Model Concepts

| Term | Definition |
|---|---|
| **vCPU demand multiplier (M)** | A scaling factor on total fleet vCPU demand. `M < 1.0` represents demand compression: users needing fewer vCPUs because each vCPU delivers more performance. `M = 0.75` means 25% fewer vCPUs are needed (a 25% demand discount). |
| **vCPU demand discount** | The percentage reduction in vCPU demand under no-SMT, defined as `1 - M`. If the no-SMT/SMT peak throughput ratio is *r*, then `M = 1/r` and the discount is `1 - 1/r`. A throughput ratio of 1.36x implies a 26.5% discount. |
| **Proportional scaling assumption** | The assumption that users scale their VM count (or VM size) proportionally with per-vCPU performance. If no-SMT delivers 1.36x the throughput per vCPU, users are assumed to need 1/1.36 = 0.735x as many vCPUs. **[Requires validation]**: This assumption has not been empirically validated against real fleet behavior; see Section 14 for the specific validation questions. |
| **Power proportionality** | How linearly server power scales with CPU utilization. A perfectly power-proportional server would draw zero watts at idle and maximum at full load. Real servers have significant idle power draw (30-40% of max). |
| **Embodied carbon** | CO2-equivalent emissions from manufacturing the server hardware (CPU die, memory DIMMs, SSDs, NIC, chassis, rack). Scales with the number of servers in the fleet. |
| **Operational carbon** | CO2-equivalent emissions from electricity consumed during the server's operational lifetime. Scales with per-server power draw and fleet size. |
| **TCO (Total Cost of Ownership)** | Capital cost (server purchase) plus operational cost (electricity) over the server's lifetime. |
| **Iso-LP comparison** | Holding the LP pool size fixed between SMT and no-SMT (e.g., both get 8 LPs). Useful for isolating the pure scheduling constraint cost, but not physically realistic since disabling SMT on the same hardware halves the LPs. |
| **Iso-physical-core comparison** | Holding physical cores fixed and letting SMT expose its full LP count (e.g., 8 physical cores yield 16 LPs with SMT and 8 LPs without). This is the operationally realistic "disable SMT on the same CPU" scenario. |

---

## 4. Modeling Framework

### 4.1 What the Model Computes

The model answers a steady-state question: given a fleet of servers that must host a fixed pool of vCPUs at a fixed average utilization, **what is the total carbon footprint and TCO over the server lifetime?**

It proceeds in five deterministic steps:

**Step 1 --- Server count.** How many servers are needed?

```
effective_vcpus = total_vcpus x M
vcpu_capacity_per_server = available_pCPUs x R
num_servers = ceil(effective_vcpus / vcpu_capacity_per_server)
```

**Step 2 --- Utilization.** What is the average per-server CPU utilization?

```
effective_utilization = min(1.0, (effective_vcpus x avg_vm_util) / (num_servers x available_pCPUs))
```

**Step 3 --- Power.** What does each server draw at that utilization?

```
P(u) = P_idle + (P_max - P_idle) x f(u)
```

where `f(u)` is a curve mapping utilization in [0, 1] to a power factor in [0, 1]. Each server component (CPU, memory, SSD, NIC, chassis) has its own idle power, max power, and curve shape. Total server power is the sum of all component powers at the effective utilization.

**Step 4 --- Carbon.** Embodied (manufacturing) plus operational (electricity) emissions.

```
embodied_carbon = num_servers x embodied_carbon_per_server
energy_kwh = num_servers x power_per_server_W x lifetime_hours / 1000
operational_carbon = energy_kwh x carbon_intensity_g_per_kwh / 1000
total_carbon = embodied_carbon + operational_carbon
```

**Step 5 --- TCO.** Capital (purchase) plus operational (electricity) cost.

```
capital_cost = num_servers x server_cost
operational_cost = energy_kwh x electricity_cost_per_kwh
total_cost = capital_cost + operational_cost
```

The model is deterministic: identical inputs always produce identical outputs. There is no simulation, randomness, or time-varying behavior.

### 4.2 What the Model Does Not Include

- **PUE (Power Usage Effectiveness)**: Cooling and datacenter overhead are not explicitly modeled. If needed, PUE can be folded into power values (e.g., multiply by 1.12 for a PUE of 1.12).
- **Non-electricity operational costs**: Staffing, networking, software licensing, and maintenance are excluded. Only server purchase cost and electricity cost are modeled.
- **Time-varying behavior**: The model uses a single average utilization for the entire fleet lifetime. It does not capture diurnal patterns, workload bursts, or autoscaling.
- **Redundancy and fault tolerance**: The model computes the minimum server count. No N+1 spare capacity or failure domain overhead is included.

### 4.3 Reference Platform: AMD EPYC Genoa 80-Core

All analyses use a single-socket AMD EPYC Genoa 80-core server as the reference platform. The "no-SMT" configuration is the same physical CPU with SMT disabled --- not a different chip design. Both configurations use the same die, same process node, and same server chassis.

| Parameter | SMT enabled | SMT disabled |
|---|---|---|
| Physical cores | 80 | 80 |
| Threads per core | 2 | 1 |
| HW threads | 160 | 80 |
| Thread overhead (10%) | 16 | 8 |
| Available pCPUs | 144 | 72 |

**Key modeling choice (purpose-built no-SMT)**: For the primary analysis, the no-SMT server is assumed to have memory and SSD scaled proportionally to HW thread count --- half the DIMMs and SSDs, since there are half as many threads to serve. This models ordering servers configured for no-SMT vCPU density, where each vCPU is entitled to a fixed allocation of memory and storage. A separate analysis (Section 11) examines the alternative: disabling SMT on existing SMT-provisioned servers without changing the installed memory or storage.

### 4.4 Per-Component Cost and Carbon Data

All per-component data (power, embodied carbon, server cost) comes from the **GreenSKU** dataset, a per-component server carbon and cost model developed through a prior Microsoft research collaboration [^greensku]. GreenSKU provides validated, industry-sourced data for datacenter server hardware and has been used in published research.

[^greensku]: GreenSKU is a per-component server lifecycle model that provides validated per-component power draw, embodied carbon (manufacturing emissions), and capital cost data for datacenter server hardware. The specific extraction uses per-component YAML data files from the published dataset.

The reference server is the `Genoa-default-1S` configuration:

| Component | Specification |
|---|---|
| CPU | AMD EPYC Genoa, 80 physical cores, 1 socket |
| Memory | 12 x 64 GB DDR5 4800 MHz DIMMs = 768 GB (SMT) / 6 DIMMs = 384 GB (no-SMT) |
| Storage | 6 x E1.S 2 TB SSDs = 12 TB (SMT) / 3 SSDs = 6 TB (no-SMT) |
| NIC | 200G Glacier Peak |
| Chassis | C2195 1U server (includes 2% repair overhead) |
| Rack | E4000 rack, 34 servers per rack |

#### Per-Component Embodied Carbon (kg CO2-equivalent per server)

| Component | Per unit | Count (SMT) | Count (no-SMT) | Total (SMT) | Total (no-SMT) |
|---|---|---|---|---|---|
| CPU die | 34.2 kg | 1 | 1 | 34.2 kg | 34.2 kg |
| Memory (DIMM) | 59.1 kg | 12 | 6 | 709.2 kg | 354.6 kg |
| SSD | 103.0 kg | 6 | 3 | 618.0 kg | 309.0 kg |
| NIC | 115.0 kg | 1 | 1 | 115.0 kg | 115.0 kg |
| Chassis | 255.5 kg | 1 | 1 | 255.5 kg | 255.5 kg |
| Rack share | 51.9 kg | 1 | 1 | 51.9 kg | 51.9 kg |
| **Total** | | | | **1,784 kg** | **1,120 kg** |

Rack share = 1,765 kg total rack carbon / 34 servers per rack.

**Structural observation**: The per-server fixed costs (CPU die + NIC + chassis + rack = 457 kg) are identical for both configurations. They represent 26% of SMT per-server carbon but 41% of no-SMT per-server carbon. This fixed-cost dilution is a structural advantage of SMT: more threads per server means fixed overhead is amortized over more vCPUs.

#### Per-Component Server Cost (USD per server)

| Component | Per unit | Count (SMT) | Count (no-SMT) | Total (SMT) | Total (no-SMT) |
|---|---|---|---|---|---|
| CPU | $1,487 | 1 | 1 | $1,487 | $1,487 |
| Memory (DIMM) | $440 | 12 | 6 | $5,280 | $2,640 |
| SSD | $273 | 6 | 3 | $1,636 | $818 |
| NIC | $1,022 | 1 | 1 | $1,022 | $1,022 |
| Chassis | $1,505 | 1 | 1 | $1,505 | $1,505 |
| Rack share | $510 | 1 | 1 | $510 | $510 |
| **Total** | | | | **$11,440** | **$7,982** |

#### Per-Component Power (Watts)

Power is modeled per component, with each component having an idle and max power draw. Idle power is derived from component TDP using derate factors at zero utilization (from GreenSKU spec_derate curves). All values include a 1.05x PSU efficiency loss factor.

| Component | Idle (W) | Max (W) | Notes |
|---|---|---|---|
| **SMT (12 DIMMs, 6 SSDs)** | | | |
| CPU | 94 | 315 | 300W TDP x 1.05 PSU |
| Memory | 39 | 131 | 12 DIMMs x 10.4W TDP x 1.05 |
| SSD | 19 | 94 | 6 SSDs x 15W TDP x 1.05 |
| NIC | 32 | 107 | 102W TDP x 1.05 |
| Chassis | 60 | 116 | DC-SCM 35W + Fan 75W x 1.05 |
| **Total** | **244 W** | **763 W** | |
| **No-SMT (6 DIMMs, 3 SSDs)** | | | |
| CPU | 94 | 315 | Same die; disabling SMT does not change the power envelope |
| Memory | 20 | 66 | 6 DIMMs (half of SMT) |
| SSD | 9 | 47 | 3 SSDs (half of SMT) |
| NIC | 32 | 107 | Same NIC |
| Chassis | 60 | 116 | Same chassis |
| **Total** | **215 W** | **651 W** | |

### 4.5 Power Curve Models

Two power curve models are used for the CPU component:

1. **Polynomial (default for SMT)**: A sublinear (concave) curve reflecting the well-documented pattern where power rises steeply at low utilization then flattens. This is the standard shape observed in SPECpower benchmarks.

2. **Linear (used for no-SMT)**: `P(u) = P_idle + (P_max - P_idle) x u`. Motivated by experimental power measurements on a c6620 server showing that no-SMT CPU power scales nearly linearly with utilization (R^2 = 0.998 for linear fit), while SMT follows a concave curve (linear R^2 = 0.896, requiring a quadratic fit for R^2 = 0.999) [^power-exp].

[^power-exp]: Experimental power measurements from stress-ng load sweeps on a c6620 (Intel Xeon, 28 cores). At matched realized host utilization, no-SMT uses 8--29W less CPU package power. The qualitative finding (no-SMT is more linear) is expected to be architecture-general, but the exact coefficients are specific to that platform.

Non-CPU components (memory, SSD, NIC, chassis) use the polynomial curve by default.

**Caveat**: The power measurements are from a c6620 (Intel Xeon, 28 cores), not a Genoa (AMD EPYC, 80 cores). The model uses Genoa-derived power breakdown values with only the curve *shape* informed by the c6620 experiment. Validating no-SMT power linearity on AMD Genoa hardware is noted as a desirable cross-check.

### 4.6 Cost and Environmental Constants

These values are held constant across all analyses:

| Parameter | Value | Rationale |
|---|---|---|
| Carbon intensity | 175 g CO2/kWh | Moderate mixed-grid value derived from the GreenSKU pipeline. For reference: very clean grids are ~20--50 g/kWh, US average is ~400 g/kWh. At 175 g/kWh, the embodied/operational carbon split is roughly balanced. |
| Electricity cost | $0.28/kWh | Representative commercial/industrial rate from GreenSKU defaults. |
| Server lifetime | 6 years (52,560 hours) | Within the typical 4--7 year range for cloud datacenter hardware. Consistent with GreenSKU defaults. |
| Total vCPUs | 100,000 | Fixed fleet size. Large enough that ceiling-function rounding effects are small (<1% of server count). |

---

## 5. The Contention Threshold: From Experiments to Oversubscription Limits

All oversubscription ratios used in this analysis are derived from experimental measurements of steal time (CPU wait time) on real hardware running real applications. This section explains the experimental setup, what the "contention threshold" is, and the assumptions it introduces.

### 5.1 What the Contention Threshold Is

The **contention threshold** is a quality-of-service boundary: the maximum acceptable steal time for VMs on an oversubscribed server. In this analysis, it is set at **1% steal time**, which is the standard used in Azure production environments. Below this threshold, VMs experience negligible performance degradation from CPU sharing. Above it, contention begins to materially affect VM performance.

At a given average VM utilization, the contention threshold determines the **maximum safe oversubscription ratio (R)**. This is the highest R at which the measured steal time stays below 1%. As utilization increases, less oversubscription headroom is available, so the safe R decreases.

The relationship between utilization and safe R is highly nonlinear. At low utilization (e.g., 10%), VMs rarely compete for CPU time, so aggressive oversubscription is safe. At moderate utilization (e.g., 30%), VMs compete more frequently, and the safe R drops sharply.

### 5.2 Experimental Setup

The oversubscription ratios are derived from experiments on a CloudLab c6620 server using a synthetic CPU-bound benchmark (go-cpu). The experimental setup uses 8-vCPU/8-core VMs in a client-server configuration with realistic load generation. VMs are pinned to specific LP sets to create controlled oversubscription scenarios.

Two scheduling regimes are compared:

- **SMT with VP constraints**: Core scheduling is enabled with topology-aware VP constraints --- the cloud-realistic minimum. vCPUs from different VMs cannot co-execute on sibling SMT threads.
- **No-SMT**: Core scheduling is irrelevant since each LP is the sole thread on its core.

The experiments measure the VP/LP rate (effectively the oversubscription ratio) at which steal time crosses the 1% threshold, for each regime at fixed per-VM utilization targets (5%, 10%, 20%, 30%).

### 5.3 Iso-LP vs Iso-Physical-Core Calibration

The experiments support two distinct comparisons, corresponding to two different questions:

**Iso-LP (8 LPs for both)**: Both SMT and no-SMT are given the same LP pool size (8 LPs from 4 physical cores under SMT, 8 LPs from 8 physical cores under no-SMT). This isolates the pure cost of VP scheduling constraints but is not physically realistic, since disabling SMT on 8 physical cores would yield 8 LPs while keeping SMT enabled would yield 16 LPs.

**Iso-physical-core (same 8 physical cores, SMT gets 16 LPs)**: SMT exposes its full 16 LPs from 8 physical cores, while no-SMT exposes 8 LPs from the same cores. This is the operationally realistic "disable SMT on the same CPU" scenario. SMT benefits from a larger LP pool, which gives it more scheduling flexibility and higher safe R values.

The oversubscription ratios used in this analysis (and the resulting headline numbers) come from the **iso-physical-core** calibration, as it represents the realistic deployment decision:

| Avg VM utilization | SMT safe R (16 LP) | No-SMT safe R (8 LP) |
|---|---|---|
| 10% | 3.32 | 5.58 |
| 20% | 1.66 | 2.79 |
| 30% | 1.11 | 1.86 |

These values are interpolated from the experimental steal-time curves at the 1% threshold, using go-cpu operating-point tables.

### 5.4 Assumptions and Limitations of the Contention Threshold

The contention threshold approach introduces several assumptions that should be noted:

**Assumption 1: Single application calibration.** The safe R values are derived from go-cpu, a synthetic CPU-bound benchmark. Go-cpu sits near the cross-application median in terms of steal-time behavior (its no-SMT/VP ratio of 2.16x is close to the 13-application mean of 2.34x), but different applications produce different steal-time curves at the same utilization. For example, across 13 tested applications at 10% utilization under the iso-physical-core regime, the no-SMT safe VP/LP rate ranges from 3.0x (Elasticsearch, worst case) to 7.9x (PgBench, best case), with go-cpu at 5.6x. The model uses a single application's calibration as representative; a production deployment would encounter a mix of applications.

**Assumption 2: Single machine.** The experiments are conducted on a c6620 server (Intel Xeon, 28 physical cores). The qualitative findings (VP constraints limit SMT oversubscription, no-SMT permits higher R) are expected to generalize, but the exact R values at each utilization level will differ on other hardware --- particularly on the Genoa 80-core platform used in the model, which has a significantly larger LP pool. Larger LP pools generally permit higher safe oversubscription due to statistical smoothing effects (law of large numbers applied to VP demand).

**Assumption 3: Fixed 1% threshold.** The analyses assume a single, fixed steal-time threshold of 1%. In practice, different cloud providers or service tiers might use different thresholds. A more permissive threshold (e.g., 5%) would allow higher R values for both configurations but would disproportionately benefit SMT (which loses more headroom to VP constraints); a stricter threshold (e.g., 0.1%) would reduce both. The sensitivity of the results to different threshold values has not yet been evaluated. **[Future work]**: Evaluate how the headline savings change across a range of contention thresholds.

**Assumption 4: Steady-state utilization.** The experiments measure steal time at fixed per-VM utilization targets. Real cloud workloads have time-varying utilization, which means the effective safe R at any moment depends on instantaneous --- not average --- demand patterns.

---

## 6. Layer 1: Baseline Penalty --- Disabling SMT with No Oversubscription

### Question

> If you take an SMT-enabled server and simply disable SMT --- with no oversubscription in either case and no adjustment to vCPU demand --- what is the raw carbon and TCO penalty?

### Approach

Both configurations run at R = 1.0. The same total vCPU demand (100,000) must be served by both. This is the most conservative comparison and the most favorable framing for SMT: no scheduling constraints are relevant (since there is no oversubscription), and no-SMT gets no credit for its higher per-vCPU performance.

### Results

| Metric | SMT (R = 1.0) | No-SMT polynomial (R = 1.0) | No-SMT linear (R = 1.0) |
|---|---|---|---|
| Servers | 70 | 139 (+99%) | 139 (+99%) |
| Carbon | 350,000 kg | 544,900 kg (**+56%**) | 515,100 kg (**+47%**) |
| TCO | $1.16M | $1.73M (**+49%**) | $1.68M (**+45%**) |

No-SMT needs nearly **2x as many servers** because each server exposes only 72 available pCPUs versus 144 for SMT. The carbon penalty is driven more by operational carbon (2x servers drawing idle power) than by embodied carbon (no-SMT servers are cheaper to manufacture per unit).

The linear power curve model reduces the penalty by ~8 percentage points on carbon compared to the polynomial model, reflecting no-SMT's more power-proportional behavior.

> **Takeaway**: Without oversubscription or demand compression, disabling SMT incurs a **~47--56% carbon penalty and ~45--49% TCO penalty**. This is the "hole" that the subsequent mechanisms must fill.

---

## 7. Modeling Choice: How Oversubscription Changes Per-Server Resources

### Question

> Once CPU is oversubscribed (`R > 1.0`), how should memory, SSD, and per-server costs be modeled, and how far do real savings fall below the naive ideal?

Before introducing the oversubscription layers, one modeling choice needs to be explicit. A higher oversubscription ratio reduces server count, but that alone does not determine what happens to memory, SSD, embodied carbon, or power. The answer depends on how per-server resources are treated once more vCPUs are packed onto each server.

This section is therefore not a separate causal "layer" in the SMT vs no-SMT story. Instead, it defines the accounting framework that the later layers use.

### 7.1 Three Resource Models

When a server hosts more vCPUs than pCPUs, each vCPU still implies some memory and storage entitlement. The model uses three distinct resource treatments:

| Resource model | What stays fixed per server | What scales or caps | Intended interpretation |
|---|---|---|---|
| **Fixed-resource model** | CPU, memory, SSD, NIC, chassis, rack | Nothing; only server count changes | Optimistic upper bound, useful as a structural sensitivity check |
| **Purpose-built scaling model** | CPU, NIC, chassis, rack | Memory and SSD scale with hosted vCPU count | Default model for projected no-SMT deployment |
| **Same-hardware constrained model** | Full SMT-provisioned server configuration | Effective R is capped by the first resource bottleneck | "Disable SMT on existing servers" transition case |

Two clarifications matter for the later sections:

- In the **purpose-built scaling model**, the no-SMT `R = 1.0` base server from Section 4.3 starts with 6 DIMMs and 3 SSDs. As `R` rises, the model scales memory and SSD capacity, cost, carbon, and power linearly with hosted vCPU count. It does **not** model whole-device step effects such as adding one DIMM or one SSD at a time.
- In the **same-hardware constrained model**, no-SMT inherits the full SMT-provisioned server (12 DIMMs and 6 SSDs). This can help at low oversubscription, but it also creates stranded-resource overhead and an eventual hard ceiling. For example, at `R = 1.0`, 72 no-SMT vCPUs at 4 GB/vCPU use only 288 GB of the installed 768 GB, so much of the memory footprint is paid for but unused; at high requested `R`, that same 768 GB caps the server at 192 vCPUs, so effective `R` cannot rise above `192 / 72 = 2.67`.

### 7.2 Ideal Savings vs Modeled Savings as R Increases

To see why the resource model matters, it helps to step back from the SMT vs no-SMT comparison and run a structural sweep on no-SMT alone. This is meant as a structural example for one fixed baseline deployment, not as a claim that the exact percentages are the same for every processor type or utilization level. The tables below use no-SMT at 30% average VM utilization and compare higher `R` values against the no-SMT `R = 1.0` baseline because 30% is the upper end of the main 10/20/30 utilization range used throughout the analysis and makes the non-idealities easy to see.

The **ideal savings** column is the naive upper bound. In simple terms: if `R = 2.0` cuts server count in half, the ideal line assumes that carbon and TCO also fall by exactly 50%. If `R = 5.0` cuts server count by 80%, the ideal line assumes 80% savings. Real modeled savings fall below that line.

The exact gap below ideal does depend on the baseline operating point, especially average utilization, because higher `R` pushes the surviving servers further up the power curve. Additional no-SMT validation sweeps at 10%, 20%, and 30% utilization keep the same qualitative ordering (`ideal > fixed-resource > purpose-built scaling > same-hardware constrained`) and the same memory-driven ceiling in the constrained case, but the exact savings move by several percentage points. For example, at `R = 2.0` under the purpose-built scaling model, carbon savings are `-29.3%` at 10% utilization, `-25.3%` at 20% utilization, and `-23.8%` at 30% utilization, all well below the ideal `-50.0%`.

#### Carbon Savings Across Oversubscription Points

| Requested R | Ideal savings | Fixed-resource model | Purpose-built scaling model | Same-hardware constrained model | Effective R in same-hardware model |
|---|---|---|---|---|---|
| 1.0 | 0.0% | +0.0% | +0.0% | +28.1% | 1.00 |
| 1.5 | -33.3% | -24.1% | -16.8% | -3.4% | 1.50 |
| 2.0 | -50.0% | -37.1% | -23.8% | -20.5% | 2.00 |
| 2.5 | -60.0% | -45.7% | -28.3% | -31.8% | 2.50 |
| 3.0 | -66.7% | -51.9% | -31.8% | -34.9% | 2.67 |
| 5.0 | -80.0% | -70.1% | -45.1% | -34.9% | 2.67 |

#### TCO Savings Across Oversubscription Points

| Requested R | Ideal savings | Fixed-resource model | Purpose-built scaling model | Same-hardware constrained model | Effective R in same-hardware model |
|---|---|---|---|---|---|
| 1.0 | 0.0% | +0.0% | +0.0% | +32.8% | 1.00 |
| 1.5 | -33.3% | -28.2% | -20.2% | -5.3% | 1.50 |
| 2.0 | -50.0% | -42.9% | -28.6% | -25.1% | 2.00 |
| 2.5 | -60.0% | -52.1% | -33.8% | -37.5% | 2.50 |
| 3.0 | -66.7% | -58.5% | -37.6% | -40.7% | 2.67 |
| 5.0 | -80.0% | -74.5% | -48.6% | -40.7% | 2.67 |

These tables show four distinct effects:

1. **The ideal line is very optimistic.** At `R = 2.0`, the naive expectation is 50% lower carbon and TCO. None of the modeled cases achieve that.
2. **The fixed-resource model still falls short of ideal.** Even when memory and SSD do not scale, fewer servers run at higher utilization, so operational energy per surviving server rises.
3. **The purpose-built scaling model falls further below ideal.** The removed servers eliminate fixed components, but memory and SSD are partly redistributed to the surviving servers rather than eliminated.
4. **The same-hardware constrained model has both an initial overhead and a hard ceiling.** At `R = 1.0`, it is worse than the no-SMT baseline because the full SMT-sized memory and SSD footprint remains installed. At higher requested `R`, savings improve, but once memory caps effective `R` at `2.67`, requesting `R = 3.0` or `R = 5.0` produces the same result.

The most important practical comparison is the one used later in the main narrative: at `R = 2.0`, ideal carbon savings would be `-50.0%`, but the **purpose-built scaling model** yields only `-23.8%`. At `R = 5.0`, ideal carbon savings would be `-80.0%`, but the purpose-built result is `-45.1%` and the same-hardware constrained result plateaus at `-34.9%`.

### 7.3 Default Resource Model Used in the Main Results

From this point onward, the main layer-by-layer narrative uses the **purpose-built scaling model** unless stated otherwise. That choice is deliberate:

- It is the appropriate model for the main deployment question: what would a real no-SMT fleet look like if the servers were provisioned for that fleet?
- It preserves per-vCPU memory and storage entitlement instead of silently assuming that denser CPU packing comes "for free."
- It makes the later savings numbers harder to obtain than in the fixed-resource model, which is the conservative choice for the main story.
- It cleanly separates the **same-hardware constrained model** into its own transition-focused analysis in Section 11.

> **Takeaway**: Once `R > 1.0`, oversubscription savings are not equal to server reduction. The later sections therefore use the **purpose-built scaling model** as the default and treat the fixed-resource and same-hardware constrained models as explicit comparison cases.

---

## 8. Layer 2: Scheduling Constraints and Oversubscription Headroom

### Question

> VP scheduling constraints limit how much SMT can safely oversubscribe. No-SMT, free of these constraints, can oversubscribe more aggressively. Under the default resource model from Section 7, does this higher no-SMT oversubscription headroom close the carbon/TCO gap?

### Approach

Using the **purpose-built scaling model** from Section 7 as the default resource treatment, replace `R = 1.0` with the experimentally measured safe `R` values from Section 5.3. SMT uses the VP-constrained iso-physical-core safe `R`; no-SMT uses the unconstrained safe `R`. No-SMT uses the linear CPU power curve. The vCPU demand multiplier is still `M = 1.0` (no demand discount yet).

### Why No-SMT Can Oversubscribe More

Under SMT, VP constraints prevent cross-VM sibling thread co-execution. When the hypervisor cannot place two VMs' vCPUs on the same physical core's two threads simultaneously, scheduling becomes constrained: even at moderate overall utilization, some physical cores sit idle while others are overloaded, because the scheduler cannot freely fill all LP slots. This drives up steal time at lower aggregate utilization levels.

No-SMT eliminates this constraint entirely. Each LP is the sole thread on its physical core, so there are no sibling threads to conflict with. The scheduler can freely assign any vCPU to any available LP. This means steal time rises purely as a function of aggregate demand exceeding aggregate capacity, without the combinatorial scheduling overhead of SMT.

### Results (Iso-Physical-Core, Purpose-Built, No Demand Discount)

| Avg VM util | SMT R | No-SMT R | No-SMT carbon vs SMT | No-SMT TCO vs SMT | No-SMT servers vs SMT |
|---|---|---|---|---|---|
| 10% | 3.32 | 5.58 | **+17.4%** | **+10.8%** | +18.6% |
| 20% | 1.66 | 2.79 | **+19.8%** | **+13.7%** | +18.9% |
| 30% | 1.11 | 1.86 | **+21.2%** | **+15.6%** | +19.3% |

Under the iso-physical-core calibration and the purpose-built scaling model, no-SMT at `M = 1.0` is still clearly behind SMT: about `+17--21%` on carbon and `+11--16%` on TCO. The oversubscription headroom advantage closes a large share of the ~50% gap from Layer 1, but once the non-CPU resource accounting from Section 7 is included, it does not bring no-SMT close to parity by itself.

For context, the same safe-`R` values would look much better under the **fixed-resource model** from Section 7: roughly `+2--3%` on carbon and `-5%` on TCO. That difference is not a change in scheduling behavior; it is purely a change in how memory and SSD are accounted for once more vCPUs are packed onto each server.

> **Takeaway**: Oversubscription headroom alone closes the ~50% baseline gap to about a **~17--21% carbon** and **~11--16% TCO** penalty under the default purpose-built scaling model. The scheduling constraint disadvantage is still a major part of SMT's cost under oversubscription, but it is not sufficient by itself to make no-SMT a win.

---

## 9. Layer 3: vCPU Demand Discount --- The Headline Result

### Question

> Since no-SMT logical processors deliver more performance per vCPU (no co-running overhead from a sibling thread), users may need fewer vCPUs for the same workload. How does this demand compression change the picture?

### 9.1 Experimental Basis for the Demand Discount

Peak throughput was measured for 30 applications (14 services, 16 batch workloads) on a c6620 server, comparing 2-core SMT (4 HW threads / 4 vCPUs) versus 4-core no-SMT (4 HW threads / 4 vCPUs). The key aggregate results:

| Application scope | Count | Geomean no-SMT/SMT ratio | Implied vCPU discount |
|---|---|---|---|
| Services | 14 | 1.31x | 23.7% |
| Batch | 16 | 1.41x | 28.9% |
| All applications | 30 | 1.36x | 26.5% |

A discount of 26.5% means that a workload needing 100 vCPUs under SMT would need only ~74 vCPUs under no-SMT for the same throughput. This translates to `M = 0.735` (rounded to `0.75` for the modeling reference point).

The discount varies widely across applications:

| Application | Perf ratio | vCPU discount | Application type |
|---|---|---|---|
| Neo4j | 1.50x | 33.1% | CPU-bound service |
| Keycloak | 1.43x | 30.3% | CPU-bound service |
| Go-CPU | 1.43x | 30.1% | CPU-bound synthetic |
| MediaWiki | 1.42x | 29.4% | CPU-bound service |
| Postgres | 1.36x | 26.3% | CPU-bound database |
| Elasticsearch | 1.29x | 22.7% | CPU-bound search |
| InfluxDB | 1.17x | 14.1% | CPU-bound time-series DB |
| Go-MemChase | 1.07x | 6.4% | Memory-latency-bound |
| Memcached | 1.02x | 2.1% | Memory-latency-bound |

CPU-intensive workloads consistently show 20--33% discounts. Memory-latency-dominated workloads (Memcached, Go-MemChase) see near-zero discounts because SMT co-running has minimal impact when the bottleneck is memory access latency rather than execution unit contention.

**[Requires validation]**: The proportional scaling assumption --- that users would actually re-size their VMs or reduce VM count in proportion to the per-vCPU performance gain --- has not been empirically validated. In practice, VM sizing depends on pricing models, application architecture, and organizational inertia. Validating this assumption is a priority; one approach is to examine how first-party cloud workloads have historically scaled VM count when migrating between hardware generations with different per-vCPU performance (e.g., Ice Lake to Emerald Rapids). Specific validation questions are noted in Section 14.

### 9.2 Contextualizing the Discount Against Industry SMT Claims

Vendor and academic sources report SMT throughput gains in the range of +10% to +60%, with a cross-source median around +30%.[^smt-survey] However, these figures are typically framed as "aggregate throughput gain on the *same physical core budget*" (e.g., 2 cores with SMT vs 2 cores without SMT), which is a different quantity than the "per-vCPU performance at fixed visible vCPU count" used in this model.

[^smt-survey]: A survey of vendor and academic SMT performance claims finds: Intel reports ~10--30% depending on source and workload class ([Intel HT Technical Guide](https://read.seas.harvard.edu/cs161/2022/pdf/intel-hyperthreading.pdf); [Intel Technology Journal 2002](https://www.intel.com/content/dam/www/public/us/en/documents/research/2002-vol06-iss-1-intel-technology-journal.pdf); [Intel virtualization white paper](https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/virtualization-xeon-core-count-impacts-performance-paper.pdf)); AMD claims "often 30--50%" for EPYC ([AMD EPYC SMT Technology Brief](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/amd-epyc-smt-technology-brief.pdf)); IBM reports 25--40% throughput and 30--60% instructions executed for SMT-2/SMT-4 ([IBM support documentation](https://www.ibm.com/support/pages/power-cpu-memory-affinity-3-scheduling-processes-smt-and-virtual-processors); [IBM AIX docs](https://www.ibm.com/docs/en/aix/7.2.0?topic=concepts-simultaneous-multithreading)). Academic measurements on SPEC CPU2000 multiprogrammed pairs show average speedups of ~1.20x with ranges from 0.86x to 1.58x ([Tuck & Tullsen 2003](https://users.cs.utah.edu/~rajeev/cs7810/papers/tuck03.pdf); [Bulpin 2004](https://pharm.ece.wisc.edu/wddd/2004/06_bulpin.pdf)). HPC applications show 0--22% gains depending on workload ([Saini et al. 2011](https://www.nas.nasa.gov/assets/nas/pdf/papers/saini_s_impact_hyper_threading_2011.pdf)).

To connect vendor-style claims to this model's framing, a first-order translation is needed. If enabling SMT on `N` physical cores raises aggregate throughput by a factor `(1 + g)`, and linear core scaling holds for no-SMT, then the implied no-SMT/SMT ratio at fixed vCPU count is approximately `2 / (1 + g)`, giving `M = (1 + g) / 2`.

| Reported SMT gain (same core budget) | Implied M | Implied vCPU discount |
|---|---|---|
| +10% | 0.55 | 45% |
| +20% | 0.60 | 40% |
| +30% | 0.65 | 35% |
| +40% | 0.70 | 30% |
| +50% | 0.75 | 25% |
| +60% | 0.80 | 20% |

Under this translation:

- The **M = 0.65 marker** ("high discount" / ~35%) corresponds to the **literature median** SMT throughput gain of ~+30%.
- The **M = 0.75 geomean marker** (~25% discount) corresponds to a same-core SMT gain of ~+50%, toward the upper end of mainstream claims but within the AMD and IBM ranges.
- The **M = 0.85 marker** ("low discount" / ~15%) corresponds to ~+70% same-core gain, which is stronger than most reported values and serves as a deliberately conservative bound from the no-SMT perspective.

The experimental geomean of 1.36x (`M = 0.735`, ~26.5% discount) back-translates to an equivalent same-core SMT gain of ~+47%. This is above Intel's typical claims of +10--30% but is explainable: the experiment measures *peak throughput at fixed visible vCPU count* on a small VM (4 vCPUs), where each vCPU is highly sensitive to whether it maps to a full core or a sibling thread. Vendor numbers are typically *machine throughput* claims using full core budgets and multi-socket systems.

**[Requires validation]**: Connecting the per-vCPU peak throughput measurement to real fleet vCPU demand changes requires validating the proportional scaling assumption (see Section 9.1 and Section 14).

### 9.3 Three Reference Points for Sensitivity

The analysis evaluates savings at three reference points that bracket the range of realistic vCPU discounts:

| Label | M | Implied perf ratio | Rationale |
|---|---|---|---|
| **High discount** | 0.65 | ~1.54x | Near the strongest measured service gains; near the literature-median SMT claim |
| **Geomean** | 0.75 | ~1.33x | Close to the all-app experimental geomean (1.36x) |
| **Low discount** | 0.85 | ~1.18x | Conservative; near weakest measured gains (InfluxDB, Go-MemChase) |

### 9.4 Sanity Check: Demand Discount Alone Is Not Sufficient

Before presenting the full results, a sanity check confirms that the demand discount alone --- without oversubscription headroom --- does not make no-SMT obviously better. With both configurations at `R = 1.0` and no-SMT using a linear power curve, this check isolates only the demand-discount mechanism; because there is no oversubscription, it is effectively unchanged by the resource-model choice:

| M | Implied discount | No-SMT carbon vs SMT | No-SMT TCO vs SMT |
|---|---|---|---|
| 0.85 | 15% | +25.8% | +24.1% |
| 0.75 | 25% | +11.0% | +9.5% |
| 0.65 | 35% | -3.8% | -5.1% |

At the geomean discount, no-SMT is still 11% *worse* on carbon without oversubscription. No-SMT only breaks even in this setting at a ~32% discount (`M = 0.676`), which is stronger than the geomean. This confirms that the projected savings in the headline results require *both* the oversubscription headroom advantage (Section 8) and the demand discount --- neither alone is sufficient at realistic geomean values.

### 9.5 Headline Results Under the Default Resource Model

With all three mechanisms active --- the purpose-built scaling model from Section 7, scheduling-constraint-limited oversubscription from Section 8, and experimentally grounded vCPU demand discounts --- the iso-physical-core comparison produces:

**At geomean discount (`M = 0.75`, ~1.33x performance ratio):**

| Avg VM utilization | Carbon savings | TCO savings | Server reduction |
|---|---|---|---|
| 10% | **-11.8%** | **-16.8%** | -11.0% |
| 20% | **-10.1%** | **-14.6%** | -10.7% |
| 30% | **-9.0%** | **-13.2%** | -10.4% |

**At high discount (`M = 0.65`, ~1.54x performance ratio):**

| Avg VM utilization | Carbon savings | TCO savings |
|---|---|---|
| 10% | -23.6% | -27.9% |
| 20% | -22.1% | -26.1% |
| 30% | -21.2% | -24.8% |

**At low discount (`M = 0.85`, ~1.18x performance ratio):**

| Avg VM utilization | Carbon savings | TCO savings |
|---|---|---|
| 10% | -0.1% | -5.7% |
| 20% | +1.9% | -3.2% |
| 30% | +3.0% | -1.8% |

The carbon breakeven `M` (the demand multiplier at which no-SMT exactly matches SMT on carbon) is approximately **0.83--0.85** across utilization levels under the iso-physical-core basis with the purpose-built scaling model. This means a discount of ~15--17% is sufficient for no-SMT to break even on carbon; the geomean discount of 26.5% pushes well past breakeven.

### 9.6 How the Three Mechanisms Stack Up

| Mechanism | Effect on no-SMT competitiveness |
|---|---|
| LP count disadvantage (Section 6) | ~50% carbon penalty --- the baseline "hole" |
| Scheduling constraint headroom only (Section 8) | Reduces the gap to a **~17--21% carbon** and **~11--16% TCO** penalty |
| vCPU demand discount only (Section 9.4, 10% util sanity check) | Still **+11.0% carbon** and **+9.5% TCO** worse at geomean discount |
| Combined headline (Section 9.5) | **~10% carbon, ~15% TCO savings** |

The progression is therefore: Section 6 establishes the baseline penalty; Section 7 defines the non-ideal resource accounting that later sections must obey; Section 8 shows that higher no-SMT oversubscription closes a substantial but incomplete share of the gap; Section 9.4 shows that demand discount alone is also insufficient at the geomean; and Section 9.5 shows that the two mechanisms together tip the result into savings.

> **Takeaway**: With experimentally grounded oversubscription ratios, linear no-SMT power, and the geomean vCPU demand discount, a homogeneous no-SMT fleet saves approximately **10% on carbon and 15% on TCO** at 10--20% average VM utilization, using purpose-built servers under the iso-physical-core comparison.

---

## 10. Scheduling Input Basis Sensitivity

### Question

> How much do the conclusions change depending on whether the oversubscription ratios come from the iso-LP calibration or the iso-physical-core calibration?

### 10.1 Background

As described in Section 5.3, the experimental data supports two calibrations:

| Basis | SMT R at 10% util | No-SMT R at 10% util |
|---|---|---|
| Iso-LP (8 LP for both) | 2.59 | 5.58 |
| Iso-physical-core (16 LP SMT, 8 LP no-SMT) | 3.32 | 5.58 |

The difference is entirely on the SMT side: the iso-physical-core calibration gives SMT a higher safe R because a larger LP pool provides more scheduling flexibility.

### 10.2 Impact on Headline Numbers

Under the **purpose-built scaling model** at `M = 0.75`:

| Basis | 10% util carbon | 20% util carbon | 30% util carbon |
|---|---|---|---|
| Iso-LP | **-13.6%** | **-15.2%** | **-15.4%** |
| Iso-physical-core | **-11.8%** | **-10.1%** | **-9.0%** |
| Difference (pp) | +1.8 | +5.1 | +6.4 |

The iso-physical-core basis reduces no-SMT savings by 2--6 percentage points relative to iso-LP, with the largest impact at 30% utilization where the LP pool size matters most. At 10% utilization, the difference is modest because both calibrations already permit aggressive oversubscription.

The iso-LP calibration makes no-SMT look **more favorable** because it does not give SMT credit for its larger LP pool. The iso-physical-core calibration is the **more operationally realistic** basis for the "should we disable SMT?" decision on the same physical hardware. This is why the headline numbers in Section 1 use the iso-physical-core basis.

### 10.3 Updated vs Legacy Calibration

A separate sensitivity check shows that updating the 8LP baseline itself (from older discrete-point measurements to new interpolated operating-point tables) changes the story materially --- more so than the iso-LP to iso-physical-core step:

- Updating the 8LP baseline shifts carbon by **-12 to -62 pp** (more favorable to no-SMT, depending on utilization)
- The subsequent iso-physical-core correction gives back **+7 to +17 pp** to SMT

The updated calibration is consistently more favorable to no-SMT than the original. The iso-physical-core correction partially recovers SMT's position but does not restore the earlier, less favorable picture.

> **Takeaway**: The scheduling-input basis is a first-order model choice. Updated 8LP calibration alone can flip the raw baseline from a no-SMT penalty to a no-SMT savings case, while the iso-physical-core correction gives a real but incomplete share of that gain back to SMT. **The headline numbers use the most SMT-favorable realistic basis** (iso-physical-core).

---

## 11. Same-Hardware Deployment: Disabling SMT on Existing Servers

### Question

> Instead of purpose-building no-SMT servers with reduced memory and SSD, what if you disable SMT on existing SMT-provisioned hardware? How do the savings change?

### 11.1 Setup

Both SMT and no-SMT run on identical physical servers: 80-core Genoa with 12 DIMMs (768 GB) and 6 SSDs (12 TB). Disabling SMT halves the HW threads (160 to 80) but the installed memory and storage remain. Resource constraints limit effective oversubscription: each vCPU is assumed to need 4 GB memory and 50 GB SSD.

### 11.2 How Constraints Interact with Each Configuration

| Configuration | 10% util | 20% util | 30% util |
|---|---|---|---|
| **SMT**: Requested R | 3.32 | 1.66 | 1.11 |
| **SMT**: Effective R (after memory cap) | 1.33 | 1.66 | 1.11 |
| **SMT**: Memory-constrained? | **Yes** | No | No |
| **No-SMT**: Requested R | 5.58 | 2.79 | 1.86 |
| **No-SMT**: Effective R | 2.67 | 2.79 | 1.86 |
| **No-SMT**: Memory-constrained? | **Yes** | No | No |

At 10% utilization, memory constrains both configurations. But the impact is asymmetric: SMT loses 60% of its requested oversubscription (3.32 to 1.33) while no-SMT loses 52% (5.58 to 2.67). This is because SMT has more pCPUs (144) sharing the same 768 GB of memory, giving only 5.33 GB per pCPU. No-SMT has 72 pCPUs sharing the same memory, giving 10.67 GB per pCPU. Since each vCPU demands 4 GB, SMT can only oversubscribe to 5.33/4.0 = 1.33x before memory runs out, while no-SMT can reach 10.67/4.0 = 2.67x.

### 11.3 Results (Iso-Physical-Core, M = 0.75)

| Avg VM util | Carbon savings (purpose-built scaling) | Carbon savings (same-hardware constrained) | TCO savings (purpose-built scaling) | TCO savings (same-hardware constrained) |
|---|---|---|---|---|
| 10% | -11.8% | **-16.5%** | -16.8% | **-20.8%** |
| 20% | -10.1% | **-13.9%** | -14.6% | **-19.0%** |
| 30% | -9.0% | **-2.4%** | -13.2% | **-5.9%** |

**At 10% utilization, the same-hardware constrained scenario saves *more* than the purpose-built scaling scenario.** This counterintuitive result arises because memory constraints eliminate SMT's oversubscription advantage at low utilization. Both configurations are capped at the same absolute vCPU count per server (192, determined by memory), but no-SMT reaches this with a higher effective R (2.67 vs 1.33), leveling the playing field. Additionally, SMT strands significant SSD capacity (40--50% unused) that no-SMT does not.

**At 30% utilization, same-hardware constrained savings shrink to marginal levels (-2.4% carbon).** Here, neither configuration is memory-constrained, but no-SMT carries the full embodied cost of 12 DIMMs and 6 SSDs per server (1,784 kg instead of 1,120 kg for the purpose-built scaling case). This ~60% higher per-server embodied cost nearly offsets the server count reduction.

> **Takeaway**: Disabling SMT on existing hardware delivers the **largest relative savings at low utilization** (-16.5% carbon, -20.8% TCO at 10%) because memory constraints disproportionately limit SMT. At moderate utilization (30%), same-hardware savings are marginal (-2.4% carbon) because of the overhead from carrying unused resources. **For a "just flip the switch" pilot, low-utilization fleet segments produce the most favorable numbers.**

---

## 12. Mixed Fleet Partitioning

### Question

> Instead of converting the entire fleet to no-SMT, can a mixed fleet --- with both SMT and no-SMT deployment options and some fraction of demand choosing each --- capture additional savings?

### 12.1 Methodology

The fleet's workloads are assumed to have a distribution of vCPU demand discounts, modeled as a uniform distribution from M = 0.50 to M = 1.00 (fleet average M = 0.75, matching the experimental geomean). The mixed-fleet model then uses a **split point** as a proxy for adoption: workloads with M below the split (strong no-SMT advantage) are treated as choosing the no-SMT option, while those above (weak or no advantage) are treated as staying on SMT.

The split point is set by a per-workload breakeven calculation. For carbon, the split is the **carbon breakeven M** --- the discount level at which a single workload evaluated on no-SMT matches the same workload on SMT for carbon. For TCO, the split is the analogous **TCO breakeven M**. Both are computed automatically via binary search.

Each resulting demand pool is evaluated independently with its own processor type, oversubscription ratio, and resource model. Fleet-level totals are the sum across pools.

### 12.2 Results

**Purpose-built no-SMT servers, iso-physical-core basis:**

| Avg VM util | Carbon split | TCO split | Homogeneous no-SMT carbon | Mixed fleet carbon | Carbon improvement (pp) | Homogeneous no-SMT TCO | Mixed fleet TCO | TCO improvement (pp) |
|---|---|---|---|---|---|---|---|---|
| 10% | 0.852 | 0.902 | -11.8% | **-14.6%** | +2.8 | -16.8% | **-17.9%** | +1.1 |
| 20% | 0.834 | 0.879 | -10.1% | **-14.0%** | +3.9 | -14.6% | **-16.7%** | +2.1 |
| 30% | 0.824 | 0.865 | -9.0% | **-13.5%** | +4.5 | -13.2% | **-15.9%** | +2.7 |

**Same-hardware constrained model, iso-physical-core basis:**

| Avg VM util | Carbon split | TCO split | Homogeneous no-SMT carbon | Mixed fleet carbon | Carbon improvement (pp) | Homogeneous no-SMT TCO | Mixed fleet TCO | TCO improvement (pp) |
|---|---|---|---|---|---|---|---|---|
| 10% | 0.898 | 0.946 | -16.5% | **-17.9%** | +1.4 | -20.8% | **-21.2%** | +0.4 |
| 20% | 0.871 | 0.926 | -13.9% | **-16.2%** | +2.3 | -19.0% | **-19.8%** | +0.8 |
| 30% | 0.770 | 0.797 | -2.4% | **-10.1%** | +7.7 | -5.9% | **-11.7%** | +5.8 |

### 12.3 When the Mixed Fleet Matters Most

The mixed fleet advantage is **largest where homogeneous no-SMT is weakest**: at 30% utilization under the iso-physical-core same-hardware constrained scenario, where homogeneous no-SMT saves only 2.4% on carbon and 5.9% on TCO, while the mixed fleet saves 10.1% on carbon and 11.7% on TCO. That 30% case is best read as an upper-end stress case for the 10/20/30 sweep, not as the most representative fleet-wide planning point when average utilization is closer to ~10--11%.

The mechanism is the same in both metrics: the homogeneous approach applies the fleet-average discount (M = 0.75) to every workload, including workloads with little true no-SMT benefit. The mixed fleet avoids that mismatch by treating only the stronger-benefit workloads as choosing no-SMT. The gain is largest when SMT remains competitive enough that this selectivity matters.

The carbon-selected and TCO-selected splits are close, but not identical. The TCO-selected split is usually slightly higher, meaning TCO prefers sending a bit more demand to no-SMT. Even so, the cross-metric sensitivity is small: for example, in the purpose-built case at 20% utilization, the mixed fleet saves 14.0% on carbon at the carbon-selected split and 13.7% on carbon at the TCO-selected split. This indicates that the mixed-fleet conclusion is not fragile to the exact split choice.

The split point is also not highly sensitive near the optimum. Sweep analyses show broad plateaus of near-optimal savings across a range of split points (typically 0.70--0.95), meaning the exact adoption threshold does not need to be precisely calibrated to capture most of the benefit.

### 12.4 Practical Interpretation and Requirements

This section should **not** be interpreted as a live cloud scheduler that transparently places or migrates an existing VM between SMT and no-SMT hosts. That is not the scenario being modeled here.

That kind of dynamic assignment is not realistic for two reasons. First, the difference would not be transparent to the user: SMT vs no-SMT changes the effective per-vCPU performance and can often be inferred from inside the VM through observed throughput, latency, and visible CPU topology or behavior. Second, the current model assumes that some workloads would also reduce requested vCPU count when choosing no-SMT. A cloud provider generally cannot infer the correct resize automatically for the user or silently apply it.

The intended interpretation is instead a **user-visible choice model**. The provider offers both SMT and no-SMT options, potentially with a no-SMT price discount that reflects the provider-side TCO or carbon benefit. The mixed-fleet model then asks: if workloads that see an aggregate benefit from no-SMT choose that option, and if those workloads scale their requested vCPUs according to the modeled performance gain, what steady-state split of fleet demand results?

Under that interpretation, the split point is a proxy for **which workloads would choose no-SMT**, not a policy for live VM placement. What matters operationally is estimating the distribution of workload discounts and how users would respond to a no-SMT offering.

Potential ways to ground that uptake model include:

- Historical evidence on how workloads resized when moved across CPU generations with different per-vCPU performance
- Application-class heuristics or benchmark-backed estimates of which workloads are likely to benefit from no-SMT
- Pricing experiments or product modeling for a no-SMT offering whose discount reflects provider-side TCO or carbon savings

**[Requires validation]**: The uniform discount distribution (M = 0.50 to 1.00) is a modeling assumption, and so is the implied customer-choice rule that workloads with downstream benefit choose no-SMT. A right-skewed discount distribution (most workloads near M = 1.0) would increase the value of offering both options; a left-skewed one would decrease it. The pass-through from provider benefit to customer pricing, and the extent to which users would actually resize VMs when selecting no-SMT, both remain open validation questions.

> **Takeaway**: For a fleet whose average utilization is closer to ~10--11%, the **10% utilization row is the most representative planning point**. Under the **purpose-built scaling model**, the mixed fleet improves on homogeneous no-SMT by **+2.8 pp on carbon** and **+1.1 pp on TCO** at 10% utilization (rising to **+3.9 pp** and **+2.1 pp** at 20%). Under the **same-hardware constrained model**, the incremental gain at 10% utilization is smaller: **+1.4 pp on carbon** and **+0.4 pp on TCO**, because memory already limits both pools. The much larger 30% same-hardware gain (**+7.7 pp carbon**, **+5.8 pp TCO**) is an upper-end case where homogeneous no-SMT becomes weak, not the default fleet-wide expectation. The mixed fleet is therefore an incremental improvement, not a replacement for the homogeneous switch.

---

## 13. Comprehensive Summary of Assumptions

The assumptions underlying the analysis fall into three categories: those grounded in experimental data, those grounded in external datasets, and those requiring further validation.

### 13.1 Experimentally Grounded

| Assumption | Basis | Limitation |
|---|---|---|
| Oversubscription ratios (R values) | Interpolated from steal-time vs utilization curves at 1% CWT threshold | Single application (go-cpu), single machine (c6620). Different applications and larger machines may produce different R values. |
| No-SMT safe R > SMT safe R at matched utilization | Measured across 13 applications, consistent across all | Magnitude varies by application; go-cpu is near-median. |
| Linear no-SMT CPU power curve | R^2 = 0.998 linear fit on c6620 | Different CPU architecture (Intel Xeon) than model platform (AMD Genoa). |
| Peak throughput ratio geomean ~1.36x | Measured across 30 applications on c6620 | Small VM size (4 vCPUs); throughput metric may not generalize to all workload types. |

### 13.2 Grounded in External Data (GreenSKU)

| Assumption | Basis | Limitation |
|---|---|---|
| Per-component embodied carbon | GreenSKU validated dataset | Specific to the `Genoa-default-1S` configuration; other server configurations would produce different breakdowns. |
| Per-component server cost | GreenSKU validated dataset | Pricing may vary by volume, region, and time. |
| Per-component power breakdown | GreenSKU TDP data with derate factors | Derate factors are approximations; real idle power varies with workload and configuration. |
| Carbon intensity (175 g CO2/kWh) | Derived from GreenSKU pipeline | Fleet average; individual datacenters range from ~20 to ~1000 g/kWh. |

### 13.3 Requiring Further Validation

| Assumption | What it affects | How to validate |
|---|---|---|
| **Proportional scaling**: Users scale VM count proportionally with per-vCPU performance | Demand multiplier M; the single most impactful parameter | Examine first-party Azure workload migration data: when workloads move between SKUs with different per-vCPU performance, do they rescale VM count accordingly? Formulate specific questions for first-party workload teams (see Section 14). |
| **Single contention threshold (1% CWT)** | All oversubscription ratios | Evaluate sensitivity by repeating the analysis with thresholds at 0.5%, 2%, 5%. |
| **Go-CPU representativeness** | R values at each utilization level | Repeat with cross-application median R values or per-application R curves. |
| **Power curve generalization** | No-SMT power advantage magnitude | Measure SMT vs no-SMT power proportionality on AMD Genoa hardware. |
| **4 GB/vCPU memory demand** | Memory constraint ceiling in the same-hardware constrained analysis | Validate against actual fleet VM memory size distributions. |
| **Uniform vCPU discount distribution** | Mixed fleet analysis savings magnitude | Characterize the actual distribution of per-workload discounts from fleet data. |

---

## 14. Open Questions and Future Work

### 14.1 Validating the Proportional Scaling Assumption

The key open question is whether the per-vCPU performance advantage of no-SMT translates to proportional demand compression in practice. Specific validation questions for first-party cloud workload teams:

1. When migrating between hardware generations with different per-vCPU performance (e.g., Ice Lake to Emerald Rapids), what effective performance-per-vCPU improvement did workload teams observe?
2. Did those teams scale VM count (or VM size) proportionally to the performance change? If not, what fraction of the performance gain was captured as demand reduction vs absorbed as headroom?
3. Are there workload categories where proportional scaling is a better or worse approximation?

**[Requires validation]**: Connecting peak throughput measurements to fleet-level vCPU demand changes is the most critical open assumption. The modeling results are robust in the sense that they can be re-evaluated at any demand multiplier, but the "headline number" depends on which M is considered realistic.

### 14.2 Contention Threshold Sensitivity

All analyses are conditioned on a single contention threshold (1% steal time) measured for a single application. Future work should:

1. Evaluate how the headline savings change across a range of thresholds (0.5%, 1%, 2%, 5%).
2. Determine whether different applications require different thresholds or whether a single conservative threshold (covering worst-case application behavior) is sufficient.
3. Investigate how the threshold maps to VM-visible performance degradation for different application types.

This work connects to the broader oversubscription research question: developing a practical threshold-setting policy that adapts to hardware configuration and workload characteristics, rather than relying on a static value.

### 14.3 Cross-Application and Cross-Hardware Generalization

- **Application diversity in R values**: The analysis uses go-cpu as the calibration workload. A natural extension is to repeat with per-application R curves or a fleet-weighted average.
- **Hardware scaling**: The c6620 experiments use up to 40 LPs and 6--8 VMs. Production servers have 200+ cores and 100+ VMs. At larger scale, application-level variance may be smoothed by the law of large numbers, potentially allowing higher safe R for both configurations.
- **AMD Genoa validation**: The power curve shape and steal-time behavior should be validated on the actual model platform.

### 14.4 Deployment and Operational Considerations

- **Transition path**: The same-hardware constrained analysis (Section 11) provides a starting point for pilot evaluations. The most favorable scenario for a pilot is low-utilization fleet segments.
- **Mixed fleet operations**: The mixed fleet (Section 12) requires workload classification infrastructure. The sensitivity analysis shows the split point need not be precise --- a coarse classification (e.g., "CPU-bound" vs "memory-bound") may be sufficient.
- **Dynamic threshold setting**: Developing a practical, profiling-informed threshold-setting policy that can adapt to different hardware configurations and workload mixes. This connects the SMT vs no-SMT modeling story to the broader oversubscription characterization research.

---

## Appendix A: Results Summary Tables

### A.1 Homogeneous No-SMT Savings (Carbon % vs SMT Baseline)

All values use iso-physical-core basis, the purpose-built scaling model for no-SMT servers, and linear no-SMT CPU power.

| Avg VM util | M = 0.65 | M = 0.75 (geomean) | M = 0.85 | M = 1.0 (no discount) |
|---|---|---|---|---|
| 10% | -23.6% | **-11.8%** | -0.1% | +17.4% |
| 20% | -22.1% | **-10.1%** | +1.9% | +19.8% |
| 30% | -21.2% | **-9.0%** | +3.0% | +21.2% |

### A.2 Homogeneous No-SMT Savings (TCO % vs SMT Baseline)

| Avg VM util | M = 0.65 | M = 0.75 (geomean) | M = 0.85 | M = 1.0 (no discount) |
|---|---|---|---|---|
| 10% | -27.9% | **-16.8%** | -5.7% | +10.8% |
| 20% | -26.1% | **-14.6%** | -3.2% | +13.7% |
| 30% | -24.8% | **-13.2%** | -1.8% | +15.6% |

### A.3 Carbon Breakeven M (Iso-Physical-Core, Purpose-Built Scaling Model)

The demand multiplier M at which no-SMT exactly matches SMT on carbon:

| Avg VM util | Carbon breakeven M |
|---|---|
| 10% | 0.851 |
| 20% | 0.834 |
| 30% | 0.825 |

Values below 1.0 mean no-SMT needs *some* demand discount to break even. These breakeven values (0.83--0.85) indicate that a discount of ~15--17% is sufficient.

### A.4 Mixed Fleet Summary (Carbon % vs SMT Baseline)

| Avg VM util | Purpose-built homogeneous | Purpose-built mixed | Same-hardware constrained homogeneous | Same-hardware constrained mixed |
|---|---|---|---|---|
| 10% | -11.8% | -14.6% | -16.5% | -17.9% |
| 20% | -10.1% | -14.0% | -13.9% | -16.2% |
| 30% | -9.0% | -13.5% | -2.4% | -10.1% |

---

## Appendix B: Notation Reference

| Symbol | Meaning | Typical values |
|---|---|---|
| R | Oversubscription ratio (vCPUs / pCPUs) | 1.0--5.6 |
| M | vCPU demand multiplier | 0.50--1.00 |
| CWT | CPU wait time (steal time), % | 0--5% |
| u | Average VM utilization | 0.05--0.50 |
| P(u) | Server power as a function of utilization | 215--763 W |
| f(u) | Power curve shape function, [0,1] -> [0,1] | Linear or polynomial |

---

*Document prepared April 2026. All modeling results are reproducible from the fleet carbon/TCO model using the processor specifications, experimental oversubscription ratios, and cost parameters described herein.*
