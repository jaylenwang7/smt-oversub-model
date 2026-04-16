# SMT vs No-SMT Fleet Modeling: Results, Methodology, and Assumptions

---

## 1. Executive Summary and Headline Numbers

This document presents the current results of the fleet-level carbon and TCO model that compares SMT-enabled and SMT-disabled cloud server deployments. The central question is:

> On the same physical CPU, does disabling SMT and leveraging the resulting higher safe oversubscription headroom produce net carbon and cost savings at fleet scale?

**Scope note**: This document does **not** address the separate (but related) question of whether a processor designed from the ground up without SMT would have lower carbon or cost than an SMT-capable processor. That comparison would likely require a processor-design study, not just an SMT-on vs SMT-off study. Existing no-SMT processors can be compared in the market, but they usually differ from SMT processors in other ways besides SMT itself (e.g., ISA, core design, cache and memory system, frequency targets, etc.). A hypothetical "same processor, but redesigned for no-SMT" would also require specifying what changes the redesign makes with the saved area and power budget: more cores, different caches, higher frequency, lower package power, or some combination. That kind of redesign is possible and processors without SMT have existed in industry, so people have already considered such questions. For now, the analysis here takes the narrower same-CPU comparison: **on the same CPU**, what changes when SMT is enabled vs. disabled?

**Headline result (homogeneous no-SMT fleet, purpose-built servers):**

Here are the headline results of the analysis so far, in terms of the modeled carbon and TCO savings of using no-SMT vs. SMT; the rest of the document builds up to them and explains what "homogeneous no-SMT fleet, purpose-built servers" means.

| Average VM utilization | Carbon savings vs SMT | TCO savings vs SMT |
|---|---|---|
| 10% | **~12%** | **~17%** |
| 20% | **~10%** | **~15%** |

These numbers assume:

- Iso-physical-core comparison (same 80-core CPU, SMT gets 160 LPs while no-SMT gets 80 LPs)
- Linear CPU power proportionality for no-SMT (experimentally motivated)
- A vCPU demand discount for no-SMT of ~26.5% (the geomean from performance profiling of a 30-application benchmark suite)
- Purpose-built no-SMT servers with memory/SSD scaled to vCPU capacity/demand
- Oversubscription ratios derived from a 1% steal-time contention threshold measured on a single benchmark application (go-cpu)

The rest of this document builds up to these numbers layer by layer, starting from a ~50% no-SMT *penalty* when neither oversubscription nor demand compression is considered, and progressively introducing the mechanisms that reverse the picture.

**A note on the contention threshold**: All oversubscription ratios in this analysis are derived from a single CPU Wait Time (CWT) profile while assuming that the maximum tolerable CWT threshold is 1%. In other words, the results are derived from the server load vs. CWT curves measured for one application (go-cpu) on one machine (a 28-core, 54-thread Intel EMR server in CloudLab, which has the machine ID "c6620" as will be referred to throughout). This curve implicitly sets the "maximum safe oversubscription ratio" at each utilization level. For instance, at 10% average VM utilization, no-SMT can safely oversubscribe at ~5.6x, while at 20% utilization the safe ratio drops to ~2.8x. This is a single-threshold approximation: it treats one observed steal-time cutoff as the fleet-wide limit at each utilization level. In practice, as we have seen through other experiments, the safe oversubscription limit depends on application behavior, hardware configuration, and scheduling policy, and would need to be calibrated per deployment. Evaluating the sensitivity of the results to different threshold values is noted as future work in Section 13.

---

## 2. Document Context

This document serves three purposes:

1. **Results communication**: Present the current headline numbers from the SMT vs no-SMT fleet modeling so that we can understand what the model currently projects in terms of carbon/TCO savings, how significant those savings are, and how sensitive those savings projections are.

2. **Methodology transparency**: Lay out every modeling assumption, data source, and calculation step so others can verify, sanity-check, and critique the approach.

3. **Progress checkpoint**: Record the current state of the modeling work as of now, including what has been completed, what requires further validation, and what remains as future work.

### Guiding Research Question

This modeling is part of our broader investigation into CPU oversubscription. Where the overarching question can be framed as:

> If a cloud provider has historical VM-level observations but does not know the applications running inside VMs, how can it safely oversubscribe CPU while capturing real TCO and carbon reductions?

A key sub-question --- the one addressed here in this doc --- is:

> Should a cloud provider enable or disable SMT on its servers, and how does that choice interact with oversubscription headroom, power consumption, and customer vCPU demand to affect total fleet carbon and TCO?

The experimental characterization work (conducted on real hardware with real applications) has established that SMT scheduling constraints limit oversubscription headroom, and that disabling SMT removes those constraints while also providing better per-vCPU performance. This document quantifies the fleet-level carbon and cost implications of those experimental findings.

---

## 3. Definitions and Recurring Terminology

This section sets up definitions/clarifications for terms that are used consistently throughout this document.

### 3.1 Hardware Concepts

| Term | Definition |
|---|---|
| **Physical core** | One CPU core on the die, containing a shared set of execution units, caches, and other microarchitectural resources. |
| **Hardware thread / LP (Logical Processor)** | One schedulable execution context exposed by the hardware. With SMT enabled, each physical core exposes 2 LPs (sibling threads sharing execution resources). With SMT disabled, each core exposes 1 LP. |
| **SMT (Simultaneous Multi-Threading)** | A hardware feature that exposes multiple hardware threads per physical core, allowing the threads to share resources. In this document we focus on the dominant 2-way SMT implementation (2 LP siblings). |

### 3.2 Virtualization / Scheduling Concepts

| Term | Definition |
|---|---|
| **vCPU (Virtual CPU)** | A virtual processor assigned to a VM. The hypervisor schedules vCPUs onto LPs. From the VM's perspective, a vCPU is an independent processor. |
| **pCPU (Physical CPU)** | A logical processor available for VM scheduling, after subtracting those reserved for the hypervisor. `pCPUs = HW_threads - thread_overhead`. |
| **Thread overhead** | LPs reserved for host/hypervisor use, not available for VM scheduling. Modeled as 10% of HW threads (e.g., 16 of 160 for SMT, 8 of 80 for no-SMT). |
| **LP pool** | The set of LPs the hypervisor scheduler is using to place vCPUs in a given scheduling configuration. In the main fleet model this equals the available pCPUs per server. In the controlled oversubscription experiments (Section 5), it is the smaller LP set a group of VMs is pinned to (e.g., 8 LPs from 4 physical cores under SMT vs 8 LPs from 8 physical cores under no-SMT). A larger LP pool generally gives the scheduler more flexibility and permits a higher safe oversubscription ratio at a given contention threshold. |
| **Oversubscription ratio (R)** | The ratio of allocated vCPUs to available pCPUs per server: `R = total_allocated_vCPUs / available_pCPUs`. R = 1.0 means no oversubscription (each vCPU maps to one pCPU); R = 2.0 means two vCPUs share each pCPU on average. |
| **Steal time / CPU wait time (CWT)** | The fraction of time a vCPU is runnable (wants to execute) but cannot because its LP is occupied by another vCPU. This is the primary quality-of-service metric for oversubscription safety. |
| **Contention threshold** | The maximum tolerable CWT, set at **1%** in this analysis. The contention threshold directly impacts the maximum safe oversubscription ratio: the highest R at which CWT remains below 1%. See Section 5 for details. |
| **VP constraints** | A scheduling constraint that prevents non-sibling vCPUs/VPs (from the guest VM's perspective) from co-running on sibling SMT LPs (i.e., on the same physical core). This constraint is enforced due to cloud security concerns. In this analysis, it is the assumed that when SMT is enabled, VP constraints must also be enforced by the hypervisor vCPU-to-LP scheduler. This constraint limits SMT oversubscription by creating scheduling conflicts. It is irrelevant for no-SMT, where each LP is the sole thread on its core. |

### 3.3 Model Concepts

| Term | Definition |
|---|---|
| **vCPU demand multiplier (M)** | A scaling factor on total fleet vCPU demand. `M < 1.0` represents demand compression: users needing fewer vCPUs because each vCPU delivers more performance. `M = 0.75` means 25% fewer vCPUs are needed (a 25% demand discount). |
| **vCPU demand discount** | The percentage reduction in vCPU demand under no-SMT, defined as `1 - M`. As will be discussed, if the no-SMT/SMT peak throughput ratio is *r*, then `M = 1/r` and the discount is `1 - 1/r`. A throughput ratio of 1.36x implies a 26.5% discount. |
| **Proportional scaling assumption** | The assumption that users scale their VM count (and/or VM size) proportionally with per-vCPU performance. If no-SMT delivers 1.36x the throughput per vCPU, users are assumed to need 1/1.36 = 0.735x as many vCPUs. **[Requires validation]**: This assumption has not been empirically validated against real fleet behavior; see Section 13 for the specific validation questions. |
| **Power proportionality** | How linearly server power scales with server-level CPU utilization. A perfectly power-proportional server would draw zero watts at idle and maximum at full load. |
| **Embodied carbon** | CO2-equivalent emissions from manufacturing the server hardware/supporting infrastructure (racks etc.). |
| **Operational carbon** | CO2-equivalent emissions from electricity consumed during the server's operational lifetime. |
| **TCO (Total Cost of Ownership)** | Capex (HW purchase) plus opex (electricity) over the server's lifetime. |
| **Iso-LP comparison** | Holding the LP pool size fixed between SMT and no-SMT (e.g., both get 8 LPs). Useful for isolating the pure scheduling constraint cost, but it is not the same-hardware comparison studied here: on the same 8 physical cores, SMT would expose 16 LPs and no-SMT would expose 8 LPs. |
| **Iso-physical-core comparison** | Holding physical cores fixed and letting SMT expose its full LP count (e.g., 8 physical cores yield 16 LPs with SMT and 8 LPs without). This is the same-hardware comparison studied in the rest of the document. |

---

## 4. Modeling Framework

### 4.1 What the Model Computes

The model answers a steady-state question: given a fleet of servers that must host a fixed, nominal demand of vCPUs at a fixed average utilization, **what is the total carbon footprint and TCO over the servers' lifetime?**

The model answers this question by proceeding in five deterministic steps:

**Step 1 --- Server count.** How many servers are needed?

```
effective_vcpus = total_vcpus x M
vcpu_capacity_per_server = available_pCPUs x R
num_servers = ceil(effective_vcpus / vcpu_capacity_per_server)
```

Where `total_vcpus` is the nominal fleet-wide vCPU demand under the SMT baseline, and `M` is the vCPU demand multiplier (see Section 3.3) that shrinks demand when each vCPU delivers more performance (so `effective_vcpus` is the actual demand after any no-SMT performance discount is applied). `available_pCPUs` is the per-server pool of LPs available for VM scheduling after subtracting hypervisor-reserved threads, and `R` is the oversubscription ratio, so `vcpu_capacity_per_server` is how many vCPUs one server can host.

This is how oversubscription enters the model mechanically. A higher `R` increases `vcpu_capacity_per_server`, which reduces `num_servers` in Step 1. The model then concentrates the same aggregate CPU utilization/work onto fewer servers in Step 2.

**Step 2 --- Utilization.** What is the average per-server CPU utilization?

```
effective_utilization = min(1.0, (effective_vcpus x avg_vm_util) / (num_servers x available_pCPUs))
```

Where `avg_vm_util` is the average per-vCPU CPU utilization observed at the VM level (e.g., 0.10 for a fleet averaging 10% CPU utilization). The numerator is the total vCPU-hours of CPU work demanded across the fleet; the denominator is the total pCPU-hours of CPU capacity available across the fleet. Because `num_servers` appears in the denominator, reducing server count through a higher `R` means the same fleet-wide work is spread across fewer servers, so average per-server utilization rises. The `min(1.0, ...)` caps effective per-server utilization at 100%.

**Step 3 --- Power.** What does each server draw at that utilization?

```
P(u) = P_idle + (P_max - P_idle) x f(u)
```

Where `u` is the effective utilization from Step 2, `P_idle` is the server's power at 0% utilization, `P_max` is its power at 100% utilization, and `f(u)` is a function/curve that maps utilization in [0, 1] to a factor of peak power/TDP in [0, 1] (see Section 4.5 for further discussion). Each server component (CPU, memory, SSD, NIC, etc.) has its own `P_idle`, `P_max`, and curve shape. Total server power is the sum of all component powers at the effective utilization.

**Step 4 --- Carbon.** Embodied plus operational emissions.

```
embodied_carbon = num_servers x embodied_carbon_per_server
energy_kwh = num_servers x power_per_server_W x lifetime_hours / 1000
operational_carbon = energy_kwh x carbon_intensity_g_per_kwh / 1000
total_carbon = embodied_carbon + operational_carbon
```

Where `embodied_carbon_per_server` is the manufacturing CO2e attributed to one server (summed across CPU, DIMMs, SSDs, NIC, chassis, and rack share; see Section 4.4), `power_per_server_W` is the per-server power from Step 3, `lifetime_hours` is the server's assumed service life in hours (e.g., 5 years ~= 43,800 hours) and is used to convert per-server power into total lifetime electricity consumption, and `carbon_intensity_g_per_kwh` is the grams of CO2e emitted per kWh of electricity consumed.

**Step 5 --- TCO.** Capex (HW purchase) plus opex (electricity) cost.

```
capex = num_servers x server_cost
opex = energy_kwh x electricity_cost_per_kwh
total_cost = capex + opex
```

Where `server_cost` is the one-time capex to purchase one server (see Section 4.4 for the per-component breakdown), `electricity_cost_per_kwh` is the price for a kWh of electricity over the server's lifetime, and `energy_kwh` is carried over from Step 4.

The model is deterministic: identical inputs always produce identical outputs. There is no simulation, randomness, or time-varying behavior, as of now.

### 4.2 What the Model Does Not Include

- **PUE (Power Usage Effectiveness)**: Cooling and data center overhead are not explicitly modeled. Though if desired, PUE can easily be folded into power values (e.g., multiply by PUE factor of 1.12).
- **Non-electricity opex**: We assume staffing, networking, software licensing, maintenance, etc. is wrapped into the opex rate (i.e., hosting rate).
- **Time-varying behavior**: The model uses a single average CPU utilization for the entire fleet lifetime. It does not explicitly capture diurnal patterns, workload bursts, or autoscaling.
- **Redundancy and fault tolerance**: The model computes the minimum server count. No buffer capacity is included.

### 4.3 Reference Platform: AMD Genoa 80-Core

All analyses use a single-socket AMD Genoa 80-core server as the reference platform, which was one of the baseline SKUs from GreenSKU. The "no-SMT" configuration is the same physical CPU with SMT disabled --- not a different processor design. Both configurations use the same die, same process node, and same server chassis.

| Parameter | SMT enabled | SMT disabled |
|---|---|---|
| Physical cores | 80 | 80 |
| Threads per core | 2 | 1 |
| HW threads | 160 | 80 |
| Thread overhead (10%) | 16 | 8 |
| Available pCPUs | 144 | 72 |

### 4.4 Per-Component Cost and Carbon Data

All per-component data (power, embodied carbon, server cost) comes from the dataset we used in GreenSKU. So this section essentially is just an organized version of the data we already used in GreenSKU for the reference server. The reference server is the `Genoa-default-1S` configuration:

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

**Structural observation**: The per-server fixed costs (CPU die + NIC + chassis + rack = 457 kg) are identical for both configurations. They represent 26% of SMT per-server carbon but 41% of no-SMT per-server carbon. This fixed-cost amortization is a structural advantage of SMT.

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

Power is modeled per component, with each component having an idle and max power. Idle power is derived from component TDP using derate factors at zero utilization (from GreenSKU spec_derate curves). All values include a 1.05x PSU efficiency loss factor.

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

**Per-server resource scaling**: The memory/SSD halving shown in the three tables above (12 -> 6 DIMMs, 6 -> 3 SSDs) reflects one specific modeling choice --- each vCPU is entitled to a fixed allocation of memory and storage, so no-SMT at `R = 1.0` gets half of each because it has half the vCPUs. This is the **purpose-built scaling** resource model, which the primary analyses use throughout. As `R` increases beyond 1.0, memory, SSD, and their associated carbon, cost, and power scale linearly with `vcpu_capacity_per_server`. Section 7 describes this model in detail and contrasts it with two alternatives (a fixed-resource upper bound, and a same-hardware constrained case), and Section 11 uses the same-hardware constrained case to examine disabling SMT on existing SMT-provisioned servers.

### 4.5 Power Curve Models

Two power curve model options are used for the CPU component:

1. **Polynomial (default for SMT)**: A sublinear (concave) curve reflecting the well-documented pattern where power rises steeply at low utilization then flattens. This is the standard shape observed in SPECpower benchmarks.

2. **Linear (used for no-SMT)**: `P(u) = P_idle + (P_max - P_idle) x u`. Motivated by experimental power measurements on the c6620 server showing that no-SMT CPU power scales almost perfectly linearly with utilization (R^2 = 0.998 for linear fit), while SMT follows a concave curve (linear R^2 = 0.896, requiring a quadratic fit for R^2 = 0.999) [^power-exp].

[^power-exp]: Experimental power measurements from stress-ng load sweeps on c6620. At iso-host utilization, no-SMT uses 8--29W less CPU package power. The qualitative finding (no-SMT is more linear) is expected to be architecture-general, but the exact coefficients are specific to that platform.

Non-CPU components (memory, SSD, NIC, chassis) use the polynomial curve by default across SMT and no-SMT.

**Caveat**: The power measurements are from the c6620 platform (Intel Xeon EMR, 28 cores), not a Genoa (AMD EPYC, 80 cores). The model uses Genoa-derived power breakdown values with only the curve *shape* informed by the c6620 experiment. Validating no-SMT power linearity on other hardware would be a good cross-check.

### 4.6 Cost and Environmental Constants

These values are held constant across all analyses:

| Parameter | Value | Rationale |
|---|---|---|
| Carbon intensity | 175 g CO2/kWh | Moderate mixed-grid value derived from the GreenSKU data center carbon intensity data. For reference: very clean grids are ~20--50 g/kWh, US average is ~400 g/kWh. At 175 g/kWh, the embodied/operational carbon split is roughly balanced. |
| Electricity/hosting cost | $0.28/kWh | Representative commercial/industrial rate from GreenSKU defaults. |
| Server lifetime | 6 years | Within the typical 4--7 year range for cloud data center hardware. Consistent with GreenSKU defaults. |
| Total vCPUs | 100,000 | Fixed fleet size. The number itself doesn't matter much (as it's a nominal demand). Large enough that ceiling-function rounding effects are small (<1% of server count). |

---

## 5. The Contention Threshold: From Experiments to Oversubscription Limits

All oversubscription ratios used in this analysis are derived from experimental measurements of CWT on real hardware running real applications. This section explains the experimental setup, what the "contention threshold" is, and the assumptions it introduces.

### 5.1 What the Contention Threshold Is

The **contention threshold** is a quality-of-service (QoS) boundary: the maximum acceptable CWT for VMs on an oversubscribed server. In this analysis, it is set at **1% CWT**, which is the standard for Azure production environments.

At a given average VM utilization, the contention threshold determines the **maximum safe oversubscription ratio (R)**. This is the highest R at which the measured CWT stays below 1%. As utilization increases, less oversubscription headroom is available, so the safe R decreases.

The safe R at each utilization level is read off of the measured server-load vs CWT curve, which is itself highly nonlinear. Because CWT rises sharply well before the server is fully loaded, the effective host-level utilization that can be achieved while staying under the 1% CWT threshold is much less than 100%. This is what prevents the naive upper bound of `R = 1 / avg_vm_util`: at 20% average VM utilization, the naive bound would be `R = 5` (which would drive host-level utilization to 100%), but the CWT curve caps the safe R well below that because CWT crosses 1% long before the host saturates. A full characterization of that curve is not the focus of this document; the analysis here only consumes the resulting maximum safe R values at each utilization level.

### 5.2 Experimental Setup

The oversubscription ratios are derived from experiments on the previously-described CloudLab c6620 server using a synthetic CPU-bound benchmark (go-cpu). This application has a load vs CWT curve that is similar to other tested applications. The experimental setup uses 8-vCPU/8-core VMs in a client-server configuration with load (QPS) generated to hit fixed utilization targets. VMs are pinned to specific LP sets to create controlled oversubscription scenarios.

Two scheduling regimes are compared:

- **SMT with VP constraints**: The Linux kernel's "core scheduling" is enabled, allowing for enforcing VP constraints.
- **No-SMT**: Constraints are irrelevant since each LP is the sole thread on its core.

The experiments measure the VP/LP rate (the oversubscription rate/ratio) at which CWT crosses the 1% threshold, for each regime at fixed per-VM utilization targets (5%, 10%, 20%, 30%).

### 5.3 Iso-LP vs Iso-Physical-Core Calibration

The experiments support two distinct comparisons, corresponding to two different questions:

**Iso-LP (8 LPs for both)**: Both SMT and no-SMT are given the same LP pool size (8 LPs from 4 physical cores under SMT, 8 LPs from 8 physical cores under no-SMT). This isolates the cost of VP scheduling constraints, but it is not the more realistic same-hardware comparison studied in the rest of this document: on the same 8 physical cores, SMT would expose 16 LPs and no-SMT would expose 8 LPs.

**Iso-physical-core (same 8 physical cores, SMT gets 16 LPs)**: SMT exposes its full 16 LPs from 8 physical cores, while no-SMT exposes 8 LPs from the same cores. This is the same-hardware "disable SMT on the same CPU" scenario studied in the main results. SMT benefits from a larger LP pool, which gives it more scheduling flexibility and moderately higher safe R values.

The oversubscription ratios used the resulting headline numbers come from the **iso-physical-core** calibration, because it matches the same-CPU decision being modeled here:

| Avg VM utilization | SMT safe R (16 LP) | No-SMT safe R (8 LP) |
|---|---|---|
| 10% | 3.32 | 5.58 |
| 20% | 1.66 | 2.79 |
| 30% | 1.11 | 1.86 |

These values are interpolated from the experimental steal-time curves at the 1% threshold, using go-cpu operating-point tables.

### 5.4 Assumptions and Limitations of the Contention Threshold

The contention threshold approach introduces several assumptions that should be noted:

**Assumption 1: Single application calibration.** The safe R values are derived from go-cpu. Go-cpu sits near the cross-application median in terms of steal-time behavior, but different applications produce different steal-time curves at the same utilization. For example, across 13 tested applications at 10% utilization under the iso-physical-core regime, the no-SMT safe VP/LP rate ranges from 3.0x (Elasticsearch, worst case) to 7.9x (PgBench, best case), with go-cpu at 5.6x. The model uses go-cpu as the single calibration point for all workloads, even though a production deployment would encounter a mix of applications.

**Assumption 2: Single machine.** The experiments are conducted on the c6620 server. The model assumes the direction of the effect persists on other hardware --- VP constraints still limit SMT oversubscription and no-SMT still permits higher R --- but the exact R values at each utilization level may differ on other hardware, particularly on the Genoa 80-core platform used in the model, which has a significantly larger LP pool. However, the proportional gain from scheduling constraints and from the smaller LP pool for no-SMT should still hold, but whether they are slightly or significantly different is not yet known.

**Assumption 3: Fixed 1% threshold.** The analyses assume a single, fixed steal-time threshold of 1%. In practice, different cloud providers or VM service tiers might use different thresholds. The sensitivity of the results to different threshold values has not yet been evaluated. **[Future work]**: Evaluate how the headline savings change across a range of contention thresholds.

---

## 6. Layer 1: Baseline Penalty --- Disabling SMT with No Oversubscription

### Question

> If you take an SMT-enabled server and simply disable SMT --- with no oversubscription in either case and no adjustment to vCPU demand --- what is the raw carbon and TCO penalty?

### Approach

Both configurations run at R = 1.0. The same total vCPU demand (100,000) must be served by both. This is the most conservative comparison and the most favorable framing for SMT: no scheduling constraints are relevant (since there is no oversubscription), and no-SMT gets no credit for its higher per-vCPU performance.

Two no-SMT variants are reported: **no-SMT polynomial** uses the same concave polynomial CPU power curve as SMT, while **no-SMT linear** uses the linear CPU power curve motivated by the c6620 experiments (see Section 4.5 for both curve definitions). The polynomial variant is the "worst case" for no-SMT (it inherits SMT's power shape), and the linear variant is the default used in the rest of the document.

### Results

| Metric | SMT (R = 1.0) | No-SMT polynomial (R = 1.0) | No-SMT linear (R = 1.0) |
|---|---|---|---|
| Servers | 695 | 1,389 (+99.9%) | 1,389 (+99.9%) |
| Carbon | 3,479,500 kg | 5,445,500 kg (**+56.5%**) | 5,148,300 kg (**+48.0%**) |
| TCO | $11.54M | $17.31M (**+50.1%**) | $16.84M (**+45.9%**) |

No-SMT needs **2x as many servers** because each server exposes only 72 available pCPUs versus 144 for SMT. The gap between the two no-SMT columns is the effect of the power-curve assumption alone: using the linear no-SMT CPU power model lowers the no-SMT penalty from `+56.5%` to `+48.0%` on carbon and from `+50.1%` to `+45.9%` on TCO. The rest of this breakdown uses the linear case, since that is the default model used in the rest of the document. The Layer 1 comparison already assumes the same **purpose-built scaling** resource model that Section 7 later makes explicit: at `R = 1.0`, no-SMT gets half the memory and SSD footprint of SMT, while the CPU die, NIC, chassis, and rack share remain fixed per server. The table below shows the **fleet-scale** carbon breakdown between embodied and operational emissions and the **fleet-scale** TCO breakdown between capex and opex for both SMT and no-SMT:

| Fleet-scale breakdown item | SMT (R = 1.0) | No-SMT linear (R = 1.0) | No-SMT vs SMT |
|---|---|---|---|
| Embodied carbon (share of carbon) | 1,239,200 kg (35.6%) | 1,555,400 kg (30.2%) | +25.5% |
| Operational carbon (share of carbon) | 2,240,300 kg (64.4%) | 3,593,000 kg (69.8%) | +60.4% |
| Capex (share of TCO) | $7,951,400 (68.9%) | $11,087,600 (65.9%) | +39.4% |
| Opex (share of TCO) | $3,584,500 (31.1%) | $5,748,700 (34.1%) | +60.4% |

At fleet scale, the simplest way to read the table is to separate the components that scale with exposed vCPU capacity from the components that stay fixed per server. Under the purpose-built scaling assumption, no-SMT uses half the memory and SSD per server, but 2x as many servers. That means the **fleet total** of memory and SSD stays unchanged. So the no-SMT penalty is **not** coming from memory and SSD totals becoming much larger at fleet scale.

What does become much larger at fleet scale is the total of the hardware terms that stay fixed per server: CPU die, NIC, chassis, and rack share. Those terms are repeated across 2x as many servers. That is why no-SMT has a positive penalty on both embodied carbon and capex even though each no-SMT server is smaller. The embodied-carbon penalty is smaller (`+25.5%`) because memory plus SSD account for a larger share of embodied carbon, so keeping their fleet total roughly constant offsets more of the extra-server penalty. The capex penalty is larger (`+39.4%`) because the fixed per-server hardware terms make up a larger share of capex. Put differently, the fixed hardware share is about `25.6%` of SMT embodied carbon but about `39.5%` of SMT capex, so replicating those fixed terms across more servers hurts capex more than it hurts embodied carbon.

The operational side follows the same high-level pattern, but through power rather than one-time hardware accounting. Memory and SSD power scale down with the smaller no-SMT server, but CPU package power, NIC power, and chassis/fan power remain large per-server terms. Running `1,389` no-SMT servers instead of `695` SMT servers therefore repeats that per-server power floor across the fleet, so total energy rises sharply. The reason why operational carbon and opex show the **same** no-SMT increase (`+60.4%`) is that they are both just the same fleet-energy increase expressed in different units. If no-SMT uses `1.604x` the fleet energy of SMT, then multiplying both sides by `carbon_intensity` makes operational carbon `1.604x` as large, and multiplying both sides by `electricity/hosting cost` makes opex `1.604x` as large. Embodied carbon and capex do **not** have that property. They are both computed from the same hardware inventory, but they are not obtained by taking one shared fleet total and multiplying by two different constants. Instead, they sum the same component counts using different per-component coefficients for carbon and for dollars. Because no-SMT changes some component counts (memory and SSD) while leaving others fixed per server (CPU die, NIC, chassis, rack share), and because those components carry different weights in carbon and in capex, the no-SMT increase can differ between embodied carbon and capex.

The total penalty in each metric is therefore a weighted average of the increases in that metric's breakdown, using the SMT baseline shares as the weights. For carbon, the `+48.0%` total comes from combining `+25.5%` embodied carbon with `+60.4%` operational carbon, weighted by SMT's `35.6% / 64.4%` split. For TCO, the `+45.9%` total comes from combining `+39.4%` capex with `+60.4%` opex, weighted by SMT's `68.9% / 31.1%` split. Carbon ends up slightly worse because the SMT baseline carbon mix is more operational-heavy, while the SMT baseline TCO mix is more capex-heavy.

The same weighting logic matters throughout the rest of the document and is not specific to the carbon-vs-TCO comparison. Any parameter change that shifts the embodied/operational split or the capex/opex split changes how much the total inherits from each part of the breakdown. For example, higher carbon intensity or longer lifetime would increase the operational share of carbon and pull the total carbon penalty toward the operational `+60.4%` increase, while lower carbon intensity, shorter lifetime, or higher embodied-carbon factors would push it toward the embodied `+25.5%` increase. Similarly, higher electricity/hosting cost would increase the opex share of TCO and pull the total TCO penalty toward `+60.4%`, while lower electricity/hosting cost would make TCO more capex-heavy and push it toward `+39.4%`. The later headline numbers therefore depend not only on the SMT/no-SMT mechanisms, but also on the fixed cost and carbon parameters from Section 4.6.

> **Takeaway**: Without oversubscription or demand compression, disabling SMT incurs a **~48--57% carbon penalty and ~46--50% TCO penalty**. This is the baseline penalty against which the later mechanisms are evaluated.

---

## 7. Modeling Choice: How Oversubscription Changes Per-Server Resources

### Question

> Once CPU is oversubscribed (`R > 1.0`), how should server resources like memory, SSD, and per-server costs be modeled, and how far do real savings fall below the naive ideal?

Before introducing layers discussing oversubscription, one modeling choice needs to be explicit. A higher oversubscription ratio reduces server count, but that alone does not determine what happens to memory, SSD, embodied carbon, or power. The answer depends on how per-server resources are treated once more vCPUs are packed onto each server.

Thus, this section defines the accounting framework that the later layers use.

### 7.1 Three Resource Models

When a server hosts more vCPUs than pCPUs, each vCPU still needs some memory and storage. The model uses three distinct resource treatments:

| Resource model | What stays fixed per server | What scales or caps | Intended interpretation |
|---|---|---|---|
| **Fixed-resource model** | CPU, memory, SSD, NIC, chassis, rack | Nothing; only server count changes | Optimistic upper bound, assumes all resources can be oversubscribed |
| **Purpose-built scaling model** | CPU, NIC, chassis, rack | Memory and SSD scale with hosted vCPU count | Default model for a projected oversubscribed deployment |
| **Same-hardware constrained model** | Full SMT-provisioned server configuration | Effective R is capped by the first resource bottleneck | "Disable SMT on existing servers" transition case |

Two clarifications matter for the later sections:

- In the **purpose-built scaling model**, the no-SMT `R = 1.0` base server from Section 4.3 starts with 6 DIMMs and 3 SSDs. As `R` rises, the model scales memory and SSD capacity, cost, carbon, and power linearly with hosted vCPU count. It does **not** model whole-device step effects such as adding one DIMM or one SSD at a time.
- In the **same-hardware constrained model**, no-SMT inherits the full SMT-provisioned server (12 DIMMs and 6 SSDs). This can help at low oversubscription, but it also creates stranded-resource overhead and an eventual hard ceiling. For example, at `R = 1.0`, 72 no-SMT vCPUs at 4 GB/vCPU use only 288 GB of the installed 768 GB, so much of the memory footprint is paid for but unused; at high requested `R`, that same 768 GB caps the server at 192 vCPUs, so effective `R` cannot rise above `192 / 72 = 2.67`.

### 7.2 Ideal Savings vs Modeled Savings as R Increases

To see why the resource model matters, it helps to step back from the SMT vs no-SMT comparison and look at how the model behaves for no-SMT alone as `R` increases. This section does two things. First, it shows how the resource model changes the savings. Second, it shows that the savings from oversubscription do **not** scale in direct proportion to the amount of oversubscription. As described in Section 4.1, a higher `R` reduces server count and therefore concentrates the same fleet-wide CPU work onto fewer servers, which raises per-server utilization. That utilization concentration is one reason real savings fall below the naive ideal, even before accounting for memory/SSD scaling or resource constraints. The goal here is to illustrate those model mechanics, not to claim that the exact percentages from this example apply to every processor type or utilization level. The tables below use no-SMT at 30% average VM utilization and compare higher `R` values against the no-SMT `R = 1.0` baseline because 30% is the upper end of the main 10/20/30 utilization range used throughout the analysis and makes the non-idealities easy to see.

The **ideal savings** column is the naive upper bound. In simple terms: if `R = 2.0` cuts server count in half, the ideal line assumes that carbon and TCO also fall by exactly 50%. If `R = 5.0` cuts server count by 80%, the ideal line assumes 80% savings. Real modeled savings fall below that line because higher `R` does more than reduce server count: it also pushes more work onto each remaining server, and the treatment of memory, SSD, embodied carbon, cost, and power depends on the resource model.

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

The comparison used later in the main narrative is: at `R = 2.0`, ideal carbon savings would be `-50.0%`, but the **purpose-built scaling model** yields only `-23.8%`. At `R = 5.0`, ideal carbon savings would be `-80.0%`, but the purpose-built result is `-45.1%` and the same-hardware constrained result plateaus at `-34.9%`.

### 7.3 Default Resource Model Used in the Main Results

From this point onward, the main layer-by-layer narrative uses the **purpose-built scaling model** unless stated otherwise. That choice is deliberate:

- It is the appropriate model for the main deployment question: what would a real no-SMT fleet look like if the servers were provisioned for that fleet?
- It preserves per-vCPU memory and storage needs instead of assuming that denser CPU packing comes "for free" or that memory/SSD oversubscription is readily deployable at the same rates.
- It treats the **same-hardware constrained model** separately in Section 11 as a transition-focused analysis.

> **Takeaway**: Once `R > 1.0`, oversubscription savings are not equal to server reduction. The later sections use the **purpose-built scaling model** as the default and treat the fixed-resource and same-hardware constrained models as explicit comparison cases.

---

## 8. Layer 2: Scheduling Constraints and Oversubscription Headroom

### Question

> VP scheduling constraints limit how much SMT can safely oversubscribe. No-SMT, free of these constraints, can oversubscribe more aggressively. Under the default resource model from Section 7, does this higher no-SMT oversubscription headroom close the carbon/TCO gap?

### Approach

Using the **purpose-built scaling model** from Section 7 as the default resource scaling model, replace `R = 1.0` with the experimentally measured safe `R` values from Section 5.3. SMT uses the VP-constrained iso-physical-core safe `R`; no-SMT uses its own no-constraints safe `R`. No-SMT uses the linear CPU power curve. The vCPU demand multiplier is still `M = 1.0` (no demand discount applied yet).

### Results (Iso-Physical-Core, Purpose-Built, No Demand Discount)

| Avg VM util | SMT R | No-SMT R | No-SMT carbon vs SMT | No-SMT TCO vs SMT | No-SMT servers vs SMT |
|---|---|---|---|---|---|
| 10% | 3.32 | 5.58 | **+17.4%** | **+10.8%** | +18.6% |
| 20% | 1.66 | 2.79 | **+19.8%** | **+13.7%** | +18.9% |
| 30% | 1.11 | 1.86 | **+21.2%** | **+15.6%** | +19.3% |

Under the iso-physical-core calibration and the purpose-built scaling model, no-SMT at `M = 1.0` still does not provide carbon or TCO savings vs. SMT: it provides a `+17--21%` increase in terms of carbon and `+11--16%` in terms of TCO. The oversubscription headroom advantage closes a large share of the ~50% gap from Layer 1, but once the non-CPU resource accounting from Section 7 is included, it does not bring no-SMT to breaking even/saving by itself.

For context, the same safe-`R` values would look much better under the **fixed-resource model** from Section 7: roughly `+2--3%` on carbon and `-5%` on TCO. That difference is not a change in scheduling behavior; it is purely a change in how memory and SSD are accounted for once more vCPUs are packed onto each server.

> **Takeaway**: Oversubscription headroom alone closes the ~50% baseline gap to about a **~17--21% carbon** and **~11--16% TCO** penalty under the default purpose-built server scaling model. The scheduling constraint disadvantage is still a major part of SMT's cost under oversubscription, but it is not sufficient by itself to make no-SMT a carbon/TCO win.

---

## 9. Layer 3: vCPU Demand Discount --- The Headline Result

### Question

> Since no-SMT LPs deliver more performance per vCPU (no co-running overhead from a sibling thread), users may need fewer vCPUs for the same workload. How does this demand compression change the picture?

### 9.1 Experimental Basis for the Demand Discount

Peak throughput was measured for 30 applications (14 latency-sensitive, request-serving services, 16 batch workloads) on the c6620 server, comparing 2-core SMT (4 HW threads / 4 vCPUs) versus 4-core no-SMT (4 HW threads / 4 vCPUs). The key aggregate results:

| Application scope | Count | Geomean no-SMT/SMT ratio | Implied vCPU discount |
|---|---|---|---|
| Services | 14 | 1.31x | 23.7% |
| Batch | 16 | 1.41x | 28.9% |
| All applications | 30 | 1.36x | 26.5% |

A discount of 26.5% means that a workload needing 100 vCPUs under SMT would need only ~74 vCPUs under no-SMT for the same throughput. This translates to `M = 0.735` (rounded to be slightly pessimistic to `0.75` for the modeling reference point). A lower `M` value is more advantageous for no-SMT.

The demand discount varies widely across applications:

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

**[Requires validation]**: The proportional scaling assumption --- that users would actually re-size their VMs or reduce VM count in proportion to the per-vCPU performance gain --- has not been empirically validated. In practice, VM sizing depends on pricing models, application architecture, and organizational inertia. Validating this assumption is a next step; one approach, as discussed, is to examine how first-party cloud workloads in Azure have historically scaled VM count when migrating between hardware generations with different per-vCPU performance (e.g., Ice Lake to Emerald Rapids). Specific validation questions are noted in Section 13.

### 9.2 Contextualizing the Discount Against Industry SMT Claims

This subsection is a brief aside. Its goal is to connect the model's vCPU-demand-discount framing to the more common industry framing of SMT gains, so the discount values above are put into context.

Vendor and academic sources report SMT "performance" gains in the range of +10% to +60%, with a cross-source median around +30%.[^smt-survey] However, these figures are typically framed as "aggregate performance/throughput gain on the *same physical core budget*" (e.g., 2 cores with SMT vs 2 cores without SMT), which is a different quantity than the "per-vCPU performance at fixed visible vCPU count" used in this model.

[^smt-survey]: A survey of vendor and academic SMT performance claims so far finds: Intel reports ~10--30% depending on source and workload class ([Intel HT Technical Guide](https://read.seas.harvard.edu/cs161/2022/pdf/intel-hyperthreading.pdf); [Intel Technology Journal 2002](https://www.intel.com/content/dam/www/public/us/en/documents/research/2002-vol06-iss-1-intel-technology-journal.pdf); [Intel virtualization white paper](https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/virtualization-xeon-core-count-impacts-performance-paper.pdf)); AMD claims "often 30--50%" for EPYC ([AMD EPYC SMT Technology Brief](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/amd-epyc-smt-technology-brief.pdf)); IBM reports 25--40% throughput and 30--60% instructions executed for SMT-2/SMT-4 ([IBM support documentation](https://www.ibm.com/support/pages/power-cpu-memory-affinity-3-scheduling-processes-smt-and-virtual-processors); [IBM AIX docs](https://www.ibm.com/docs/en/aix/7.2.0?topic=concepts-simultaneous-multithreading)). Academic measurements on SPEC CPU2000 multiprogrammed pairs show average speedups of ~1.20x with ranges from 0.86x to 1.58x ([Tuck & Tullsen 2003](https://users.cs.utah.edu/~rajeev/cs7810/papers/tuck03.pdf); [Bulpin 2004](https://pharm.ece.wisc.edu/wddd/2004/06_bulpin.pdf)). HPC applications show 0--22% gains depending on workload ([Saini et al. 2011](https://www.nas.nasa.gov/assets/nas/pdf/papers/saini_s_impact_hyper_threading_2011.pdf)).

To connect the vendor-style claims to this model's framing, an approximate translation is needed. If enabling SMT on `N` physical cores raises aggregate throughput by a factor `(1 + g)`, then the implied no-SMT/SMT performance ratio (assuming 2-way SMT) at fixed vCPU count is approximately `2 / (1 + g)`, giving `M = (1 + g) / 2`.

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

The experimental geomean of 1.36x (`M = 0.735`, ~26.5% discount) back-translates to an equivalent same-core SMT gain of ~+47%. This is above Intel's typical claims of +10--30%; however, Intel (and other vendors) does not specify whether such gains are peak throughput, instructions, IPC, etc. In the same experiments, if using instruction throughput-per-physical core on c6620, we find that the gain from SMT is on average about 30%, which is lower than the peak throughput gain from SMT, but in line with Intel's numbers. So depending on what "performance" means for the vendors, our numbers may be right in line or in the upper range in terms of SMT's performance overhead.

**[Requires validation]**: Connecting the per-vCPU peak throughput measurement to real fleet vCPU demand changes requires validating the proportional scaling assumption (see Section 9.1 and Section 13).

### 9.3 Three Reference Points for Sensitivity

The analysis evaluates savings at three reference points for vCPU-demand sensitivity:

| Label | M | Implied perf ratio | Rationale |
|---|---|---|---|
| **High discount** | 0.65 | ~1.54x | Near the strongest measured service gains; near the literature-median SMT claim |
| **Geomean** | 0.75 | ~1.33x | Close to the all-app experimental geomean (1.36x) |
| **Low discount** | 0.85 | ~1.18x | Conservative; near weakest measured gains (InfluxDB, Go-MemChase) |

### 9.4 Sanity Check: Demand Discount Alone Is Not Sufficient

Before presenting the full results, a sanity check confirms that the demand discount alone --- without oversubscription headroom --- does not make no-SMT obviously better. This analysis is a sanity check as the demand discount applies without oversubscription, so if just with the discount no-SMT won in carbon/TCO, then it would mean cloud providers should already disable SMT even without considering the oversubscription benefits; which is not what we see in practice. With both configurations at `R = 1.0` and no-SMT using a linear power curve, this check isolates only the demand-discount mechanism; because there is no oversubscription, it is effectively unchanged by the resource-model choice:

| M | Implied discount | No-SMT carbon vs SMT | No-SMT TCO vs SMT |
|---|---|---|---|
| 0.85 | 15% | +25.8% | +24.1% |
| 0.75 | 25% | +11.0% | +9.5% |
| 0.65 | 35% | -3.8% | -5.1% |

At the geomean discount, no-SMT is still 11% *worse* on carbon without oversubscription. No-SMT only breaks even in this setting at a ~32% discount (`M = 0.676`), which is stronger than the geomean. This confirms that the projected savings in the headline results require *both* the oversubscription headroom advantage (Section 8) and the demand discount --- neither alone is sufficient at the geomean value used here.

### 9.5 Headline Results Under the Default Resource Model

With all three mechanisms active --- the purpose-built scaling model from Section 7, scheduling-constraint-limited oversubscription from Section 8, and experimentally grounded vCPU demand discounts from this section --- the iso-physical-core comparison produces:

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

In the same homogeneous iso-physical-core setup used here, combining the purpose-built scaling model from Section 7, the scheduling-constraint-limited oversubscription ratios from Section 8, and the vCPU demand discount from this section, the carbon breakeven `M` is `0.869`, `0.844`, and `0.782` at 10%, 20%, and 30% utilization, while the TCO breakeven `M` is `0.938`, `0.889`, and `0.808`. This means no-SMT needs about a 13--22% demand discount to break even on carbon and about a 6--19% discount to break even on TCO; the geomean discount of 26.5% is past breakeven on both metrics.

### 9.6 How the Three Mechanisms Stack Up

To summarize:

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

As described in Section 5.3, the experimental data supports two types of experimental setups with regards to LPs:

| Basis | SMT R at 10% util | No-SMT R at 10% util |
|---|---|---|
| Iso-LP (8 LP for both) | 2.59 | 5.58 |
| Iso-physical-core (16 LP SMT, 8 LP no-SMT) | 3.32 | 5.58 |

The difference is entirely on the SMT side: the iso-physical-core setup gives SMT a higher safe R because a larger LP pool provides more scheduling flexibility.

### 10.2 Impact on Headline Numbers

Under the **purpose-built scaling model** at `M = 0.75`:

| Basis | 10% util carbon | 20% util carbon | 30% util carbon |
|---|---|---|---|
| Iso-LP | **-13.6%** | **-15.2%** | **-15.4%** |
| Iso-physical-core | **-11.8%** | **-10.1%** | **-9.0%** |
| Difference (pp) | +1.8 | +5.1 | +6.4 |

The iso-physical-core basis reduces no-SMT savings by 2--6 percentage points relative to iso-LP, with the largest impact at 30% utilization where the LP pool size matters most. At 10% utilization, the difference is only 1.8 pp because both calibrations already permit aggressive oversubscription.

The iso-LP calibration makes no-SMT look **more favorable** because it does not give SMT credit for its larger LP pool. The iso-physical-core calibration is the same-hardware basis for the "should we disable SMT?" decision on one physical CPU. This is why the headline numbers in Section 1 use the iso-physical-core basis.

> **Takeaway**: The scheduling-input setup is a major model choice. The iso-physical-core correction gives a real but incomplete share of the no-SMT gain to SMT. **The headline numbers use the most SMT-favorable same-hardware basis considered here** (iso-physical-core).

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

**At 10% utilization, the same-hardware constrained scenario saves *more* than the purpose-built scaling scenario.** This counterintuitive result arises because memory constraints severely limits SMT's oversubscription headroom at low utilization. Both configurations are capped at the same absolute vCPU count per server (192, determined by memory), but no-SMT reaches this with a higher effective R (2.67 vs 1.33), leveling the playing field. Additionally, SMT strands significant SSD capacity (40--50% unused) that no-SMT does not.

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

Across these cases, the mixed fleet adds `+1.4` to `+7.7` percentage points on carbon and `+0.4` to `+5.8` percentage points on TCO relative to homogeneous no-SMT. In the main 10--20% utilization range, those additional savings are modest: `+1.4` to `+3.9` percentage points on carbon and `+0.4` to `+2.1` percentage points on TCO.

### 12.3 When the Mixed Fleet Matters Most

The mixed fleet advantage is **largest where homogeneous no-SMT is weakest**: at 30% utilization under the iso-physical-core same-hardware constrained scenario, where homogeneous no-SMT saves only 2.4% on carbon and 5.9% on TCO, while the mixed fleet saves 10.1% on carbon and 11.7% on TCO. That 30% case is best read as an upper-end stress case for the 10/20/30 sweep, not as the most representative fleet-wide planning point when average utilization is closer to ~10--11%.

The mechanism is the same in both metrics: the homogeneous approach applies the fleet-average discount (M = 0.75) to every workload, including workloads with little true no-SMT benefit. The mixed fleet avoids that mismatch by treating only the stronger-benefit workloads as choosing no-SMT. The gain is largest when SMT remains competitive enough that this selectivity matters.

The carbon-selected and TCO-selected splits are close, but not identical. The TCO-selected split is usually slightly higher, meaning TCO prefers sending a bit more demand to no-SMT. Even so, the cross-metric sensitivity is small: for example, in the purpose-built case at 20% utilization, the mixed fleet saves 14.0% on carbon at the carbon-selected split and 13.7% on carbon at the TCO-selected split, a difference of 0.3 pp.

### 12.4 Practical Interpretation and Requirements

This section should **not** be interpreted as a live cloud scheduler that transparently places or migrates an existing VM between SMT and no-SMT hosts. That is not the scenario being modeled here.

That kind of dynamic assignment is not the scenario modeled here and would be hard to offer transparently for two reasons. First, the difference would likely be transparent to the user: SMT vs no-SMT changes the effective per-vCPU performance and can often be inferred from inside the VM through observed throughput, latency, and visible CPU topology or behavior. Second, the current model assumes that some workloads would also reduce requested vCPU count when choosing no-SMT. It would be very difficult for a cloud provider to infer the correct resize automatically for the user or silently apply it.

The intended interpretation is instead a **user-visible choice model**. The provider offers both SMT and no-SMT options, potentially with a no-SMT price discount that reflects the provider-side TCO or carbon benefit. The mixed-fleet model then asks: if workloads that see an aggregate benefit from no-SMT choose that option, and if those workloads scale their requested vCPUs according to the modeled performance gain, what steady-state split of fleet demand results?

Under that interpretation, the split point is a proxy for **which workloads would choose no-SMT**, not a policy for live VM placement. What matters operationally is estimating the distribution of workload discounts and how users would respond to a no-SMT offering.

Potential ways to ground that uptake model include:

- Historical evidence on how workloads resized when moved across CPU generations with different per-vCPU performance
- Application-class heuristics or benchmark-based estimates of which applications are likely to benefit from no-SMT

**[Requires validation]**: The uniform discount distribution (M = 0.50 to 1.00) is a modeling assumption, and so is the implied customer-choice rule that workloads with downstream benefit choose no-SMT. A right-skewed discount distribution (most workloads near M = 1.0) would increase the value of offering both options; a left-skewed one would decrease it. The pass-through from provider benefit to customer pricing, and the extent to which users would actually resize VMs when selecting no-SMT, both remain open validation questions.

> **Takeaway**: For a fleet whose average utilization is closer to ~10--11%, the **10% utilization row is the most representative planning point**. Under the **purpose-built scaling model**, the mixed fleet improves on homogeneous no-SMT by **+2.8 pp on carbon** and **+1.1 pp on TCO** at 10% utilization (rising to **+3.9 pp** and **+2.1 pp** at 20%). Under the **same-hardware constrained model**, the incremental gain at 10% utilization is smaller: **+1.4 pp on carbon** and **+0.4 pp on TCO**, because memory already limits both pools. The much larger 30% same-hardware gain (**+7.7 pp carbon**, **+5.8 pp TCO**) is an upper-end case where homogeneous no-SMT becomes weak, not the default fleet-wide expectation. The mixed fleet is therefore an incremental improvement, not a replacement for the homogeneous switch.

---

## 13. Open Questions and Future Work

### 13.1 Validating the Proportional Scaling Assumption

One key open question is whether the per-vCPU performance advantage of no-SMT translates to proportional demand compression in practice. However, this is a similar assumption tht we made in GreenSKU, so to some extent using a representative set of applications and deriving "scaling factors" has prior precedent that we could cite as established practice. However, if we want to be more rigorous: specific validation questions that would be nice to get some info from first-party cloud workload teams:

1. When migrating between hardware generations with different per-vCPU performance (e.g., Ice Lake to Emerald Rapids), what effective performance-per-vCPU improvement did workload teams observe?
2. Did those teams scale VM count (or VM size) proportionally to the performance change? If not, what fraction of the performance gain was captured as demand reduction vs absorbed as headroom?
3. Are there workload categories where proportional scaling is a better or worse approximation?

**[Requires validation]**: Connecting peak throughput measurements to fleet-level vCPU demand changes is the most critical open assumption. The model can be re-evaluated at any demand multiplier, but the headline savings change directly with the chosen `M`.

### 13.2 Contention Threshold Sensitivity

All analyses are conditioned on a single contention threshold (1% CWT) measured for a single application. Future work should:

1. Evaluate how the headline savings change across a range of thresholds (0.5%, 1%, 2%, 5%).
2. Determine how different applications requiring different thresholds impacts projected savings.

This work connects to the broader oversubscription research question: developing a practical threshold-setting policy that adapts to hardware configuration and workload characteristics, rather than relying on a static value.

### 13.3 Cross-Application and Cross-Hardware Generalization

- **Application diversity in R values**: The analysis uses go-cpu as the calibration workload. A natural extension is to repeat with per-application R curves or a fleet-weighted average.
- **Hardware scaling**: The c6620 experiments use up to 40 LPs and 6--8 VMs. Production servers have 200+ cores and 100+ VMs.
- **Power curve validation**: The power curve shape should be validated on more platforms.

---

## Appendix A: Results Summary Tables

This appendix is a compact reference for the main numerical results already discussed in the body of the document. It does not introduce a new analysis layer or new assumptions. Instead, it puts the headline homogeneous, breakeven, and mixed-fleet results in one place so the exact values used in Sections 9, 11, and 12 can be compared quickly.

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

### A.3 Homogeneous Breakeven M (Iso-Physical-Core, Purpose-Built Scaling Model)

The demand multiplier `M` at which no-SMT exactly matches SMT in the homogeneous iso-physical-core setup:

| Avg VM util | Carbon breakeven M | TCO breakeven M |
|---|---|---|
| 10% | 0.869 | 0.938 |
| 20% | 0.844 | 0.889 |
| 30% | 0.782 | 0.808 |

Values below 1.0 mean no-SMT needs *some* demand discount to break even. For the homogeneous setup used in Section 9.5, these breakeven values imply a required discount of about 13--22% for carbon and about 6--19% for TCO.

### A.4 Mixed Fleet Summary (Carbon % vs SMT Baseline)

| Avg VM util | Purpose-built homogeneous | Purpose-built mixed | Same-hardware constrained homogeneous | Same-hardware constrained mixed |
|---|---|---|---|---|
| 10% | -11.8% | -14.6% | -16.5% | -17.9% |
| 20% | -10.1% | -14.0% | -13.9% | -16.2% |
| 30% | -9.0% | -13.5% | -2.4% | -10.1% |
