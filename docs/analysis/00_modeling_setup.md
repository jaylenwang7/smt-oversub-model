# 00: Modeling Setup and Assumptions

This document describes the model, its parameters, and the specific values chosen
for the analyses that follow. It is not an analysis itself but a reference for
understanding what underlies the numbers in [01] through [03a].

Read this first if you want to understand **what the model computes**, **what
numbers go in**, and **where those numbers come from**.

## Prerequisites

- Read the [spine document](SMT_VS_NOSMT_ANALYSIS.md) for high-level context,
  terminology, and the analysis progression.

---

## The Model in Brief

The model (`smt_oversub_model`) answers a simple question: given a fleet of
servers that must host a fixed pool of vCPUs, what is the **total carbon
footprint** and **total cost of ownership (TCO)** over the server lifetime?

It does this in five steps:

1. **Server count**: How many servers are needed?
   ```
   effective_vcpus = total_vcpus * vcpu_demand_multiplier
   vcpu_capacity_per_server = available_pcpus * oversub_ratio
   num_servers = ceil(effective_vcpus / vcpu_capacity_per_server)
   ```

2. **Utilization**: What is the average per-server utilization?
   ```
   avg_util = (effective_vcpus * avg_util) / (num_servers * available_pcpus)
   effective_util = min(1.0, avg_util + util_overhead)
   ```

3. **Power**: What does each server draw at that utilization?
   ```
   P(u) = P_idle + (P_max - P_idle) * f(u)
   ```
   where `f(u)` is a curve function mapping [0,1] utilization to [0,1] power
   factor. With per-component power breakdowns, total server power is the sum
   of component-level power curves.

4. **Carbon**: Embodied (manufacturing) + operational (electricity) emissions.
   ```
   embodied_carbon = num_servers * embodied_carbon_per_server
   operational_carbon = num_servers * power_per_server * lifetime_hours / 1000 * carbon_intensity / 1000
   total_carbon = embodied_carbon + operational_carbon
   ```

5. **TCO**: Capital (server purchase) + operational (electricity) cost.
   ```
   embodied_cost = num_servers * server_cost_per_server
   operational_cost = total_energy_kwh * electricity_cost_per_kwh
   total_cost = embodied_cost + operational_cost
   ```

The model is deterministic: same inputs always produce the same outputs. There is
no simulation, no randomness, and no time-varying behavior -- it computes
steady-state totals for a fleet running at a fixed average utilization over a fixed
lifetime.

### What the model does not include

- **PUE (Power Usage Effectiveness)**: Cooling and datacenter overhead are not
  explicitly modeled. If needed, PUE can be folded into the power values (e.g.,
  multiply max power by 1.12 for PUE=1.12).
- **Maintenance / operational costs**: Only capital (server purchase) and
  electricity costs are included. No staffing, networking, or software costs.
- **Time-varying behavior**: The model uses a single average utilization. It does
  not model diurnal patterns, workload bursts, or server scaling.
- **Redundancy**: The model calculates the minimum number of servers. No N+1
  spare capacity or failure domains.

---

## Parameter Categories

The model takes four categories of input. Each analysis config specifies all of
them.

### 1. Processor Configuration

Defines the physical server hardware: core count, threading, power envelope, and
per-component embodied carbon and cost.

| Parameter | What it controls |
|---|---|
| `physical_cores` | CPU cores on the die |
| `threads_per_core` | 2 for SMT, 1 for no-SMT |
| `thread_overhead` | pCPUs reserved for hypervisor (not available for VMs) |
| `power_breakdown` | Per-component power at idle and max (CPU, memory, SSD, NIC, chassis) |
| `embodied_carbon` | Per-thread + per-server carbon breakdown (kg CO2e) |
| `server_cost` | Per-thread + per-server cost breakdown (USD) |

The per-thread/per-server split is important: components like memory and SSD scale
with HW thread count (per-thread), while CPU die, NIC, chassis, and rack share are
fixed per server (per-server). This means that when SMT doubles the thread count,
memory and SSD carbon/cost double per server, but the fixed components do not.

**Derived quantities:**
- `hw_threads = physical_cores * threads_per_core`
- `available_pcpus = hw_threads - thread_overhead`
- `embodied_carbon_per_server = sum(per_thread * hw_threads) + sum(per_server)`
- `server_cost_per_server = sum(per_thread * hw_threads) + sum(per_server)`

### 2. Scenario Parameters

Define how the server is deployed: oversubscription level, scheduling overhead,
and demand adjustments.

| Parameter | What it controls |
|---|---|
| `oversub_ratio` (R) | vCPU-to-pCPU packing ratio. R=1.0 means no oversubscription. |
| `util_overhead` | Additive utilization overhead (e.g., 0.05 for SMT contention) |
| `vcpu_demand_multiplier` | Scales total vCPU demand. Values < 1.0 represent demand compression from higher per-vCPU performance. |
| `resource_scaling` | Which embodied components scale with vCPU count (for R > 1.0) |
| `resource_constraints` | Fixed hardware capacities that may limit effective R |

These are the parameters that **vary across analyses**. Each sub-document changes
one or two of these to isolate the effect of a specific mechanism.

### 3. Workload Parameters

Define the demand the fleet must serve.

| Parameter | Value used | Notes |
|---|---|---|
| `total_vcpus` | 100,000 | Fixed across all analyses. Large enough that rounding effects from `ceil()` are small. |
| `avg_util` | Varies (0.10, 0.20, 0.30) | Swept across utilization levels in most analyses. Represents average VM utilization. |

### 4. Cost / Environmental Parameters

Define the economic and environmental constants.

| Parameter | Value used | What it represents |
|---|---|---|
| `carbon_intensity_g_kwh` | 175 | Grid carbon intensity in grams CO2 per kWh of electricity |
| `electricity_cost_usd_kwh` | 0.28 | Electricity price in USD per kWh |
| `lifetime_years` | 6 | Server operational lifetime (= 52,560 hours) |

These are **held constant** across all analyses in this spine. They define the
economic and environmental backdrop against which all comparisons are made.

---

## Data Source: GreenSKU Numbers

All processor-level numbers (power, embodied carbon, cost, component counts) come
from a single source: the **GreenSKU** dataset, accessed via the `frugal-model`
repository (`/Users/jaylenw/Code/frugal-model/`).

GreenSKU is a per-component server carbon and cost model developed through a
Microsoft research collaboration and published in prior work. It provides
validated, industry-sourced data for per-component power draw, embodied carbon
(manufacturing emissions), and capital cost for datacenter server hardware. The
numbers have been used in published research, making them a credible,
externally-validated basis for modeling.

The specific extraction pipeline is in
[`/Users/jaylenw/Code/frugal-model/2026_analysis.py`](/Users/jaylenw/Code/frugal-model/2026_analysis.py),
which reads per-component YAML data files from
`frugal-model/data_sources_paper/` and computes derived quantities like carbon
intensity. The processor configurations used in this analysis spine are defined
in [`configs/shared/genoa_processors.jsonc`](../../configs/shared/genoa_processors.jsonc),
which documents the exact derivation of each value in its comments.

---

## Reference Platform: AMD EPYC Genoa 80-Core

All analyses use a single-socket AMD EPYC Genoa 80-core server as the reference
platform. This is the `Genoa-default-1S` configuration from the GreenSKU dataset.

### Physical Platform

| Component | Specification | Source |
|---|---|---|
| CPU | AMD EPYC Genoa, 80 physical cores, 1 socket | CPU.yaml |
| Memory | 12 x 64GB DDR5 4800MHz DIMMs = 768 GB | DRAM.yaml |
| Storage | 6 x E1.S 2TB SSDs = 12 TB | SSD.yaml |
| NIC | 200G Glacier Peak | NIC.yaml |
| Chassis | C2195 1U server (includes 2% repair overhead) | server.yaml |
| Rack | E4000 rack, 34 servers per rack | rack.yaml |

### SMT vs No-SMT Configurations

The "no-SMT" configuration is the **same physical CPU** with SMT disabled, not a
different chip. The key modeling choice is how memory and SSD scale:

- **Purpose-built no-SMT** (`genoa_nosmt`): Memory and SSD are halved
  proportionally to HW threads (6 DIMMs, 3 SSDs for 80 threads vs 12 DIMMs, 6
  SSDs for 160 threads). This models ordering servers configured for no-SMT
  density.
- **Same hardware** (`genoa_nosmt_smt_hw`): Full 12 DIMMs and 6 SSDs retained.
  This models disabling SMT on existing SMT-provisioned servers.

The purpose-built configuration is the default in most analyses ([01]-[03]).
The same-hardware configuration is used specifically in [03a] to model the
"just flip the switch" deployment scenario.

### Per-Component Embodied Carbon

| Component | Per-unit | Count (SMT) | Count (no-SMT) | Per-server (SMT) | Per-server (no-SMT) |
|---|---|---|---|---|---|
| CPU die | 34.2 kg | 1 | 1 | 34.2 kg | 34.2 kg |
| Memory (DIMM) | 59.1 kg | 12 | 6 | 709.2 kg | 354.6 kg |
| SSD | 103.0 kg | 6 | 3 | 618.0 kg | 309.0 kg |
| NIC | 115.0 kg | 1 | 1 | 115.0 kg | 115.0 kg |
| Chassis | 255.5 kg | 1 | 1 | 255.5 kg | 255.5 kg |
| Rack share | 51.9 kg | 1 | 1 | 51.9 kg | 51.9 kg |
| **Total** | | | | **1,783.8 kg** | **1,120.2 kg** |

Rack share = 1,765.2 kg total rack carbon / 34 servers per rack.

**Structural observation**: The per-server fixed costs (CPU + NIC + chassis +
rack = 456.6 kg) are identical for both configurations. They represent 26% of
SMT per-server carbon but 41% of no-SMT per-server carbon. This fixed-cost
dilution is a structural advantage of SMT: more threads per server means the
fixed overhead is amortized over more vCPUs.

### Per-Component Server Cost

| Component | Per-unit | Count (SMT) | Count (no-SMT) | Per-server (SMT) | Per-server (no-SMT) |
|---|---|---|---|---|---|
| CPU | $1,487 | 1 | 1 | $1,487 | $1,487 |
| Memory (DIMM) | $440 | 12 | 6 | $5,280 | $2,640 |
| SSD | $272.71 | 6 | 3 | $1,636 | $818 |
| NIC | $1,022 | 1 | 1 | $1,022 | $1,022 |
| Chassis | $1,505 | 1 | 1 | $1,505 | $1,505 |
| Rack share | $510 | 1 | 1 | $510 | $510 |
| **Total** | | | | **$11,440** | **$7,982** |

### Per-Component Power

Power is modeled per-component, with each component having an idle and max power
draw. The model interpolates between idle and max using a utilization-dependent
curve function.

| Component | Idle (W) | Max (W) | Notes |
|---|---|---|---|
| **SMT (12 DIMMs, 6 SSDs)** | | | |
| CPU | 94 | 315 | 300W TDP * 1.05 PSU loss |
| Memory | 39 | 131 | 12 DIMMs * 10.4W TDP * 1.05 |
| SSD | 19 | 94 | 6 SSDs * 15W TDP * 1.05 |
| NIC | 32 | 107 | 102W TDP * 1.05 |
| Chassis | 60 | 116 | DC-SCM 35W + Fan 75W, * 1.05 |
| **Total** | **244** | **763** | |
| **No-SMT (6 DIMMs, 3 SSDs)** | | | |
| CPU | 94 | 315 | Same die, SMT off doesn't change power envelope |
| Memory | 20 | 66 | 6 DIMMs (half of SMT) |
| SSD | 9 | 47 | 3 SSDs (half of SMT) |
| NIC | 32 | 107 | Same NIC |
| Chassis | 60 | 116 | Same chassis |
| **Total** | **215** | **651** | |

Idle power is derived from TDP using derate factors at zero utilization (from
the GreenSKU spec_derate curves): CPU 0.30, DRAM 0.30, SSD 0.20, NIC 0.30, Fan
0.30. All values include a 1.05x PSU efficiency loss factor.

### Thread Overhead

The hypervisor reserves a fraction of HW threads for host use. Following the
GreenSKU `Genoa-default-1S` configuration, this is set to **10% of HW threads**:

- SMT: 10% of 160 = **16** reserved pCPUs, leaving **144** available for VMs
- No-SMT: 10% of 80 = **8** reserved pCPUs, leaving **72** available for VMs

### Power Curve Models

Two power curve models are used across the analyses:

1. **Polynomial (default)**: A frequency-dependent polynomial fit that captures the
   sublinear (concave) relationship between utilization and power. This is applied
   to the CPU component by default and reflects the well-documented SPECpower-like
   behavior where power rises steeply at low utilization then flattens.

2. **Linear CPU power**: The CPU component uses `P(u) = P_idle + (P_max - P_idle) * u`.
   Motivated by experimental measurements on a c6620 server showing that no-SMT
   power scales nearly linearly with utilization (R^2 = 0.998 for linear fit). See
   [01: Naive Comparison](01_naive_comparison.md) for the experimental basis.

Non-CPU components (memory, SSD, NIC, chassis) default to the global power curve
(polynomial) unless overridden. The `genoa_nosmt_linear` processor variant applies
the linear curve to CPU only.

### Processor Configuration Summary

| Config name | SMT | DIMMs/SSDs | CPU power curve | Embodied/server | Cost/server | Used in |
|---|---|---|---|---|---|---|
| `genoa_smt` | On | 12/6 | polynomial | 1,784 kg | $11,440 | All |
| `genoa_nosmt` | Off | 6/3 | polynomial | 1,120 kg | $7,982 | [01]-[03] |
| `genoa_nosmt_linear` | Off | 6/3 | linear CPU | 1,120 kg | $7,982 | [01]-[03] |
| `genoa_nosmt_smt_hw` | Off | 12/6 | polynomial | 1,784 kg | $11,440 | [03a] |
| `genoa_nosmt_smt_hw_linear` | Off | 12/6 | linear CPU | 1,784 kg | $11,440 | [03a] |
| `genoa_nosmt_5pct_area` | Off | 6/3 | polynomial | 1,118 kg | $7,982 | Sensitivity |
| `genoa_nosmt_5pct_area_linear` | Off | 6/3 | linear CPU | 1,118 kg | $7,982 | Sensitivity |

The 5% area variants model a hypothetical CPU die shrink from removing SMT logic,
reducing CPU embodied carbon by 5% (34.2 -> 32.49 kg). CPU cost is unchanged
(same wafer/packaging process).

Full definitions with derivation comments:
[`configs/shared/genoa_processors.jsonc`](../../configs/shared/genoa_processors.jsonc)

---

## Cost and Environmental Constants

These values are fixed across all analyses.

### Carbon Intensity: 175 g CO2/kWh

This represents the grid carbon intensity -- grams of CO2 emitted per kilowatt-hour
of electricity consumed by the datacenter.

This value is derived from the GreenSKU pipeline
(`frugal-model/2026_analysis.py`), which back-calculates an effective carbon
intensity from a weighted-average Scope 2 percentage across a fleet of
datacenters. The calculation uses datacenter-level Scope 2 emissions data
(`dc_data.csv`) and a baseline server configuration to solve for the carbon
intensity that produces the observed embodied/operational carbon split.

175 g/kWh is a moderate value, roughly representative of a mixed-grid datacenter
fleet (not exclusively clean or dirty). For reference:
- Very clean grids (e.g., hydro/nuclear-heavy regions): ~20-50 g/kWh
- US average grid: ~400 g/kWh
- Coal-heavy grids: ~800-1000 g/kWh

At 175 g/kWh, the analyses produce a roughly balanced embodied/operational carbon
split (neither strongly dominated by one or the other), which means conclusions
about the SMT tradeoff are not artifacts of an extreme carbon intensity assumption.

### Electricity Cost: $0.28/kWh

A representative commercial/industrial electricity rate. The GreenSKU dataset uses
this as the default electricity price.

### Server Lifetime: 6 Years

Servers are assumed to operate for 6 years (52,560 hours). This is consistent with
the GreenSKU default (`SERVER_LIFETIME_HOURS = 52560` in `2026_analysis.py`) and
falls within the typical 4-7 year range for cloud datacenter hardware.

A longer lifetime amplifies operational costs/carbon relative to embodied, since
the same servers consume electricity for more years while the one-time
manufacturing carbon is fixed. At 6 years, operational carbon is the larger
contributor for most configurations at 175 g/kWh.

---

## What Varies Across Analyses

The parameters above (processor specs, carbon intensity, electricity cost,
lifetime, total vCPUs) are **held constant** across all analyses. What changes
from one sub-document to the next is the set of **scenario parameters** that
define how the servers are deployed:

| Analysis | What varies | What is introduced |
|---|---|---|
| [01: Naive Comparison](01_naive_comparison.md) | Nothing -- both SMT and no-SMT at R=1.0 | Baseline penalty of disabling SMT |
| [02: Scheduling Constraints](02_scheduling_constraints_oversub.md) | `oversub_ratio` (from experimental steal-time data) | SMT constrained to lower R; no-SMT can oversubscribe more |
| [02a: Resource Modeling](02a_resource_modeling.md) | `resource_scaling`, `resource_constraints` | How memory/SSD costs change at R > 1.0 |
| [02b: Savings Scaling](02b_oversub_savings_scaling.md) | `oversub_ratio` swept from 1.0 to 5.0 | Sublinear returns from oversubscription |
| [02c: Input Sensitivity](02c_scheduling_input_basis_sensitivity.md) | `oversub_ratio` (recalibrated experimental values) | Sensitivity to experimental input basis |
| [03: vCPU Demand Discount](03_vcpu_demand_discount.md) | `vcpu_demand_multiplier` | No-SMT's higher per-vCPU performance reduces demand |
| [03a: Constrained Savings](03a_constrained_savings.md) | `resource_constraints` + `vcpu_demand_multiplier` | Same hardware (no DIMM/SSD reduction) with demand compression |

This layered approach isolates the effect of each mechanism. When reading a result,
you can trace exactly which parameters produced it and what assumptions underlie
the comparison.

---

## Reproducing the Numbers

All processor configurations are defined in a single shared file:

```bash
# View processor definitions (with derivation comments)
cat configs/shared/genoa_processors.jsonc
```

Every analysis config references these processors and specifies the cost/workload
parameters inline. To see the full resolved configuration for any analysis:

```bash
# Run any analysis -- the output includes a config.json with all values expanded
python -m smt_oversub_model configs/oversub_analysis/genoa/no_oversub_comparison.jsonc

# Check the resolved config
cat results/oversub_analysis/genoa/no_oversub_comparison/config.json
```

For detailed model documentation (calculation formulas, resource scaling math,
analysis type specifications), see [`docs/MODELING.md`](../MODELING.md).
