# SMT Oversubscription Model: Technical Documentation

This document describes the mathematical model used to evaluate carbon and TCO tradeoffs between SMT-enabled and non-SMT processor configurations in cloud environments.

## Table of Contents

1. [Overview](#overview)
2. [Input Parameters](#input-parameters)
3. [Calculations](#calculations)
4. [Assumptions](#assumptions)
5. [Breakeven Analysis](#breakeven-analysis)
6. [Worked Examples](#worked-examples)
7. [Parameter Sensitivity](#parameter-sensitivity)

---

## Overview

The model evaluates carbon footprint and TCO for server deployments, enabling comparisons across different processor configurations, oversubscription strategies, and workload assumptions. The generalized breakeven analysis can find the value of **any numeric parameter** at which two scenarios match on carbon or TCO.

**Example questions the model can answer:**
- At what oversubscription ratio does non-SMT match SMT on carbon?
- What vCPU demand reduction would make a non-SMT fleet equivalent?
- How does carbon intensity affect the embodied vs operational balance?
- At what utilization does oversubscription stop being beneficial?

### Why This Matters

SMT (Simultaneous Multi-Threading) provides 2 logical threads per physical core, effectively doubling the vCPU capacity per server. However, SMT introduces scheduling constraints in virtualized environments:

1. **Anti-affinity constraints**: vCPUs from different VMs often cannot share sibling SMT threads for security/performance isolation
2. **Topology constraints**: Guest VMs may expect specific CPU topology
3. **Contention overhead**: Co-running threads on the same core increases effective utilization

These constraints limit achievable oversubscription with SMT. Non-SMT configurations avoid these constraints but have fewer logical CPUs per server, requiring more servers for the same workload.

### Model Flow

```
Inputs                    Calculations                 Outputs
─────────────────────────────────────────────────────────────────
Workload                  Server Count           ──▶   # Servers
  - total_vcpus           Utilization            ──▶   Effective Util
  - avg_util              Power                  ──▶   Power/Server

Processor                 Embodied Carbon        ──▶   Embodied CO2
  - physical_cores        Operational Carbon     ──▶   Operational CO2
  - threads_per_core
  - power_curve           Embodied Cost          ──▶   Server Cost
                          Operational Cost       ──▶   Energy Cost
Scenario
  - oversub_ratio
  - util_overhead         ════════════════════════════════════════
                          Total Carbon (kg CO2e)
Cost                      Total Cost (USD)
  - embodied_carbon_kg    Carbon per vCPU
  - carbon_intensity      Cost per vCPU
  - electricity_cost
```

---

## Input Parameters

### Processor Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `physical_cores` | int | Physical CPU cores per server (e.g., 48) |
| `threads_per_core` | int | Logical threads per core: 2 for SMT, 1 for non-SMT |
| `power_curve.p_idle` | float | Server power at 0% utilization (Watts) |
| `power_curve.p_max` | float | Server power at 100% utilization (Watts) |
| `power_curve.curve_fn` | function | Power-utilization relationship (see [Power Curves](#power-curves)) |
| `core_overhead` | int | pCPUs reserved for hypervisor/host (not oversubscribable) |

**Derived:**
- `pcpus = physical_cores × threads_per_core` (total logical CPUs)
- `available_pcpus = pcpus - core_overhead` (pCPUs available for VMs)

### Scenario Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `oversub_ratio` | float | vCPU:pCPU ratio. 1.0 = no oversubscription, 2.0 = 2:1 oversub |
| `util_overhead` | float | Additive utilization overhead (e.g., 0.05 for 5% SMT contention) |
| `vcpu_demand_multiplier` | float | Scales total vCPU demand (e.g., 0.7 = 30% less demand) |

### Workload Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_vcpus` | int | Total vCPU demand across all VMs |
| `avg_util` | float | Average VM utilization [0.0, 1.0] (e.g., 0.3 = 30%) |

### Cost Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embodied_carbon_kg` | float | Embodied carbon per server (kg CO2e) |
| `server_cost_usd` | float | Capital cost per server (USD) |
| `carbon_intensity_g_kwh` | float | Grid carbon intensity (g CO2/kWh) |
| `electricity_cost_usd_kwh` | float | Electricity price ($/kWh) |
| `lifetime_hours` | float | Server operational lifetime (hours) |

---

## Calculations

### Step 1: Server Count

The number of servers needed to serve the workload:

```
effective_vcpus = total_vcpus × vcpu_demand_multiplier

vcpu_capacity_per_server = available_pcpus × oversub_ratio

num_servers = ⌈effective_vcpus / vcpu_capacity_per_server⌉
```

**Example:**
- 10,000 vCPUs, 48 cores × 2 threads = 96 pCPUs, 8 reserved → 88 available
- At R=1.0: `⌈10000 / 88⌉ = 114 servers`
- At R=1.5: `⌈10000 / 132⌉ = 76 servers`

### Step 2: Server Utilization

Average utilization per server based on total work distributed across capacity:

```
total_work = effective_vcpus × avg_util    (in pCPU-equivalents)

total_capacity = num_servers × available_pcpus

avg_util_per_server = total_work / total_capacity

effective_util = min(1.0, avg_util_per_server + util_overhead)
```

**Key insight:** Higher oversubscription → fewer servers → higher per-server utilization.

**Example:**
- 10,000 vCPUs at 30% util → 3,000 pCPU-equivalents of work
- 114 servers × 88 pCPUs = 10,032 capacity → 30% avg util
- With 5% overhead → 35% effective util

### Step 3: Power Consumption

Power per server at the effective utilization:

```
P(u) = P_idle + (P_max - P_idle) × f(u)
```

Where `f(u)` is the power curve function mapping utilization [0,1] to power factor [0,1].

#### Power Curves

| Curve | Function | Description |
|-------|----------|-------------|
| Linear | `f(u) = u` | Simple linear scaling (default) |
| SPECpower | `f(u) = u^0.9` | Slightly sublinear, matches empirical data |
| Polynomial | Complex formula | Empirically-fit curve from server measurements |

**SPECpower curve** reflects that power doesn't scale perfectly linearly with utilization—there's diminishing returns at higher utilization.

### Step 4: Carbon Footprint

**Embodied Carbon:**
```
embodied_carbon = num_servers × embodied_carbon_kg
```

**Operational Carbon:**
```
total_energy_kwh = num_servers × power_per_server_w × lifetime_hours / 1000

operational_carbon = total_energy_kwh × carbon_intensity_g_kwh / 1000
```

**Total Carbon:**
```
total_carbon_kg = embodied_carbon + operational_carbon
```

### Step 5: Total Cost of Ownership

**Embodied Cost:**
```
embodied_cost = num_servers × server_cost_usd
```

**Operational Cost:**
```
operational_cost = total_energy_kwh × electricity_cost_usd_kwh
```

**Total TCO:**
```
total_cost_usd = embodied_cost + operational_cost
```

---

## Assumptions

### Server Assumptions

1. **Homogeneous fleet**: All servers have identical specifications
2. **Full deployment**: All servers run for the entire lifetime (no scaling)
3. **No redundancy**: Model calculates minimum servers needed (no N+1, etc.)
4. **Linear pCPU equivalence**: 1 vCPU = 1 pCPU worth of work when at 100% utilization

### Power Assumptions

1. **Steady-state**: Power calculated at average utilization, not time-varying
2. **No power-off**: Servers remain on regardless of load (can be changed)
3. **Uniform utilization**: Work distributed evenly across servers
4. **Power curve fidelity**: Real power may deviate from modeled curve

### SMT Assumptions

1. **2:1 threading**: SMT provides exactly 2 threads per core
2. **Full pCPU exposure**: All logical CPUs are visible to the hypervisor
3. **Overhead is additive**: SMT contention modeled as flat utilization overhead

### Cost Assumptions

1. **Linear embodied carbon**: Carbon scales linearly with server count
2. **Fixed carbon intensity**: Grid carbon intensity constant over lifetime
3. **No cooling overhead**: PUE not explicitly modeled (can be folded into power)
4. **No maintenance costs**: Only capital and energy costs included


---

## Worked Examples

### Example 1: Oversubscription Ratio Breakeven

**Question:** At what oversubscription ratio does non-SMT match SMT+oversub on carbon?

**Configuration:**
- 10,000 vCPUs at 30% average utilization
- 48 physical cores per server
- SMT: 2 threads/core, 400W max, 100W idle
- Non-SMT: 1 thread/core, 340W max (85% of SMT), 90W idle
- 5-year lifetime, 1000 kg embodied carbon, 400 g/kWh grid

### SMT Baseline (R=1.0)

```
pCPUs per server: 48 × 2 = 96
Servers needed: ⌈10000 / 96⌉ = 105

Average util: (10000 × 0.3) / (105 × 96) = 29.8%
Power: 100 + 300 × (0.298)^0.9 = 198W

Embodied carbon: 105 × 1000 = 105,000 kg
Energy: 105 × 198W × 43800h = 909,000 kWh
Operational carbon: 909,000 × 0.4 = 363,600 kg

Total carbon: 468,600 kg
```

### SMT + Oversub (R=1.3)

```
Servers needed: ⌈10000 / (96 × 1.3)⌉ = 81

Average util: (10000 × 0.3) / (81 × 96) = 38.6%
With 5% overhead: 43.6%
Power: 100 + 300 × (0.436)^0.9 = 241W

Embodied carbon: 81 × 1000 = 81,000 kg
Energy: 81 × 241W × 43800h = 855,000 kWh
Operational carbon: 855,000 × 0.4 = 342,000 kg

Total carbon: 423,000 kg
Savings vs baseline: 9.7%
```

### Non-SMT Breakeven

To match 423,000 kg with non-SMT (48 pCPUs per server):

Binary search finds **R ≈ 2.58**

```
Servers needed: ⌈10000 / (48 × 2.58)⌉ = 81

Average util: (10000 × 0.3) / (81 × 48) = 77.2%
Power: 90 + 250 × (0.772)^0.9 = 288W

Embodied carbon: 81 × 1000 = 81,000 kg
Energy: 81 × 288W × 43800h = 1,022,000 kWh
Operational carbon: 1,022,000 × 0.4 = 408,800 kg

Total carbon: 489,800 kg  (still higher due to power)
```

This example shows that finding breakeven on oversubscription ratio alone may not always yield a solution — the target metric depends on multiple interacting factors.

---

### Example 2: vCPU Demand Multiplier Breakeven

**Question:** If non-SMT has a fixed oversubscription ratio (R=1.4), what reduction in vCPU demand would make it match SMT+oversub on carbon?

This flips the question: instead of asking "how much more oversub does non-SMT need?", we ask "how much less demand could non-SMT serve while matching carbon?"

**Setup:**
- Reference: SMT with R=1.1 and 5% util overhead
- Target: Non-SMT with R=1.4, vary `vcpu_demand_multiplier` in [0.5, 1.0]

Binary search finds **vcpu_demand_multiplier ≈ 0.73**, meaning non-SMT at R=1.4 matches SMT+oversub carbon when serving 73% of the vCPU demand.

**Interpretation:** Non-SMT would need 27% fewer vCPUs (or equivalently, could serve 27% fewer VMs) to achieve carbon parity with SMT+oversub at the same server count.

---

## Parameter Sensitivity

Key parameters and their effects on carbon/TCO outcomes:

| Parameter | Effect When Increased |
|-----------|----------------------|
| `carbon_intensity_g_kwh` | Operational carbon dominates → power efficiency matters more |
| `embodied_carbon_kg` | Embodied dominates → server count matters more |
| `avg_util` | Higher base util → less headroom, oversub less effective |
| `oversub_ratio` | Fewer servers → lower embodied, but higher util and power |
| `vcpu_demand_multiplier` | More demand → more servers needed |
| `util_overhead` | Higher effective util → more power per server |

Use **sweep analyses** to explore how breakeven values change across parameter ranges. For example, sweep `carbon_intensity_g_kwh` from 100-800 to see how grid cleanliness affects the embodied vs operational tradeoff.
