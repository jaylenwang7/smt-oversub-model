# SMT Oversubscription Model: Technical Documentation

This document describes the mathematical model used to evaluate carbon and TCO tradeoffs between SMT-enabled and non-SMT processor configurations in cloud environments.

## Table of Contents

1. [Overview](#overview)
2. [Input Parameters](#input-parameters)
3. [Calculations](#calculations)
4. [Power Modeling](#power-modeling)
5. [Cost Modeling](#cost-modeling)
6. [Resource Modeling](#resource-modeling)
7. [Assumptions](#assumptions)
8. [Analysis Types](#analysis-types)
9. [Composite Scenarios](#composite-scenarios)
10. [Worked Examples](#worked-examples)
11. [Parameter Sensitivity](#parameter-sensitivity)

---

## Overview

The model evaluates carbon footprint and TCO for server deployments, enabling comparisons across different processor configurations, oversubscription strategies, and workload assumptions. The generalized breakeven analysis can find the value of **any numeric parameter** at which two scenarios match on carbon or TCO.

**Example questions the model can answer:**
- At what oversubscription ratio does non-SMT match SMT on carbon?
- What vCPU demand reduction would make a non-SMT fleet equivalent?
- How does carbon intensity affect the embodied vs operational balance?
- At what utilization does oversubscription stop being beneficial?
- What is the effective oversubscription ratio when memory constrains packing?
- Should workloads be split across a heterogeneous SMT + non-SMT fleet?
- How do savings vary across utilization levels and vCPU demand discounts?

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
  - power_breakdown       Operational Cost       ──▶   Energy Cost

Scenario                  Resource Constraints   ──▶   Effective R
  - oversub_ratio         Resource Scaling       ──▶   Scaled Costs
  - util_overhead         ════════════════════════════════════════
  - resource_scaling      Total Carbon (kg CO2e)
  - resource_constraints  Total Cost (USD)
                          Carbon per vCPU
Cost                      Cost per vCPU
  - embodied_carbon
  - server_cost           Breakdown by component
  - carbon_intensity        (CPU, memory, SSD, chassis, NIC)
  - electricity_cost
```

---

## Input Parameters

### Processor Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `physical_cores` | int | Physical CPU cores per server (e.g., 80 for Genoa) |
| `threads_per_core` | int | Logical threads per core: 2 for SMT, 1 for non-SMT |
| `thread_overhead` | int | pCPUs reserved for hypervisor/host (not oversubscribable) |
| `power_idle_w` / `power_max_w` | float | Server power at 0% / 100% utilization (Watts) — flat mode |
| `power_breakdown` | dict | Per-component power (CPU, memory, SSD, NIC, chassis) — structured mode |
| `power_curve` | dict | Power-utilization curve type (per-processor or per-component) |
| `embodied_carbon` | dict | Per-thread + per-server carbon breakdown (kg CO2e) |
| `server_cost` | dict | Per-thread + per-server cost breakdown (USD) |

**Derived:**
- `hw_threads = physical_cores × threads_per_core` (total logical CPUs)
- `available_pcpus = hw_threads - thread_overhead` (pCPUs available for VMs)

### Scenario Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `oversub_ratio` | float | vCPU:pCPU ratio. 1.0 = no oversubscription, 2.0 = 2:1 oversub |
| `util_overhead` | float | Additive utilization overhead (e.g., 0.05 for 5% SMT contention) |
| `vcpu_demand_multiplier` | float | Scales total vCPU demand (e.g., 0.7 = 30% less demand due to performance gap) |
| `resource_scaling` | dict | Components that scale with vCPU count instead of HW threads |
| `resource_constraints` | dict | Fixed resource capacities that limit effective packing |

### Workload Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_vcpus` | int | Total vCPU demand across all VMs |
| `avg_util` | float | Average VM utilization [0.0, 1.0] (e.g., 0.3 = 30%) |
| `traits` | dict | Optional trait distributions for composite scenario allocation |

### Cost Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embodied_carbon_kg` | float | Embodied carbon per server (kg CO2e) — flat mode |
| `server_cost_usd` | float | Capital cost per server (USD) — flat mode |
| `carbon_intensity_g_kwh` | float | Grid carbon intensity (g CO2/kWh) |
| `electricity_cost_usd_kwh` | float | Electricity price ($/kWh) |
| `lifetime_years` | float | Server operational lifetime (years) |

---

## Calculations

### Step 1: Server Count

The number of servers needed to serve the workload:

```
effective_vcpus = total_vcpus × vcpu_demand_multiplier

vcpu_capacity_per_server = available_pcpus × oversub_ratio

num_servers = ⌈effective_vcpus / vcpu_capacity_per_server⌉
```

**Example (Genoa, 80 cores, 2 tpc, 8 overhead):**
- hw_threads = 160, available_pcpus = 152
- 100,000 vCPUs at R=1.0: `⌈100000 / 152⌉ = 659 servers`
- 100,000 vCPUs at R=2.34: `⌈100000 / (152 × 2.34)⌉ = 282 servers`

### Step 2: Server Utilization

Average utilization per server based on total work distributed across capacity:

```
total_work = effective_vcpus × avg_util    (in pCPU-equivalents)

total_capacity = num_servers × available_pcpus

avg_util_per_server = total_work / total_capacity

effective_util = min(1.0, avg_util_per_server + util_overhead)
```

**Key insight:** Higher oversubscription → fewer servers → higher per-server utilization.

### Step 3: Power Consumption

Power per server at the effective utilization:

```
P(u) = P_idle + (P_max - P_idle) × f(u)
```

Where `f(u)` is the power curve function mapping utilization [0,1] to power factor [0,1].

With `power_breakdown`, the composite power is the sum of all component power curves:

```
P_total(u) = Σ_component P_component(u)
           = Σ_component [P_idle_c + (P_max_c - P_idle_c) × f_c(u)]
```

### Step 4: Carbon Footprint

**Embodied Carbon:**
```
embodied_carbon = num_servers × embodied_carbon_kg
```

With structured breakdown:
```
embodied_carbon_per_server = Σ(per_thread_components × hw_threads) + Σ(per_server_components)
embodied_carbon = num_servers × embodied_carbon_per_server
```

**Operational Carbon:**
```
lifetime_hours = lifetime_years × 8760
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

With structured breakdown:
```
server_cost_per_server = Σ(per_thread_components × hw_threads) + Σ(per_server_components)
embodied_cost = num_servers × server_cost_per_server
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

## Power Modeling

### Power Curve Types

| Curve | Function | Use Case |
|-------|----------|----------|
| `linear` | `f(u) = u` | Simple first-order estimate |
| `specpower` | `f(u) = u^0.9` | Slightly sublinear, matches SPECpower empirical data |
| `power` | `f(u) = u^exponent` | Custom exponent for specific hardware |
| `polynomial` | Frequency-dependent fit | Empirically-fit from server measurements |

### Per-Component Power Breakdown

Rather than a single P_idle/P_max per server, processors can specify per-component power:

```json
{
  "power_breakdown": {
    "cpu": {"idle_w": 30, "max_w": 234, "power_curve": {"type": "polynomial"}},
    "memory": {"idle_w": 20, "max_w": 66, "power_curve": {"type": "linear"}},
    "ssd": {"idle_w": 6, "max_w": 12, "power_curve": {"type": "linear"}},
    "nic": {"idle_w": 10, "max_w": 15, "power_curve": {"type": "linear"}},
    "chassis": {"idle_w": 10, "max_w": 10, "power_curve": {"type": "linear"}}
  }
}
```

Each component can have its own power curve type. The model builds a composite `PowerCurve` by summing all component curves. This allows:
- CPU to follow a polynomial/specpower curve while memory scales linearly
- Fixed-power components (chassis fans) with constant idle_w = max_w
- Resource scaling to selectively scale only affected components (e.g., memory power increases with more DIMMs)

---

## Cost Modeling

### Structured Per-Thread/Per-Server Breakdown

Embodied carbon and server cost support structured breakdowns:

```json
{
  "embodied_carbon": {
    "per_thread": {"cpu_die": 2.37, "memory": 4.43, "ssd": 1.25},
    "per_server": {"chassis": 100.0, "nic": 15.0}
  }
}
```

- **per_thread** components scale with `hw_threads` (physical_cores × threads_per_core)
- **per_server** components are flat per server
- Total per server: `Σ(per_thread × hw_threads) + Σ(per_server)`

### Priority Chain

Cost values are resolved with this precedence (highest to lowest):

1. Processor-level structured (`embodied_carbon: {per_thread, per_server}`)
2. Processor-level flat (`embodied_carbon_kg: 980`)
3. Global cost structured (`cost.embodied_carbon: {...}`)
4. Global cost flat (`cost.embodied_carbon_kg: 1000`)

### Ratio-Based Cost Mode

Instead of specifying raw parameters, specify operational/embodied ratios:

```json
{
  "cost": {
    "mode": "ratio_based",
    "reference_scenario": "baseline",
    "operational_carbon_fraction": 0.75,
    "embodied_carbon_kg": 2000.0,
    "lifetime_years": 5.0
  }
}
```

**Embodied Anchor Math:**
```
Given: operational_carbon_fraction (f_op), embodied_carbon_kg, reference scenario
1. embodied_total = num_servers × embodied_carbon_kg
2. operational_total = embodied_total × f_op / (1 - f_op)
3. carbon_intensity_g_kwh = operational_total × 1000 / energy_kwh
```

**Total Anchor Math:**
```
Given: operational_carbon_fraction (f_op), total_carbon_kg, reference scenario
1. operational_total = total_carbon_kg × f_op
2. embodied_total = total_carbon_kg × (1 - f_op)
3. embodied_carbon_kg = embodied_total / num_servers
4. carbon_intensity_g_kwh = operational_total × 1000 / energy_kwh
```

Parameters are derived **once** from the reference scenario, then applied consistently to all scenarios.

---

## Resource Modeling

### Resource Scaling (Purpose-Built Servers)

When oversubscription packs more vCPUs than HW threads, resources like memory and SSD must scale with vCPU count. This models purpose-built non-SMT servers provisioned for the target vCPU density.

```json
{
  "resource_scaling": {
    "scale_with_vcpus": ["memory", "ssd"],
    "scale_power": true
  }
}
```

**Math:**
```
vcpus_per_server = max(hw_threads, available_pcpus × oversub_ratio)
scale_factor = vcpus_per_server / hw_threads
```

- Named components move from `per_thread` to `per_vcpu` (multiplied by `vcpus_per_server` instead of `hw_threads`)
- When `scale_power` is true, matching power_breakdown components have idle_w/max_w multiplied by `scale_factor`
- Composite power curve is rebuilt with scaled component values

**Example (Genoa nosmt, 80 cores, 1 tpc, thread_overhead=8):**
- `hw_threads = 80`, `available_pcpus = 72`
- At R=2.0: `vcpus_per_server = max(80, 72 × 2.0) = 144`, `scale_factor = 1.8`
- CAPEX: memory `per_thread=4.43` → `per_vcpu: 4.43 × 144 = 637.9 kg/server` (was `4.43 × 80 = 354.4`)
- OPEX: memory power `idle=20W, max=66W` scaled by 1.8 → `idle=36W, max=119W`

### Resource Constraints (Fixed-Capacity Servers)

Servers have fixed resource capacities that independently limit how many vCPUs can be packed. This models running non-SMT workloads on existing SMT hardware where memory/SSD capacity is already determined.

```json
{
  "resource_constraints": {
    "memory_gb": {"capacity_per_thread": 4.8, "demand_per_vcpu": 4.0},
    "ssd_gb": {"capacity_per_server": 6000, "demand_per_vcpu": 50}
  }
}
```

**Math:**
```
1. core_limit = available_pcpus × oversub_ratio
2. For each resource: max_vcpus = capacity / demand_per_vcpu
3. effective_vcpus = min(core_limit, all resource limits)
4. effective_R = effective_vcpus / available_pcpus
5. bottleneck = resource with lowest max_vcpus
6. stranded_pct = (1 - effective_vcpus / max_vcpus) × 100  (per resource)
```

The effective R replaces the requested R for the rest of the calculation pipeline. The model reports the bottleneck resource and stranded capacity for each resource.

### Resource Scaling vs Resource Constraints

These are mutually exclusive and model opposite deployment strategies:

| Aspect | Resource Scaling | Resource Constraints |
|--------|-----------------|---------------------|
| Server type | Purpose-built for target density | Existing/fixed hardware |
| Memory/SSD | Scales with vCPU count | Fixed capacity limits packing |
| Effective R | Always equals requested R | May be lower than requested R |
| Cost impact | Higher per-server embodied carbon | Fixed per-server, but more servers |
| Use case | "Build new non-SMT servers with right DIMM count" | "Run non-SMT on existing SMT hardware" |

---

## Assumptions

### Server Assumptions

1. **Homogeneous fleet** (within a scenario): All servers have identical specifications
2. **Full deployment**: All servers run for the entire lifetime (no scaling up/down)
3. **No redundancy**: Model calculates minimum servers needed (no N+1, etc.)
4. **Linear pCPU equivalence**: 1 vCPU = 1 pCPU worth of work at 100% utilization

### Power Assumptions

1. **Steady-state**: Power calculated at average utilization, not time-varying
2. **No power-off**: Servers remain on regardless of load
3. **Uniform utilization**: Work distributed evenly across servers
4. **Component additivity**: Per-component power sums to total server power

### SMT Assumptions

1. **2:1 threading**: SMT provides exactly 2 threads per core (configurable via `threads_per_core`)
2. **Full pCPU exposure**: All logical CPUs visible to hypervisor minus `thread_overhead`
3. **Overhead is additive**: SMT contention modeled as flat utilization overhead

### Cost Assumptions

1. **Linear embodied carbon**: Carbon/cost scales linearly with server count
2. **Fixed carbon intensity**: Grid carbon intensity constant over lifetime
3. **No cooling overhead**: PUE not explicitly modeled (can be folded into power)
4. **No maintenance costs**: Only capital and energy costs included

---

## Analysis Types

### find_breakeven

Binary search to find a single parameter value where target matches reference.

**Inputs:** baseline scenario, reference scenario, target scenario, `vary_parameter`, `match_metric`, `search_bounds`

**Output:** The breakeven value, search convergence history, scenario results at breakeven.

**Use case:** "What oversubscription ratio does non-SMT need to match SMT on carbon?"

### compare

Evaluate and compare multiple scenarios side-by-side. No search — just evaluate each scenario and show differences.

**Output:** Per-scenario results with absolute and % differences vs baseline.

### sweep

Run `find_breakeven` repeatedly across different values of a second parameter.

**Inputs:** All `find_breakeven` fields plus `sweep_parameter` and `sweep_values`.

**Output:** Table of (sweep_value → breakeven_value) pairs.

**Use case:** "How does the breakeven oversubscription ratio change as carbon intensity varies from 100-800 g/kWh?"

### compare_sweep

Sweep a parameter showing % change vs baseline at each value. Supports single or multi-scenario comparison.

**Inputs:** baseline scenario, sweep scenario(s), `sweep_parameter`, `sweep_values`

**Output:** Table and plot of % change in carbon/TCO at each sweep value, with breakeven markers.

**Options:**
- `show_breakeven_marker`: Mark where savings cross 0% (default: true)
- `separate_metric_plots`: Separate plots for carbon and TCO (default: false)
- `show_ideal_scaling_line`: Show theoretical 1/R scaling reference (default: false)

**Use case:** "How do no-SMT savings change as vCPU demand discount varies from 0.5 to 1.0?"

### breakeven_curve

Aggregate breakeven values from multiple sub-config files into a single curve plot.

**Inputs:** List of sub-config files (each a `find_breakeven` analysis), shared parameters.

**Output:** Multi-line plot of breakeven values (Y) vs a shared parameter (X, e.g., utilization).

**Use case:** "Plot breakeven vCPU discount vs utilization, with and without resource scaling."

### savings_curve

Aggregate `compare_sweep` results from multiple sub-configs into a multi-line savings plot.

**Inputs:** List of sub-config files (each a `compare_sweep`), shared parameters, optional marker positions.

**Output:** Multi-line savings curves across a shared X-axis (e.g., utilization).

**Use case:** "Show no-SMT savings across utilization 10-50% for three different vCPU discount assumptions."

### per_server_comparison

Compare per-server metrics across configurations and utilization levels.

**Output:** Per-server capacity (vCPUs, memory, SSD), embodied carbon/cost breakdown by component.

**Use case:** "What's the per-server embodied carbon breakdown (CPU vs memory vs SSD) for SMT vs non-SMT at different utilization levels?"

### resource_packing

Visualize resource utilization, bottlenecks, and stranded capacity under resource constraints.

**Output:** Per-resource max vCPU capacity, bottleneck identification, stranded capacity percentages.

**Use case:** "At R=2.0, which resource limits packing first — CPU cores, memory, or SSD?"

### fleet_comparison

Aggregate fleet-level totals from multiple scenario sets across a shared axis.

**Output:** Total fleet carbon and TCO across configurations with optional marker annotations.

**Use case:** "Compare total fleet carbon for scaled vs constrained deployments across utilization 10-30%."

---

## Composite Scenarios

Composite scenarios model heterogeneous fleets where different workload segments are routed to different server pools.

### Explicit Allocation

Directly specify what fraction of vCPU demand goes to each pool:

```json
{
  "mixed_fleet": {
    "composite": {
      "smt_pool": {"vcpu_fraction": 0.4},
      "nosmt_pool": {"vcpu_fraction": 0.6}
    }
  }
}
```

Fractions must sum to 1.0. Each pool is evaluated independently with its fraction of total_vcpus, then results are aggregated.

### Trait-Based Allocation

Workloads have a trait distribution (e.g., vCPU performance discount), and a split point determines which pool gets which workloads:

```json
{
  "workload": {
    "traits": {
      "vcpu_discount": {
        "type": "discrete",
        "bins": [
          {"value": 0.5, "vcpu_fraction": 0.10},
          {"value": 0.7, "vcpu_fraction": 0.25},
          {"value": 0.9, "vcpu_fraction": 0.30},
          {"value": 1.0, "vcpu_fraction": 0.35}
        ]
      }
    }
  }
}
```

- `below_split` pool gets all bins with `value < split_point`
- `above_split` pool gets all bins with `value >= split_point`
- `parameter_effects: {"vcpu_demand_multiplier": "weighted_average"}` sets the parameter to the weighted average of trait values in that partition

### Auto-Breakeven Split Point

Instead of a fixed split point, compute it automatically via breakeven search:

```json
{
  "split_point": {
    "auto_breakeven": {
      "baseline_scenario": "smt_pool",
      "target_scenario": "nosmt_pool",
      "target_parameter": "vcpu_demand_multiplier",
      "match_metric": "carbon",
      "search_bounds": [0.5, 1.0]
    }
  }
}
```

### Aggregation Rules

- **Servers**: Sum across pools
- **Carbon/Cost**: Sum across pools (embodied + operational)
- **Utilization/Power**: Weighted average by server count
- **Per-vCPU metrics**: Use original total_vcpus (not per-pool)

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

#### SMT Baseline (R=1.0)

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

#### SMT + Oversub (R=1.3)

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

#### Non-SMT Breakeven

To match 423,000 kg with non-SMT (48 pCPUs per server), binary search finds **R ≈ 2.58**:

```
Servers needed: ⌈10000 / (48 × 2.58)⌉ = 81

Average util: (10000 × 0.3) / (81 × 48) = 77.2%
Power: 90 + 250 × (0.772)^0.9 = 288W

Embodied carbon: 81 × 1000 = 81,000 kg
Energy: 81 × 288W × 43800h = 1,022,000 kWh
Operational carbon: 1,022,000 × 0.4 = 408,800 kg

Total carbon: 489,800 kg  (still higher due to power)
```

This shows that breakeven on oversubscription alone may not always yield a solution — the target metric depends on multiple interacting factors (server count, utilization, power).

---

### Example 2: vCPU Demand Multiplier Breakeven

**Question:** If non-SMT has a fixed oversubscription ratio (R=1.4), what reduction in vCPU demand would make it match SMT+oversub on carbon?

This flips the question: instead of "how much more oversub does non-SMT need?", we ask "how much less demand could non-SMT serve while matching carbon?"

**Setup:**
- Reference: SMT with R=1.1 and 5% util overhead
- Target: Non-SMT with R=1.4, vary `vcpu_demand_multiplier` in [0.5, 1.0]

Binary search finds **vcpu_demand_multiplier ≈ 0.73**, meaning non-SMT at R=1.4 matches SMT+oversub carbon when serving 73% of the vCPU demand.

**Interpretation:** Non-SMT needs 27% fewer vCPUs (or equivalently, could serve 27% fewer VMs) to achieve carbon parity with SMT+oversub at equivalent server count.

---

### Example 3: Resource-Constrained Packing

**Question:** On existing SMT hardware (80 cores, 160 HW threads, 12 DIMMs at 64GB, 6 SSDs at 1TB), what's the effective oversubscription ratio for non-SMT at R=2.0?

**Setup:**
- hw_threads = 80 (non-SMT), available_pcpus = 72, requested R=2.0
- Memory: 12 × 64GB = 768 GB total, demand = 4 GB/vCPU
- SSD: 6 × 1TB = 6000 GB total, demand = 50 GB/vCPU

```
core_limit = 72 × 2.0 = 144 vCPUs
memory_limit = 768 / 4 = 192 vCPUs
ssd_limit = 6000 / 50 = 120 vCPUs

effective_vcpus = min(144, 192, 120) = 120 (SSD bottleneck)
effective_R = 120 / 72 = 1.67

Stranded: cores 16.7%, memory 37.5%, SSD 0% (bottleneck)
```

The effective R is 1.67, not the requested 2.0, because SSD capacity limits packing.

---

### Example 4: Heterogeneous Fleet with Trait-Based Allocation

**Question:** Given a workload where 35% of VMs have no performance penalty (discount=1.0), 30% have a 10% penalty (0.9), and 35% have larger penalties (0.5-0.7), what's the optimal split between SMT and non-SMT pools?

**Setup:**
- Split point at 0.75: VMs with discount < 0.75 → non-SMT pool, discount >= 0.75 → SMT pool
- Non-SMT pool gets 35% of vCPUs (discount 0.5 and 0.7 bins), weighted avg discount = 0.614
- SMT pool gets 65% of vCPUs (discount 0.9 and 1.0 bins), no discount applied

Each pool is evaluated independently, then results are summed. Sweeping `split_point` from 0.5 to 1.0 reveals the optimal allocation threshold.

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
| `scale_factor` (resource scaling) | Higher per-server embodied carbon and power |

### Key Tradeoffs

**Embodied vs Operational Carbon**: In clean-grid regions (low carbon_intensity), embodied carbon dominates and reducing server count is paramount. In carbon-heavy grids, operational efficiency (power per server) matters more.

**Server Count vs Utilization**: Higher oversubscription reduces server count (lower embodied) but increases per-server utilization and power (higher operational). The optimal point depends on the embodied/operational ratio.

**Resource Scaling vs Constraints**: Purpose-built servers (resource scaling) achieve the full requested R but at higher per-server cost. Constrained servers may not reach the target R but avoid the cost of additional memory/SSD.

Use **sweep analyses** to explore how breakeven values change across parameter ranges. Use **savings_curve** analyses to visualize savings across multiple utilization levels simultaneously.

---

## Supported Parameter Paths

The `vary_parameter` and `sweep_parameter` fields support dot-notation for nested access:

- **Direct**: `oversub_ratio`, `util_overhead`, `vcpu_demand_multiplier`
- **Processor**: `processor.physical_cores`, `processor.power_curve.p_max`
- **Workload**: `workload.avg_util`, `workload.total_vcpus`
- **Cost (raw)**: `cost.embodied_carbon_kg`, `cost.carbon_intensity_g_kwh`, `cost.server_cost_usd`, `cost.electricity_cost_usd_kwh`, `cost.lifetime_years`
- **Cost (ratio)**: `cost.operational_carbon_fraction`, `cost.operational_cost_fraction`, `cost.total_carbon_kg`, `cost.total_cost_usd`
- **Composite**: `split_point`

### Match Conditions

Control how breakeven is determined:

- `"carbon"` — match carbon exactly
- `"tco"` — match TCO exactly
- `{"carbon": "match", "tco": "within_5%"}` — compound condition
- `{"carbon": "<="}` — carbon at or below reference
