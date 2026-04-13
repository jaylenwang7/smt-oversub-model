# 02a: Resource Modeling for Oversubscription

## Question

> When a server oversubscribes CPU (packing more vCPUs than physical CPUs), what
> happens to per-server memory, SSD, and power costs -- and how should the model
> account for this?

This is not a new layer of the SMT vs no-SMT analysis. It is a **modeling
clarification** that explains an important assumption choice affecting all results
where oversubscription ratio R > 1.0. Documents [02](02_scheduling_constraints_oversub.md)
and [03](03_vcpu_demand_discount.md) reference results "with" and "without"
resource scaling -- this document defines what those terms mean, why the
distinction matters, and which framing is most appropriate for the analysis.

## Prerequisites

- [01: Naive Comparison](01_naive_comparison.md) for per-server cost breakdown
  (per-thread vs per-server components)
- [02: Scheduling Constraints](02_scheduling_constraints_oversub.md) for
  oversubscription ratios

## The Problem

At R=1.0 (no oversubscription), each vCPU maps to one physical CPU. Memory, SSD,
and other resources are provisioned to match: each hardware thread gets a fixed
amount of memory and storage, and VMs are sized accordingly.

When R > 1.0, the server hosts more vCPUs than it has physical CPUs. Each of
those vCPUs still belongs to a VM that expects a certain amount of memory and
storage. The question is: **where does that additional memory and storage come
from?**

There are three coherent answers, each corresponding to a different deployment
scenario. The model supports all three.

## Three Resource Modeling Approaches

### 1. Fixed Resources (Unscaled)

**Assumption**: Per-server embodied carbon, cost, and power remain the same
regardless of R. The server is the same physical hardware whether it runs 72 or
144 vCPUs.

**When this applies**: This is appropriate if you assume memory and SSD are
already over-provisioned relative to CPU, or if VMs are so small that the
existing per-server resources are never the bottleneck. It is the simplest model
and produces the most optimistic savings projections for oversubscription.

**In the model**: This is the default behavior -- a scenario with just
`oversub_ratio > 1.0` and no `resource_scaling` or `resource_constraints`
configured.

**Limitation**: At high R values, this becomes unrealistic. If a server with 80
HW threads is running 200 vCPUs, those VMs collectively need ~2.5x the memory
they would at R=1.0. Pretending the server's resources don't change overstates
the savings.

### 2. Scaled Resources (Purpose-Built)

**Assumption**: Memory and SSD are provisioned proportionally to the number of
vCPUs the server actually hosts. A server running 2x as many vCPUs has 2x the
DIMMs and SSDs.

**When this applies**: This models a **purpose-built deployment** where servers
are configured for their target oversubscription level. If you know you will run
no-SMT at R=2.0, you order servers with enough DIMMs and SSDs for 144 vCPUs
instead of 72. The CPU, NIC, chassis, and rack are the same physical hardware;
only memory and SSD scale.

**In the model**: Configured via `resource_scaling` on the scenario:
```json
{
  "resource_scaling": {
    "scale_with_vcpus": ["memory", "ssd"]
  }
}
```

This moves the named components from per-thread scaling (based on HW threads) to
per-vCPU scaling (based on `vcpus_per_server`). Components not listed (CPU, NIC,
chassis, rack) remain per-server and do not scale.

**Effect**: At R=2.0 on a no-SMT server (72 pCPUs, 80 HW threads), the server
hosts 144 vCPUs. Memory and SSD embodied carbon and cost are multiplied by
144/80 = 1.8x compared to R=1.0. Power for memory and SSD components also scales
by 1.8x. This increases per-server cost but the server is also hosting 2x the
vCPUs, so the fleet still shrinks -- just not as much as the fixed-resource model
would suggest.

**Why this is the most appropriate framing**: In the context of this analysis, we
are asking: "what does it cost to serve 100,000 vCPUs under no-SMT with
oversubscription?" Each VM needs a certain amount of memory and SSD. We are not
assuming memory/SSD oversubscription -- only CPU oversubscription. So the per-vCPU
resource demand is fixed, and total server resources must scale with vCPU count.
This gives the most accurate picture of the actual cost of oversubscription.

(Memory and SSD *could* also be oversubscribed, and in practice sometimes are,
but that is a separate analysis with its own QoS implications. The purpose-built
framing isolates the CPU oversubscription effect.)

### 3. Resource-Constrained (Existing Hardware)

**Assumption**: The server has fixed resource capacities (e.g., 12 DIMM slots,
6 SSD bays) that cannot be expanded. Oversubscription is limited by whichever
resource runs out first.

**When this applies**: This models **deploying no-SMT on existing SMT-provisioned
hardware** -- you disable SMT on servers that were originally configured for SMT
workloads. The memory and SSD capacity is whatever the server was built with, and
that may limit how many vCPUs you can pack even if CPU oversubscription could go
higher.

**In the model**: Configured via `resource_constraints` on the scenario:
```json
{
  "resource_constraints": {
    "memory_gb": {
      "capacity_per_thread": 9.6,
      "demand_per_vcpu": 4.0
    },
    "ssd_gb": {
      "capacity_per_server": 6000,
      "demand_per_vcpu": 50
    }
  }
}
```

The model computes the maximum vCPUs allowed by each resource, takes the minimum,
and uses that as the effective oversubscription ratio. If you request R=4.0 but
memory only supports R=2.67, the effective R is 2.67.

**Effect**: Savings plateau once the bottleneck resource is exhausted. Beyond that
point, increasing the requested R has no effect on server count, carbon, or TCO.
The model reports the bottleneck resource, effective R, and stranded capacity for
each non-bottleneck resource.

## Comparison at a Glance

| Approach | Per-server cost | vCPU capacity | Best for |
|---|---|---|---|
| **Fixed** | Constant | Scales with R | Upper-bound estimates; memory-insensitive workloads |
| **Scaled** | Grows with vCPU count | Scales with R | Purpose-built deployments; accurate TCO/carbon |
| **Constrained** | Constant | Capped by resources | Existing hardware reuse; transition planning |

The key difference between fixed and scaled is in per-server embodied cost.
Both allow the same server count (they are not capped). The constrained model
caps effective R, which limits both server reduction and savings.

## How This Affects Results

The impact is most visible at high oversubscription ratios. At R=1.0 all three
approaches are identical. As R increases:

- **Fixed** resources show savings proportional to server reduction (approaches
  the theoretical 1/R ideal)
- **Scaled** resources show smaller savings because per-server cost grows (the
  savings curve flattens relative to fixed)
- **Constrained** resources show savings up to the capacity ceiling, then
  flat-line

See [02b: Oversubscription Savings Scaling](02b_oversub_savings_scaling.md) for
quantitative results showing exactly how the three approaches diverge across the
R=1.0 to R=5.0 range.

## Which Approach Is Used Where

| Document | Approach | Rationale |
|---|---|---|
| [02](02_scheduling_constraints_oversub.md) | Fixed | Isolates the scheduling constraint effect without resource cost complications |
| [03](03_vcpu_demand_discount.md) breakeven curves | Both fixed and scaled | Shows how resource scaling shifts breakeven in no-SMT's favor |
| [03](03_vcpu_demand_discount.md) savings curves | Scaled | Purpose-built framing for projected savings |
| [02b](02b_oversub_savings_scaling.md) | All three | Direct comparison of how resource modeling affects savings |

## Connection to Per-Server Cost Structure

Understanding the resource modeling approaches requires recalling the per-server
cost structure from [01](01_naive_comparison.md):

**Per-thread components** (scale with HW threads at baseline, with vCPUs under
resource scaling):
- Memory: 4.43 kg CO2 / $33 per thread
- SSD: 3.86 kg CO2 / $10.23 per thread

**Per-server components** (fixed regardless of R):
- CPU die: 34.2 kg / $1,487
- NIC: 115.0 kg / $1,022
- Chassis: 255.5 kg / $1,505
- Rack share: 51.9 kg / $510

At R=1.0 on a no-SMT server (80 HW threads), per-thread components contribute
663 kg out of 1,120 kg total (59%). At R=2.0 with scaled resources, the per-vCPU
components would contribute 1,194 kg out of 1,651 kg total (72%). The per-server
fixed costs become a smaller fraction as vCPU density increases -- this is why
scaled savings are less than fixed savings but still substantial.
