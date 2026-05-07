# 03b: Oversubscription Ratio Breakeven

## Question

> Given the headline 10% utilization setup, is there a single breakeven ratio
> between SMT and no-SMT oversubscription rates, or does breakeven depend on the
> absolute R values? If it depends on absolute values, hold SMT at the current
> headline R and find the no-SMT R needed for breakeven.

## Prerequisites

- [02c: Scheduling Input Basis Sensitivity](02c_scheduling_input_basis_sensitivity.md)
- [03: vCPU Demand Discount](03_vcpu_demand_discount.md)
- [04: Mixed Fleet Partitioning](04_mixed_fleet_partitioning.md)

## Short Answer

There is **not** one clean breakeven ratio that works independent of the actual
R values.

Using the headline 10% **iso-physical-core** resource-scaling setup:

| Input | Value |
|---|---:|
| SMT R | `3.32` |
| no-SMT R | `5.58` |
| no-SMT / SMT R ratio | `1.681x` |
| no-SMT vCPU demand multiplier | `0.75` |
| no-SMT carbon vs SMT | `-11.8%` |
| no-SMT TCO vs SMT | `-16.8%` |

Holding SMT fixed at `R=3.32`, no-SMT reaches breakeven at:

| Metric | no-SMT breakeven R | no-SMT / SMT R ratio | SMT : no-SMT R ratio |
|---|---:|---:|---:|
| Carbon | `2.457` | `0.740x` | `1 : 0.740` |
| TCO | `2.504` | `0.754x` | `1 : 0.754` |

The ratio being less than `1.0` does **not** mean no-SMT wins without
oversubscription. It means no-SMT's breakeven R is lower than SMT's already
oversubscribed headline R. The breakeven no-SMT R itself is still above `1.0`,
so oversubscription is still active.

For comparison, if both sides run at `R=1.0` with the same `0.75` demand
multiplier, no-SMT is still worse by about `+11.0%` carbon and `+9.5%` TCO. And
if SMT keeps the headline `R=3.32` while no-SMT is set to `R=1.0`, no-SMT is much
worse: about `+50.5%` carbon and `+59.7%` TCO.

## Headline Setup Used Here

This note uses the updated [02c](02c_scheduling_input_basis_sensitivity.md)
`iso_physical_core` resource-scaling config at 10% utilization:

- Config: `configs/oversub_analysis/genoa/scheduling_input_sensitivity/iso_physical_core/resource_scaling/util_10_pct.jsonc`
- Workload: `100,000` total vCPUs, `10%` average utilization
- SMT: Genoa SMT, `R=3.32`
- no-SMT: Genoa no-SMT linear power curve, `R=5.58`
- no-SMT vCPU demand multiplier: `0.75`
- Resource scaling: memory and SSD scale with vCPUs per server
- Cost: `175 gCO2/kWh`, `$0.28/kWh`, `6` year lifetime

Note: [03](03_vcpu_demand_discount.md) calls `0.75` the "geomean" marker. The
exact all-application measured geomean multiplier is closer to `0.735`, but the
headline tables use the rounded `0.75` marker.

## Why the Ratio Alone Is Not Enough

The model does not use R only as a capacity ratio. R changes several quantities
directly:

```text
vcpu_capacity_per_server = available_pcpus * R
num_servers = ceil(effective_vcpus / vcpu_capacity_per_server)
avg_util_per_server = effective_vcpus * avg_util / (num_servers * available_pcpus)
power_per_server = power_curve(avg_util_per_server)
```

With resource scaling enabled, R also changes per-server memory and SSD
provisioning:

```text
vcpus_per_server = max(hw_threads, available_pcpus * R)
```

That means multiplying both SMT and no-SMT R by the same factor while keeping
their ratio fixed does not leave the model unchanged:

- server counts fall,
- per-server utilization rises,
- power per server changes along the power curve,
- scaled memory/SSD embodied carbon, cost, and power change per server,
- integer server counts introduce small step changes.

So the breakeven ratio is an operating-point result, not a universal conversion
factor.

## Results

### Fixed at the Current SMT R

Holding SMT fixed at the headline `R=3.32`:

| Metric | no-SMT breakeven R | no-SMT / SMT R ratio | Difference at threshold |
|---|---:|---:|---:|
| Carbon | `2.457` | `0.740x` | `-0.087%` |
| TCO | `2.504` | `0.754x` | `-0.049%` |

The no-SMT breakeven R is lower than the SMT R because this comparison is not an
oversubscription-only result: it also includes the no-SMT vCPU demand multiplier
of `0.75` and no-SMT's lower per-server carbon/cost at this R. It should not be
read as saying no-SMT beats SMT at `R=1.0`.

For intuition, if only server count mattered, a no-SMT server has half as many
available pCPUs (`72` vs `144`), and the `0.75` demand multiplier would imply
server-count parity at:

```text
no-SMT R / SMT R = 0.75 * 144 / 72 = 1.50x
```

But carbon/TCO parity happens earlier than server-count parity because the
purpose-built no-SMT server is cheaper and lower-carbon per server even after
resource scaling.

### Same-R Sanity Check

If both SMT and no-SMT use `R=3.32`, no-SMT is better **only when the headline
vCPU demand multiplier is included**:

| no-SMT vCPU multiplier | SMT servers | no-SMT servers | Carbon change | TCO change |
|---:|---:|---:|---:|---:|
| `1.00` | `210` | `419` | `+24.8%` | `+22.8%` |
| `0.75` | `210` | `314` | `-6.5%` | `-8.0%` |

This is the key distinction:

- Same `R` with equal vCPU demand (`m=1.00`) is still bad for no-SMT because
  no-SMT has half the available pCPUs per server (`72` vs `144`) and therefore
  needs about `2x` as many servers.
- Same `R` with the headline demand multiplier (`m=0.75`) reduces that server
  penalty to about `1.5x`.
- At `R=3.32`, no-SMT's per-server embodied carbon is much lower (`2,438 kg`
  vs `4,420 kg`) and its per-server power is lower (`566 W` vs `848 W`). Those
  advantages overcome the remaining `1.5x` server-count penalty on total
  carbon/TCO.

So this does not contradict the no-oversub sanity check. It says the combination
of same oversubscription **plus** demand compression **plus** lower no-SMT
per-server resource footprint can be favorable.

### Why the Same Relative Ratio Flips Between R=1 and R=3.32

Using the same R on both sides means the relative R ratio is `1.0` in both rows.
The model still changes because increasing R changes the absolute operating
point:

| Same R on both sides | Server ratio (no-SMT / SMT) | no-SMT / SMT per-server embodied carbon | no-SMT / SMT fleet embodied carbon | no-SMT / SMT per-server power | no-SMT / SMT fleet power | Carbon change | TCO change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.00` | `1.499x` | `0.628x` | `0.942x` | `0.802x` | `1.203x` | `+11.0%` | `+9.5%` |
| `3.32` | `1.495x` | `0.552x` | `0.825x` | `0.667x` | `0.998x` | `-6.5%` | `-8.0%` |

The server-count penalty is essentially the same in both rows: no-SMT needs about
`1.5x` as many servers because it has half the available pCPUs per server but
only `0.75x` as much vCPU demand.

What changes as R rises:

1. **The extra no-SMT idle-power penalty shrinks.** At `R=1.0`, both fleets are
   near `10%` utilization, so the extra no-SMT servers carry a large idle-power
   burden. At `R=3.32`, both fleets are near `33%` utilization, so power is
   better amortized across fewer, busier servers.
2. **no-SMT's per-server power ratio improves.** In this config, no-SMT has a
   more linear/lower power curve and fewer memory/SSD devices per server. Its
   per-server power is `80%` of SMT at `R=1.0`, but only `67%` of SMT at
   `R=3.32`.
3. **Resource scaling increases SMT's per-server resource footprint faster in
   absolute terms.** At the same R, SMT packs twice as many vCPUs per server
   because it has twice as many available pCPUs (`144` vs `72`). With memory and
   SSD scaled per packed vCPU, SMT's per-server embodied carbon grows more in
   absolute terms. The no-SMT/SMT per-server embodied-carbon ratio falls from
   `0.628x` at `R=1.0` to `0.552x` at `R=3.32`.

The flip is therefore not caused by the relative R ratio. It is caused by the
absolute R value moving the fleet from an idle-heavy regime into a busier regime,
while resource scaling and no-SMT's lower per-server footprint improve no-SMT's
relative economics.

This is not a coding bug, but it is an important modeling sensitivity. The result
depends on assuming that a `0.75` vCPU demand multiplier also reduces the
resource footprint of the no-SMT fleet through fewer VM vCPUs. If real workloads
need the same memory or SSD capacity even after using fewer vCPUs, then this
purpose-built resource-scaling model is optimistic for no-SMT. The
same-hardware/resource-constrained analyses in [03a](03a_constrained_savings.md)
are the relevant counterpoint for that assumption.

## How to Reproduce

Run:

```bash
python tools/compute_oversub_ratio_breakeven.py
```

The script reuses
`configs/oversub_analysis/genoa/scheduling_input_sensitivity/iso_physical_core/resource_scaling/util_10_pct.jsonc`
and rebuilds the declarative config at each candidate no-SMT R, so resource
scaling is re-resolved at the tested R.

Because server counts are integer-valued, the reported breakeven is a practical
threshold: the first no-SMT R where the target metric is no worse than SMT,
rounded to three decimals.

## Interpretation

For the headline 10% iso-physical-core case, the current oversubscription-rate
ratio is much more favorable to no-SMT than breakeven requires:

```text
current no-SMT / SMT R ratio = 5.58 / 3.32 = 1.681x
carbon breakeven ratio       = 2.457 / 3.32 = 0.740x
TCO breakeven ratio          = 2.504 / 3.32 = 0.754x
```

The right way to phrase the conclusion is:

> In the headline 10% iso-physical-core resource-scaling setup, there is no
> universal SMT:no-SMT oversubscription-rate breakeven ratio. At the current SMT
> R of `3.32`, no-SMT needs about `0.74x` SMT's R for carbon breakeven and
> `0.75x` for TCO breakeven, while the headline uses `1.68x`.

