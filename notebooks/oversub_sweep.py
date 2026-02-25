import sys
sys.path.insert(0, '..')

from smt_oversub_model import (
    compare_smt_vs_nosmt,
    compare_oversub_ratios,
    ScenarioBuilder,
    plot_scenarios,
    polynomial_power_curve_fn,
)
from smt_oversub_model.analysis import ProcessorDefaults, CostDefaults
import numpy as np
import matplotlib.pyplot as plt

def compute_oversub_savings_sweep(
    total_vcpus: int = 10000,
    avg_util: float = 0.3,
    oversub_min: float = 1.0,
    oversub_max: float = 2.0,
    step: float = 0.1,
    util_overhead: float = 0.0,
    smt: bool = True,
    processor_defaults: ProcessorDefaults = None,
    cost_defaults: CostDefaults = None,
    power_curve_fn=None,
    **kwargs,
):
    """
    Compute actual and theoretical savings for a sweep of oversubscription ratios.

    Returns:
        dict with 'ratios', 'carbon_savings_pct', 'tco_savings_pct', 'theoretical_savings_pct'
    """
    ratios = np.arange(oversub_min, oversub_max + step/2, step)

    # Create builder with custom defaults if provided
    builder = ScenarioBuilder(
        processor_defaults=processor_defaults,
        cost_defaults=cost_defaults,
        power_curve_fn=power_curve_fn,
    )
    workload = builder.build_workload_params(total_vcpus, avg_util)
    cost = builder.build_cost_params(**kwargs)
    
    # Get baseline (R=1.0)
    baseline_scenario = builder.build_scenario(
        name="Baseline",
        smt=smt,
        oversub_ratio=1.0,
        util_overhead=0.0,  # No overhead at baseline
    )
    from smt_oversub_model.analysis import evaluate_scenarios
    baseline_result = evaluate_scenarios([baseline_scenario], workload, cost)[0]
    
    baseline_carbon = baseline_result['total_carbon_kg']
    baseline_tco = baseline_result['total_cost_usd']
    
    # Compute savings for each ratio
    carbon_savings = []
    tco_savings = []
    theoretical_savings = []
    
    for r in ratios:
        # Theoretical: 1 - 1/R (perfect linear server reduction)
        theoretical = (1 - 1/r) * 100
        theoretical_savings.append(theoretical)
        
        # Actual
        scenario = builder.build_scenario(
            name=f"R={r:.1f}",
            smt=smt,
            oversub_ratio=r,
            util_overhead=util_overhead if r > 1.0 else 0.0,
        )
        result = evaluate_scenarios([scenario], workload, cost)[0]
        
        carbon_pct = (1 - result['total_carbon_kg'] / baseline_carbon) * 100
        tco_pct = (1 - result['total_cost_usd'] / baseline_tco) * 100
        
        carbon_savings.append(carbon_pct)
        tco_savings.append(tco_pct)
    
    return {
        'ratios': ratios,
        'carbon_savings_pct': np.array(carbon_savings),
        'tco_savings_pct': np.array(tco_savings),
        'theoretical_savings_pct': np.array(theoretical_savings),
    }

def plot_oversub_savings_sweep(sweep_result, title_suffix=""):
    """
    Plot carbon and TCO savings vs oversubscription ratio.
    
    Creates two side-by-side plots showing actual vs theoretical savings.
    """
    ratios = sweep_result['ratios']
    carbon = sweep_result['carbon_savings_pct']
    tco = sweep_result['tco_savings_pct']
    theoretical = sweep_result['theoretical_savings_pct']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Carbon savings plot
    ax1 = axes[0]
    ax1.plot(ratios, carbon, 'b-o', linewidth=2, markersize=6, label='Actual Carbon Savings')
    ax1.plot(ratios, theoretical, 'k--', linewidth=2, alpha=0.7, label='Theoretical (1 - 1/R)')
    ax1.set_xlabel('Oversubscription Ratio (R)', fontsize=12)
    ax1.set_ylabel('Carbon Savings (%)', fontsize=12)
    ax1.set_title(f'Carbon Savings vs Oversubscription{title_suffix}', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(ratios[0], ratios[-1])
    ax1.set_ylim(bottom=0)
    
    # TCO savings plot
    ax2 = axes[1]
    ax2.plot(ratios, tco, 'g-o', linewidth=2, markersize=6, label='Actual TCO Savings')
    ax2.plot(ratios, theoretical, 'k--', linewidth=2, alpha=0.7, label='Theoretical (1 - 1/R)')
    ax2.set_xlabel('Oversubscription Ratio (R)', fontsize=12)
    ax2.set_ylabel('TCO Savings (%)', fontsize=12)
    ax2.set_title(f'TCO Savings vs Oversubscription{title_suffix}', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(ratios[0], ratios[-1])
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Configure processor and cost defaults
processor_defaults = ProcessorDefaults(
    smt_physical_cores=84,
    smt_power_idle_w=300.0,
    smt_power_max_w=800.0,
    smt_thread_overhead=8,
    nosmt_physical_cores=96,
    nosmt_power_ratio=1.0,
    nosmt_idle_ratio=1.0,
    nosmt_thread_overhead=9,
)

cost_defaults = CostDefaults(
    embodied_carbon_kg=4000.0,
    server_cost_usd=10000.0,
    carbon_intensity_g_kwh=200.0,
    electricity_cost_usd_kwh=0.28,
    lifetime_years=6.0,
)

# Run the sweep with custom parameters (SMT, no utilization overhead)
# Use polynomial power curve for more realistic power modeling
sweep = compute_oversub_savings_sweep(
    total_vcpus=10000,
    avg_util=0.3,
    oversub_min=1.0,
    oversub_max=2.0,
    step=0.1,
    util_overhead=0.0,  # No additional overhead
    smt=True,
    processor_defaults=processor_defaults,
    cost_defaults=cost_defaults,
    power_curve_fn=polynomial_power_curve_fn(freq_mhz=3500),
)

# Print the data
print("Oversubscription Savings Analysis (SMT)")
print("=" * 60)
print(f"{'Ratio':<8} {'Carbon %':<12} {'TCO %':<12} {'Theoretical %':<12}")
print("-" * 60)
for i, r in enumerate(sweep['ratios']):
    print(f"{r:<8.1f} {sweep['carbon_savings_pct'][i]:<12.1f} {sweep['tco_savings_pct'][i]:<12.1f} {sweep['theoretical_savings_pct'][i]:<12.1f}")

# Plot the results
fig = plot_oversub_savings_sweep(sweep, " (SMT)")