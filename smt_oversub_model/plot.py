"""
Plotting utilities for visualizing SMT oversubscription model results.

Requires the 'plot' optional dependency: pip install -e ".[plot]"

Usage:
    from smt_oversub_model import Runner, load_config
    from smt_oversub_model.plot import plot_scenario_comparison

    config = load_config("configs/my_experiment.json")
    runner = Runner(config)
    result = runner.run()

    # Plot and show
    plot_scenario_comparison(result)

    # Or save to file
    plot_scenario_comparison(result, save_path="my_plot.png")
"""

from typing import Optional, Union, Dict, Any, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Raise helpful error if matplotlib is not installed."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install -e '.[plot]'"
        )


def _extract_scenario_data(result) -> Tuple[dict, dict, Optional[dict], dict]:
    """
    Extract scenario data from RunResult or dict.

    Returns:
        (baseline, smt_oversub, nosmt_breakeven, savings) dicts
    """
    # Handle both RunResult object and dict
    if hasattr(result, 'results') and result.results is not None:
        scenarios = result.results.scenarios
        savings = result.results.savings
    elif isinstance(result, dict):
        if 'results' in result:
            scenarios = result['results']['scenarios']
            savings = result['results']['savings']
        else:
            raise ValueError("Dict must contain 'results' key for single run data")
    else:
        raise ValueError("Expected RunResult or dict with single run results")

    baseline = scenarios.get('baseline', {})
    smt_oversub = scenarios.get('smt_oversub', {})
    nosmt_breakeven = scenarios.get('nosmt_breakeven')

    return baseline, smt_oversub, nosmt_breakeven, savings


def _format_value(value: float, is_carbon: bool = False) -> str:
    """Format a value for display."""
    if is_carbon:
        if value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}k"
        else:
            return f"{value:.0f}"
    else:
        if value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:.0f}k"
        else:
            return f"${value:.0f}"


def plot_scenario_comparison(
    result,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (14, 6),
    show: bool = True,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """
    Create stacked bar charts comparing scenarios for TCO and Carbon.

    Shows:
    - Baseline (SMT, no oversub)
    - SMT + Oversub
    - Non-SMT at breakeven (if achievable)

    Each bar is stacked showing embodied (capex) vs operational (opex) portions,
    with percentage labels and relative savings annotations.

    Args:
        result: RunResult object or dict from a single run
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        show: Whether to display the plot (default True)
        title: Optional custom title
        colors: Optional dict with 'embodied' and 'operational' color overrides

    Returns:
        matplotlib Figure object if matplotlib is available
    """
    _check_matplotlib()

    baseline, smt_oversub, nosmt_breakeven, savings = _extract_scenario_data(result)

    # Default colors
    if colors is None:
        colors = {
            'embodied': '#2ecc71',      # Green for embodied/capex
            'operational': '#3498db',    # Blue for operational/opex
        }

    # Prepare data for both metrics
    metrics = {
        'tco': {
            'label': 'Total Cost of Ownership (USD)',
            'embodied_key': 'embodied_cost_usd',
            'operational_key': 'operational_cost_usd',
            'total_key': 'total_cost_usd',
            'is_carbon': False,
            'savings_key': 'smt_tco_savings_vs_baseline_pct',
        },
        'carbon': {
            'label': 'Total Carbon (kg CO2e)',
            'embodied_key': 'embodied_carbon_kg',
            'operational_key': 'operational_carbon_kg',
            'total_key': 'total_carbon_kg',
            'is_carbon': True,
            'savings_key': 'smt_carbon_savings_vs_baseline_pct',
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax_idx, (metric_name, metric) in enumerate(metrics.items()):
        ax = axes[ax_idx]

        # Build scenario data
        scenarios_to_plot = []
        labels = []

        # Baseline
        if baseline:
            scenarios_to_plot.append({
                'name': 'Baseline\n(SMT, R=1.0)',
                'embodied': baseline.get(metric['embodied_key'], 0),
                'operational': baseline.get(metric['operational_key'], 0),
                'total': baseline.get(metric['total_key'], 0),
                'servers': baseline.get('num_servers', 0),
            })
            labels.append('Baseline\n(SMT, R=1.0)')

        # SMT + Oversub
        if smt_oversub:
            scenarios_to_plot.append({
                'name': 'SMT+Oversub',
                'embodied': smt_oversub.get(metric['embodied_key'], 0),
                'operational': smt_oversub.get(metric['operational_key'], 0),
                'total': smt_oversub.get(metric['total_key'], 0),
                'servers': smt_oversub.get('num_servers', 0),
            })
            labels.append('SMT+Oversub')

        # Non-SMT at breakeven
        if nosmt_breakeven:
            scenarios_to_plot.append({
                'name': 'Non-SMT\n@ Breakeven',
                'embodied': nosmt_breakeven.get(metric['embodied_key'], 0),
                'operational': nosmt_breakeven.get(metric['operational_key'], 0),
                'total': nosmt_breakeven.get(metric['total_key'], 0),
                'servers': nosmt_breakeven.get('num_servers', 0),
            })
            labels.append('Non-SMT\n@ Breakeven')

        if not scenarios_to_plot:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        # Create bar positions
        x = np.arange(len(scenarios_to_plot))
        bar_width = 0.6

        # Extract values
        embodied_vals = [s['embodied'] for s in scenarios_to_plot]
        operational_vals = [s['operational'] for s in scenarios_to_plot]
        total_vals = [s['total'] for s in scenarios_to_plot]

        # Plot stacked bars
        bars_embodied = ax.bar(x, embodied_vals, bar_width,
                               label='Embodied (CapEx)', color=colors['embodied'],
                               edgecolor='white', linewidth=1)
        bars_operational = ax.bar(x, operational_vals, bar_width,
                                  bottom=embodied_vals,
                                  label='Operational (OpEx)', color=colors['operational'],
                                  edgecolor='white', linewidth=1)

        # Add percentage labels inside bars
        for i, (emb, ops, total) in enumerate(zip(embodied_vals, operational_vals, total_vals)):
            if total > 0:
                emb_pct = (emb / total) * 100
                ops_pct = (ops / total) * 100

                # Embodied percentage (bottom part)
                if emb_pct > 5:  # Only show if visible
                    ax.text(i, emb / 2, f'{emb_pct:.0f}%',
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='white')

                # Operational percentage (top part)
                if ops_pct > 5:  # Only show if visible
                    ax.text(i, emb + ops / 2, f'{ops_pct:.0f}%',
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='white')

        # Add total value on top of bars
        max_total = max(total_vals) if total_vals else 1
        for i, (total, scenario) in enumerate(zip(total_vals, scenarios_to_plot)):
            formatted = _format_value(total, metric['is_carbon'])
            ax.text(i, total * 1.02, formatted,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Add server count below x-axis labels
            servers = scenario['servers']
            ax.annotate(f"({servers} servers)", xy=(i, 0), xytext=(0, -35),
                        textcoords='offset points', ha='center', va='top',
                        fontsize=8, color='gray')

        # Add savings annotations
        if len(scenarios_to_plot) >= 2:
            baseline_total = total_vals[0]

            for i in range(1, len(scenarios_to_plot)):
                scenario_total = total_vals[i]
                if baseline_total > 0:
                    savings_pct = (1 - scenario_total / baseline_total) * 100
                    color = 'green' if savings_pct > 0 else 'red'
                    sign = '+' if savings_pct < 0 else '-'
                    if savings_pct != 0:
                        # Draw arrow and text
                        ax.annotate(
                            f'{sign}{abs(savings_pct):.1f}%',
                            xy=(i, scenario_total),
                            xytext=(i - 0.4, baseline_total * 0.85),
                            fontsize=10, fontweight='bold', color=color,
                            ha='center', va='center',
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        )

        # Customize axes
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(metric['label'], fontsize=11)
        ax.set_ylim(0, max(total_vals) * 1.25)  # Leave room for annotations
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add legend (shared between subplots)
    handles = [
        mpatches.Patch(color=colors['embodied'], label='Embodied (CapEx)'),
        mpatches.Patch(color=colors['operational'], label='Operational (OpEx)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=10)

    # Title
    if title is None:
        # Try to get experiment name from result
        if hasattr(result, 'meta'):
            title = f"Scenario Comparison: {result.meta.get('experiment_name', 'Unnamed')}"
        elif isinstance(result, dict) and 'meta' in result:
            title = f"Scenario Comparison: {result['meta'].get('experiment_name', 'Unnamed')}"
        else:
            title = "Scenario Comparison"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_sweep_breakeven(
    result,
    metric: str = 'carbon',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    title: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot breakeven oversubscription ratio across a parameter sweep.

    Args:
        result: RunResult object or dict from a sweep run
        metric: 'carbon' or 'tco'
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        show: Whether to display the plot (default True)
        title: Optional custom title

    Returns:
        matplotlib Figure object if matplotlib is available
    """
    _check_matplotlib()

    # Extract sweep results
    if hasattr(result, 'sweep_results') and result.sweep_results:
        sweep_results = result.sweep_results
        config = result.config if hasattr(result, 'config') else {}
        meta = result.meta if hasattr(result, 'meta') else {}
    elif isinstance(result, dict):
        if 'sweep_results' not in result:
            raise ValueError("Dict must contain 'sweep_results' key for sweep data")
        sweep_results = result['sweep_results']
        config = result.get('config', {})
        meta = result.get('meta', {})
    else:
        raise ValueError("Expected RunResult or dict with sweep results")

    # Extract parameter name
    sweep_config = config.get('sweep', {})
    param_name = sweep_config.get('parameter', 'Parameter')

    # Extract data
    param_values = []
    breakeven_values = []

    for point in sweep_results:
        if hasattr(point, 'parameter_value'):
            param_values.append(point.parameter_value)
            be = point.breakeven_carbon if metric == 'carbon' else point.breakeven_tco
        else:
            param_values.append(point['parameter_value'])
            be = point.get('breakeven_carbon') if metric == 'carbon' else point.get('breakeven_tco')
        breakeven_values.append(be)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line with markers, handling None values
    valid_x = []
    valid_y = []
    none_x = []

    for x, y in zip(param_values, breakeven_values):
        if y is not None:
            valid_x.append(x)
            valid_y.append(y)
        else:
            none_x.append(x)

    ax.plot(valid_x, valid_y, 'o-', linewidth=2, markersize=8,
            color='#3498db', label=f'Breakeven ({metric.upper()})')

    # Mark where breakeven is not achievable
    if none_x:
        for x in none_x:
            ax.axvline(x, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.plot([], [], 'r--', label='Not achievable')

    # Reference line at R=1.0 (no oversub)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label='No oversubscription (R=1.0)')

    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(f'Breakeven Oversubscription Ratio ({metric.upper()})', fontsize=11)

    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title is None:
        exp_name = meta.get('experiment_name', 'Sweep')
        title = f"Breakeven Analysis: {exp_name}"
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_result(
    result,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> Optional[Any]:
    """
    Automatically plot the appropriate visualization based on result type.

    For single runs: plots scenario comparison stacked bar chart.
    For sweeps: plots breakeven curve.

    Args:
        result: RunResult object or dict
        save_path: Optional path to save the figure
        show: Whether to display the plot
        **kwargs: Additional arguments passed to the specific plot function

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Determine result type
    is_sweep = False
    if hasattr(result, 'sweep_results') and result.sweep_results:
        is_sweep = True
    elif isinstance(result, dict) and 'sweep_results' in result and result['sweep_results']:
        is_sweep = True

    if is_sweep:
        return plot_sweep_breakeven(result, save_path=save_path, show=show, **kwargs)
    else:
        return plot_scenario_comparison(result, save_path=save_path, show=show, **kwargs)
