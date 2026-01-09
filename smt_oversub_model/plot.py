"""
Plotting utilities for visualizing SMT oversubscription model results.

Requires the 'plot' optional dependency: pip install -e ".[plot]"

Usage:
    from smt_oversub_model import Runner, load_config
    from smt_oversub_model.plot import plot_scenario_comparison, plot_scenarios

    config = load_config("configs/my_experiment.json")
    runner = Runner(config)
    result = runner.run()

    # Plot and show
    plot_scenario_comparison(result)

    # Or use the flexible API for custom comparisons
    scenarios = [
        {"name": "SMT", "tco": {...}, "carbon": {...}},
        {"name": "Non-SMT", "tco": {...}, "carbon": {...}},
    ]
    plot_scenarios(scenarios, baseline_idx=0)
"""

from typing import Optional, Union, Dict, Any, Tuple, List
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Professional color palette
COLORS = {
    'embodied': '#1a5276',      # Dark blue for embodied/capex
    'operational': '#5dade2',    # Light blue for operational/opex
    'positive': '#27ae60',       # Green for positive savings
    'negative': '#e74c3c',       # Red for negative savings
    'neutral': '#7f8c8d',        # Gray for neutral
}


def _check_matplotlib():
    """Raise helpful error if matplotlib is not installed."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install -e '.[plot]'"
        )


def _format_value(value: float, is_carbon: bool = False) -> str:
    """Format a value for display."""
    if is_carbon:
        if abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}k"
        else:
            return f"{value:.0f}"
    else:
        if abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.0f}k"
        else:
            return f"${value:.0f}"


def _format_diff(value: float, baseline: float, is_carbon: bool = False) -> Tuple[str, str]:
    """
    Format the difference from baseline.

    Returns:
        (formatted_string, color) tuple
    """
    if baseline == 0:
        return ("", COLORS['neutral'])

    diff = value - baseline
    pct = (diff / baseline) * 100

    if abs(pct) < 0.1:
        return ("baseline", COLORS['neutral'])

    sign = "+" if diff > 0 else ""
    color = COLORS['negative'] if diff > 0 else COLORS['positive']

    return (f"{sign}{pct:.1f}%", color)


def plot_scenarios(
    scenarios: List[Dict[str, Any]],
    baseline_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 5),
    show: bool = True,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    show_server_count: bool = True,
) -> Optional[Any]:
    """
    Create professional stacked bar charts comparing arbitrary scenarios.

    This is the flexible API for custom comparisons. Each scenario should have:
    - name: Display name for x-axis
    - For TCO: embodied_cost_usd, operational_cost_usd, total_cost_usd
    - For Carbon: embodied_carbon_kg, operational_carbon_kg, total_carbon_kg
    - Optionally: num_servers

    Bars show embodied vs operational breakdown. Annotations show the percentage
    difference from the baseline scenario (not raw totals).

    Args:
        scenarios: List of scenario dicts with cost/carbon data
        baseline_idx: Index of baseline scenario for diff calculations (default 0)
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        show: Whether to display the plot (default True)
        title: Optional custom title
        colors: Optional dict with 'embodied' and 'operational' color overrides
        show_server_count: Whether to show server count under labels

    Returns:
        matplotlib Figure object if matplotlib is available
    """
    _check_matplotlib()

    if not scenarios:
        raise ValueError("No scenarios to plot")

    # Use provided colors or defaults
    if colors is None:
        colors = COLORS.copy()

    # Metrics to plot
    metrics = {
        'tco': {
            'label': 'Total Cost of Ownership',
            'embodied_key': 'embodied_cost_usd',
            'operational_key': 'operational_cost_usd',
            'total_key': 'total_cost_usd',
            'is_carbon': False,
            'unit': 'USD',
        },
        'carbon': {
            'label': 'Total Carbon',
            'embodied_key': 'embodied_carbon_kg',
            'operational_key': 'operational_carbon_kg',
            'total_key': 'total_carbon_kg',
            'is_carbon': True,
            'unit': 'kg CO₂e',
        },
    }

    # Set up professional style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    labels = [s.get('name', f'Scenario {i}') for i, s in enumerate(scenarios)]
    x = np.arange(len(scenarios))
    bar_width = 0.5

    for ax_idx, (metric_name, metric) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        ax.set_facecolor('white')

        # Extract values
        embodied_vals = [s.get(metric['embodied_key'], 0) for s in scenarios]
        operational_vals = [s.get(metric['operational_key'], 0) for s in scenarios]
        total_vals = [s.get(metric['total_key'], 0) for s in scenarios]

        if not any(total_vals):
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color=COLORS['neutral'])
            continue

        baseline_total = total_vals[baseline_idx] if baseline_idx < len(total_vals) else total_vals[0]

        # Plot stacked bars
        bars_embodied = ax.bar(
            x, embodied_vals, bar_width,
            label='Embodied (CapEx)',
            color=colors.get('embodied', COLORS['embodied']),
            edgecolor='white',
            linewidth=0.5,
        )
        bars_operational = ax.bar(
            x, operational_vals, bar_width,
            bottom=embodied_vals,
            label='Operational (OpEx)',
            color=colors.get('operational', COLORS['operational']),
            edgecolor='white',
            linewidth=0.5,
        )

        # Add percentage labels inside bars (only if segment is large enough)
        for i, (emb, ops, total) in enumerate(zip(embodied_vals, operational_vals, total_vals)):
            if total > 0:
                emb_pct = (emb / total) * 100
                ops_pct = (ops / total) * 100

                # Embodied percentage (bottom part)
                if emb_pct > 10:
                    ax.text(i, emb / 2, f'{emb_pct:.0f}%',
                            ha='center', va='center', fontsize=9,
                            fontweight='medium', color='white')

                # Operational percentage (top part)
                if ops_pct > 10:
                    ax.text(i, emb + ops / 2, f'{ops_pct:.0f}%',
                            ha='center', va='center', fontsize=9,
                            fontweight='medium', color='white')

        # Add diff from baseline annotation above bars
        max_total = max(total_vals) if total_vals else 1
        for i, total in enumerate(total_vals):
            diff_text, diff_color = _format_diff(total, baseline_total, metric['is_carbon'])
            ax.text(i, total + max_total * 0.03, diff_text,
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=diff_color)

        # Add server count below x-axis labels if available
        if show_server_count:
            for i, s in enumerate(scenarios):
                servers = s.get('num_servers')
                if servers is not None:
                    ax.annotate(f"({servers} servers)", xy=(i, 0), xytext=(0, -28),
                                textcoords='offset points', ha='center', va='top',
                                fontsize=8, color=COLORS['neutral'])

        # Customize axes
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(f"{metric['label']} ({metric['unit']})", fontsize=11)
        ax.set_ylim(0, max(total_vals) * 1.15)

        # Clean grid
        ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')

        # Y-axis formatting
        ax.tick_params(axis='y', colors='#666666')
        ax.tick_params(axis='x', colors='#333333')

    # Add legend (shared between subplots)
    handles = [
        mpatches.Patch(color=colors.get('embodied', COLORS['embodied']), label='Embodied (CapEx)'),
        mpatches.Patch(color=colors.get('operational', COLORS['operational']), label='Operational (OpEx)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=10)

    # Title
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98, color='#333333')

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


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


def plot_scenario_comparison(
    result,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 5),
    show: bool = True,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """
    Create stacked bar charts comparing scenarios from a RunResult.

    Shows:
    - Baseline (SMT, no oversub)
    - SMT + Oversub
    - Non-SMT at breakeven (if achievable)

    Each bar is stacked showing embodied (capex) vs operational (opex) portions.
    Annotations show the percentage difference from baseline.

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

    # Build scenarios list for the general plotter
    scenarios = []

    if baseline:
        scenarios.append({
            'name': 'Baseline\n(SMT, R=1.0)',
            **baseline,
        })

    if smt_oversub:
        scenarios.append({
            'name': 'SMT+Oversub',
            **smt_oversub,
        })

    if nosmt_breakeven:
        scenarios.append({
            'name': 'Non-SMT\n@ Breakeven',
            **nosmt_breakeven,
        })

    # Default title from result metadata
    if title is None:
        if hasattr(result, 'meta'):
            title = f"Scenario Comparison: {result.meta.get('experiment_name', 'Unnamed')}"
        elif isinstance(result, dict) and 'meta' in result:
            title = f"Scenario Comparison: {result['meta'].get('experiment_name', 'Unnamed')}"
        else:
            title = "Scenario Comparison"

    return plot_scenarios(
        scenarios,
        baseline_idx=0,
        save_path=save_path,
        figsize=figsize,
        show=show,
        title=title,
        colors=colors,
    )


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

    # Set up professional style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

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
            color=COLORS['embodied'], label=f'Breakeven ({metric.upper()})')

    # Mark where breakeven is not achievable
    if none_x:
        for x in none_x:
            ax.axvline(x, color=COLORS['negative'], linestyle='--', alpha=0.5, linewidth=1)
        ax.plot([], [], color=COLORS['negative'], linestyle='--', label='Not achievable')

    # Reference line at R=1.0 (no oversub)
    ax.axhline(1.0, color=COLORS['neutral'], linestyle=':', alpha=0.7, linewidth=1.5,
               label='No oversubscription (R=1.0)')

    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(f'Breakeven Oversubscription Ratio ({metric.upper()})', fontsize=11)

    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    if title is None:
        exp_name = meta.get('experiment_name', 'Sweep')
        title = f"Breakeven Analysis: {exp_name}"
    ax.set_title(title, fontsize=13, fontweight='bold', color='#333333')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

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
