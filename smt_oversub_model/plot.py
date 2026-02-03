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


def _format_parameter_label(param_name: str) -> str:
    """
    Format parameter name for display, handling special cases like vCPU.
    
    Args:
        param_name: Raw parameter name (e.g., 'vcpu_demand_multiplier')
    
    Returns:
        Formatted label with proper capitalization
    """
    # Replace underscores and dots with spaces
    formatted = param_name.replace('_', ' ').replace('.', ' ')
    
    # Split into words
    words = formatted.split()
    
    # Handle special cases
    result_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower == 'vcpu':
            result_words.append('vCPU')
        elif word_lower == 'vcpus':
            result_words.append('vCPUs')
        else:
            # Title case for other words
            result_words.append(word.capitalize())
    
    return ' '.join(result_words)


def plot_scenarios(
    scenarios: List[Dict[str, Any]],
    baseline_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 5),
    show: bool = True,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    show_server_count: bool = True,
    show_plot_title: bool = True,
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
        show_plot_title: Whether to show the title (default True)

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
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    # Title (only if show_plot_title is True)
    if title and show_plot_title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98, color='#333333')

    # Adjust layout based on whether title is shown
    if show_plot_title and title:
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])

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

    ax.set_xlabel(_format_parameter_label(param_name), fontsize=11)
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


def plot_breakeven_search(
    breakeven_result,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    title: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot the convergence of a breakeven search.

    Shows how the parameter value and error converged over iterations.

    Args:
        breakeven_result: BreakevenResult from GeneralizedBreakevenFinder
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        show: Whether to display the plot (default True)
        title: Optional custom title

    Returns:
        matplotlib Figure object if matplotlib is available
    """
    _check_matplotlib()

    # Extract search history
    if hasattr(breakeven_result, 'search_history'):
        history = breakeven_result.search_history
    elif isinstance(breakeven_result, dict):
        history = breakeven_result.get('search_history', [])
    else:
        raise ValueError("Expected BreakevenResult or dict with search_history")

    if not history:
        return None

    # Extract data
    iterations = list(range(len(history)))
    values = []
    errors = []

    for entry in history:
        if hasattr(entry, 'value'):
            values.append(entry.value)
            errors.append(entry.error)
        else:
            values.append(entry.get('value', 0))
            errors.append(entry.get('error', 0))

    # Set up professional style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor('white')

    # Plot parameter values
    ax1.set_facecolor('white')
    ax1.plot(iterations, values, 'o-', linewidth=2, markersize=6,
             color=COLORS['embodied'], label='Parameter Value')

    # Mark final value if breakeven achieved
    final_value = None
    if hasattr(breakeven_result, 'breakeven_value'):
        final_value = breakeven_result.breakeven_value
    elif isinstance(breakeven_result, dict):
        final_value = breakeven_result.get('breakeven_value')

    if final_value is not None:
        ax1.axhline(final_value, color=COLORS['positive'], linestyle='--',
                   alpha=0.7, linewidth=1.5, label=f'Breakeven: {final_value:.4f}')

    ax1.set_ylabel('Parameter Value', fontsize=11)
    ax1.legend(loc='best', frameon=True, fontsize=9)
    ax1.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot errors
    ax2.set_facecolor('white')
    ax2.plot(iterations, errors, 'o-', linewidth=2, markersize=6,
             color=COLORS['operational'], label='Error')
    ax2.axhline(0, color=COLORS['neutral'], linestyle='-', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Normalized Error', fontsize=11)
    ax2.legend(loc='best', frameon=True, fontsize=9)
    ax2.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    if title is None:
        title = "Breakeven Search Convergence"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98, color='#333333')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def _find_breakeven_point(x_values: List[float], y_values: List[float]) -> Optional[float]:
    """Find where y crosses 0 using linear interpolation."""
    for i in range(len(x_values) - 1):
        y1, y2 = y_values[i], y_values[i + 1]
        x1, x2 = x_values[i], x_values[i + 1]

        if (y1 <= 0 <= y2) or (y2 <= 0 <= y1):
            if y2 == y1:
                return x1
            t = -y1 / (y2 - y1)
            return x1 + t * (x2 - x1)
    return None


def _interpolate_y_at_x(x_values: List[float], y_values: List[float], target_x: float) -> Optional[float]:
    """Interpolate y-value at a given x position using linear interpolation.
    
    Args:
        x_values: List of x coordinates (must be sorted)
        y_values: List of corresponding y coordinates
        target_x: The x position to interpolate at
    
    Returns:
        Interpolated y-value, or None if target_x is outside the range
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        return None
    
    # Check if target_x is within range
    if target_x < min(x_values) or target_x > max(x_values):
        return None
    
    # Find the two points to interpolate between
    for i in range(len(x_values) - 1):
        x1, x2 = x_values[i], x_values[i + 1]
        y1, y2 = y_values[i], y_values[i + 1]
        
        if x1 <= target_x <= x2:
            # Linear interpolation
            if x2 == x1:
                return y1
            t = (target_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    
    # If exact match
    if target_x in x_values:
        idx = x_values.index(target_x)
        return y_values[idx]
    
    return None


def plot_compare_sweep(
    result,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    title: Optional[str] = None,
    show_breakeven_marker: Optional[bool] = None,
    metric: str = 'carbon',
    scenario_filter: Optional[List[str]] = None,
    show_plot_title: Optional[bool] = None,
    x_axis_markers: Optional[List[float]] = None,
    x_axis_marker_labels: Optional[List[str]] = None,
) -> Optional[Any]:
    """
    Plot compare_sweep results showing % difference vs sweep parameter.

    Creates a line plot with the sweep parameter on the x-axis and % change
    from baseline on the y-axis. Supports single and multi-scenario sweeps.

    Args:
        result: AnalysisResult with compare_sweep_results, or dict
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        show: Whether to display the plot (default True)
        title: Optional custom title
        show_breakeven_marker: Whether to show breakeven markers (default from config or True)
        metric: Which metric to plot: 'carbon', 'tco', 'servers', or 'all' (default 'carbon')
        scenario_filter: Optional list of scenario names to include (for individual plots)
        show_plot_title: Whether to show the title (default from config or True)
        x_axis_markers: List of x-values to draw vertical lines and label intersections (default from config or None)
        x_axis_marker_labels: Labels for x_axis_markers, displayed at top of vertical lines (default from config or None)

    Returns:
        matplotlib Figure object if matplotlib is available
    """
    _check_matplotlib()

    # Extract compare_sweep results
    if hasattr(result, 'compare_sweep_results'):
        compare_sweep_results = result.compare_sweep_results
    elif isinstance(result, dict):
        compare_sweep_results = result.get('compare_sweep_results', [])
    else:
        raise ValueError("Expected result with compare_sweep_results")

    if not compare_sweep_results:
        return None

    # Get config options
    if show_breakeven_marker is None:
        if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
            show_breakeven_marker = result.config.analysis.show_breakeven_marker
        elif isinstance(result, dict) and 'config' in result:
            analysis = result['config'].get('analysis', {})
            show_breakeven_marker = analysis.get('show_breakeven_marker', True)
        else:
            show_breakeven_marker = True

    if show_plot_title is None:
        if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
            show_plot_title = result.config.analysis.show_plot_title
        elif isinstance(result, dict) and 'config' in result:
            analysis = result['config'].get('analysis', {})
            show_plot_title = analysis.get('show_plot_title', True)
        else:
            show_plot_title = True

    if x_axis_markers is None:
        if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
            x_axis_markers = result.config.analysis.x_axis_markers
        elif isinstance(result, dict) and 'config' in result:
            analysis = result['config'].get('analysis', {})
            x_axis_markers = analysis.get('x_axis_markers')

    if x_axis_marker_labels is None:
        if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
            x_axis_marker_labels = result.config.analysis.x_axis_marker_labels
        elif isinstance(result, dict) and 'config' in result:
            analysis = result['config'].get('analysis', {})
            x_axis_marker_labels = analysis.get('x_axis_marker_labels')

    # Get labels from config
    labels = {}
    param_label = None
    if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
        labels = result.config.analysis.labels or {}
        param_label = result.config.analysis.sweep_parameter_label
    elif isinstance(result, dict) and 'config' in result:
        analysis = result['config'].get('analysis', {})
        labels = analysis.get('labels', {})
        param_label = analysis.get('sweep_parameter_label')

    def get_label(name: str) -> str:
        return labels.get(name, name)

    # Check if multi-scenario
    first_point = compare_sweep_results[0]
    is_multi = 'scenarios' in first_point and len(first_point.get('scenarios', {})) > 1

    # Get scenario names
    if is_multi:
        scenario_names = list(first_point['scenarios'].keys())
    else:
        scenario_names = ['default']

    # Apply filter if provided
    if scenario_filter:
        scenario_names = [s for s in scenario_names if s in scenario_filter]
        if not scenario_names:
            return None

    # Get parameter name from config
    param_name = 'Parameter'
    if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
        param_name = result.config.analysis.sweep_parameter or 'Parameter'
    elif isinstance(result, dict) and 'config' in result:
        analysis = result['config'].get('analysis', {})
        param_name = analysis.get('sweep_parameter', 'Parameter')

    # Format parameter name for display
    if param_label:
        # Use custom label if provided
        display_param_name = param_label
    else:
        # Auto-format with proper vCPU capitalization
        display_param_name = _format_parameter_label(param_name)

    # Extract data
    param_values = [p.get('parameter_value', 0) if isinstance(p, dict) else getattr(p, 'parameter_value', 0)
                    for p in compare_sweep_results]

    # Color palette for multiple scenarios
    scenario_colors = ['#1a5276', '#e74c3c', '#27ae60', '#8e44ad', '#f39c12', '#16a085']
    metric_styles = {
        'carbon': {'marker': 'o', 'label_suffix': ' Carbon'},
        'tco': {'marker': 's', 'label_suffix': ' TCO'},
        'servers': {'marker': '^', 'label_suffix': ' Servers'},
    }

    # Set up professional style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Determine which metrics to plot
    if metric == 'all':
        metrics_to_plot = ['carbon', 'tco', 'servers']
    else:
        metrics_to_plot = [metric]

    breakeven_points = []  # Collect for annotation

    for scenario_idx, scenario_name in enumerate(scenario_names):
        base_color = scenario_colors[scenario_idx % len(scenario_colors)]
        scenario_label = get_label(scenario_name)

        for metric_name in metrics_to_plot:
            metric_key = f'{metric_name}_diff_pct'
            style = metric_styles[metric_name]

            # Extract values for this scenario and metric
            if is_multi:
                y_values = [p['scenarios'].get(scenario_name, {}).get(metric_key, 0)
                           for p in compare_sweep_results]
            else:
                y_values = [p.get(metric_key, 0) for p in compare_sweep_results]

            # Build label using display labels
            # Use label_suffix (which has proper capitalization like ' TCO', ' Carbon')
            metric_display = style['label_suffix'].strip()  # e.g., 'Carbon', 'TCO'
            if is_multi and len(metrics_to_plot) > 1:
                plot_label = f"{scenario_label}{style['label_suffix']}"
            elif is_multi:
                plot_label = scenario_label
            elif len(metrics_to_plot) > 1:
                plot_label = metric_display
            else:
                plot_label = f"{metric_display} %"

            # Adjust color shade for different metrics within same scenario
            if len(metrics_to_plot) > 1:
                metric_idx = metrics_to_plot.index(metric_name)
                # Lighten color for subsequent metrics
                color = _adjust_color_brightness(base_color, 1.0 - metric_idx * 0.2)
            else:
                color = base_color

            # Plot line
            line, = ax.plot(param_values, y_values, f"{style['marker']}-",
                           linewidth=2, markersize=8, color=color, label=plot_label)

            # Find and mark breakeven point
            if show_breakeven_marker:
                breakeven_x = _find_breakeven_point(param_values, y_values)
                if breakeven_x is not None:
                    # Interpolate y value (should be ~0)
                    ax.plot(breakeven_x, 0, 'D', markersize=12, color=color,
                           markeredgecolor='white', markeredgewidth=2, zorder=15)
                    breakeven_points.append((breakeven_x, scenario_label, metric_name, color))

    # Reference line at 0%
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1.5,
               label='Baseline (0%)')

    # Shade regions
    y_min, y_max = ax.get_ylim()
    ax.axhspan(y_min, 0, alpha=0.1, color=COLORS['positive'], label='_nolegend_')
    ax.axhspan(0, y_max, alpha=0.1, color=COLORS['negative'], label='_nolegend_')

    # Add annotations for savings/cost regions
    ax.text(0.02, 0.02, 'Savings', transform=ax.transAxes, fontsize=9,
            color=COLORS['positive'], fontweight='bold', alpha=0.7)
    ax.text(0.02, 0.98, 'Increase', transform=ax.transAxes, fontsize=9,
            color=COLORS['negative'], fontweight='bold', alpha=0.7, va='top')

    # Add breakeven labels (high zorder to appear above x_axis_marker labels)
    if show_breakeven_marker and breakeven_points:
        # Offset labels to avoid overlap
        for i, (bx, slabel, mname, color) in enumerate(breakeven_points):
            y_offset = 5 + (i % 3) * 15  # Stagger labels
            if is_multi or len(scenario_names) > 1:
                label_text = f"{slabel}: {bx:.3f}"
            else:
                label_text = f"Breakeven: {bx:.3f}"
            ax.annotate(label_text, xy=(bx, 0), xytext=(0, y_offset),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=color, alpha=0.9),
                       zorder=20)

    ax.set_xlabel(display_param_name, fontsize=11)
    ax.set_ylabel('Change vs Baseline (%)', fontsize=11)
    
    # Draw x-axis markers if specified
    if x_axis_markers:
        for marker_idx, marker_x in enumerate(x_axis_markers):
            # Draw vertical line
            ax.axvline(marker_x, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)

            # Add label at top of vertical line if provided
            if x_axis_marker_labels and marker_idx < len(x_axis_marker_labels):
                marker_label = x_axis_marker_labels[marker_idx]
                # Position label at top of plot area
                y_top = ax.get_ylim()[1]
                ax.annotate(marker_label, xy=(marker_x, y_top),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=9, color='#555555', fontweight='bold',
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor='gray', alpha=0.9))

            # Find and label intersection points for each scenario
            for scenario_idx, scenario_name in enumerate(scenario_names):
                base_color = scenario_colors[scenario_idx % len(scenario_colors)]

                for metric_name in metrics_to_plot:
                    metric_key = f'{metric_name}_diff_pct'

                    # Extract values for this scenario and metric
                    if is_multi:
                        y_values = [p['scenarios'].get(scenario_name, {}).get(metric_key, 0)
                                   for p in compare_sweep_results]
                    else:
                        y_values = [p.get(metric_key, 0) for p in compare_sweep_results]

                    # Interpolate y-value at marker_x
                    y_at_marker = _interpolate_y_at_x(param_values, y_values, marker_x)
                    if y_at_marker is not None:
                        # Determine color shade
                        if len(metrics_to_plot) > 1:
                            metric_idx = metrics_to_plot.index(metric_name)
                            color = _adjust_color_brightness(base_color, 1.0 - metric_idx * 0.2)
                        else:
                            color = base_color

                        # Plot marker point
                        ax.plot(marker_x, y_at_marker, 'o', markersize=8, color=color,
                               markeredgecolor='white', markeredgewidth=1.5, zorder=10)

                        # Add label with y-value
                        label_text = f"{y_at_marker:.1f}%"
                        ax.annotate(label_text, xy=(marker_x, y_at_marker),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, color=color, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           edgecolor=color, alpha=0.8))
    
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set title only if show_plot_title is True
    if show_plot_title:
        if title is None:
            config_name = None
            if hasattr(result, 'config') and hasattr(result.config, 'name'):
                config_name = result.config.name
            elif isinstance(result, dict) and 'config' in result:
                config_name = result['config'].get('name')
            title = f"Compare Sweep: {config_name}" if config_name else "Compare Sweep Analysis"
        
        ax.set_title(title, fontsize=13, fontweight='bold', color='#333333')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def _adjust_color_brightness(hex_color: str, factor: float) -> str:
    """Adjust color brightness. Factor > 1 lightens, < 1 darkens."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)

    if factor > 1:
        # Lighten
        r = int(r + (255 - r) * (factor - 1))
        g = int(g + (255 - g) * (factor - 1))
        b = int(b + (255 - b) * (factor - 1))
    else:
        # Darken
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

    r, g, b = min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b))
    return f'#{r:02x}{g:02x}{b:02x}'


def plot_analysis_result(
    result,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> Optional[Any]:
    """
    Auto-detect analysis type and plot appropriate visualization.

    For declarative AnalysisResult:
    - find_breakeven: plots scenario comparison and optionally breakeven search
    - compare: plots scenario comparison
    - sweep: plots sweep results
    - compare_sweep: plots % difference vs sweep parameter

    Args:
        result: AnalysisResult from DeclarativeAnalysisEngine, or dict
        save_path: Optional path to save the figure
        show: Whether to display the plot
        **kwargs: Additional arguments passed to specific plot function

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    # Determine analysis type
    analysis_type = None
    if hasattr(result, 'analysis_type'):
        analysis_type = result.analysis_type
    elif isinstance(result, dict):
        analysis_type = result.get('analysis_type')

    if analysis_type == 'find_breakeven':
        # Get scenario results for comparison plot
        if hasattr(result, 'scenario_results'):
            scenario_results = result.scenario_results
        else:
            scenario_results = result.get('scenario_results', {})

        scenarios = [{'name': k, **v} for k, v in scenario_results.items()]

        if scenarios:
            config_name = None
            if hasattr(result, 'config') and hasattr(result.config, 'name'):
                config_name = result.config.name
            elif isinstance(result, dict) and 'config' in result:
                config_name = result['config'].get('name')

            return plot_scenarios(
                scenarios,
                baseline_idx=0,
                save_path=save_path,
                show=show,
                title=f"Scenario Comparison: {config_name}" if config_name else None,
                **kwargs,
            )

    elif analysis_type == 'compare':
        if hasattr(result, 'scenario_results'):
            scenario_results = result.scenario_results
        else:
            scenario_results = result.get('scenario_results', {})

        # Get labels and show_plot_title from config
        labels = {}
        show_plot_title = True
        if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
            labels = result.config.analysis.labels or {}
            show_plot_title = result.config.analysis.show_plot_title
        elif isinstance(result, dict) and 'config' in result:
            analysis = result['config'].get('analysis', {})
            labels = analysis.get('labels', {})
            show_plot_title = analysis.get('show_plot_title', True)

        # Apply custom labels if provided
        scenarios = []
        for k, v in scenario_results.items():
            name = labels.get(k, k)  # Use custom label or fallback to key
            scenarios.append({'name': name, **v})

        if scenarios:
            # Remove title from kwargs if show_plot_title is False
            plot_kwargs = kwargs.copy()
            if 'title' not in plot_kwargs and not show_plot_title:
                plot_kwargs['title'] = None
            
            return plot_scenarios(
                scenarios,
                baseline_idx=0,
                save_path=save_path,
                show=show,
                show_plot_title=show_plot_title,
                **plot_kwargs,
            )

    elif analysis_type == 'sweep':
        return plot_sweep_analysis(result, save_path=save_path, show=show, **kwargs)

    elif analysis_type == 'compare_sweep':
        return plot_compare_sweep(result, save_path=save_path, show=show, **kwargs)

    return None


def plot_sweep_analysis(
    result,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    title: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot sweep analysis results showing breakeven values across parameter sweep.

    Args:
        result: AnalysisResult with sweep_results, or dict
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        show: Whether to display the plot (default True)
        title: Optional custom title

    Returns:
        matplotlib Figure object if matplotlib is available
    """
    _check_matplotlib()

    # Extract sweep results
    if hasattr(result, 'sweep_results'):
        sweep_results = result.sweep_results
    elif isinstance(result, dict):
        sweep_results = result.get('sweep_results', [])
    else:
        raise ValueError("Expected result with sweep_results")

    if not sweep_results:
        return None

    # Extract data
    param_values = []
    breakeven_values = []
    achieved = []

    for point in sweep_results:
        if isinstance(point, dict):
            param_values.append(point.get('parameter_value', 0))
            breakeven_values.append(point.get('breakeven_value'))
            achieved.append(point.get('achieved', False))
        else:
            param_values.append(getattr(point, 'parameter_value', 0))
            breakeven_values.append(getattr(point, 'breakeven_value', None))
            achieved.append(getattr(point, 'achieved', False))

    # Get parameter name from config
    param_name = 'Parameter'
    if hasattr(result, 'config') and hasattr(result.config, 'analysis'):
        param_name = result.config.analysis.sweep_parameter or 'Parameter'
    elif isinstance(result, dict) and 'config' in result:
        analysis = result['config'].get('analysis', {})
        param_name = analysis.get('sweep_parameter', 'Parameter')
    
    # Format parameter label
    display_param_name = _format_parameter_label(param_name)

    # Set up professional style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Separate achieved and not achieved
    valid_x = []
    valid_y = []
    none_x = []

    for x, y, ach in zip(param_values, breakeven_values, achieved):
        if y is not None and ach:
            valid_x.append(x)
            valid_y.append(y)
        else:
            none_x.append(x)

    # Plot achieved points
    if valid_x:
        ax.plot(valid_x, valid_y, 'o-', linewidth=2, markersize=8,
                color=COLORS['embodied'], label='Breakeven Value')

    # Mark not achieved
    if none_x:
        for x in none_x:
            ax.axvline(x, color=COLORS['negative'], linestyle='--', alpha=0.5, linewidth=1)
        ax.plot([], [], color=COLORS['negative'], linestyle='--', label='Not achievable')

    # Reference line at R=1.0
    ax.axhline(1.0, color=COLORS['neutral'], linestyle=':', alpha=0.7, linewidth=1.5,
               label='No oversubscription (R=1.0)')

    ax.set_xlabel(display_param_name, fontsize=11)
    ax.set_ylabel('Breakeven Value', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title is None:
        config_name = None
        if hasattr(result, 'config') and hasattr(result.config, 'name'):
            config_name = result.config.name
        elif isinstance(result, dict) and 'config' in result:
            config_name = result['config'].get('name')
        title = f"Sweep Analysis: {config_name}" if config_name else "Sweep Analysis"

    ax.set_title(title, fontsize=13, fontweight='bold', color='#333333')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig
