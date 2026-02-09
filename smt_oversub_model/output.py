"""
Output management for declarative analysis results.

Provides structured directory output with:
- results.json: Full analysis results
- config.json: Echoed input config
- summary.md: Human-readable summary
- comparison.txt: Detailed comparison table (for compare analysis)
- comparison.csv: CSV version of comparison data
- scenarios/: Per-scenario JSON files
- plots/: Generated visualizations (if matplotlib available)
"""

import csv
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .declarative import AnalysisResult


class OutputWriter:
    """
    Write analysis results to structured directory.

    Output structure:
        output_dir/
            results.json        # Full analysis results
            config.json         # Echoed input config
            summary.md          # Human-readable summary
            scenarios/          # Per-scenario details
                baseline.json
                smt_oversub.json
                nosmt_oversub.json
            plots/              # Generated visualizations
                comparison.png
                breakeven_search.png (if find_breakeven)
    """

    def __init__(self, output_dir: str):
        """
        Initialize writer.

        Args:
            output_dir: Path to output directory (will be created if needed)
        """
        self.output_dir = Path(output_dir)

    def write(
        self,
        result: 'AnalysisResult',
        generate_plots: bool = True,
    ) -> None:
        """
        Write all output files.

        Args:
            result: AnalysisResult to write
            generate_plots: Whether to generate plots (requires matplotlib)
        """
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        scenarios_dir = self.output_dir / 'scenarios'
        scenarios_dir.mkdir(exist_ok=True)

        # Write main results
        self._write_results(result)
        self._write_config(result)
        self._write_summary(result)
        self._write_scenarios(result, scenarios_dir)

        # Write detailed comparison files for compare analysis
        if result.analysis_type == 'compare':
            self._write_comparison_table(result)

        if generate_plots:
            self._write_plots(result)

    def _write_results(self, result: 'AnalysisResult') -> None:
        """Write full results.json."""
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _write_config(self, result: 'AnalysisResult') -> None:
        """Write echoed config.json."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(result.config.to_dict(), f, indent=2)

    def _write_summary(self, result: 'AnalysisResult') -> None:
        """Write human-readable summary.md."""
        summary_path = self.output_dir / 'summary.md'
        with open(summary_path, 'w') as f:
            f.write(result.summary)

    def _write_scenarios(
        self,
        result: 'AnalysisResult',
        scenarios_dir: Path,
    ) -> None:
        """Write per-scenario JSON files."""
        for name, scenario_data in result.scenario_results.items():
            # Sanitize filename
            safe_name = name.replace(' ', '_').replace('/', '_')
            scenario_path = scenarios_dir / f'{safe_name}.json'
            with open(scenario_path, 'w') as f:
                json.dump(scenario_data, f, indent=2, default=str)

    def _write_comparison_table(self, result: 'AnalysisResult') -> None:
        """Write detailed comparison table as text and CSV files."""
        baseline_name = result.config.analysis.baseline
        scenarios = result.scenario_results

        if baseline_name not in scenarios:
            return

        baseline = scenarios[baseline_name]

        # Build comparison data
        comparison_data = self._build_comparison_data(scenarios, baseline_name, baseline)

        # Write text file
        txt_path = self.output_dir / 'comparison.txt'
        with open(txt_path, 'w') as f:
            f.write(self._format_comparison_text(result.config.name, comparison_data, baseline_name))

        # Write CSV file
        csv_path = self.output_dir / 'comparison.csv'
        with open(csv_path, 'w', newline='') as f:
            self._write_comparison_csv(f, comparison_data)

    def _build_comparison_data(
        self,
        scenarios: Dict[str, Dict[str, Any]],
        baseline_name: str,
        baseline: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build structured comparison data for all scenarios."""
        data = []

        for name, scenario in scenarios.items():
            row = {
                'scenario': name,
                'is_baseline': name == baseline_name,
                # Raw values
                'servers': scenario.get('num_servers', 0),
                'embodied_carbon_kg': scenario.get('embodied_carbon_kg', 0),
                'operational_carbon_kg': scenario.get('operational_carbon_kg', 0),
                'total_carbon_kg': scenario.get('total_carbon_kg', 0),
                'embodied_cost_usd': scenario.get('embodied_cost_usd', 0),
                'operational_cost_usd': scenario.get('operational_cost_usd', 0),
                'total_cost_usd': scenario.get('total_cost_usd', 0),
            }

            # Extract breakdown data if present
            breakdown = scenario.get('embodied_breakdown')
            if breakdown:
                row['has_breakdown'] = True
                carbon_bd = breakdown.get('carbon')
                cost_bd = breakdown.get('cost')
                if carbon_bd:
                    row['carbon_breakdown'] = carbon_bd
                if cost_bd:
                    row['cost_breakdown'] = cost_bd
            else:
                row['has_breakdown'] = False

            # Calculate savings vs baseline (negative = reduction/savings)
            if name != baseline_name:
                base_total_carbon = baseline.get('total_carbon_kg', 1)
                base_total_cost = baseline.get('total_cost_usd', 1)

                # Component savings (vs same component in baseline)
                row['servers_diff'] = scenario.get('num_servers', 0) - baseline.get('num_servers', 0)
                row['embodied_carbon_diff'] = scenario.get('embodied_carbon_kg', 0) - baseline.get('embodied_carbon_kg', 0)
                row['operational_carbon_diff'] = scenario.get('operational_carbon_kg', 0) - baseline.get('operational_carbon_kg', 0)
                row['total_carbon_diff'] = scenario.get('total_carbon_kg', 0) - baseline.get('total_carbon_kg', 0)
                row['embodied_cost_diff'] = scenario.get('embodied_cost_usd', 0) - baseline.get('embodied_cost_usd', 0)
                row['operational_cost_diff'] = scenario.get('operational_cost_usd', 0) - baseline.get('operational_cost_usd', 0)
                row['total_cost_diff'] = scenario.get('total_cost_usd', 0) - baseline.get('total_cost_usd', 0)

                # Percentage savings vs same component in baseline
                row['servers_pct'] = self._pct_change(scenario.get('num_servers', 0), baseline.get('num_servers', 1))
                row['embodied_carbon_pct'] = self._pct_change(scenario.get('embodied_carbon_kg', 0), baseline.get('embodied_carbon_kg', 1))
                row['operational_carbon_pct'] = self._pct_change(scenario.get('operational_carbon_kg', 0), baseline.get('operational_carbon_kg', 1))
                row['total_carbon_pct'] = self._pct_change(scenario.get('total_carbon_kg', 0), baseline.get('total_carbon_kg', 1))
                row['embodied_cost_pct'] = self._pct_change(scenario.get('embodied_cost_usd', 0), baseline.get('embodied_cost_usd', 1))
                row['operational_cost_pct'] = self._pct_change(scenario.get('operational_cost_usd', 0), baseline.get('operational_cost_usd', 1))
                row['total_cost_pct'] = self._pct_change(scenario.get('total_cost_usd', 0), baseline.get('total_cost_usd', 1))

                # Component savings as percentage of TOTAL baseline
                row['embodied_carbon_of_total_pct'] = (row['embodied_carbon_diff'] / base_total_carbon) * 100 if base_total_carbon else 0
                row['operational_carbon_of_total_pct'] = (row['operational_carbon_diff'] / base_total_carbon) * 100 if base_total_carbon else 0
                row['embodied_cost_of_total_pct'] = (row['embodied_cost_diff'] / base_total_cost) * 100 if base_total_cost else 0
                row['operational_cost_of_total_pct'] = (row['operational_cost_diff'] / base_total_cost) * 100 if base_total_cost else 0
            else:
                # Baseline has no diff
                for key in ['servers_diff', 'embodied_carbon_diff', 'operational_carbon_diff', 'total_carbon_diff',
                           'embodied_cost_diff', 'operational_cost_diff', 'total_cost_diff',
                           'servers_pct', 'embodied_carbon_pct', 'operational_carbon_pct', 'total_carbon_pct',
                           'embodied_cost_pct', 'operational_cost_pct', 'total_cost_pct',
                           'embodied_carbon_of_total_pct', 'operational_carbon_of_total_pct',
                           'embodied_cost_of_total_pct', 'operational_cost_of_total_pct']:
                    row[key] = 0.0

            data.append(row)

        return data

    def _pct_change(self, new: float, old: float) -> float:
        """Calculate percentage change from old to new."""
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100

    def _format_comparison_text(
        self,
        analysis_name: str,
        data: List[Dict[str, Any]],
        baseline_name: str,
    ) -> str:
        """Format comparison data as a readable text file."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"SCENARIO COMPARISON: {analysis_name}")
        lines.append("=" * 80)
        lines.append("")

        # Find baseline for reference
        baseline_data = next((d for d in data if d['is_baseline']), data[0])

        # Section 1: Raw Values Table
        lines.append("-" * 80)
        lines.append("RAW VALUES")
        lines.append("-" * 80)
        lines.append("")

        # Header
        scenario_names = [d['scenario'] for d in data]
        col_width = max(25, max(len(n) for n in scenario_names) + 2)

        lines.append(f"{'Metric':<30}" + "".join(f"{name:>{col_width}}" for name in scenario_names))
        lines.append("-" * (30 + col_width * len(scenario_names)))

        # Rows
        metrics = [
            ('Servers', 'servers', '{:,.0f}'),
            ('', '', ''),
            ('Embodied Carbon (kg)', 'embodied_carbon_kg', '{:,.0f}'),
            ('Operational Carbon (kg)', 'operational_carbon_kg', '{:,.0f}'),
            ('Total Carbon (kg)', 'total_carbon_kg', '{:,.0f}'),
            ('', '', ''),
            ('Embodied Cost ($)', 'embodied_cost_usd', '{:,.0f}'),
            ('Operational Cost ($)', 'operational_cost_usd', '{:,.0f}'),
            ('Total Cost ($)', 'total_cost_usd', '{:,.0f}'),
        ]

        for label, key, fmt in metrics:
            if not key:
                lines.append("")
                continue
            row = f"{label:<30}"
            for d in data:
                val = d.get(key, 0)
                row += f"{fmt.format(val):>{col_width}}"
            lines.append(row)

        lines.append("")
        lines.append("")

        # Section 2: Savings vs Baseline (by component)
        lines.append("-" * 80)
        lines.append(f"CHANGE VS BASELINE ({baseline_name})")
        lines.append("-" * 80)
        lines.append("")

        lines.append(f"{'Metric':<30}" + "".join(f"{name:>{col_width}}" for name in scenario_names))
        lines.append("-" * (30 + col_width * len(scenario_names)))

        change_metrics = [
            ('Servers', 'servers_diff', 'servers_pct'),
            ('', '', ''),
            ('Embodied Carbon', 'embodied_carbon_diff', 'embodied_carbon_pct'),
            ('Operational Carbon', 'operational_carbon_diff', 'operational_carbon_pct'),
            ('Total Carbon', 'total_carbon_diff', 'total_carbon_pct'),
            ('', '', ''),
            ('Embodied Cost', 'embodied_cost_diff', 'embodied_cost_pct'),
            ('Operational Cost', 'operational_cost_diff', 'operational_cost_pct'),
            ('Total Cost', 'total_cost_diff', 'total_cost_pct'),
        ]

        for label, diff_key, pct_key in change_metrics:
            if not diff_key:
                lines.append("")
                continue
            row = f"{label:<30}"
            for d in data:
                if d['is_baseline']:
                    row += f"{'--':>{col_width}}"
                else:
                    pct = d.get(pct_key, 0)
                    row += f"{pct:>+{col_width-1}.1f}%"
            lines.append(row)

        lines.append("")
        lines.append("")

        # Section 3: Component savings as % of total baseline
        lines.append("-" * 80)
        lines.append(f"COMPONENT SAVINGS AS % OF TOTAL BASELINE")
        lines.append("-" * 80)
        lines.append("(How much each component contributes to overall savings)")
        lines.append("")

        lines.append(f"{'Metric':<30}" + "".join(f"{name:>{col_width}}" for name in scenario_names))
        lines.append("-" * (30 + col_width * len(scenario_names)))

        contribution_metrics = [
            ('Embodied Carbon', 'embodied_carbon_of_total_pct'),
            ('Operational Carbon', 'operational_carbon_of_total_pct'),
            ('', ''),
            ('Embodied Cost', 'embodied_cost_of_total_pct'),
            ('Operational Cost', 'operational_cost_of_total_pct'),
        ]

        for label, key in contribution_metrics:
            if not key:
                lines.append("")
                continue
            row = f"{label:<30}"
            for d in data:
                if d['is_baseline']:
                    row += f"{'--':>{col_width}}"
                else:
                    val = d.get(key, 0)
                    row += f"{val:>+{col_width-1}.1f}%"
            lines.append(row)

        # Section 4: Embodied Breakdown (if any scenario has breakdown data)
        has_breakdown = any(d.get('has_breakdown', False) for d in data)
        if has_breakdown:
            lines.append("")
            lines.append("-" * 80)
            lines.append("EMBODIED BREAKDOWN (per-core vs per-server)")
            lines.append("-" * 80)
            lines.append("")

            for d in data:
                name = d['scenario']
                lines.append(f"  {name}:")

                carbon_bd = d.get('carbon_breakdown')
                if carbon_bd:
                    per_core = carbon_bd.get('per_core', {})
                    per_server = carbon_bd.get('per_server', {})
                    phys = carbon_bd.get('physical_cores', 0)
                    tpc = carbon_bd.get('threads_per_core', 1)
                    hw_threads = phys * tpc
                    servers = d.get('servers', 0)

                    lines.append(f"    Embodied Carbon:")
                    per_core_total = 0.0
                    for comp, val in per_core.items():
                        comp_fleet = val * hw_threads * servers
                        per_core_total += val * hw_threads
                        lines.append(f"      per_core.{comp}: {val:.1f}/thread x {hw_threads} threads x {servers} servers = {comp_fleet:,.0f} kg")
                    per_server_total = 0.0
                    for comp, val in per_server.items():
                        comp_fleet = val * servers
                        per_server_total += val
                        lines.append(f"      per_server.{comp}: {val:.1f}/server x {servers} servers = {comp_fleet:,.0f} kg")
                    total_per_srv = per_core_total + per_server_total
                    lines.append(f"      Total per server: {total_per_srv:,.0f} kg ({per_core_total:,.0f} per-core + {per_server_total:,.0f} per-server)")

                cost_bd = d.get('cost_breakdown')
                if cost_bd:
                    per_core = cost_bd.get('per_core', {})
                    per_server = cost_bd.get('per_server', {})
                    phys = cost_bd.get('physical_cores', 0)
                    tpc = cost_bd.get('threads_per_core', 1)
                    hw_threads = phys * tpc
                    servers = d.get('servers', 0)

                    lines.append(f"    Server Cost:")
                    per_core_total = 0.0
                    for comp, val in per_core.items():
                        comp_fleet = val * hw_threads * servers
                        per_core_total += val * hw_threads
                        lines.append(f"      per_core.{comp}: ${val:.1f}/thread x {hw_threads} threads x {servers} servers = ${comp_fleet:,.0f}")
                    per_server_total = 0.0
                    for comp, val in per_server.items():
                        comp_fleet = val * servers
                        per_server_total += val
                        lines.append(f"      per_server.{comp}: ${val:.1f}/server x {servers} servers = ${comp_fleet:,.0f}")
                    total_per_srv = per_core_total + per_server_total
                    lines.append(f"      Total per server: ${total_per_srv:,.0f} (${per_core_total:,.0f} per-core + ${per_server_total:,.0f} per-server)")

                if not carbon_bd and not cost_bd:
                    lines.append("    (no breakdown data)")

                lines.append("")

        lines.append("=" * 80)
        lines.append("")

        return "\n".join(lines)

    def _write_comparison_csv(self, f, data: List[Dict[str, Any]]) -> None:
        """Write comparison data to CSV file."""
        if not data:
            return

        # Define columns for CSV
        columns = [
            'scenario',
            'is_baseline',
            'servers',
            'embodied_carbon_kg',
            'operational_carbon_kg',
            'total_carbon_kg',
            'embodied_cost_usd',
            'operational_cost_usd',
            'total_cost_usd',
            'servers_diff',
            'embodied_carbon_diff',
            'operational_carbon_diff',
            'total_carbon_diff',
            'embodied_cost_diff',
            'operational_cost_diff',
            'total_cost_diff',
            'servers_pct',
            'embodied_carbon_pct',
            'operational_carbon_pct',
            'total_carbon_pct',
            'embodied_cost_pct',
            'operational_cost_pct',
            'total_cost_pct',
            'embodied_carbon_of_total_pct',
            'operational_carbon_of_total_pct',
            'embodied_cost_of_total_pct',
            'operational_cost_of_total_pct',
        ]

        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    def _write_plots(self, result: 'AnalysisResult') -> None:
        """Generate and save plots."""
        try:
            from .plot import HAS_MATPLOTLIB
            if not HAS_MATPLOTLIB:
                return
        except ImportError:
            return

        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        if result.analysis_type == 'find_breakeven':
            self._plot_breakeven(result, plots_dir)
            self._plot_comparison(result, plots_dir)
        elif result.analysis_type == 'compare':
            self._plot_comparison(result, plots_dir)
        elif result.analysis_type == 'sweep':
            self._plot_sweep(result, plots_dir)
        elif result.analysis_type == 'compare_sweep':
            self._plot_compare_sweep(result, plots_dir)

    def _plot_comparison(
        self,
        result: 'AnalysisResult',
        plots_dir: Path,
    ) -> None:
        """Generate comparison bar chart."""
        try:
            from .plot import plot_scenarios
        except ImportError:
            return

        # Get labels and show_plot_title from config
        labels = {}
        show_plot_title = True
        if hasattr(result.config, 'analysis'):
            labels = result.config.analysis.labels or {}
            show_plot_title = result.config.analysis.show_plot_title

        # Build scenario list for plotting with custom labels
        scenarios = []
        for name, data in result.scenario_results.items():
            display_name = labels.get(name, name)  # Use custom label or fallback to key
            scenarios.append({
                'name': display_name,
                **data,
            })

        if not scenarios:
            return

        save_path = plots_dir / 'comparison.png'
        plot_scenarios(
            scenarios,
            baseline_idx=0,
            save_path=str(save_path),
            show=False,
            title=f"Scenario Comparison: {result.config.name}",
            show_plot_title=show_plot_title,
        )

    def _plot_breakeven(
        self,
        result: 'AnalysisResult',
        plots_dir: Path,
    ) -> None:
        """Generate breakeven search convergence plot."""
        if not result.breakeven or not result.breakeven.search_history:
            return

        try:
            from .plot import plot_breakeven_search
        except ImportError:
            return

        save_path = plots_dir / 'breakeven_search.png'
        plot_breakeven_search(
            result.breakeven,
            save_path=str(save_path),
            show=False,
            title=f"Breakeven Search: {result.config.name}",
        )

    def _plot_sweep(
        self,
        result: 'AnalysisResult',
        plots_dir: Path,
    ) -> None:
        """Generate sweep results plot."""
        if not result.sweep_results:
            return

        try:
            from .plot import plot_sweep_analysis
        except ImportError:
            return

        save_path = plots_dir / 'sweep.png'
        plot_sweep_analysis(
            result,
            save_path=str(save_path),
            show=False,
        )

    def _plot_compare_sweep(
        self,
        result: 'AnalysisResult',
        plots_dir: Path,
    ) -> None:
        """Generate compare_sweep results plot(s).

        For multi-scenario sweeps, generates:
        - One combined plot with all lines
        - Individual plots for each scenario

        When separate_metric_plots is True, generates separate carbon and TCO plots.
        """
        if not result.compare_sweep_results:
            return

        try:
            from .plot import plot_compare_sweep
        except ImportError:
            return

        # Check if multi-scenario
        first_point = result.compare_sweep_results[0]
        is_multi = 'scenarios' in first_point and len(first_point.get('scenarios', {})) > 1

        # Check if separate metric plots requested
        separate_metrics = False
        if hasattr(result.config, 'analysis') and result.config.analysis.separate_metric_plots:
            separate_metrics = True

        if separate_metrics:
            # Generate separate plots for carbon and TCO
            for metric, filename_suffix in [('carbon', 'carbon'), ('tco', 'TCO')]:
                save_path = plots_dir / f'compare_sweep_{filename_suffix}.png'
                plot_compare_sweep(
                    result,
                    save_path=str(save_path),
                    show=False,
                    metric=metric,
                )
        else:
            # Generate combined plot
            save_path = plots_dir / 'compare_sweep.png'
            plot_compare_sweep(
                result,
                save_path=str(save_path),
                show=False,
            )

        # For multi-scenario, also generate individual plots
        if is_multi:
            scenario_names = list(first_point['scenarios'].keys())
            labels = {}
            if hasattr(result.config, 'analysis') and result.config.analysis.labels:
                labels = result.config.analysis.labels

            for scenario_name in scenario_names:
                # Create safe filename
                safe_name = scenario_name.replace(' ', '_').replace('/', '_').replace('=', '_')

                # Get display label for title
                display_label = labels.get(scenario_name, scenario_name)

                if separate_metrics:
                    # Generate separate plots for each metric
                    for metric, filename_suffix in [('carbon', 'carbon'), ('tco', 'TCO')]:
                        save_path = plots_dir / f'compare_sweep_{safe_name}_{filename_suffix}.png'
                        plot_compare_sweep(
                            result,
                            save_path=str(save_path),
                            show=False,
                            scenario_filter=[scenario_name],
                            title=f"Compare Sweep: {display_label}",
                            metric=metric,
                        )
                else:
                    save_path = plots_dir / f'compare_sweep_{safe_name}.png'
                    plot_compare_sweep(
                        result,
                        save_path=str(save_path),
                        show=False,
                        scenario_filter=[scenario_name],
                        title=f"Compare Sweep: {display_label}",
                    )


def save_result(result: 'AnalysisResult', output_dir: str) -> None:
    """
    Convenience function to save analysis result.

    Args:
        result: AnalysisResult to save
        output_dir: Output directory path
    """
    writer = OutputWriter(output_dir)
    writer.write(result)


def load_result(output_dir: str) -> Dict[str, Any]:
    """
    Load results from output directory.

    Args:
        output_dir: Output directory path

    Returns:
        Dict containing the results
    """
    results_path = Path(output_dir) / 'results.json'
    with open(results_path, 'r') as f:
        return json.load(f)
