"""
Output management for declarative analysis results.

Provides structured directory output with:
- results.json: Full analysis results
- config.json: Echoed input config
- summary.md: Human-readable summary
- scenarios/: Per-scenario JSON files
- plots/: Generated visualizations (if matplotlib available)
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
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

        # Build scenario list for plotting
        scenarios = []
        for name, data in result.scenario_results.items():
            scenarios.append({
                'name': name,
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
