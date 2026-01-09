"""
Command-line interface for running experiments.

Usage:
    python -m smt_oversub_model.cli configs/my_experiment.json
    python -m smt_oversub_model.cli configs/*.json --output-dir results/
    python -m smt_oversub_model.cli configs/quick.json --stdout
    python -m smt_oversub_model.cli configs/basic.json --plot
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from .config import load_config, validate_config, ExperimentConfig
from .runner import Runner, RunResult, save_result, generate_output_filename

# Optional plotting support
try:
    from .plot import plot_result
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    plot_result = None


def format_single_result_summary(result: RunResult) -> str:
    """Format a human-readable summary of single run results."""
    lines = []
    r = result.results

    lines.append(f"Experiment: {result.meta['experiment_name']}")
    lines.append("-" * 50)

    # Breakeven results
    if r.breakeven_carbon is not None:
        lines.append(f"Breakeven oversub (carbon): {r.breakeven_carbon:.3f}")
    else:
        lines.append("Breakeven oversub (carbon): Not achievable")

    if r.breakeven_tco is not None:
        lines.append(f"Breakeven oversub (TCO):    {r.breakeven_tco:.3f}")
    else:
        lines.append("Breakeven oversub (TCO):    Not achievable")

    lines.append("")

    # Savings summary
    lines.append("SMT+Oversub vs Baseline:")
    for key, val in r.savings.items():
        label = key.replace("_", " ").replace("pct", "%").title()
        lines.append(f"  {label}: {val:.2f}%")

    lines.append("")

    # Server counts
    baseline = r.scenarios.get("baseline", {})
    smt = r.scenarios.get("smt_oversub", {})
    nosmt = r.scenarios.get("nosmt_breakeven", {})

    lines.append("Server Counts:")
    lines.append(f"  Baseline (SMT, no oversub): {baseline.get('num_servers', 'N/A')}")
    lines.append(f"  SMT + Oversub:              {smt.get('num_servers', 'N/A')}")
    if nosmt:
        lines.append(f"  Non-SMT @ breakeven:        {nosmt.get('num_servers', 'N/A')}")

    return "\n".join(lines)


def format_sweep_result_summary(result: RunResult) -> str:
    """Format a human-readable summary of sweep results."""
    lines = []
    sweep = result.config.get("sweep", {})
    param = sweep.get("parameter", "unknown")

    lines.append(f"Experiment: {result.meta['experiment_name']}")
    lines.append(f"Sweep parameter: {param}")
    lines.append("-" * 60)

    # Header
    lines.append(f"{'Value':>12} | {'BE Carbon':>10} | {'BE TCO':>10} | {'Servers':>8}")
    lines.append("-" * 60)

    for point in result.sweep_results:
        be_carbon = f"{point.breakeven_carbon:.3f}" if point.breakeven_carbon else "N/A"
        be_tco = f"{point.breakeven_tco:.3f}" if point.breakeven_tco else "N/A"
        servers = point.scenarios.get("smt_oversub", {}).get("num_servers", "N/A")
        lines.append(f"{point.parameter_value:>12.4f} | {be_carbon:>10} | {be_tco:>10} | {servers:>8}")

    return "\n".join(lines)


def run_single_config(
    config_path: Path,
    output_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    stdout: bool = False,
    quiet: bool = False,
    plot: bool = False,
    plot_save_path: Optional[Path] = None,
) -> bool:
    """
    Run a single config file.

    Returns True on success, False on failure.
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}", file=sys.stderr)
        return False

    # Validate
    errors = validate_config(config)
    if errors:
        print(f"Error: Invalid config {config_path}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    # Run
    try:
        runner = Runner(config, config_path=str(config_path))
        result = runner.run()
    except Exception as e:
        print(f"Error running {config_path}: {e}", file=sys.stderr)
        return False

    # Output
    if stdout:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Determine output path
        if output_path is None:
            if output_dir is None:
                output_dir = Path("results")
            filename = generate_output_filename(config, result.meta["timestamp"])
            output_path = output_dir / filename

        save_result(result, output_path)

        if not quiet:
            print(f"Results saved to: {output_path}")
            print()
            if result.results:
                print(format_single_result_summary(result))
            elif result.sweep_results:
                print(format_sweep_result_summary(result))

    # Plot if requested
    if plot:
        if not HAS_PLOT:
            print("Warning: --plot requires matplotlib. Install with: pip install -e '.[plot]'",
                  file=sys.stderr)
        else:
            # Determine plot save path
            if plot_save_path is None and output_path is not None:
                # Default: same name as output but with .png extension
                plot_save_path = output_path.with_suffix('.png')

            plot_result(result, save_path=plot_save_path, show=(plot_save_path is None))

            if plot_save_path and not quiet:
                print(f"Plot saved to: {plot_save_path}")

    return True


def main(argv: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SMT oversubscription model experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s configs/basic.json
  %(prog)s configs/*.json --output-dir results/
  %(prog)s configs/quick.json --stdout
  %(prog)s configs/sweep.json -o custom_output.json
  %(prog)s configs/basic.json --plot
  %(prog)s configs/basic.json --plot --plot-save my_plot.png
        """,
    )

    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Config file(s) to run",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (only valid with single config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON result to stdout instead of saving",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress summary output (only save/print JSON)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plot (requires matplotlib)",
    )
    parser.add_argument(
        "--plot-save",
        type=Path,
        default=None,
        help="Save plot to file (defaults to output path with .png extension)",
    )

    args = parser.parse_args(argv)

    # Validate args
    if args.output and len(args.configs) > 1:
        parser.error("--output can only be used with a single config file")

    if args.stdout and args.output:
        parser.error("Cannot use --stdout with --output")

    # Run configs
    success_count = 0
    fail_count = 0

    for config_path in args.configs:
        success = run_single_config(
            config_path,
            output_path=args.output,
            output_dir=args.output_dir,
            stdout=args.stdout,
            quiet=args.quiet,
            plot=args.plot,
            plot_save_path=args.plot_save,
        )
        if success:
            success_count += 1
        else:
            fail_count += 1

        # Print separator between multiple configs
        if len(args.configs) > 1 and not args.stdout and not args.quiet:
            print("\n" + "=" * 60 + "\n")

    # Summary for multiple configs
    if len(args.configs) > 1 and not args.quiet:
        print(f"Completed: {success_count} succeeded, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
