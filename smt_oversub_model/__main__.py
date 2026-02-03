"""
Main entry point for running the SMT oversubscription model.

Usage:
    python -m smt_oversub_model configs/analysis.json
    python -m smt_oversub_model configs/util_oversub_comparison/
    
For a single file, runs the declarative analysis and outputs results.
For a directory, runs all valid analysis configs found within.
"""

import sys
from pathlib import Path

from .declarative import (
    run_analysis,
    run_analysis_batch,
    is_valid_analysis_config,
)

# Optional: OutputWriter for saving results
try:
    from .output import OutputWriter
    _HAS_OUTPUT = True
except ImportError:
    _HAS_OUTPUT = False


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m smt_oversub_model <config.json | directory>")
        print()
        print("Examples:")
        print("  python -m smt_oversub_model configs/analysis.json")
        print("  python -m smt_oversub_model configs/util_oversub_comparison/")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_dir():
        # Run all configs in directory
        batch_result = run_analysis_batch(input_path)
        print(batch_result.summary)

        # Print individual summaries for successful runs
        if batch_result.results:
            print("\n" + "=" * 60 + "\n")
            for path, result in batch_result.results.items():
                print(f"## {path}\n")
                print(result.summary)
                print("\n" + "-" * 40 + "\n")

                # Save results if output_dir specified
                if result.config.output_dir and _HAS_OUTPUT:
                    writer = OutputWriter(result.config.output_dir)
                    writer.write(result)
                    print(f"Results saved to: {result.config.output_dir}\n")
    else:
        # Single file
        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        if not is_valid_analysis_config(input_path):
            print(f"Error: Not a valid analysis config: {input_path}", file=sys.stderr)
            print("A valid config must have 'name', 'scenarios', and 'analysis' keys.")
            sys.exit(1)

        result = run_analysis(input_path)

        # Print summary
        print(result.summary)

        # Save results if output_dir specified
        if result.config.output_dir and _HAS_OUTPUT:
            writer = OutputWriter(result.config.output_dir)
            writer.write(result)
            print(f"\nResults saved to: {result.config.output_dir}")


if __name__ == "__main__":
    main()
