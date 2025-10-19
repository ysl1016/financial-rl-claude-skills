#!/usr/bin/env python3
"""
Trading Analysis Skill - Report Generator Script

This script is invoked by Claude Code when the trading-analysis skill is used.
It wraps the existing Python modules to maintain compatibility.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
# Path: .claude/skills/trading-analysis/scripts/generate_report.py
# We need to go up 5 levels: scripts -> trading-analysis -> skills -> .claude -> project_root
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import existing modules
import importlib.util
report_script_path = project_root / "scripts" / "reports" / "generate_investment_report.py"
spec = importlib.util.spec_from_file_location(
    "generate_investment_report",
    str(report_script_path)
)
generate_module = importlib.util.module_from_spec(spec)
sys.modules['generate_investment_report'] = generate_module
spec.loader.exec_module(generate_module)
generate_complete_report = generate_module.generate_complete_report

# Change to project root directory after imports
os.chdir(str(project_root))


def main():
    """Main entry point for the skill"""
    parser = argparse.ArgumentParser(
        description='Generate professional investment reports'
    )
    parser.add_argument(
        'symbol',
        type=str,
        help='Stock ticker symbol (e.g., SPY, AAPL, TSLA)'
    )
    parser.add_argument(
        '--client',
        type=str,
        default='Institutional Investors',
        help='Client name for the report'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom report title'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='6mo',
        help='Historical data period (e.g., 1mo, 3mo, 6mo, 1y)'
    )

    args = parser.parse_args()

    # Generate report using existing infrastructure
    try:
        print(f"\n{'='*60}")
        print(f"  TRADING ANALYSIS SKILL - REPORT GENERATION")
        print(f"{'='*60}\n")
        print(f"Symbol:        {args.symbol}")
        print(f"Client:        {args.client}")
        print(f"Period:        {args.period}")
        if args.title:
            print(f"Title:         {args.title}")
        print(f"\n{'='*60}\n")

        # Call the existing report generation function
        success = generate_complete_report(
            symbol=args.symbol,
            client_name=args.client,
            report_title=args.title
        )

        if success:
            print(f"\n{'='*60}")
            print(f"  ✓ SKILL EXECUTION SUCCESSFUL")
            print(f"{'='*60}\n")
            return 0
        else:
            print(f"\n{'='*60}")
            print(f"  ✗ SKILL EXECUTION FAILED")
            print(f"{'='*60}\n")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
