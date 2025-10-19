# Claude Skills Integration Guide

## Overview

This project now includes **Claude Skills** integration, allowing natural language interaction with the investment report generation system directly through Claude Code.

## What Are Claude Skills?

Claude Skills are modular capabilities that extend Claude's functionality. They enable:
- **Automatic Discovery**: Claude automatically finds and uses skills based on your requests
- **Natural Language Interface**: No coding required - just ask in plain English
- **Knowledge Packaging**: Expert workflows bundled into reusable modules
- **Team Collaboration**: Skills can be shared via Git

## Project Architecture

This project supports **dual modes** of operation:

### 1. Claude Skills (New - Conversational)
```
User: "Generate an investment report for SPY"
  ↓
Claude Code (auto-discovers trading-analysis skill)
  ↓
Skill executes → Reports generated
```

**Best for:**
- Ad-hoc analysis requests
- Non-programmers
- Interactive exploration
- Quick one-off reports

### 2. Python API (Existing - Programmatic)
```python
from src.reporting import InvestmentReportGenerator
report = generator.generate_market_analysis_report(...)
```

**Best for:**
- Automated pipelines
- Scheduled batch processing
- Custom integrations
- Production systems

## Skill Location

```
financial-rl-claude-skills/
└── .claude/
    └── skills/
        └── trading-analysis/
            ├── SKILL.md              # Skill definition (auto-discovered by Claude)
            ├── reference.md          # Technical documentation
            ├── examples.md           # Usage examples
            ├── scripts/
            │   └── generate_report.py  # Execution wrapper
            └── templates/            # Future: report templates
```

## Using the Trading Analysis Skill

### Basic Usage

Simply ask Claude in natural language:

```
"Generate an investment report for SPY"
```

```
"Analyze AAPL and create a report"
```

```
"I need a market analysis for TSLA"
```

### Advanced Usage

**Custom client name:**
```
"Create a Tesla report for Goldman Sachs"
```

**Custom title:**
```
"Generate NVDA report titled 'Q4 2025 Growth Analysis'"
```

**Multiple symbols:**
```
"Generate reports for SPY, QQQ, and DIA"
```

**Comparison:**
```
"Compare AAPL and MSFT - which should I buy?"
```

### What Gets Generated

For each request, the skill generates:

1. **Markdown Report** (`reports/{SYMBOL}_analysis_report_{timestamp}.md`)
   - Executive summary
   - Market overview
   - Technical analysis
   - AI insights
   - Investment recommendations
   - Risk assessment

2. **JSON Data** (`reports/{SYMBOL}_analysis_report_{timestamp}_data.json`)
   - Structured data export
   - All metrics and indicators

3. **Four Professional Charts** (300 DPI PNG)
   - Price chart with moving averages
   - Technical indicators (RSI & MACD)
   - Historical volatility
   - Performance summary dashboard

## How It Works

### Automatic Discovery

Claude Code automatically discovers the `trading-analysis` skill by reading `.claude/skills/trading-analysis/SKILL.md`:

```yaml
---
name: trading-analysis
description: Generate professional investment reports for stocks and ETFs...
---
```

The `description` field tells Claude:
- **What** the skill does (investment reports)
- **When** to use it (market analysis, stock analysis)
- **How** to invoke it (stock ticker symbols)

### Execution Flow

```
1. User makes natural language request
   ↓
2. Claude analyzes request intent
   ↓
3. Claude matches "trading-analysis" skill
   ↓
4. Claude executes: python3 scripts/generate_report.py {SYMBOL} --client "{CLIENT}"
   ↓
5. Script loads existing Python modules
   ↓
6. Report generation pipeline executes:
   - Download market data (Yahoo Finance)
   - Calculate technical indicators
   - Claude AI analysis
   - Generate charts
   - Compile report
   ↓
7. Results returned to Claude
   ↓
8. Claude presents results to user
```

### Integration with Existing Code

The skill **wraps** existing Python modules - it doesn't replace them:

```python
# .claude/skills/trading-analysis/scripts/generate_report.py
from generate_investment_report import generate_complete_report

# Calls the existing function
success = generate_complete_report(
    symbol=args.symbol,
    client_name=args.client,
    report_title=args.title
)
```

**Benefits:**
- ✅ Preserves existing functionality
- ✅ Maintains code quality
- ✅ Enables both programmatic AND conversational use
- ✅ No code duplication

## Examples

### Example 1: Portfolio Review

**User:**
```
Review my portfolio: 40% SPY, 30% QQQ, 20% GLD, 10% TLT
```

**Claude:**
```
I'll generate investment reports for all four holdings...

[Generates 4 reports]

Portfolio Analysis Complete:
- SPY (40%): NEUTRAL → HOLD
- QQQ (30%): BULLISH → BUY
- GLD (20%): NEUTRAL → HOLD
- TLT (10%): BEARISH → SELL

Recommendation: Increase QQQ allocation, reduce TLT
```

### Example 2: Market Update

**User:**
```
Daily market update for major indices
```

**Claude:**
```
[Generates SPY, QQQ, DIA, IWM reports]

Market Summary - October 19, 2025:
- Large Cap (SPY): +0.57% | HOLD
- Tech (QQQ): +1.23% | BUY
- Dow (DIA): +0.34% | HOLD
- Small Cap (IWM): +0.89% | BUY

Tech and small caps leading today.
```

### Example 3: Stock Comparison

**User:**
```
Compare AAPL and MSFT for tech allocation
```

**Claude:**
```
[Generates 2 reports]

Comparison:
- AAPL: BULLISH (BUY) - Stronger momentum, higher volatility
- MSFT: NEUTRAL (HOLD) - More stable, weaker trend

Recommendation: AAPL for short-term, MSFT for stability
```

## Configuration

### Environment Setup

The skill uses the same `.env` configuration as the Python API:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-xxx...

# Optional (defaults provided)
CLAUDE_MODEL=claude-3-7-sonnet-20250219
CLAUDE_MAX_TOKENS=2048
CLAUDE_TEMPERATURE=0.7
```

### Customization

To customize the skill:

1. **Modify description** (`.claude/skills/trading-analysis/SKILL.md`)
   - Change when Claude invokes the skill
   - Add/remove keywords

2. **Update script** (`.claude/skills/trading-analysis/scripts/generate_report.py`)
   - Add new command-line options
   - Modify default parameters

3. **Extend Python modules** (`src/reporting/`)
   - Add new chart types
   - Enhance analysis logic
   - Changes automatically available to skill

## Sharing Skills

### Team Collaboration

Since skills are in the `.claude/` directory:

```bash
# Commit skills to Git
git add .claude/skills/trading-analysis/
git commit -m "Add trading-analysis skill"
git push

# Team members get skill automatically
git pull
# Skill now available in their Claude Code!
```

### Personal vs Project Skills

- **Project Skills**: `.claude/skills/` (shared via Git)
- **Personal Skills**: `~/.claude/skills/` (your machine only)
- **Priority**: Project > Personal > Plugins

## Comparison: Skills vs Python API

| Feature | Claude Skills | Python API |
|---------|--------------|------------|
| **Interface** | Natural language | Code |
| **Users** | Everyone | Programmers |
| **Invocation** | "Analyze SPY" | `generate_report(...)` |
| **Discovery** | Automatic | Manual import |
| **Sharing** | Git (effortless) | Documentation |
| **Customization** | Limited | Full control |
| **Automation** | Interactive | Scriptable |
| **Best For** | Ad-hoc requests | Production pipelines |

## Advantages of Skills Integration

### 1. Accessibility
- Financial analysts (non-coders) can generate reports
- No Python knowledge required
- Natural conversation interface

### 2. Consistency
- Everyone uses same methodology
- Best practices built-in
- Uniform output quality

### 3. Efficiency
- No repetitive prompting
- Fast turnaround
- Parallel processing for multiple symbols

### 4. Knowledge Transfer
- New team members onboard instantly
- Expertise encoded in skill
- Self-documenting workflow

### 5. Flexibility
- Works alongside Python API
- Choose mode based on use case
- No vendor lock-in

## Troubleshooting

### Skill Not Found

If Claude doesn't recognize the skill:

1. **Check file location:**
   ```bash
   ls .claude/skills/trading-analysis/SKILL.md
   ```

2. **Verify SKILL.md format:**
   - YAML frontmatter with `name` and `description`
   - Clear description of what/when

3. **Restart Claude Code** (if running)

### Execution Errors

If skill execution fails:

1. **Check Python environment:**
   ```bash
   python3 --version  # Should be 3.8+
   ```

2. **Verify dependencies:**
   ```bash
   pip list | grep -E "(yfinance|anthropic|matplotlib)"
   ```

3. **Check API key:**
   ```bash
   cat .env | grep ANTHROPIC_API_KEY
   ```

4. **Test script directly:**
   ```bash
   python3 .claude/skills/trading-analysis/scripts/generate_report.py SPY
   ```

### Permission Issues

```bash
chmod +x .claude/skills/trading-analysis/scripts/generate_report.py
```

## Migration from API-Only

If you were using the Python API exclusively:

**Before (Python API):**
```python
from src.reporting import InvestmentReportGenerator
report = generator.generate_market_analysis_report(
    symbol='SPY',
    market_data=data,
    # ... 10+ parameters
)
```

**After (Claude Skills):**
```
"Generate SPY investment report"
```

**Both still work!** Choose based on context:
- Interactive work → Use Skills
- Automation → Use Python API

## Future Enhancements

Planned skill improvements:

1. **Portfolio Analysis Skill**
   - Multi-asset optimization
   - Correlation analysis
   - Rebalancing recommendations

2. **Backtesting Skill**
   - Strategy testing
   - Performance metrics
   - Risk analysis

3. **Alert Skill**
   - Price monitoring
   - Technical signal alerts
   - Email notifications

4. **Report Customization**
   - Template selection
   - Branding options
   - Format preferences (PDF, HTML)

## Technical Details

### Skill Definition (SKILL.md)

The skill is defined in YAML frontmatter:

```yaml
---
name: trading-analysis
description: Generate professional investment reports for stocks and ETFs. Use when user requests market analysis, investment reports, stock analysis, or trading recommendations for financial instruments (SPY, AAPL, TSLA, etc.). Creates institutional-grade reports with technical indicators, AI-powered insights, charts, and investment recommendations.
---
```

**Key elements:**
- `name`: Unique identifier
- `description`: When/how to use (critical for auto-discovery)

### Wrapper Script

Located at `.claude/skills/trading-analysis/scripts/generate_report.py`:

- Handles command-line arguments
- Loads existing Python modules
- Calls `generate_complete_report()` function
- Returns success/failure status

### Dependencies

Same as Python API:
- yfinance (market data)
- pandas (data processing)
- numpy (calculations)
- anthropic (Claude AI)
- matplotlib/seaborn (charts)

## Documentation Files

- **SKILL.md**: Skill definition (auto-discovered)
- **reference.md**: Technical architecture details
- **examples.md**: Comprehensive usage examples
- **This file**: Integration overview

## Support

For issues or questions:

1. Check `examples.md` for usage patterns
2. Review `reference.md` for technical details
3. Test with Python API directly to isolate issues
4. Verify environment configuration (`.env`)

## Summary

Claude Skills integration provides:

✅ **Natural language interface** for investment reports
✅ **Automatic discovery** by Claude Code
✅ **No coding required** for end users
✅ **Preserves existing API** for automation
✅ **Team collaboration** via Git
✅ **Institutional-quality output**

Best of both worlds: conversational ease + programmatic power!

---

**Version**: 1.0.0
**Created**: October 19, 2025
**Updated**: October 19, 2025
