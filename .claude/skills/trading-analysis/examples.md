# Trading Analysis Skill - Usage Examples

This document provides practical examples of how to use the trading-analysis skill in Claude Code.

## Basic Usage

### Example 1: Simple Report Generation

**User request:**
```
Generate an investment report for SPY
```

**Claude response:**
```
I'll generate a comprehensive investment report for SPY using the trading-analysis skill.

[Runs the skill]

I've generated a complete investment report for SPY. Here's what was created:

📄 Report: reports/SPY_analysis_report_20251019_011642.md
📊 Data: reports/SPY_analysis_report_20251019_011642_data.json

Charts:
- Price chart with moving averages
- Technical indicators (RSI & MACD)
- Historical volatility
- Performance summary dashboard

Key findings:
- Market Sentiment: NEUTRAL
- Recommendation: HOLD
- Current Price: $664.39 (+0.57%)
- Risk Level: MEDIUM

Would you like me to explain any specific section of the report?
```

### Example 2: Custom Client Name

**User request:**
```
Create a market analysis report for AAPL for Goldman Sachs
```

**Claude response:**
```
I'll generate an investment report for AAPL prepared for Goldman Sachs.

[Executes: python3 scripts/generate_report.py AAPL --client "Goldman Sachs"]

Report generated successfully for Goldman Sachs:
- Symbol: AAPL
- Client: Goldman Sachs
- Files created in reports/ directory
```

### Example 3: Custom Title

**User request:**
```
Generate a Tesla investment report titled "Q4 2025 TSLA Growth Strategy"
```

**Claude response:**
```
I'll create a detailed TSLA report with your custom title.

[Executes: python3 scripts/generate_report.py TSLA --title "Q4 2025 TSLA Growth Strategy"]

Created "Q4 2025 TSLA Growth Strategy" report for TSLA with:
- Technical analysis
- AI-powered insights
- Professional charts
- Investment recommendations
```

## Advanced Usage

### Example 4: Multiple Symbols

**User request:**
```
Generate investment reports for SPY, QQQ, and DIA
```

**Claude response:**
```
I'll generate three separate investment reports for these major ETFs.

[Executes skill 3 times, once for each symbol]

Generated 3 investment reports:

1. SPY (S&P 500 ETF)
   - Sentiment: NEUTRAL
   - Recommendation: HOLD

2. QQQ (Nasdaq-100 ETF)
   - Sentiment: BULLISH
   - Recommendation: BUY

3. DIA (Dow Jones ETF)
   - Sentiment: NEUTRAL
   - Recommendation: HOLD

All reports saved to reports/ directory with timestamps.
```

### Example 5: Sector Analysis

**User request:**
```
Analyze the tech sector by generating reports for AAPL, MSFT, GOOGL, and META
```

**Claude response:**
```
I'll perform a tech sector analysis by generating individual reports for each major tech stock.

[Generates 4 reports]

Tech Sector Analysis Complete:

📱 AAPL: BULLISH → BUY (Confidence: 78%)
💻 MSFT: NEUTRAL → HOLD (Confidence: 65%)
🔍 GOOGL: BEARISH → SELL (Confidence: 72%)
📘 META: BULLISH → BUY (Confidence: 81%)

Sector Overview:
- 2 Buy recommendations
- 1 Hold recommendation
- 1 Sell recommendation
- Average sector confidence: 74%

All detailed reports available in reports/ directory.
```

### Example 6: Comparative Analysis

**User request:**
```
Compare SPY and QQQ - which one should I invest in?
```

**Claude response:**
```
I'll generate detailed reports for both SPY and QQQ and provide a comparison.

[Generates 2 reports]

Comparative Analysis: SPY vs QQQ

📊 SPY (S&P 500 ETF):
- Price: $664.39 (+0.57%)
- Sentiment: NEUTRAL
- RSI: 50.63 (Neutral zone)
- Volatility: 13.07% (Low)
- Recommendation: HOLD

📊 QQQ (Nasdaq-100 ETF):
- Price: $485.21 (+1.23%)
- Sentiment: BULLISH
- RSI: 62.18 (Moderately bullish)
- Volatility: 18.45% (Medium)
- Recommendation: BUY

Recommendation:
Based on the analysis, QQQ shows stronger bullish momentum but higher volatility.
- For growth-focused investors: QQQ (higher upside potential)
- For conservative investors: SPY (lower volatility, broader diversification)

See detailed reports for comprehensive analysis.
```

## Real-World Scenarios

### Scenario 1: Portfolio Review

**User request:**
```
I need to review my portfolio holdings: 40% SPY, 30% QQQ, 20% GLD, 10% TLT.
Generate reports for all four.
```

**Claude response:**
```
I'll generate investment reports for all four holdings in your portfolio.

[Generates 4 reports]

Portfolio Holdings Analysis Complete:

🏦 SPY (40% allocation) - S&P 500 ETF
   Status: NEUTRAL → HOLD
   Current: $664.39 (+0.57%)

📱 QQQ (30% allocation) - Tech-heavy Nasdaq
   Status: BULLISH → BUY
   Current: $485.21 (+1.23%)

🥇 GLD (20% allocation) - Gold ETF
   Status: NEUTRAL → HOLD
   Current: $218.45 (-0.32%)

📊 TLT (10% allocation) - Long-term Treasury
   Status: BEARISH → SELL/REDUCE
   Current: $91.34 (-1.12%)

Portfolio Recommendation:
- Maintain SPY and GLD positions
- Consider increasing QQQ (strong momentum)
- Consider reducing TLT (bearish trend)

Detailed reports available for each holding.
```

### Scenario 2: Daily Market Update

**User request:**
```
Give me a daily market update with reports for the major indices
```

**Claude response:**
```
I'll generate today's market update for the major indices.

[Generates reports for SPY, QQQ, DIA, IWM]

📅 Daily Market Update - October 19, 2025

Large Cap (SPY): $664.39 ↑ +0.57%
   📊 NEUTRAL → HOLD
   RSI: 50.63 | Volatility: 13.07% (Low)

Tech-Heavy (QQQ): $485.21 ↑ +1.23%
   📊 BULLISH → BUY
   RSI: 62.18 | Volatility: 18.45% (Medium)

Dow Jones (DIA): $423.12 ↑ +0.34%
   📊 NEUTRAL → HOLD
   RSI: 48.92 | Volatility: 11.23% (Low)

Small Cap (IWM): $218.76 ↑ +0.89%
   📊 BULLISH → BUY
   RSI: 58.34 | Volatility: 22.11% (High)

Market Summary:
- Broad market strength with tech leading
- Small caps showing strong momentum
- Low to medium volatility environment
- 2 Buy signals, 2 Hold signals

Detailed analysis available in reports/ directory.
```

### Scenario 3: Risk Assessment Request

**User request:**
```
I'm concerned about market volatility. Analyze SPY and tell me the risk level.
```

**Claude response:**
```
I'll generate a comprehensive risk analysis for SPY.

[Generates SPY report with focus on risk metrics]

SPY Risk Assessment:

🎯 Overall Risk Level: MEDIUM

Volatility Metrics:
- 20-day volatility: 13.07% (Low)
- 50-day volatility: 10.48% (Low)
- Regime: Low volatility environment

Risk Factors Identified:
1. MACD below signal line (potential momentum deterioration)
2. Elevated volume could amplify movements
3. Price struggling near 20-day SMA resistance

Risk Mitigation Recommendations:
- Implement collar strategies for downside protection
- Consider reducing position sizes
- Set stop-loss at 50-day SMA ($653.63)
- Avoid new long positions above $666

Current Status: Low volatility with some caution signals
Action: HOLD current positions, avoid adding at current levels

Full risk analysis available in the generated report.
```

## Question-Based Examples

### Example 7: "Should I buy...?"

**User:**
```
Should I buy NVDA right now?
```

**Claude:**
```
Let me generate a detailed analysis for NVDA to help you make an informed decision.

[Generates NVDA report]

NVDA Analysis:

Current Status:
- Price: $892.45 (+2.34%)
- Sentiment: BULLISH
- RSI: 68.42 (Approaching overbought)
- Recommendation: BUY with caution

⚠️ Key Considerations:
- Strong upward momentum (BULLISH)
- RSI nearing overbought zone (70)
- High volatility (24.32%)

💡 Recommendation:
- For aggressive investors: BUY (momentum is strong)
- For conservative investors: WAIT for pullback to $860-870
- Set stop-loss at $850 (below 50-day SMA)
- Take partial profits if RSI exceeds 75

Entry Strategy:
- Option 1: Enter now with 50% position, add on pullback
- Option 2: Wait for dip to support level ($860-870)

See full report for detailed technical analysis.
```

### Example 8: "What's happening with...?"

**User:**
```
What's happening with the market today? Analyze SPY.
```

**Claude:**
```
I'll generate a current market analysis for SPY.

[Generates SPY report]

Today's Market Analysis (SPY):

📈 Price Action:
- Current: $664.39
- Change: +$3.75 (+0.57%)
- Range: $658.14 - $665.76

🔍 What's Happening:
1. Modest upward movement (+0.57%)
2. Trading below 20-day SMA ($665.06) - minor resistance
3. Volume elevated +19.88% above average
4. Neutral RSI (50.63) - balanced momentum

💭 Claude AI Insights:
"SPY is showing short-term consolidation with modest buying interest.
The elevated volume without decisive price movement suggests institutional
positioning. MACD bearish divergence warrants caution."

🎯 Current Market State:
- Trend: Sideways with bullish bias
- Momentum: Moderate
- Risk: Medium
- Action: HOLD - Mixed signals

The market is in a consolidation phase. Not a strong buy or sell signal.

Full analysis in the generated report.
```

### Example 9: "Compare these stocks..."

**User:**
```
I'm choosing between AAPL and MSFT for my tech allocation. Compare them.
```

**Claude:**
```
I'll generate detailed reports for both AAPL and MSFT to help you compare.

[Generates 2 reports]

AAPL vs MSFT Comparison:

🍎 AAPL (Apple Inc.):
- Price: $187.23 (+1.45%)
- Sentiment: BULLISH
- RSI: 64.12 (Bullish momentum)
- Volatility: 21.34% (Medium)
- Recommendation: BUY
- Confidence: 76%

💻 MSFT (Microsoft Corp.):
- Price: $412.89 (+0.87%)
- Sentiment: NEUTRAL
- RSI: 52.34 (Neutral)
- Volatility: 19.12% (Medium)
- Recommendation: HOLD
- Confidence: 68%

Head-to-Head:
┌─────────────┬────────┬────────┐
│ Metric      │  AAPL  │  MSFT  │
├─────────────┼────────┼────────┤
│ Momentum    │   ✓    │   -    │
│ Volatility  │ Higher │ Lower  │
│ Trend       │ Strong │ Weak   │
│ Recommend   │  BUY   │  HOLD  │
└─────────────┴────────┴────────┘

Recommendation:
- Short-term (1-3 months): AAPL (stronger momentum)
- Long-term/Conservative: MSFT (more stable)
- Diversification: Consider 50/50 allocation

Both are quality tech stocks with different risk/reward profiles.

See detailed reports for comprehensive analysis.
```

## Integration Examples

### Example 10: Scheduled Daily Reports

**User:**
```
Can you set up a daily report for my watchlist: SPY, AAPL, TSLA?
```

**Claude:**
```
I can generate reports for your watchlist right now. For automated daily
generation, here's a simple bash script you can schedule with cron:

[Provides automation script]

For now, let me generate today's reports for your watchlist.

[Generates 3 reports]

Watchlist Reports Generated:

1. SPY: NEUTRAL → HOLD
2. AAPL: BULLISH → BUY
3. TSLA: NEUTRAL → HOLD

Daily reports saved to reports/ directory.

To automate:
1. Save the provided script as daily_watchlist.sh
2. Run: chmod +x daily_watchlist.sh
3. Schedule with cron: 0 9 * * 1-5 (weekdays at 9 AM)
```

## Error Handling Examples

### Example 11: Invalid Symbol

**User:**
```
Generate a report for XYZ123
```

**Claude:**
```
I'll attempt to generate a report for XYZ123.

[Executes skill]

❌ Error: Unable to download data for symbol 'XYZ123'

This could mean:
1. Invalid ticker symbol
2. Symbol not available on Yahoo Finance
3. Delisted or inactive symbol

Please verify the ticker symbol. Common examples:
- Stocks: AAPL, MSFT, GOOGL, TSLA
- ETFs: SPY, QQQ, DIA, IWM
- Indices: ^GSPC (S&P 500), ^DJI (Dow Jones)

Would you like to try a different symbol?
```

### Example 12: API Key Missing

**User:**
```
Generate report for SPY
```

**Claude:**
```
I'll generate a report for SPY.

[Executes skill]

⚠️ Warning: Claude AI analysis unavailable (API key not configured)

Generated report with:
✓ Market data
✓ Technical indicators
✓ Charts
✗ AI-powered insights (requires API key)

The report includes all technical analysis but lacks Claude AI market
intelligence. To enable AI analysis, configure ANTHROPIC_API_KEY in .env file.

Would you like to proceed with the technical-only report?
```

## Pro Tips

### Tip 1: Batch Processing

**User:**
```
Analyze my entire portfolio: SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
```

**Claude will automatically:**
- Detect multiple symbols
- Generate reports in parallel (if possible)
- Provide summary comparison
- Flag any errors or warnings

### Tip 2: Natural Language Queries

All of these work:
- "SPY report"
- "Analyze SPY"
- "What's the market doing? Check SPY"
- "Investment analysis for S&P 500"
- "Generate trading report for spy"

Claude understands context and intent!

### Tip 3: Follow-up Questions

After generating a report, you can ask:
- "Explain the MACD signal"
- "What does the RSI tell us?"
- "Should I buy based on this analysis?"
- "What are the main risks?"

Claude has context from the generated report!

## Summary

The trading-analysis skill provides:
- Natural language interface
- Automatic report generation
- Professional-grade analysis
- Flexible usage patterns
- Integration with existing workflows

Just ask Claude to analyze a stock, and the skill handles the rest!
