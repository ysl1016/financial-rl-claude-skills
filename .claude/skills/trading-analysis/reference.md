# Trading Analysis Skill - Technical Reference

## Architecture Overview

This skill is built on top of a comprehensive financial analysis system that combines:

1. **Data Layer**: Yahoo Finance integration via `yfinance`
2. **Analysis Layer**: Technical indicators + Claude AI market analysis
3. **Visualization Layer**: Professional chart generation with matplotlib/seaborn
4. **Reporting Layer**: Institutional-grade markdown report generation

### System Architecture

```
User Request (Natural Language)
    ↓
Claude Code (discovers trading-analysis skill)
    ↓
SKILL.md (skill definition & description)
    ↓
scripts/generate_report.py (wrapper script)
    ↓
generate_investment_report.py (main orchestrator)
    ↓
┌─────────────────────────────────────────────┐
│  Data Collection (yfinance)                 │
│  └─ Download historical market data         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Technical Analysis                         │
│  ├─ RSI (Relative Strength Index)          │
│  ├─ MACD (Moving Average Convergence)      │
│  ├─ Moving Averages (SMA 10, 20, 50)       │
│  ├─ Bollinger Bands                         │
│  ├─ Volatility (20-day, 50-day)            │
│  └─ Volume Analysis                         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Claude AI Analysis                         │
│  ├─ Market sentiment analysis               │
│  ├─ Trend identification                    │
│  ├─ Risk assessment                         │
│  ├─ Trading recommendations                 │
│  └─ Confidence scoring                      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Chart Generation (4 types)                │
│  ├─ Price Chart with Moving Averages       │
│  ├─ Technical Indicators (RSI & MACD)      │
│  ├─ Historical Volatility                   │
│  └─ Performance Summary Dashboard          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Report Compilation                         │
│  ├─ Executive Summary                       │
│  ├─ Market Overview                         │
│  ├─ Technical Analysis Section             │
│  ├─ AI-Powered Insights                    │
│  ├─ Investment Recommendations             │
│  ├─ Risk Assessment                         │
│  └─ Legal Disclaimers                       │
└─────────────────────────────────────────────┘
    ↓
Output Files (Markdown + JSON + PNG charts)
```

## Technical Indicators Calculated

### 1. Relative Strength Index (RSI)
- **Period**: 14 days
- **Range**: 0-100
- **Interpretation**:
  - RSI > 70: Overbought (potential sell signal)
  - RSI < 30: Oversold (potential buy signal)
  - RSI 40-60: Neutral zone

### 2. MACD (Moving Average Convergence Divergence)
- **Fast EMA**: 12 periods
- **Slow EMA**: 26 periods
- **Signal Line**: 9-period EMA
- **Interpretation**:
  - MACD > Signal: Bullish momentum
  - MACD < Signal: Bearish momentum
  - Crossovers indicate trend changes

### 3. Moving Averages
- **SMA 10**: Short-term trend
- **SMA 20**: Medium-term trend
- **SMA 50**: Long-term trend
- **Interpretation**:
  - Price > SMA: Bullish
  - Price < SMA: Bearish
  - Golden Cross (50 > 200): Strong bullish signal

### 4. Bollinger Bands
- **Middle Band**: 20-day SMA
- **Upper Band**: Middle + (2 × standard deviation)
- **Lower Band**: Middle - (2 × standard deviation)
- **Interpretation**:
  - Price near upper band: Overbought
  - Price near lower band: Oversold
  - Band width indicates volatility

### 5. Volatility
- **20-day volatility**: Short-term risk measure
- **50-day volatility**: Medium-term risk measure
- **Calculation**: Annualized standard deviation of returns
- **Regimes**:
  - Low: < 15%
  - Medium: 15-25%
  - High: > 25%

### 6. Volume Analysis
- **Current Volume**: Today's trading volume
- **20-day Average**: Baseline volume
- **Volume Ratio**: Current / Average
- **Interpretation**:
  - Ratio > 1.5: High volume (strong conviction)
  - Ratio < 0.5: Low volume (weak conviction)

## Claude AI Analysis Components

### Market Sentiment
Claude analyzes all technical indicators to determine overall market sentiment:
- **Bullish**: Strong upward momentum
- **Neutral**: Balanced or sideways movement
- **Bearish**: Strong downward momentum

### Confidence Level
0.0 to 1.0 scale indicating analysis confidence:
- **0.8-1.0**: High confidence (clear signals)
- **0.6-0.8**: Medium confidence (mixed signals)
- **0.0-0.6**: Low confidence (uncertain conditions)

### Trading Recommendation
Based on comprehensive analysis:
- **BUY**: Strong bullish signals, low risk
- **HOLD**: Mixed signals or neutral conditions
- **SELL**: Strong bearish signals or high risk

### Risk Assessment
Multi-factor risk evaluation:
- **Low**: Stable conditions, clear trend
- **Medium**: Some uncertainty, mixed signals
- **High**: High volatility, conflicting indicators
- **Critical**: Extreme conditions, potential major moves

### Key Observations
Claude provides 3-5 specific observations including:
- Price-indicator relationships
- Momentum analysis
- Volume interpretation
- Support/resistance levels

## Report Sections

### 1. Executive Summary
- Market sentiment (bullish/neutral/bearish)
- Confidence level percentage
- Primary recommendation (buy/hold/sell)
- 3-4 key findings in bullet points

### 2. Market Overview
**Price Tables:**
- Current price with daily change
- Open, High, Low values
- Week and month performance

**Volume Analysis:**
- Current volume vs 20-day average
- Volume ratio interpretation

### 3. Technical Analysis
**Indicator Table:**
- All calculated indicators with values
- Signal interpretation (bullish/neutral/bearish)
- Visual indicators (emojis for quick reading)

**Volatility Assessment:**
- 20-day and 50-day volatility
- Volatility regime classification
- Trend analysis

### 4. AI-Powered Market Intelligence
**Claude Analysis:**
- Detailed trend analysis
- Momentum assessment
- Support/resistance identification
- Risk evaluation with specific factors
- Risk mitigation strategies

### 5. Investment Recommendation
- Recommended action with rationale
- Specific entry criteria and price levels
- Exit criteria and stop-loss levels
- Confidence analysis with supporting/contradicting evidence

### 6. Charts & Visualizations
Four embedded high-resolution PNG charts:
1. Price chart with moving averages
2. Technical indicators (RSI & MACD)
3. Historical volatility
4. Performance summary dashboard

### 7. Legal Disclaimers
- Investment advice disclaimer
- Risk warnings
- Past performance notice
- Professional consultation recommendation

## Chart Specifications

### Chart 1: Price Chart
- **Type**: Candlestick or line chart
- **Timeframe**: 60 trading days
- **Overlays**: SMA 20, SMA 50
- **Subpanel**: Volume chart with color coding
- **DPI**: 300 (print quality)
- **Size**: ~350-400 KB

### Chart 2: Technical Indicators
- **Panel 1**: RSI with overbought/oversold zones
- **Panel 2**: MACD with signal line and histogram
- **Zones**: Marked buy/sell areas
- **DPI**: 300
- **Size**: ~300-350 KB

### Chart 3: Volatility
- **Type**: Line chart
- **Data**: 20-day rolling volatility
- **Markers**: High/low volatility thresholds
- **Trend**: Volatility regime indicators
- **DPI**: 300
- **Size**: ~180-200 KB

### Chart 4: Summary Dashboard
- **Layout**: 2x2 grid
- **Panels**:
  - Top-left: Price trend (30 days)
  - Top-right: RSI gauge
  - Bottom-left: Volume (20 days)
  - Bottom-right: Volatility trend
- **Summary Box**: Claude AI analysis summary
- **DPI**: 300
- **Size**: ~330-350 KB

## File Naming Convention

All files use the following pattern:
```
{SYMBOL}_{type}_{timestamp}.{extension}

Examples:
- SPY_analysis_report_20251019_011642.md
- SPY_analysis_report_20251019_011642_data.json
- SPY_price_chart.png
- SPY_indicators_chart.png
- SPY_volatility_chart.png
- SPY_summary_dashboard.png
```

## JSON Data Structure

The JSON export contains:
```json
{
  "symbol": "SPY",
  "report_date": "2025-10-19T01:16:42",
  "price_data": {
    "current": 664.39,
    "open": 659.50,
    "high": 665.76,
    "low": 658.14,
    "daily_change": 3.75,
    "daily_change_pct": 0.57,
    "week_change_pct": 0.20,
    "month_change_pct": -0.37
  },
  "volume_data": {
    "current": 96386200,
    "avg_20d": 80401485,
    "ratio": 1.20
  },
  "technical_indicators": {
    "rsi": 50.63,
    "macd": 2.77,
    "macd_signal": 4.23,
    "sma_10": 665.35,
    "sma_20": 665.06,
    "sma_50": 653.63,
    "volatility": 0.13
  },
  "claude_analysis": {
    "market_sentiment": "neutral",
    "confidence_level": 0.65,
    "key_observations": [...],
    "trading_recommendation": {...},
    "risk_assessment": {...}
  }
}
```

## Dependencies

### Required Packages
```
yfinance>=0.2.40        # Market data
pandas>=2.0.0           # Data processing
numpy>=1.24.0           # Numerical computing
anthropic>=0.25.0       # Claude AI
matplotlib>=3.7.0       # Chart generation
seaborn>=0.13.0         # Statistical visualization
python-dotenv>=1.0.0    # Environment configuration
```

### Environment Variables
```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxx...
CLAUDE_MODEL=claude-3-7-sonnet-20250219
CLAUDE_MAX_TOKENS=2048
CLAUDE_TEMPERATURE=0.7
```

## Performance Metrics

Typical execution times for single symbol:
- Data download: ~2 seconds
- Indicator calculation: <1 second
- Claude analysis: ~5 seconds
- Chart generation: ~3 seconds
- Report compilation: <1 second
- **Total**: ~12 seconds

## Error Handling

The skill handles various error scenarios:
- **Network errors**: Retry mechanism for data download
- **Invalid symbols**: Clear error message with symbol validation
- **API errors**: Fallback to technical analysis only
- **Missing data**: Graceful degradation with warnings

## Customization Points

### Via Arguments
- Symbol selection
- Client name
- Report title
- Historical period

### Via Code (Advanced)
- Chart styling and colors
- Indicator parameters (periods, thresholds)
- Report sections and formatting
- Claude analysis prompts
- Risk assessment criteria

## Security Considerations

- API keys stored in `.env` file (not in code)
- `.env` file in `.gitignore`
- No sensitive data in reports
- All market data is public information
- Appropriate disclaimers included

## Limitations

- Historical data limited by Yahoo Finance
- Real-time quotes may have 15-minute delay
- AI analysis is technical, not fundamental
- No backtesting or strategy optimization
- Single-asset analysis (no portfolio optimization)
- Requires internet connection

## Future Enhancements

Potential improvements:
- PDF export with embedded charts
- Multi-asset portfolio analysis
- Backtesting integration
- Email report delivery
- Scheduled report generation
- Interactive web dashboard
- Fundamental analysis integration
- Sector/industry comparison
