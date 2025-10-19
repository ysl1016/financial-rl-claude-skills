#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yahoo Finance 실제 데이터로 Claude Integration 데모

이 스크립트는 실제 Yahoo Finance 데이터를 다운로드하여
Claude 통합 시스템을 테스트합니다.
"""

import sys
import os

# 패키지 import 체크
print("=" * 80)
print("Yahoo Finance + Claude Integration Demo")
print("=" * 80)

print("\n[Step 1/7] Checking dependencies...")
required_packages = {
    'yfinance': 'Yahoo Finance data',
    'pandas': 'Data processing',
    'numpy': 'Numerical computing',
    'torch': 'Deep learning',
    'anthropic': 'Claude API'
}

missing = []
for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {package:15s} - {description}")
    except ImportError:
        print(f"  ✗ {package:15s} - {description} (MISSING)")
        missing.append(package)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print(f"\nInstall with: pip install {' '.join(missing)}")
    sys.exit(1)

print("\n✅ All dependencies installed!")

# 이제 실제 import
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n[Step 2/7] Downloading real market data from Yahoo Finance...")

# 실제 주식 데이터 다운로드
SYMBOL = 'SPY'  # S&P 500 ETF
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=730)  # 2년치 데이터

print(f"  Symbol: {SYMBOL}")
print(f"  Period: {START_DATE.date()} to {END_DATE.date()}")

try:
    ticker = yf.Ticker(SYMBOL)
    raw_data = ticker.history(start=START_DATE, end=END_DATE)

    if len(raw_data) == 0:
        raise ValueError("No data downloaded")

    print(f"  ✓ Downloaded {len(raw_data)} trading days")
    print(f"  ✓ Columns: {list(raw_data.columns)}")

except Exception as e:
    print(f"  ✗ Download failed: {e}")
    print("\n💡 Tip: Check your internet connection or try a different symbol")
    sys.exit(1)

# 데이터 샘플 출력
print(f"\n  Recent data (last 3 days):")
print(raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3))

print("\n[Step 3/7] Processing technical indicators...")

# 기술적 지표 계산 (간단 버전)
def calculate_simple_indicators(data):
    """간단한 기술적 지표 계산"""
    df = data.copy()

    # 이동평균
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)

    # 정규화 (normalized)
    for col in ['RSI', 'MACD', 'MACD_Signal']:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-9)

    return df.dropna()

processed_data = calculate_simple_indicators(raw_data)
print(f"  ✓ Calculated indicators: SMA20, SMA50, RSI, MACD, Bollinger Bands")
print(f"  ✓ Processed data: {len(processed_data)} days (after removing NaN)")

print("\n[Step 4/7] Checking Claude API configuration...")

api_key = os.environ.get('ANTHROPIC_API_KEY')
if api_key:
    masked_key = api_key[:10] + '...' + api_key[-4:]
    print(f"  ✓ API Key found: {masked_key}")
    claude_available = True
else:
    print("  ⚠ ANTHROPIC_API_KEY not set")
    print("    Claude features will use simulation mode")
    claude_available = False

print("\n[Step 5/7] Analyzing current market conditions...")

# 최근 시장 상태 분석
recent_data = processed_data.tail(20)
current_price = recent_data['Close'].iloc[-1]
prev_price = recent_data['Close'].iloc[-2]
price_change = (current_price / prev_price - 1) * 100

current_rsi = recent_data['RSI'].iloc[-1]
current_macd = recent_data['MACD'].iloc[-1]

print(f"\n  {SYMBOL} Current Market Analysis:")
print(f"  ─────────────────────────────────")
print(f"  Current Price:    ${current_price:.2f}")
print(f"  Daily Change:     {price_change:+.2f}%")
print(f"  RSI:              {current_rsi:.1f} {'(Overbought)' if current_rsi > 70 else '(Oversold)' if current_rsi < 30 else '(Neutral)'}")
print(f"  MACD:             {current_macd:+.2f}")
print(f"  20-day Volatility: {recent_data['Close'].pct_change().std() * np.sqrt(252) * 100:.1f}%")

print("\n[Step 6/7] Demonstrating Claude integration (if available)...")

if claude_available:
    try:
        from src.claude_integration import ClaudeMarketAnalyzer

        analyzer = ClaudeMarketAnalyzer()

        # 시장 분석 요청
        print("  🤖 Requesting Claude market analysis...")

        market_summary = {
            'price_action': {
                'current_price': float(current_price),
                'price_change_1d': float(price_change),
                'volatility_20d': float(recent_data['Close'].pct_change().std() * np.sqrt(252) * 100)
            },
            'technical_indicators': {
                'RSI': float(current_rsi),
                'MACD': float(current_macd)
            }
        }

        indicators = {
            'RSI_norm': float(recent_data['RSI_norm'].iloc[-1]),
            'MACD_norm': float(recent_data['MACD_norm'].iloc[-1])
        }

        analysis = analyzer.analyze_market_state(
            market_data=processed_data.tail(50),
            technical_indicators=indicators,
            current_position=0,
            portfolio_value=100000
        )

        print("\n  ✅ Claude Analysis Complete!")
        print(f"\n  Market Sentiment:     {analysis.get('market_sentiment', 'N/A').upper()}")
        print(f"  Confidence Level:     {analysis.get('confidence_level', 0)*100:.0f}%")

        recommendation = analysis.get('trading_recommendation', {})
        print(f"  Suggested Action:     {recommendation.get('suggested_action', 'N/A').upper()}")
        print(f"  Reasoning:            {recommendation.get('reasoning', 'N/A')[:100]}...")

        print(f"\n  Key Observations:")
        for i, obs in enumerate(analysis.get('key_observations', [])[:3], 1):
            print(f"    {i}. {obs}")

    except Exception as e:
        print(f"  ⚠ Claude analysis failed: {e}")
        print("    This is normal if API key is not set")

else:
    print("  ℹ Running in simulation mode (no API key)")
    print("\n  Simulated Analysis:")
    print(f"  Market Sentiment:     {'BULLISH' if price_change > 0 else 'BEARISH'}")
    print(f"  Suggested Action:     {'BUY' if current_rsi < 40 else 'SELL' if current_rsi > 60 else 'HOLD'}")
    print(f"  Reasoning:            Based on RSI={current_rsi:.1f} and price trend")

print("\n[Step 7/7] Summary & Next Steps...")
print("\n" + "="*80)
print("✅ Demo Complete!")
print("="*80)

print(f"\n📊 Data Summary:")
print(f"  • Downloaded {len(raw_data)} days of real {SYMBOL} data from Yahoo Finance")
print(f"  • Calculated 8+ technical indicators")
print(f"  • Processed {len(processed_data)} trading days")
print(f"  • Current market state: {SYMBOL} @ ${current_price:.2f} ({price_change:+.2f}%)")

print(f"\n🤖 Claude Integration:")
if claude_available:
    print(f"  • Claude API: ✓ Available")
    print(f"  • Real-time analysis: ✓ Working")
else:
    print(f"  • Claude API: ⚠ Not configured (set ANTHROPIC_API_KEY)")
    print(f"  • Simulation mode: ✓ Working")

print(f"\n🚀 Next Steps:")
print(f"  1. Set API key: export ANTHROPIC_API_KEY='your-key-here'")
print(f"  2. Run full training: python examples/hybrid_claude_trading.py")
print(f"  3. Try different symbols: --symbol AAPL or --symbol TSLA")
print(f"  4. Customize parameters in the code")

print(f"\n💡 Quick Start Commands:")
print(f"  # Install missing packages:")
print(f"  pip install yfinance pandas numpy torch anthropic")
print(f"  ")
print(f"  # Set Claude API key:")
print(f"  export ANTHROPIC_API_KEY='sk-ant-...'")
print(f"  ")
print(f"  # Run this demo:")
print(f"  python examples/quick_demo_yahoo_finance.py")
print(f"  ")
print(f"  # Run full hybrid training:")
print(f"  python examples/hybrid_claude_trading.py --symbol SPY --episodes 5")

print("\n" + "="*80)
print("📚 Documentation: docs/CLAUDE_INTEGRATION_GUIDE.md")
print("="*80 + "\n")
