#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Investment Report Generator

투자 분석 보고서를 생성하는 모듈
차트, 표, 그래프를 포함한 전문적인 투자 보고서 작성
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path


class InvestmentReportGenerator:
    """
    투자 분석 보고서 생성기

    Claude 분석 결과와 시장 데이터를 기반으로
    전문적인 투자 보고서를 생성합니다.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Args:
            output_dir: 보고서 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 보고서 메타데이터
        self.report_metadata = {
            'company': 'AI Trading Analytics',
            'department': 'Quantitative Research',
            'disclaimer': 'This report is for informational purposes only and does not constitute investment advice.'
        }

    def generate_market_analysis_report(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_indicators: Dict[str, float],
        claude_analysis: Dict[str, Any],
        report_title: Optional[str] = None,
        client_name: Optional[str] = None
    ) -> str:
        """
        시장 분석 보고서 생성

        Args:
            symbol: 종목 심볼
            market_data: 시장 데이터 (OHLCV)
            technical_indicators: 기술적 지표
            claude_analysis: Claude AI 분석 결과
            report_title: 보고서 제목
            client_name: 고객명

        Returns:
            생성된 보고서 파일 경로
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"{symbol}_analysis_report_{timestamp}"

        # 보고서 데이터 준비
        report_data = self._prepare_report_data(
            symbol, market_data, technical_indicators, claude_analysis
        )

        # Markdown 보고서 생성
        markdown_content = self._generate_markdown_report(
            symbol, report_data, claude_analysis, report_title, client_name
        )

        # 파일 저장
        markdown_file = self.output_dir / f"{report_name}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # JSON 데이터도 함께 저장
        json_file = self.output_dir / f"{report_name}_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"✓ Report generated: {markdown_file}")
        print(f"✓ Data saved: {json_file}")

        return str(markdown_file)

    def _prepare_report_data(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_indicators: Dict[str, float],
        claude_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """보고서 데이터 준비"""

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]

        # 가격 변동 계산
        daily_change = latest['Close'] - prev['Close']
        daily_change_pct = (daily_change / prev['Close']) * 100

        # 주간/월간 변동
        week_change = None
        month_change = None
        if len(market_data) >= 5:
            week_ago = market_data.iloc[-5]
            week_change = ((latest['Close'] / week_ago['Close']) - 1) * 100
        if len(market_data) >= 20:
            month_ago = market_data.iloc[-20]
            month_change = ((latest['Close'] / month_ago['Close']) - 1) * 100

        # 변동성 계산
        returns = market_data['Close'].pct_change()
        volatility_20d = returns.tail(20).std() * np.sqrt(252) * 100
        volatility_50d = returns.tail(50).std() * np.sqrt(252) * 100 if len(returns) >= 50 else None

        # 거래량 분석
        avg_volume_20d = market_data['Volume'].tail(20).mean()
        volume_ratio = latest['Volume'] / avg_volume_20d

        # 지지/저항 레벨
        high_20d = market_data['High'].tail(20).max()
        low_20d = market_data['Low'].tail(20).min()

        return {
            'symbol': symbol,
            'report_date': datetime.now().isoformat(),
            'price_data': {
                'current': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'daily_change': float(daily_change),
                'daily_change_pct': float(daily_change_pct),
                'week_change_pct': float(week_change) if week_change else None,
                'month_change_pct': float(month_change) if month_change else None,
                'high_20d': float(high_20d),
                'low_20d': float(low_20d)
            },
            'volume_data': {
                'current': int(latest['Volume']),
                'avg_20d': int(avg_volume_20d),
                'ratio': float(volume_ratio),
                'trend': 'Above Average' if volume_ratio > 1.1 else 'Below Average' if volume_ratio < 0.9 else 'Normal'
            },
            'technical_indicators': technical_indicators,
            'volatility': {
                'volatility_20d': float(volatility_20d),
                'volatility_50d': float(volatility_50d) if volatility_50d else None,
                'regime': 'High' if volatility_20d > 25 else 'Moderate' if volatility_20d > 15 else 'Low'
            },
            'claude_analysis': claude_analysis,
            'market_stats': {
                'data_points': len(market_data),
                'date_range': {
                    'start': str(market_data.index[0].date()),
                    'end': str(market_data.index[-1].date())
                }
            }
        }

    def _generate_markdown_report(
        self,
        symbol: str,
        report_data: Dict[str, Any],
        claude_analysis: Dict[str, Any],
        report_title: Optional[str],
        client_name: Optional[str]
    ) -> str:
        """Markdown 형식 보고서 생성"""

        price = report_data['price_data']
        volume = report_data['volume_data']
        volatility = report_data['volatility']
        indicators = report_data['technical_indicators']

        # 보고서 제목
        if not report_title:
            report_title = f"{symbol} Market Analysis Report"

        # 날짜
        report_date = datetime.now().strftime('%B %d, %Y')

        md = f"""# {report_title}

---

**Report Date**: {report_date}
**Symbol**: {symbol}
**Prepared By**: {self.report_metadata['company']} - {self.report_metadata['department']}
"""

        if client_name:
            md += f"**Prepared For**: {client_name}  \n"

        md += "\n---\n\n"

        # Executive Summary
        sentiment = claude_analysis.get('market_sentiment', 'neutral').upper()
        confidence = claude_analysis.get('confidence_level', 0) * 100
        recommendation = claude_analysis.get('trading_recommendation', {}).get('suggested_action', 'hold').upper()

        sentiment_emoji = {'BULLISH': '📈', 'BEARISH': '📉', 'NEUTRAL': '➡️'}.get(sentiment, '➡️')

        md += f"""## Executive Summary

{sentiment_emoji} **Market Sentiment**: {sentiment}
**Confidence Level**: {confidence:.0f}%
**Investment Recommendation**: **{recommendation}**

"""

        # Key Findings
        if 'key_observations' in claude_analysis:
            md += "### Key Findings\n\n"
            for i, obs in enumerate(claude_analysis['key_observations'], 1):
                md += f"{i}. {obs}\n"
            md += "\n"

        md += "---\n\n"

        # Market Overview
        md += f"""## 1. Market Overview

### Current Price Action

| Metric | Value | Change |
|--------|-------|--------|
| **Current Price** | ${price['current']:,.2f} | {price['daily_change_pct']:+.2f}% (${price['daily_change']:+,.2f}) |
| **Open** | ${price['open']:,.2f} | - |
| **High** | ${price['high']:,.2f} | - |
| **Low** | ${price['low']:,.2f} | - |
"""

        if price['week_change_pct'] is not None:
            md += f"| **Week Change** | - | {price['week_change_pct']:+.2f}% |\n"
        if price['month_change_pct'] is not None:
            md += f"| **Month Change** | - | {price['month_change_pct']:+.2f}% |\n"

        md += f"""
### Price Range (20-Day)

- **Resistance Level**: ${price['high_20d']:,.2f}
- **Support Level**: ${price['low_20d']:,.2f}
- **Range**: ${price['high_20d'] - price['low_20d']:,.2f} ({((price['high_20d'] / price['low_20d']) - 1) * 100:.2f}%)

"""

        # Volume Analysis
        md += f"""### Volume Analysis

| Metric | Value | Analysis |
|--------|-------|----------|
| **Current Volume** | {volume['current']:,} | {volume['trend']} |
| **20-Day Average** | {volume['avg_20d']:,} | Baseline |
| **Volume Ratio** | {volume['ratio']:.2f}x | {'🔴 High Activity' if volume['ratio'] > 1.2 else '🟢 Normal' if volume['ratio'] > 0.8 else '🟡 Low Activity'} |

"""

        # Technical Indicators
        md += """---

## 2. Technical Analysis

### Technical Indicators

| Indicator | Value | Signal |
|-----------|-------|--------|
"""

        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            rsi_signal = '🔴 Overbought' if rsi > 70 else '🟢 Oversold' if rsi < 30 else '🟡 Neutral'
            md += f"| **RSI (14)** | {rsi:.2f} | {rsi_signal} |\n"

        # MACD
        if 'macd' in indicators:
            macd = indicators['macd']
            macd_signal = indicators.get('macd_signal', 0)
            macd_trend = '🟢 Bullish' if macd > macd_signal else '🔴 Bearish'
            md += f"| **MACD** | {macd:.2f} | {macd_trend} |\n"

        # Moving Averages
        if 'sma_20' in indicators:
            md += f"| **SMA 20** | ${indicators['sma_20']:,.2f} | {'Above' if price['current'] > indicators['sma_20'] else 'Below'} Current Price |\n"
        if 'sma_50' in indicators:
            md += f"| **SMA 50** | ${indicators['sma_50']:,.2f} | {'Above' if price['current'] > indicators['sma_50'] else 'Below'} Current Price |\n"

        # Volatility
        md += f"""
### Volatility Assessment

| Metric | Value | Regime |
|--------|-------|--------|
| **20-Day Volatility** | {volatility['volatility_20d']:.2f}% | {volatility['regime']} |
"""

        if volatility['volatility_50d']:
            md += f"| **50-Day Volatility** | {volatility['volatility_50d']:.2f}% | Trend Indicator |\n"

        md += f"""
**Interpretation**: {volatility['regime']} volatility regime indicates {'stable' if volatility['regime'] == 'Low' else 'moderate' if volatility['regime'] == 'Moderate' else 'unstable'} market conditions.

"""

        # Claude AI Analysis
        md += """---

## 3. AI-Powered Market Intelligence

### Claude AI Analysis

"""

        # Technical Assessment
        if 'technical_assessment' in claude_analysis:
            tech = claude_analysis['technical_assessment']
            md += f"""#### Market Trend Analysis

- **Trend Direction**: {tech.get('trend', 'N/A').upper()}
- **Momentum**: {tech.get('momentum', 'N/A').upper()}
- **Support & Resistance**: {tech.get('support_resistance', 'N/A')}

"""

        # Risk Assessment
        if 'risk_assessment' in claude_analysis:
            risk = claude_analysis['risk_assessment']
            risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(risk.get('risk_level', 'medium').lower(), '🟡')

            md += f"""#### Risk Assessment

{risk_emoji} **Risk Level**: {risk.get('risk_level', 'N/A').upper()}

**Identified Risks**:
"""
            for i, r in enumerate(risk.get('key_risks', []), 1):
                md += f"{i}. {r}\n"

            md += f"\n**Risk Mitigation**: {risk.get('risk_mitigation', 'N/A')}\n\n"

        # Trading Recommendation
        if 'trading_recommendation' in claude_analysis:
            rec = claude_analysis['trading_recommendation']

            rec_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡', 'reduce': '🟠'}.get(rec.get('suggested_action', 'hold').lower(), '🟡')

            md += f"""---

## 4. Investment Recommendation

### {rec_emoji} Recommended Action: **{rec.get('suggested_action', 'HOLD').upper()}**

**Rationale**:
{rec.get('reasoning', 'N/A')}

**Entry Criteria**:
{rec.get('entry_criteria', 'N/A')}

**Exit Criteria**:
{rec.get('exit_criteria', 'N/A')}

"""

        # Confidence Factors
        if 'confidence_factors' in claude_analysis:
            factors = claude_analysis['confidence_factors']

            md += """### Confidence Analysis

**Supporting Evidence**:
"""
            for i, evidence in enumerate(factors.get('supporting_evidence', []), 1):
                md += f"{i}. ✓ {evidence}\n"

            md += "\n**Contradicting Evidence**:\n"
            for i, evidence in enumerate(factors.get('contradicting_evidence', []), 1):
                md += f"{i}. ✗ {evidence}\n"

            md += "\n"

        # Data Summary
        stats = report_data['market_stats']
        md += f"""---

## 5. Data Summary

- **Analysis Period**: {stats['date_range']['start']} to {stats['date_range']['end']}
- **Data Points**: {stats['data_points']} trading days
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

"""

        # Disclaimer
        md += f"""---

## Disclaimer

{self.report_metadata['disclaimer']}

This analysis is based on historical data and AI-powered insights. Past performance does not guarantee future results. Always conduct your own due diligence and consult with a qualified financial advisor before making investment decisions.

**Risk Warning**: Trading and investing in financial markets involves substantial risk of loss. You should only invest money that you can afford to lose.

---

*Report generated by {self.report_metadata['company']} using Claude AI technology*
*© {datetime.now().year} AI Trading Analytics. All rights reserved.*
"""

        return md

    def generate_summary_table(
        self,
        symbol: str,
        report_data: Dict[str, Any]
    ) -> str:
        """요약 테이블 생성"""

        price = report_data['price_data']
        claude = report_data['claude_analysis']

        table = f"""
## Quick Reference Table

| Category | Metric | Value |
|----------|--------|-------|
| **Price** | Current | ${price['current']:,.2f} |
| | Daily Change | {price['daily_change_pct']:+.2f}% |
| **Sentiment** | Market | {claude.get('market_sentiment', 'N/A').upper()} |
| | Confidence | {claude.get('confidence_level', 0) * 100:.0f}% |
| **Recommendation** | Action | {claude.get('trading_recommendation', {}).get('suggested_action', 'N/A').upper()} |
| | Risk Level | {claude.get('risk_assessment', {}).get('risk_level', 'N/A').upper()} |
"""

        return table
