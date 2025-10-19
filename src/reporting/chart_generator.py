#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chart Generator

투자 보고서용 차트 생성 모듈
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# 한글 폰트 설정 (옵션)
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class ChartGenerator:
    """투자 보고서용 차트 생성기"""

    def __init__(self, output_dir: str = "reports/charts"):
        """
        Args:
            output_dir: 차트 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 색상 팔레트
        self.colors = {
            'bullish': '#26A69A',  # 녹색 (상승)
            'bearish': '#EF5350',  # 빨간색 (하락)
            'neutral': '#78909C',  # 회색 (중립)
            'volume': '#90A4AE',   # 회색 (거래량)
            'ma_short': '#FF6F00', # 주황 (단기 MA)
            'ma_long': '#1976D2',  # 파랑 (장기 MA)
            'signal': '#9C27B0'    # 보라 (신호선)
        }

    def generate_price_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        save_path: Optional[str] = None
    ) -> str:
        """가격 차트 생성 (캔들스틱 + 이동평균선)"""

        if save_path is None:
            save_path = self.output_dir / f"{symbol}_price_chart.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # 상위 차트: 가격 + 이동평균선
        dates = data.index
        closes = data['Close']

        # 캔들스틱 (간소화 버전 - 라인 차트)
        ax1.plot(dates, closes, color=self.colors['neutral'], linewidth=1.5, label='Price', alpha=0.7)

        # 이동평균선
        if 'SMA_20' in data.columns:
            ax1.plot(dates, data['SMA_20'], color=self.colors['ma_short'],
                    linewidth=1.5, label='SMA 20', alpha=0.8)

        if 'SMA_50' in data.columns:
            ax1.plot(dates, data['SMA_50'], color=self.colors['ma_long'],
                    linewidth=1.5, label='SMA 50', alpha=0.8)

        # 현재 가격 표시
        current_price = closes.iloc[-1]
        ax1.axhline(y=current_price, color=self.colors['signal'],
                   linestyle='--', linewidth=1, alpha=0.5)
        ax1.text(dates[-1], current_price, f'  ${current_price:.2f}',
                verticalalignment='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{symbol} - Price Chart & Moving Averages', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # 하위 차트: 거래량
        colors_volume = [self.colors['bullish'] if data['Close'].iloc[i] >= data['Open'].iloc[i]
                        else self.colors['bearish'] for i in range(len(data))]

        ax2.bar(dates, data['Volume'], color=colors_volume, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 날짜 포맷
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Price chart saved: {save_path}")
        return str(save_path)

    def generate_technical_indicators_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        save_path: Optional[str] = None
    ) -> str:
        """기술적 지표 차트 생성 (RSI, MACD)"""

        if save_path is None:
            save_path = self.output_dir / f"{symbol}_indicators_chart.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        dates = data.index

        # RSI 차트
        if 'RSI' in data.columns:
            ax1.plot(dates, data['RSI'], color=self.colors['neutral'], linewidth=1.5, label='RSI')
            ax1.axhline(y=70, color=self.colors['bearish'], linestyle='--',
                       linewidth=1, alpha=0.5, label='Overbought (70)')
            ax1.axhline(y=30, color=self.colors['bullish'], linestyle='--',
                       linewidth=1, alpha=0.5, label='Oversold (30)')
            ax1.fill_between(dates, 70, 100, alpha=0.1, color=self.colors['bearish'])
            ax1.fill_between(dates, 0, 30, alpha=0.1, color=self.colors['bullish'])

            ax1.set_ylabel('RSI', fontsize=11, fontweight='bold')
            ax1.set_title(f'{symbol} - RSI Indicator', fontsize=13, fontweight='bold')
            ax1.legend(loc='upper left', framealpha=0.9)
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)

        # MACD 차트
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            ax2.plot(dates, data['MACD'], color=self.colors['neutral'],
                    linewidth=1.5, label='MACD', alpha=0.8)
            ax2.plot(dates, data['MACD_Signal'], color=self.colors['signal'],
                    linewidth=1.5, label='Signal', alpha=0.8)

            # MACD 히스토그램
            if 'MACD_Hist' in data.columns:
                colors_hist = [self.colors['bullish'] if val >= 0 else self.colors['bearish']
                              for val in data['MACD_Hist']]
                ax2.bar(dates, data['MACD_Hist'], color=colors_hist, alpha=0.3, width=0.8)

            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel('MACD', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax2.set_title(f'{symbol} - MACD Indicator', fontsize=13, fontweight='bold')
            ax2.legend(loc='upper left', framealpha=0.9)
            ax2.grid(True, alpha=0.3)

            # 날짜 포맷
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Technical indicators chart saved: {save_path}")
        return str(save_path)

    def generate_volatility_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        save_path: Optional[str] = None
    ) -> str:
        """변동성 차트 생성"""

        if save_path is None:
            save_path = self.output_dir / f"{symbol}_volatility_chart.png"

        fig, ax = plt.subplots(figsize=(14, 6))

        dates = data.index

        if 'Volatility_20' in data.columns:
            volatility_pct = data['Volatility_20'] * 100  # 퍼센트로 변환

            ax.plot(dates, volatility_pct, color=self.colors['neutral'],
                   linewidth=2, label='20-Day Volatility', alpha=0.8)
            ax.fill_between(dates, volatility_pct, alpha=0.2, color=self.colors['neutral'])

            # 변동성 구간 표시
            ax.axhline(y=15, color=self.colors['bullish'], linestyle='--',
                      linewidth=1, alpha=0.5, label='Low Volatility (15%)')
            ax.axhline(y=25, color=self.colors['bearish'], linestyle='--',
                      linewidth=1, alpha=0.5, label='High Volatility (25%)')

            ax.set_ylabel('Volatility (%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_title(f'{symbol} - Historical Volatility', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.3)

            # 날짜 포맷
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Volatility chart saved: {save_path}")
        return str(save_path)

    def generate_performance_summary(
        self,
        data: pd.DataFrame,
        symbol: str,
        claude_analysis: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """성과 요약 대시보드 생성"""

        if save_path is None:
            save_path = self.output_dir / f"{symbol}_summary_dashboard.png"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 가격 추이 (상단 전체)
        ax1 = fig.add_subplot(gs[0, :])
        dates = data.index
        closes = data['Close']
        ax1.plot(dates, closes, color=self.colors['neutral'], linewidth=2)
        ax1.fill_between(dates, closes, alpha=0.1, color=self.colors['neutral'])
        ax1.set_title(f'{symbol} - Price Trend', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = fig.add_subplot(gs[1, 0])
        if 'RSI' in data.columns:
            latest_rsi = data['RSI'].iloc[-1]
            ax2.axhline(y=50, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax2.axhline(y=70, color=self.colors['bearish'], linestyle='--', linewidth=0.5, alpha=0.3)
            ax2.axhline(y=30, color=self.colors['bullish'], linestyle='--', linewidth=0.5, alpha=0.3)
            ax2.barh([0], [latest_rsi], color=self.colors['bullish'] if latest_rsi < 50 else self.colors['bearish'])
            ax2.set_xlim(0, 100)
            ax2.set_yticks([])
            ax2.set_xlabel('RSI Value', fontsize=9)
            ax2.set_title(f'RSI: {latest_rsi:.1f}', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')

        # 3. 거래량 추이
        ax3 = fig.add_subplot(gs[1, 1])
        volumes = data['Volume'].tail(20)
        ax3.bar(range(len(volumes)), volumes, color=self.colors['volume'], alpha=0.6)
        ax3.set_title('Volume (Last 20 Days)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Volume', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 변동성
        ax4 = fig.add_subplot(gs[1, 2])
        if 'Volatility_20' in data.columns:
            vol_data = data['Volatility_20'].tail(20) * 100
            ax4.plot(range(len(vol_data)), vol_data, color=self.colors['neutral'], linewidth=2)
            ax4.fill_between(range(len(vol_data)), vol_data, alpha=0.2, color=self.colors['neutral'])
            ax4.set_title('Volatility Trend (%)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Volatility (%)', fontsize=9)
            ax4.grid(True, alpha=0.3)

        # 5. Claude 분석 결과 (하단)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        sentiment = claude_analysis.get('market_sentiment', 'neutral').upper()
        confidence = claude_analysis.get('confidence_level', 0) * 100
        recommendation = claude_analysis.get('trading_recommendation', {}).get('suggested_action', 'hold').upper()
        risk = claude_analysis.get('risk_assessment', {}).get('risk_level', 'medium').upper()

        summary_text = f"""
AI-POWERED MARKET INTELLIGENCE

Market Sentiment: {sentiment} | Confidence: {confidence:.0f}%
Recommendation: {recommendation} | Risk Level: {risk}

Key Observations:
"""
        if 'key_observations' in claude_analysis:
            for i, obs in enumerate(claude_analysis['key_observations'][:3], 1):
                # Escape dollar signs to prevent LaTeX interpretation
                obs_escaped = obs.replace('$', r'\$')
                summary_text += f"\n{i}. {obs_escaped}"

        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f'{symbol} - Market Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Summary dashboard saved: {save_path}")
        return str(save_path)
