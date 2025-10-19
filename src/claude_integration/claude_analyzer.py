#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Claude Market Analyzer

Claude API를 활용하여 시장 데이터를 분석하고 자연어 인사이트를 제공하는 모듈입니다.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Run: pip install anthropic")


class ClaudeMarketAnalyzer:
    """
    Claude를 활용한 시장 분석 클래스

    기술적 지표와 시장 데이터를 자연어로 해석하고,
    복잡한 패턴을 인식하여 인사이트를 제공합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        cache_enabled: bool = True
    ):
        """
        Args:
            api_key: Anthropic API 키 (없으면 .env 파일에서 로드)
            model: 사용할 Claude 모델
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            cache_enabled: 프롬프트 캐싱 활성화 여부
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install: pip install anthropic")

        # .env 파일에서 설정 로드
        try:
            from src.utils.config import config
            self.api_key = api_key or config.get("ANTHROPIC_API_KEY")
            if not self.api_key or self.api_key == 'your-api-key-here':
                raise ValueError(
                    "ANTHROPIC_API_KEY must be set in .env file or passed as argument.\n"
                    "Copy .env.example to .env and add your API key."
                )
        except ImportError:
            # Fallback: 환경변수에서 직접 로드
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY must be set in environment or .env file.\n"
                    "Copy .env.example to .env and add your API key."
                )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_enabled = cache_enabled

        # 분석 히스토리
        self.analysis_history = []

        # 시스템 프롬프트
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        return """You are an expert financial market analyst with deep knowledge of:
- Technical analysis (RSI, MACD, Bollinger Bands, Moving Averages)
- Market microstructure and price action
- Risk management and portfolio theory
- Market regimes (trending, ranging, volatile)
- Behavioral finance and market psychology

Your role is to:
1. Analyze market data and technical indicators
2. Identify patterns, trends, and anomalies
3. Assess market sentiment and momentum
4. Evaluate risk levels and potential scenarios
5. Provide actionable insights for trading decisions

Always be:
- Objective and data-driven
- Clear about uncertainty and confidence levels
- Specific with numerical thresholds when relevant
- Balanced in considering both bullish and bearish scenarios

Format your analysis in JSON structure when requested."""

    def analyze_market_state(
        self,
        market_data: pd.DataFrame,
        technical_indicators: Dict[str, float],
        current_position: float,
        portfolio_value: float,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        현재 시장 상태를 종합적으로 분석

        Args:
            market_data: 최근 시장 데이터 (OHLCV)
            technical_indicators: 기술적 지표 딕셔너리
            current_position: 현재 포지션 (-1: short, 0: neutral, 1: long)
            portfolio_value: 현재 포트폴리오 가치
            context: 추가 컨텍스트 정보

        Returns:
            분석 결과 딕셔너리
        """
        # 시장 데이터 요약 생성
        market_summary = self._summarize_market_data(market_data, technical_indicators)

        # Claude에게 분석 요청
        prompt = self._create_analysis_prompt(
            market_summary,
            current_position,
            portfolio_value,
            context
        )

        try:
            response = self._call_claude(prompt)
            analysis = self._parse_analysis_response(response)

            # 히스토리에 추가
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'market_summary': market_summary,
                'analysis': analysis
            })

            return analysis

        except Exception as e:
            print(f"Error in market analysis: {e}")
            return self._get_fallback_analysis()

    def _summarize_market_data(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """시장 데이터를 요약 형식으로 변환"""
        recent_data = data.tail(20)  # 최근 20개 데이터

        summary = {
            'price_action': {
                'current_price': float(recent_data['Close'].iloc[-1]),
                'price_change_1d': float((recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-2] - 1) * 100),
                'price_change_5d': float((recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-6] - 1) * 100) if len(recent_data) >= 6 else None,
                'high_20d': float(recent_data['High'].max()),
                'low_20d': float(recent_data['Low'].min()),
                'volatility_20d': float(recent_data['Close'].pct_change().std() * np.sqrt(252) * 100)
            },
            'technical_indicators': {},
            'volume_analysis': {
                'current_volume': float(recent_data['Volume'].iloc[-1]),
                'avg_volume_20d': float(recent_data['Volume'].mean()),
                'volume_trend': 'increasing' if recent_data['Volume'].iloc[-1] > recent_data['Volume'].mean() else 'decreasing'
            }
        }

        # 기술적 지표 추가 (정규화된 값들만)
        for key, value in indicators.items():
            if key.endswith('_norm') and not pd.isna(value):
                clean_key = key.replace('_norm', '')
                summary['technical_indicators'][clean_key] = float(value)

        return summary

    def _create_analysis_prompt(
        self,
        market_summary: Dict[str, Any],
        current_position: float,
        portfolio_value: float,
        context: Optional[str] = None
    ) -> str:
        """분석 요청 프롬프트 생성"""
        position_str = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}.get(current_position, "UNKNOWN")

        prompt = f"""Analyze the following market situation and provide trading insights:

MARKET DATA:
{json.dumps(market_summary, indent=2)}

CURRENT PORTFOLIO:
- Position: {position_str}
- Portfolio Value: ${portfolio_value:,.2f}

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

Please provide a comprehensive analysis in the following JSON format:
{{
    "market_sentiment": "bullish/bearish/neutral",
    "confidence_level": 0.0-1.0,
    "key_observations": [
        "observation 1",
        "observation 2",
        "..."
    ],
    "technical_assessment": {{
        "trend": "uptrend/downtrend/sideways",
        "momentum": "strong/moderate/weak",
        "support_resistance": "description"
    }},
    "risk_assessment": {{
        "risk_level": "low/medium/high",
        "key_risks": ["risk 1", "risk 2"],
        "risk_mitigation": "suggestions"
    }},
    "trading_recommendation": {{
        "suggested_action": "buy/sell/hold/reduce",
        "reasoning": "explanation",
        "entry_criteria": "conditions for action",
        "exit_criteria": "conditions for exit"
    }},
    "confidence_factors": {{
        "supporting_evidence": ["factor 1", "factor 2"],
        "contradicting_evidence": ["factor 1", "factor 2"]
    }}
}}

Provide detailed reasoning based on the technical indicators and price action."""

        return prompt

    def _call_claude(self, prompt: str) -> str:
        """Claude API 호출"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        except anthropic.RateLimitError as e:
            print(f"Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return self._call_claude(prompt)  # Retry

        except Exception as e:
            print(f"Claude API error: {e}")
            raise

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Claude 응답 파싱"""
        try:
            # JSON 블록 추출 시도
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                json_str = response

            analysis = json.loads(json_str)

            # 필수 필드 검증
            required_fields = ['market_sentiment', 'confidence_level', 'trading_recommendation']
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            return analysis

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse Claude response: {e}")
            print(f"Raw response: {response}")
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """파싱 실패 시 기본 분석 반환"""
        return {
            'market_sentiment': 'neutral',
            'confidence_level': 0.0,
            'key_observations': ['Analysis unavailable due to parsing error'],
            'technical_assessment': {
                'trend': 'unknown',
                'momentum': 'unknown',
                'support_resistance': 'unavailable'
            },
            'risk_assessment': {
                'risk_level': 'high',
                'key_risks': ['Unable to assess'],
                'risk_mitigation': 'Recommend caution'
            },
            'trading_recommendation': {
                'suggested_action': 'hold',
                'reasoning': 'Analysis unavailable',
                'entry_criteria': 'N/A',
                'exit_criteria': 'N/A'
            },
            'confidence_factors': {
                'supporting_evidence': [],
                'contradicting_evidence': ['Analysis failed']
            },
            'error': True
        }

    def explain_indicator_divergence(
        self,
        price_data: pd.Series,
        indicator_data: pd.Series,
        indicator_name: str
    ) -> str:
        """
        가격과 지표 간 다이버전스 설명

        Args:
            price_data: 가격 시계열
            indicator_data: 지표 시계열
            indicator_name: 지표 이름

        Returns:
            다이버전스 설명 텍스트
        """
        prompt = f"""Analyze the divergence between price and {indicator_name}:

RECENT PRICE DATA (last 10 periods):
{price_data.tail(10).to_dict()}

RECENT {indicator_name} DATA (last 10 periods):
{indicator_data.tail(10).to_dict()}

Explain:
1. Is there a divergence (bullish or bearish)?
2. What does this divergence indicate?
3. How reliable is this signal?
4. What should traders watch for next?

Provide a clear, concise explanation (3-4 sentences)."""

        try:
            response = self._call_claude(prompt)
            return response.strip()
        except Exception as e:
            return f"Unable to analyze divergence: {str(e)}"

    def interpret_pattern(
        self,
        recent_candles: pd.DataFrame,
        pattern_type: str = "unknown"
    ) -> Dict[str, Any]:
        """
        차트 패턴 해석

        Args:
            recent_candles: 최근 캔들 데이터
            pattern_type: 패턴 유형 (옵션)

        Returns:
            패턴 해석 결과
        """
        candle_summary = recent_candles[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).to_dict('records')

        prompt = f"""Analyze this candlestick pattern:

RECENT CANDLES:
{json.dumps(candle_summary, indent=2)}

{f"SUSPECTED PATTERN: {pattern_type}" if pattern_type != "unknown" else ""}

Identify and explain:
1. What chart pattern(s) do you see?
2. What is the typical significance of this pattern?
3. What are the key price levels to watch?
4. What confirmation signals should we look for?

Format response as JSON:
{{
    "pattern_identified": "pattern name",
    "reliability": "high/medium/low",
    "interpretation": "detailed explanation",
    "key_levels": {{"support": price, "resistance": price}},
    "confirmation_needed": ["signal 1", "signal 2"]
}}"""

        try:
            response = self._call_claude(prompt)
            return self._parse_analysis_response(response)
        except Exception as e:
            return {
                'pattern_identified': 'unknown',
                'reliability': 'low',
                'interpretation': f'Pattern analysis failed: {str(e)}',
                'key_levels': {},
                'confirmation_needed': []
            }

    def generate_trading_narrative(
        self,
        trade_history: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> str:
        """
        트레이딩 히스토리를 자연어 내러티브로 변환

        Args:
            trade_history: 거래 내역
            performance_metrics: 성과 지표

        Returns:
            자연어 설명
        """
        prompt = f"""Create a narrative summary of this trading session:

PERFORMANCE METRICS:
{json.dumps(performance_metrics, indent=2)}

RECENT TRADES (last 5):
{json.dumps(trade_history[-5:], indent=2)}

Generate a professional trading journal entry that:
1. Summarizes overall performance
2. Explains key trading decisions
3. Identifies what worked and what didn't
4. Provides lessons learned
5. Suggests areas for improvement

Keep it concise (5-7 sentences) and actionable."""

        try:
            response = self._call_claude(prompt)
            return response.strip()
        except Exception as e:
            return f"Trading session completed with {len(trade_history)} trades. Unable to generate detailed narrative."

    def get_analysis_summary(self) -> Dict[str, Any]:
        """분석 히스토리 요약"""
        if not self.analysis_history:
            return {'message': 'No analysis history available'}

        return {
            'total_analyses': len(self.analysis_history),
            'time_range': {
                'start': self.analysis_history[0]['timestamp'],
                'end': self.analysis_history[-1]['timestamp']
            },
            'recent_sentiments': [
                analysis['analysis'].get('market_sentiment', 'unknown')
                for analysis in self.analysis_history[-5:]
            ]
        }
