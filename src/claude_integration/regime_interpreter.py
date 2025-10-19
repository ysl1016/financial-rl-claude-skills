#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Claude Regime Interpreter

시장 레짐 감지 결과를 Claude가 해석하고 전략을 제안하는 모듈입니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeRegimeInterpreter:
    """
    시장 레짐 해석기

    강화학습 모델이 감지한 시장 레짐을 Claude가 해석하고,
    각 레짐에 맞는 트레이딩 전략을 제안합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Args:
            api_key: Anthropic API 키
            model: Claude 모델
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

        # 레짐 해석 히스토리
        self.regime_history = []

        # 레짐별 전략 캐시
        self.regime_strategies = {}

    def interpret_regime(
        self,
        regime_id: int,
        regime_features: Dict[str, float],
        regime_stability: float,
        market_data: pd.DataFrame,
        historical_regimes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        감지된 시장 레짐 해석

        Args:
            regime_id: 레짐 ID (0, 1, 2, 3 등)
            regime_features: 레짐 특성 (volatility, trend, volume 등)
            regime_stability: 레짐 안정도 (0-1)
            market_data: 시장 데이터
            historical_regimes: 과거 레짐 시퀀스

        Returns:
            레짐 해석 결과
        """
        # 시장 컨텍스트 생성
        market_context = self._create_market_context(market_data)

        prompt = f"""Interpret this market regime:

REGIME INFORMATION:
- Regime ID: {regime_id}
- Stability: {regime_stability:.2f} (1.0 = very stable, 0.0 = transitioning)

REGIME CHARACTERISTICS:
{json.dumps(regime_features, indent=2)}

CURRENT MARKET CONTEXT:
{json.dumps(market_context, indent=2)}

{f"RECENT REGIME SEQUENCE: {historical_regimes[-10:]}" if historical_regimes else ""}

Provide a comprehensive interpretation in JSON format:
{{
    "regime_name": "descriptive name (e.g., 'High Volatility Bull Market')",
    "regime_type": "trending_up/trending_down/ranging/volatile/transitioning",
    "key_characteristics": [
        "characteristic 1",
        "characteristic 2",
        "..."
    ],
    "market_psychology": "description of likely market sentiment and behavior",
    "typical_duration": "how long this regime usually lasts",
    "stability_assessment": {{
        "is_stable": true/false,
        "confidence": 0.0-1.0,
        "transition_risk": "low/medium/high",
        "likely_next_regime": "prediction"
    }},
    "trading_strategy": {{
        "recommended_approach": "description",
        "position_sizing": "conservative/moderate/aggressive",
        "holding_period": "short-term/medium-term/long-term",
        "entry_signals": ["signal 1", "signal 2"],
        "exit_signals": ["signal 1", "signal 2"],
        "risk_management": "specific guidelines"
    }},
    "indicators_to_watch": [
        {{
            "indicator": "name",
            "reason": "why it's important in this regime",
            "threshold": "key levels"
        }}
    ],
    "historical_performance": "how strategies typically perform in this regime",
    "warnings": ["warning 1", "warning 2"]
}}

Consider:
- What market conditions define this regime?
- How should traders adapt their strategy?
- What are the key risk factors?
- When is this regime likely to end?
"""

        try:
            response = self._call_claude(prompt)
            interpretation = self._parse_response(response)

            # 히스토리에 추가
            self.regime_history.append({
                'timestamp': datetime.now().isoformat(),
                'regime_id': regime_id,
                'stability': regime_stability,
                'interpretation': interpretation
            })

            # 전략 캐시 업데이트
            self.regime_strategies[regime_id] = interpretation.get('trading_strategy', {})

            return interpretation

        except Exception as e:
            print(f"Regime interpretation error: {e}")
            return self._get_fallback_interpretation(regime_id)

    def detect_regime_transition(
        self,
        current_regime: int,
        previous_regime: int,
        transition_probability: float,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        레짐 전환 감지 및 대응 전략 제시

        Args:
            current_regime: 현재 레짐 ID
            previous_regime: 이전 레짐 ID
            transition_probability: 전환 확률
            market_data: 시장 데이터

        Returns:
            레짐 전환 분석
        """
        recent_volatility = market_data['Close'].pct_change().tail(20).std() * np.sqrt(252)

        prompt = f"""Analyze this market regime transition:

REGIME TRANSITION:
- From Regime: {previous_regime}
- To Regime: {current_regime}
- Transition Probability: {transition_probability:.2%}
- Recent Volatility: {recent_volatility*100:.2f}%

RECENT PRICE ACTION:
- Last 5 closes: {market_data['Close'].tail(5).tolist()}
- Price change (5d): {((market_data['Close'].iloc[-1]/market_data['Close'].iloc[-6])-1)*100:.2f}%

Analyze in JSON format:
{{
    "transition_type": "gradual/abrupt/false_signal",
    "transition_confidence": 0.0-1.0,
    "transition_drivers": [
        "driver 1: explanation",
        "driver 2: explanation"
    ],
    "implications": {{
        "market_direction": "description",
        "volatility_expectation": "increasing/stable/decreasing",
        "timeframe": "how long until new regime stabilizes"
    }},
    "immediate_actions": [
        {{
            "action": "specific action",
            "priority": "high/medium/low",
            "reasoning": "why this is important",
            "timing": "when to execute"
        }}
    ],
    "position_adjustments": {{
        "current_positions": "hold/reduce/exit",
        "new_positions": "wait/cautious/aggressive",
        "hedging": "recommended hedging strategy"
    }},
    "monitoring_checklist": [
        "what to watch 1",
        "what to watch 2"
    ],
    "false_signal_risk": {{
        "probability": 0.0-1.0,
        "indicators": "what would confirm false signal",
        "contingency": "what to do if false signal"
    }}
}}

Focus on:
- Is this a real regime change or noise?
- How should traders reposition?
- What confirms the transition?
- What invalidates it?
"""

        try:
            response = self._call_claude(prompt)
            analysis = self._parse_response(response)
            return analysis

        except Exception as e:
            print(f"Transition analysis error: {e}")
            return {
                'transition_type': 'unknown',
                'transition_confidence': 0.5,
                'immediate_actions': [{'action': 'Monitor closely', 'priority': 'high'}],
                'error': True
            }

    def recommend_strategy_adaptation(
        self,
        current_regime: int,
        regime_interpretation: Dict[str, Any],
        current_strategy_performance: Dict[str, float],
        agent_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        현재 레짐에 맞는 전략 조정 제안

        Args:
            current_regime: 현재 레짐 ID
            regime_interpretation: 레짐 해석 결과
            current_strategy_performance: 현재 전략 성과
            agent_params: 현재 에이전트 파라미터

        Returns:
            전략 조정 제안
        """
        prompt = f"""Recommend strategy adaptations for this market regime:

CURRENT REGIME: {current_regime}
Regime Type: {regime_interpretation.get('regime_type', 'unknown')}
Regime Name: {regime_interpretation.get('regime_name', 'unknown')}

CURRENT STRATEGY PERFORMANCE:
{json.dumps(current_strategy_performance, indent=2)}

CURRENT AGENT PARAMETERS:
{json.dumps(agent_params, indent=2)}

Provide specific recommendations in JSON format:
{{
    "adaptation_priority": "low/medium/high/urgent",
    "parameter_adjustments": [
        {{
            "parameter": "name",
            "current_value": value,
            "recommended_value": value,
            "reasoning": "why this adjustment helps",
            "expected_impact": "description"
        }}
    ],
    "strategy_modifications": [
        {{
            "aspect": "what to change",
            "modification": "how to change it",
            "rationale": "why this helps in current regime"
        }}
    ],
    "risk_management_changes": {{
        "stop_loss_adjustment": "tighter/unchanged/wider",
        "position_sizing": "reduce/maintain/increase",
        "diversification": "recommendation"
    }},
    "performance_expectations": {{
        "win_rate_change": "increase/decrease/stable",
        "profit_factor_change": "increase/decrease/stable",
        "drawdown_risk": "lower/same/higher"
    }},
    "implementation_plan": [
        {{
            "step": "action",
            "timing": "when",
            "validation": "how to verify success"
        }}
    ]
}}

Consider:
- What works best in this regime type?
- What parameters need adjustment?
- How to optimize risk-reward?
"""

        try:
            response = self._call_claude(prompt)
            recommendations = self._parse_response(response)
            return recommendations

        except Exception as e:
            print(f"Strategy adaptation error: {e}")
            return {
                'adaptation_priority': 'medium',
                'parameter_adjustments': [],
                'error': True
            }

    def explain_regime_sequence(
        self,
        regime_sequence: List[int],
        timestamps: List[str],
        performance_by_regime: Dict[int, Dict[str, float]]
    ) -> str:
        """
        레짐 시퀀스를 자연어로 설명

        Args:
            regime_sequence: 레짐 시퀀스
            timestamps: 타임스탬프 리스트
            performance_by_regime: 레짐별 성과

        Returns:
            자연어 설명
        """
        prompt = f"""Create a narrative explaining this market regime progression:

REGIME SEQUENCE:
{list(zip(timestamps[-20:], regime_sequence[-20:]))}

PERFORMANCE BY REGIME:
{json.dumps(performance_by_regime, indent=2)}

Write a concise narrative (5-7 sentences) that:
1. Describes the regime progression
2. Explains what market conditions drove transitions
3. Analyzes how the trading strategy performed in each regime
4. Identifies patterns or cycles
5. Provides insights for future trading

Use professional language suitable for a trading journal."""

        try:
            response = self._call_claude(prompt)
            return response.strip()

        except Exception as e:
            return f"Regime sequence analysis unavailable: {str(e)}"

    def _create_market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """시장 컨텍스트 생성"""
        recent = market_data.tail(20)

        return {
            'price_trend': 'up' if recent['Close'].iloc[-1] > recent['Close'].iloc[0] else 'down',
            'volatility': float(recent['Close'].pct_change().std() * np.sqrt(252)),
            'volume_trend': 'increasing' if recent['Volume'].iloc[-1] > recent['Volume'].mean() else 'decreasing',
            'range_20d': float((recent['High'].max() - recent['Low'].min()) / recent['Close'].mean()),
            'recent_performance': float((recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1) * 100)
        }

    def _call_claude(self, prompt: str) -> str:
        """Claude API 호출"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.6,
                system="You are a quantitative market analyst specializing in regime-based trading strategies. You understand market cycles, regime transitions, and adaptive trading systems.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        except Exception as e:
            print(f"Claude API error: {e}")
            raise

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """응답 파싱"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                json_str = response

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"Parse error: {e}")
            return {'error': True, 'raw_response': response}

    def _get_fallback_interpretation(self, regime_id: int) -> Dict[str, Any]:
        """기본 해석"""
        return {
            'regime_name': f'Regime {regime_id}',
            'regime_type': 'unknown',
            'key_characteristics': ['Unable to interpret'],
            'trading_strategy': {
                'recommended_approach': 'Proceed with caution',
                'position_sizing': 'conservative'
            },
            'error': True
        }

    def get_regime_summary(self) -> Dict[str, Any]:
        """레짐 히스토리 요약"""
        if not self.regime_history:
            return {'message': 'No regime history'}

        regime_counts = {}
        for entry in self.regime_history:
            regime_id = entry['regime_id']
            regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1

        return {
            'total_observations': len(self.regime_history),
            'unique_regimes': len(regime_counts),
            'regime_distribution': regime_counts,
            'most_common_regime': max(regime_counts, key=regime_counts.get)
        }
