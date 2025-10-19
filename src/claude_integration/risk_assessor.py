#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Claude Risk Assessor

Claude를 활용한 고급 리스크 평가 및 포트폴리오 관리 모듈입니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeRiskAssessor:
    """
    Claude 기반 리스크 평가 시스템

    포트폴리오 리스크를 다차원적으로 분석하고,
    복잡한 시나리오와 상호작용을 Claude의 추론 능력으로 평가합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        risk_tolerance: str = "moderate"
    ):
        """
        Args:
            api_key: Anthropic API 키
            model: Claude 모델
            risk_tolerance: 리스크 허용도 (conservative/moderate/aggressive)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required")

        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.risk_tolerance = risk_tolerance

        self.risk_history = []

    def assess_position_risk(
        self,
        current_position: float,
        portfolio_value: float,
        max_drawdown: float,
        volatility: float,
        market_data: pd.DataFrame,
        technical_indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        현재 포지션의 리스크 평가

        Args:
            current_position: 현재 포지션 (-1 to 1)
            portfolio_value: 포트폴리오 가치
            max_drawdown: 최대 손실률
            volatility: 변동성
            market_data: 시장 데이터
            technical_indicators: 기술적 지표

        Returns:
            리스크 평가 결과
        """
        prompt = f"""Assess the risk of this trading position:

PORTFOLIO STATUS:
- Current Position: {current_position:.2f} (-1=full short, 0=neutral, 1=full long)
- Portfolio Value: ${portfolio_value:,.2f}
- Maximum Drawdown: {max_drawdown*100:.2f}%
- Volatility (annualized): {volatility*100:.2f}%
- Risk Tolerance: {self.risk_tolerance}

RECENT MARKET DATA:
- Current Price: ${market_data['Close'].iloc[-1]:.2f}
- Price Change (1d): {((market_data['Close'].iloc[-1]/market_data['Close'].iloc[-2])-1)*100:.2f}%
- High (20d): ${market_data['High'].tail(20).max():.2f}
- Low (20d): ${market_data['Low'].tail(20).min():.2f}

TECHNICAL INDICATORS:
{json.dumps({k: float(v) for k, v in technical_indicators.items() if not pd.isna(v)}, indent=2)}

Provide a comprehensive risk assessment in JSON format:
{{
    "overall_risk_score": 0.0-1.0,
    "risk_level": "low/medium/high/critical",
    "risk_factors": [
        {{
            "factor": "factor name",
            "severity": "low/medium/high",
            "description": "explanation",
            "mitigation": "how to reduce this risk"
        }}
    ],
    "position_sizing_recommendation": {{
        "current_size_appropriate": true/false,
        "recommended_size": -1.0 to 1.0,
        "reasoning": "explanation"
    }},
    "stop_loss_recommendation": {{
        "should_set_stop": true/false,
        "stop_level": price_level,
        "reasoning": "explanation"
    }},
    "scenario_analysis": {{
        "best_case": "description and probability",
        "base_case": "description and probability",
        "worst_case": "description and probability"
    }},
    "immediate_actions": ["action 1", "action 2"],
    "monitoring_points": ["what to watch 1", "what to watch 2"]
}}

Be specific and actionable. Consider:
1. Market volatility and trend strength
2. Position concentration risk
3. Drawdown proximity to limits
4. Technical indicator divergences
5. Correlation with broader market
"""

        try:
            response = self._call_claude(prompt)
            assessment = self._parse_risk_response(response)

            # 히스토리 저장
            self.risk_history.append({
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'position': current_position,
                'assessment': assessment
            })

            return assessment

        except Exception as e:
            print(f"Risk assessment error: {e}")
            return self._get_fallback_risk_assessment()

    def evaluate_trade_risk(
        self,
        proposed_action: int,
        current_position: float,
        entry_price: float,
        current_price: float,
        position_duration: int,
        recent_returns: List[float]
    ) -> Dict[str, Any]:
        """
        특정 거래의 리스크 평가

        Args:
            proposed_action: 제안된 행동 (0:hold, 1:buy, 2:sell)
            current_position: 현재 포지션
            entry_price: 진입 가격
            current_price: 현재 가격
            position_duration: 포지션 보유 기간
            recent_returns: 최근 수익률 리스트

        Returns:
            거래 리스크 평가
        """
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_names.get(proposed_action, "UNKNOWN")

        # 현재 손익
        if current_position != 0:
            pnl_pct = ((current_price / entry_price) - 1) * current_position * 100
        else:
            pnl_pct = 0

        prompt = f"""Evaluate the risk of this proposed trade:

PROPOSED ACTION: {action_name}

CURRENT POSITION:
- Position: {current_position:.2f}
- Entry Price: ${entry_price:.2f}
- Current Price: ${current_price:.2f}
- Unrealized P&L: {pnl_pct:.2f}%
- Position Duration: {position_duration} periods

RECENT PERFORMANCE:
- Recent Returns: {recent_returns[-10:] if len(recent_returns) > 10 else recent_returns}
- Win Rate (last 20): {sum(1 for r in recent_returns[-20:] if r > 0) / min(len(recent_returns), 20) * 100:.1f}%
- Average Return: {np.mean(recent_returns)*100:.2f}%

Analyze this trade in JSON format:
{{
    "trade_risk_score": 0.0-1.0,
    "risk_level": "low/medium/high",
    "key_concerns": ["concern 1", "concern 2"],
    "approval_status": "approved/cautioned/rejected",
    "reasoning": "detailed explanation",
    "alternative_suggestions": ["suggestion 1", "suggestion 2"],
    "timing_assessment": "good/fair/poor timing",
    "expected_outcome": {{
        "probability_success": 0.0-1.0,
        "potential_gain": "estimate",
        "potential_loss": "estimate",
        "risk_reward_ratio": ratio
    }}
}}

Consider:
- Is this the right time for this action?
- Does it align with the current trend?
- Is the position being held too long or too short?
- What's the probability of success vs failure?
"""

        try:
            response = self._call_claude(prompt)
            evaluation = self._parse_risk_response(response)
            return evaluation

        except Exception as e:
            print(f"Trade risk evaluation error: {e}")
            return {
                'trade_risk_score': 0.5,
                'risk_level': 'medium',
                'approval_status': 'cautioned',
                'reasoning': f'Evaluation failed: {str(e)}',
                'error': True
            }

    def analyze_portfolio_correlation_risk(
        self,
        assets_data: Dict[str, pd.DataFrame],
        positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        다중 자산 포트폴리오의 상관관계 리스크 분석

        Args:
            assets_data: 자산별 시장 데이터
            positions: 자산별 포지션

        Returns:
            상관관계 리스크 분석
        """
        # 상관계수 계산
        returns_dict = {}
        for asset, data in assets_data.items():
            returns_dict[asset] = data['Close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = returns_df.corr().to_dict()

        prompt = f"""Analyze portfolio correlation risk:

PORTFOLIO POSITIONS:
{json.dumps(positions, indent=2)}

CORRELATION MATRIX:
{json.dumps(correlation_matrix, indent=2)}

Assess correlation risk in JSON format:
{{
    "overall_diversification_score": 0.0-1.0,
    "concentration_risk": "low/medium/high",
    "correlation_concerns": [
        {{
            "asset_pair": "ASSET1-ASSET2",
            "correlation": value,
            "risk_level": "low/medium/high",
            "explanation": "why this is concerning"
        }}
    ],
    "hedging_opportunities": [
        {{
            "strategy": "description",
            "assets_involved": ["ASSET1", "ASSET2"],
            "expected_benefit": "description"
        }}
    ],
    "rebalancing_suggestions": [
        {{
            "asset": "ASSET",
            "current_position": value,
            "suggested_position": value,
            "reasoning": "explanation"
        }}
    ]
}}

Look for:
- Over-concentration in correlated assets
- Diversification opportunities
- Natural hedges
- Systemic risk exposure
"""

        try:
            response = self._call_claude(prompt)
            analysis = self._parse_risk_response(response)
            return analysis

        except Exception as e:
            print(f"Correlation risk analysis error: {e}")
            return {
                'overall_diversification_score': 0.5,
                'concentration_risk': 'medium',
                'error': True
            }

    def generate_risk_report(
        self,
        portfolio_metrics: Dict[str, float],
        trade_history: List[Dict[str, Any]],
        current_positions: Dict[str, float]
    ) -> str:
        """
        종합 리스크 리포트 생성

        Args:
            portfolio_metrics: 포트폴리오 성과 지표
            trade_history: 거래 내역
            current_positions: 현재 포지션

        Returns:
            자연어 리스크 리포트
        """
        prompt = f"""Generate a comprehensive risk management report:

PORTFOLIO METRICS:
{json.dumps(portfolio_metrics, indent=2)}

CURRENT POSITIONS:
{json.dumps(current_positions, indent=2)}

RECENT TRADES (last 5):
{json.dumps(trade_history[-5:] if trade_history else [], indent=2)}

Create a professional risk report covering:

1. EXECUTIVE SUMMARY
   - Overall portfolio health
   - Key risk metrics
   - Risk tolerance alignment

2. RISK ANALYSIS
   - Market risk
   - Concentration risk
   - Liquidity risk
   - Operational risk

3. PERFORMANCE REVIEW
   - Risk-adjusted returns
   - Drawdown analysis
   - Volatility trends

4. RECOMMENDATIONS
   - Immediate actions needed
   - Strategic adjustments
   - Risk mitigation strategies

5. MONITORING PLAN
   - Key metrics to watch
   - Alert thresholds
   - Review frequency

Keep it concise but comprehensive (10-15 sentences total).
Use professional language suitable for a trading journal.
"""

        try:
            response = self._call_claude(prompt)
            return response.strip()

        except Exception as e:
            return f"Risk report generation failed: {str(e)}"

    def _call_claude(self, prompt: str) -> str:
        """Claude API 호출"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.5,  # 리스크 평가는 낮은 온도로
                system="You are a professional risk management analyst with expertise in quantitative finance, portfolio theory, and behavioral finance. Provide objective, data-driven risk assessments.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        except Exception as e:
            print(f"Claude API error: {e}")
            raise

    def _parse_risk_response(self, response: str) -> Dict[str, Any]:
        """Claude 응답 파싱"""
        try:
            # JSON 추출
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
            print(f"Failed to parse risk response: {e}")
            return self._get_fallback_risk_assessment()

    def _get_fallback_risk_assessment(self) -> Dict[str, Any]:
        """기본 리스크 평가"""
        return {
            'overall_risk_score': 0.5,
            'risk_level': 'medium',
            'risk_factors': [
                {
                    'factor': 'Unable to assess',
                    'severity': 'medium',
                    'description': 'Risk assessment unavailable',
                    'mitigation': 'Proceed with caution'
                }
            ],
            'immediate_actions': ['Monitor closely'],
            'error': True
        }

    def get_risk_history_summary(self) -> Dict[str, Any]:
        """리스크 히스토리 요약"""
        if not self.risk_history:
            return {'message': 'No risk history available'}

        risk_scores = [
            entry['assessment'].get('overall_risk_score', 0.5)
            for entry in self.risk_history
        ]

        return {
            'total_assessments': len(self.risk_history),
            'average_risk_score': np.mean(risk_scores),
            'max_risk_score': np.max(risk_scores),
            'min_risk_score': np.min(risk_scores),
            'risk_trend': 'increasing' if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else 'stable/decreasing'
        }
