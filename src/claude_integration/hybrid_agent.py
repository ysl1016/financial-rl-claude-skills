#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid RL-Claude Trading Agent

강화학습 에이전트와 Claude의 자연어 분석을 결합한 하이브리드 의사결정 시스템입니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
from datetime import datetime

from src.models.grpo_agent import GRPOAgent
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent
from .claude_analyzer import ClaudeMarketAnalyzer


class HybridRLClaudeAgent:
    """
    RL + Claude 하이브리드 트레이딩 에이전트

    강화학습 모델의 빠른 의사결정과 Claude의 깊은 분석 능력을 결합하여
    더 나은 트레이딩 결정을 내립니다.

    의사결정 프로세스:
    1. RL 에이전트가 초기 행동 선택
    2. Claude가 시장 상황을 분석하고 리스크 평가
    3. 두 의견을 종합하여 최종 결정
    """

    def __init__(
        self,
        rl_agent: Union[GRPOAgent, DeepSeekGRPOAgent],
        claude_analyzer: Optional[ClaudeMarketAnalyzer] = None,
        decision_mode: str = "weighted",
        rl_weight: float = 0.7,
        claude_weight: float = 0.3,
        claude_consultation_frequency: int = 10,
        risk_threshold: float = 0.7,
        enable_claude_override: bool = True
    ):
        """
        Args:
            rl_agent: 강화학습 에이전트 (GRPO 또는 DeepSeek)
            claude_analyzer: Claude 분석기 (없으면 생성)
            decision_mode: 의사결정 모드
                - "weighted": RL과 Claude 의견을 가중 평균
                - "sequential": RL이 제안하고 Claude가 승인/거부
                - "ensemble": 둘 다 동의할 때만 행동
            rl_weight: RL 에이전트 가중치 (weighted 모드)
            claude_weight: Claude 가중치 (weighted 모드)
            claude_consultation_frequency: Claude 상담 주기 (스텝 단위)
            risk_threshold: 리스크 임계값 (이상이면 보수적으로)
            enable_claude_override: Claude가 RL 결정을 덮어쓸 수 있는지
        """
        self.rl_agent = rl_agent
        self.claude_analyzer = claude_analyzer or ClaudeMarketAnalyzer()
        self.decision_mode = decision_mode
        self.rl_weight = rl_weight
        self.claude_weight = claude_weight
        self.claude_consultation_frequency = claude_consultation_frequency
        self.risk_threshold = risk_threshold
        self.enable_claude_override = enable_claude_override

        # 상태 추적
        self.step_count = 0
        self.last_claude_consultation = 0
        self.decision_history = []
        self.claude_analysis_cache = None

        # 통계
        self.stats = {
            'total_decisions': 0,
            'rl_decisions': 0,
            'claude_influenced_decisions': 0,
            'claude_overrides': 0,
            'agreed_decisions': 0,
            'disagreed_decisions': 0
        }

    def select_action(
        self,
        state: np.ndarray,
        market_data: pd.DataFrame,
        technical_indicators: Dict[str, float],
        current_position: float,
        portfolio_value: float,
        history: Optional[np.ndarray] = None,
        force_claude_consultation: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        하이브리드 행동 선택

        Args:
            state: 현재 상태
            market_data: 시장 데이터
            technical_indicators: 기술적 지표
            current_position: 현재 포지션
            portfolio_value: 포트폴리오 가치
            history: 상태 히스토리 (DeepSeek 전용)
            force_claude_consultation: 강제로 Claude 상담

        Returns:
            (action, decision_info) 튜플
        """
        self.step_count += 1
        self.stats['total_decisions'] += 1

        # RL 에이전트의 초기 행동
        if isinstance(self.rl_agent, DeepSeekGRPOAgent) and history is not None:
            rl_action = self.rl_agent.select_action(state, history, deterministic=False)
        else:
            rl_action = self.rl_agent.select_action(state)

        decision_info = {
            'rl_action': rl_action,
            'claude_consulted': False,
            'final_action': rl_action,
            'decision_mode': self.decision_mode,
            'reasoning': 'RL-only decision'
        }

        # Claude 상담 여부 결정
        should_consult_claude = (
            force_claude_consultation or
            (self.step_count - self.last_claude_consultation >= self.claude_consultation_frequency)
        )

        if not should_consult_claude:
            self.stats['rl_decisions'] += 1
            self.decision_history.append(decision_info)
            return rl_action, decision_info

        # Claude에게 시장 분석 요청
        try:
            claude_analysis = self.claude_analyzer.analyze_market_state(
                market_data=market_data,
                technical_indicators=technical_indicators,
                current_position=current_position,
                portfolio_value=portfolio_value
            )

            self.claude_analysis_cache = claude_analysis
            self.last_claude_consultation = self.step_count

            decision_info['claude_consulted'] = True
            decision_info['claude_analysis'] = claude_analysis
            self.stats['claude_influenced_decisions'] += 1

            # Claude 추천 행동 매핑
            claude_action = self._map_claude_recommendation_to_action(
                claude_analysis.get('trading_recommendation', {})
            )
            decision_info['claude_action'] = claude_action

            # 의사결정 모드에 따른 최종 행동 결정
            final_action = self._make_hybrid_decision(
                rl_action=rl_action,
                claude_action=claude_action,
                claude_analysis=claude_analysis,
                decision_info=decision_info
            )

            decision_info['final_action'] = final_action

        except Exception as e:
            print(f"Claude consultation failed: {e}. Using RL action.")
            decision_info['claude_error'] = str(e)
            final_action = rl_action

        self.decision_history.append(decision_info)
        return final_action, decision_info

    def _map_claude_recommendation_to_action(
        self,
        recommendation: Dict[str, Any]
    ) -> int:
        """
        Claude의 자연어 추천을 행동 인덱스로 변환

        Args:
            recommendation: Claude 추천 딕셔너리

        Returns:
            행동 인덱스 (0: hold, 1: buy, 2: sell)
        """
        suggested_action = recommendation.get('suggested_action', 'hold').lower()

        action_map = {
            'buy': 1,
            'long': 1,
            'accumulate': 1,
            'hold': 0,
            'wait': 0,
            'neutral': 0,
            'sell': 2,
            'short': 2,
            'reduce': 2,
            'exit': 2
        }

        return action_map.get(suggested_action, 0)  # 기본값: hold

    def _make_hybrid_decision(
        self,
        rl_action: int,
        claude_action: int,
        claude_analysis: Dict[str, Any],
        decision_info: Dict[str, Any]
    ) -> int:
        """
        RL과 Claude의 의견을 종합하여 최종 결정

        Args:
            rl_action: RL 에이전트 행동
            claude_action: Claude 추천 행동
            claude_analysis: Claude 분석 결과
            decision_info: 의사결정 정보 (업데이트됨)

        Returns:
            최종 행동
        """
        risk_level = claude_analysis.get('risk_assessment', {}).get('risk_level', 'medium')
        confidence = claude_analysis.get('confidence_level', 0.5)

        if self.decision_mode == "weighted":
            return self._weighted_decision(rl_action, claude_action, confidence, decision_info)

        elif self.decision_mode == "sequential":
            return self._sequential_decision(rl_action, claude_action, risk_level, confidence, decision_info)

        elif self.decision_mode == "ensemble":
            return self._ensemble_decision(rl_action, claude_action, decision_info)

        else:
            raise ValueError(f"Unknown decision mode: {self.decision_mode}")

    def _weighted_decision(
        self,
        rl_action: int,
        claude_action: int,
        confidence: float,
        decision_info: Dict[str, Any]
    ) -> int:
        """가중 평균 방식 의사결정"""
        # 행동을 점수로 변환: hold=0, buy=1, sell=-1
        action_to_score = {0: 0, 1: 1, 2: -1}
        rl_score = action_to_score[rl_action]
        claude_score = action_to_score[claude_action]

        # Claude 신뢰도에 따라 가중치 조정
        adjusted_claude_weight = self.claude_weight * confidence
        adjusted_rl_weight = 1 - adjusted_claude_weight

        # 가중 평균
        weighted_score = (rl_score * adjusted_rl_weight +
                         claude_score * adjusted_claude_weight)

        # 점수를 행동으로 변환
        if weighted_score > 0.3:
            final_action = 1  # buy
        elif weighted_score < -0.3:
            final_action = 2  # sell
        else:
            final_action = 0  # hold

        decision_info['reasoning'] = (
            f"Weighted decision: RL={rl_score:.2f} (weight={adjusted_rl_weight:.2f}), "
            f"Claude={claude_score:.2f} (weight={adjusted_claude_weight:.2f}), "
            f"Final score={weighted_score:.2f}"
        )

        if final_action == rl_action == claude_action:
            self.stats['agreed_decisions'] += 1
        else:
            self.stats['disagreed_decisions'] += 1
            if final_action != rl_action:
                self.stats['claude_overrides'] += 1

        return final_action

    def _sequential_decision(
        self,
        rl_action: int,
        claude_action: int,
        risk_level: str,
        confidence: float,
        decision_info: Dict[str, Any]
    ) -> int:
        """순차적 승인 방식 의사결정"""
        # 고위험 상황에서 Claude가 반대하면 hold로 변경
        if risk_level == 'high' and self.enable_claude_override:
            if (rl_action == 1 and claude_action == 2) or \
               (rl_action == 2 and claude_action == 1):
                decision_info['reasoning'] = (
                    f"High risk detected. Claude overrode RL action {rl_action} "
                    f"with safer action 0 (hold)"
                )
                self.stats['claude_overrides'] += 1
                self.stats['disagreed_decisions'] += 1
                return 0  # hold

        # Claude가 높은 신뢰도로 동의하면 해당 행동
        if rl_action == claude_action:
            decision_info['reasoning'] = (
                f"RL and Claude agree on action {rl_action} "
                f"(Claude confidence: {confidence:.2f})"
            )
            self.stats['agreed_decisions'] += 1
            return rl_action

        # 의견 불일치: Claude 신뢰도가 높으면 Claude 따름
        if confidence > 0.7 and self.enable_claude_override:
            decision_info['reasoning'] = (
                f"Disagreement resolved in favor of Claude (confidence: {confidence:.2f}). "
                f"RL suggested {rl_action}, Claude suggested {claude_action}"
            )
            self.stats['claude_overrides'] += 1
            self.stats['disagreed_decisions'] += 1
            return claude_action

        # 기본값: RL 따름
        decision_info['reasoning'] = (
            f"Disagreement resolved in favor of RL. "
            f"Claude confidence {confidence:.2f} not high enough to override"
        )
        self.stats['disagreed_decisions'] += 1
        return rl_action

    def _ensemble_decision(
        self,
        rl_action: int,
        claude_action: int,
        decision_info: Dict[str, Any]
    ) -> int:
        """앙상블 방식: 둘 다 동의할 때만 행동"""
        if rl_action == claude_action:
            decision_info['reasoning'] = (
                f"Ensemble agreement on action {rl_action}"
            )
            self.stats['agreed_decisions'] += 1
            return rl_action
        else:
            decision_info['reasoning'] = (
                f"Ensemble disagreement: RL={rl_action}, Claude={claude_action}. "
                f"Defaulting to hold (0)"
            )
            self.stats['disagreed_decisions'] += 1
            return 0  # hold

    def get_latest_claude_analysis(self) -> Optional[Dict[str, Any]]:
        """가장 최근 Claude 분석 결과 반환"""
        return self.claude_analysis_cache

    def get_decision_statistics(self) -> Dict[str, Any]:
        """의사결정 통계 반환"""
        stats = self.stats.copy()

        if stats['total_decisions'] > 0:
            stats['claude_influence_rate'] = (
                stats['claude_influenced_decisions'] / stats['total_decisions']
            )
            stats['claude_override_rate'] = (
                stats['claude_overrides'] / stats['total_decisions']
            )
            stats['agreement_rate'] = (
                stats['agreed_decisions'] /
                max(stats['claude_influenced_decisions'], 1)
            )

        return stats

    def get_decision_history(
        self,
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """의사결정 히스토리 반환"""
        if last_n is None:
            return self.decision_history
        return self.decision_history[-last_n:]

    def explain_last_decision(self) -> str:
        """마지막 의사결정에 대한 자연어 설명"""
        if not self.decision_history:
            return "No decisions made yet."

        last_decision = self.decision_history[-1]

        explanation = f"Decision #{len(self.decision_history)}:\n"
        explanation += f"RL Agent suggested: Action {last_decision['rl_action']}\n"

        if last_decision['claude_consulted']:
            claude_rec = last_decision.get('claude_analysis', {}).get('trading_recommendation', {})
            explanation += f"Claude suggested: {claude_rec.get('suggested_action', 'N/A')}\n"
            explanation += f"Reasoning: {last_decision['reasoning']}\n"
            explanation += f"Final action: {last_decision['final_action']}\n"
        else:
            explanation += "Claude was not consulted (using RL-only decision)\n"
            explanation += f"Final action: {last_decision['final_action']}\n"

        return explanation

    def reset_statistics(self):
        """통계 초기화"""
        self.stats = {
            'total_decisions': 0,
            'rl_decisions': 0,
            'claude_influenced_decisions': 0,
            'claude_overrides': 0,
            'agreed_decisions': 0,
            'disagreed_decisions': 0
        }
        self.decision_history = []
        self.step_count = 0
        self.last_claude_consultation = 0

    def save_decision_log(self, filepath: str):
        """의사결정 로그 저장"""
        import json

        log_data = {
            'statistics': self.get_decision_statistics(),
            'decision_history': self.decision_history,
            'configuration': {
                'decision_mode': self.decision_mode,
                'rl_weight': self.rl_weight,
                'claude_weight': self.claude_weight,
                'consultation_frequency': self.claude_consultation_frequency,
                'risk_threshold': self.risk_threshold,
                'claude_override_enabled': self.enable_claude_override
            }
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        print(f"Decision log saved to {filepath}")
