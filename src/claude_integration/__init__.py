"""
Claude Skills Integration Module

이 모듈은 Anthropic Claude API를 활용하여 강화학습 기반 트레이딩 모델의
의사결정 능력을 향상시킵니다.

주요 기능:
- 시장 상황 자연어 분석
- 복잡한 패턴 해석
- 리스크 평가 및 설명
- 트레이딩 전략 추천
- 시장 레짐 해석
"""

from .claude_analyzer import ClaudeMarketAnalyzer
from .hybrid_agent import HybridRLClaudeAgent
from .risk_assessor import ClaudeRiskAssessor
from .regime_interpreter import ClaudeRegimeInterpreter

__all__ = [
    'ClaudeMarketAnalyzer',
    'HybridRLClaudeAgent',
    'ClaudeRiskAssessor',
    'ClaudeRegimeInterpreter'
]
