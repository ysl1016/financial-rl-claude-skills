#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid RL-Claude Trading Example

강화학습과 Claude를 결합한 하이브리드 트레이딩 시스템 예시입니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 기존 모듈
from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent

# Claude 통합 모듈
from src.claude_integration.claude_analyzer import ClaudeMarketAnalyzer
from src.claude_integration.hybrid_agent import HybridRLClaudeAgent
from src.claude_integration.risk_assessor import ClaudeRiskAssessor
from src.claude_integration.regime_interpreter import ClaudeRegimeInterpreter


def train_hybrid_agent(
    symbol: str = 'SPY',
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
    num_episodes: int = 10,
    decision_mode: str = 'weighted',
    claude_frequency: int = 20,
    save_logs: bool = True
):
    """
    하이브리드 에이전트 학습

    Args:
        symbol: 주식 심볼
        start_date: 시작 날짜
        end_date: 종료 날짜
        num_episodes: 학습 에피소드 수
        decision_mode: 의사결정 모드 (weighted/sequential/ensemble)
        claude_frequency: Claude 상담 빈도
        save_logs: 로그 저장 여부
    """
    print("=" * 80)
    print("Hybrid RL-Claude Trading System")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Decision Mode: {decision_mode}")
    print(f"Claude Consultation Frequency: every {claude_frequency} steps")
    print("=" * 80)

    # 1. 데이터 준비
    print("\n[1/6] Processing market data...")
    try:
        data = process_data(symbol, start_date=start_date, end_date=end_date)
        print(f"✓ Loaded {len(data)} data points")
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return

    # 학습/테스트 분할
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    print(f"  Train: {len(train_data)} points, Test: {len(test_data)} points")

    # 2. 환경 생성
    print("\n[2/6] Creating trading environment...")
    env = TradingEnv(
        data=train_data,
        initial_capital=100000,
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=1.0,
        stop_loss_pct=0.02
    )
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")

    # 3. RL 에이전트 생성
    print("\n[3/6] Initializing RL agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    rl_agent = GRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        reward_scale=1.0,
        penalty_scale=0.5
    )
    print(f"✓ GRPO agent initialized")

    # 4. Claude 분석기 생성
    print("\n[4/6] Initializing Claude integration...")
    try:
        claude_analyzer = ClaudeMarketAnalyzer()
        risk_assessor = ClaudeRiskAssessor()
        print("✓ Claude analyzer initialized")
    except Exception as e:
        print(f"✗ Claude initialization failed: {e}")
        print("  Tip: Set ANTHROPIC_API_KEY environment variable")
        return

    # 5. 하이브리드 에이전트 생성
    print("\n[5/6] Creating hybrid agent...")
    hybrid_agent = HybridRLClaudeAgent(
        rl_agent=rl_agent,
        claude_analyzer=claude_analyzer,
        decision_mode=decision_mode,
        rl_weight=0.7,
        claude_weight=0.3,
        claude_consultation_frequency=claude_frequency,
        enable_claude_override=True
    )
    print(f"✓ Hybrid agent created")

    # 6. 학습 시작
    print("\n[6/6] Starting hybrid training...\n")
    print("-" * 80)

    training_metrics = {
        'episode_rewards': [],
        'portfolio_values': [],
        'claude_consultations': [],
        'decision_agreements': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        claude_consultations_this_episode = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            # 기술적 지표 추출
            current_row = train_data.iloc[env.index]
            technical_indicators = {
                col: float(current_row[col])
                for col in train_data.columns
                if col.endswith('_norm') and not pd.isna(current_row[col])
            }

            # 하이브리드 행동 선택
            action, decision_info = hybrid_agent.select_action(
                state=state,
                market_data=train_data.iloc[max(0, env.index-50):env.index+1],
                technical_indicators=technical_indicators,
                current_position=env.position,
                portfolio_value=env.portfolio_values[-1],
                force_claude_consultation=(step_count % 50 == 0 and step_count > 0)
            )

            # Claude 상담 카운트
            if decision_info['claude_consulted']:
                claude_consultations_this_episode += 1

                # 주기적으로 의사결정 설명 출력
                if claude_consultations_this_episode <= 2:
                    print(f"\n  Step {step_count}: {hybrid_agent.explain_last_decision()}")

            # 환경 스텝
            next_state, reward, done, info = env.step(action)

            # 전환 저장
            rl_agent.store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            step_count += 1

            # 주기적으로 업데이트
            if step_count % 100 == 0 and step_count > 0:
                update_info = rl_agent.update()
                if update_info:
                    print(f"  Step {step_count}: Policy Loss={update_info['policy_loss']:.4f}, "
                          f"Q Loss={update_info['q_loss']:.4f}, "
                          f"Mean Reward={update_info['mean_reward']:.4f}")

        # 에피소드 완료
        final_value = env.portfolio_values[-1]
        total_return = (final_value / env.initial_capital - 1) * 100

        print(f"\n  Episode Results:")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Final Portfolio: ${final_value:,.2f}")
        print(f"    Total Return: {total_return:.2f}%")
        print(f"    Claude Consultations: {claude_consultations_this_episode}")
        print(f"    Total Trades: {len(env.trades)}")

        # 메트릭 저장
        training_metrics['episode_rewards'].append(episode_reward)
        training_metrics['portfolio_values'].append(final_value)
        training_metrics['claude_consultations'].append(claude_consultations_this_episode)

        # 주기적으로 리스크 평가
        if (episode + 1) % 3 == 0:
            print(f"\n  Generating risk report...")
            try:
                performance_metrics = {
                    'total_return': total_return / 100,
                    'sharpe_ratio': np.mean(env.daily_returns) / (np.std(env.daily_returns) + 1e-9) * np.sqrt(252),
                    'max_drawdown': max((np.maximum.accumulate(env.portfolio_values) - env.portfolio_values) / np.maximum.accumulate(env.portfolio_values))
                }

                risk_report = risk_assessor.generate_risk_report(
                    portfolio_metrics=performance_metrics,
                    trade_history=[{'action': t[0], 'price': t[2]} for t in env.trades[-10:]],
                    current_positions={'SPY': env.position}
                )

                print(f"\n  Risk Report:\n{risk_report}")

            except Exception as e:
                print(f"  Risk assessment skipped: {e}")

        print("-" * 80)

    # 7. 학습 완료 - 통계 출력
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    decision_stats = hybrid_agent.get_decision_statistics()
    print(f"\nDecision Statistics:")
    print(f"  Total Decisions: {decision_stats['total_decisions']}")
    print(f"  Claude Influenced: {decision_stats['claude_influenced_decisions']} "
          f"({decision_stats.get('claude_influence_rate', 0)*100:.1f}%)")
    print(f"  Claude Overrides: {decision_stats['claude_overrides']} "
          f"({decision_stats.get('claude_override_rate', 0)*100:.1f}%)")
    print(f"  Agreement Rate: {decision_stats.get('agreement_rate', 0)*100:.1f}%")

    # 8. 시각화
    print("\n[7/7] Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 포트폴리오 가치
    axes[0, 0].plot(training_metrics['portfolio_values'])
    axes[0, 0].axhline(y=100000, color='r', linestyle='--', label='Initial Capital')
    axes[0, 0].set_title('Portfolio Value per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 에피소드 보상
    axes[0, 1].plot(training_metrics['episode_rewards'])
    axes[0, 1].set_title('Episode Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True)

    # Claude 상담 빈도
    axes[1, 0].bar(range(len(training_metrics['claude_consultations'])),
                    training_metrics['claude_consultations'])
    axes[1, 0].set_title('Claude Consultations per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Consultations')
    axes[1, 0].grid(True)

    # 의사결정 통계
    stats_labels = ['Total', 'Claude\nInfluenced', 'Overrides', 'Agreed', 'Disagreed']
    stats_values = [
        decision_stats['total_decisions'],
        decision_stats['claude_influenced_decisions'],
        decision_stats['claude_overrides'],
        decision_stats['agreed_decisions'],
        decision_stats['disagreed_decisions']
    ]
    axes[1, 1].bar(stats_labels, stats_values)
    axes[1, 1].set_title('Decision Statistics')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'hybrid_training_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    print(f"✓ Visualization saved")

    # 9. 로그 저장
    if save_logs:
        log_filename = f'hybrid_decisions_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        hybrid_agent.save_decision_log(log_filename)

    # 10. 모델 저장
    model_filename = f'hybrid_model_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    rl_agent.save(model_filename)
    print(f"✓ Model saved: {model_filename}")

    print("\n" + "=" * 80)
    print("Hybrid training completed successfully!")
    print("=" * 80)

    return hybrid_agent, training_metrics


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid RL-Claude Trading')
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--mode', type=str, default='weighted',
                        choices=['weighted', 'sequential', 'ensemble'],
                        help='Decision mode')
    parser.add_argument('--frequency', type=int, default=20,
                        help='Claude consultation frequency')

    args = parser.parse_args()

    # 실행
    hybrid_agent, metrics = train_hybrid_agent(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        num_episodes=args.episodes,
        decision_mode=args.mode,
        claude_frequency=args.frequency
    )

    plt.show()


if __name__ == '__main__':
    main()
