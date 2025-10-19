import sys
import os
import unittest
import numpy as np
import pandas as pd
import torch
import json
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경에서 테스트를 위한 백엔드 설정

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent

class TestRegression(unittest.TestCase):
    """모델 성능 회귀 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 환경 설정"""
        # 랜덤 시드 설정
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 테스트용 데이터 생성
        dates = pd.date_range(start='2020-01-01', periods=200)
        
        # 가격 데이터 생성
        close_prices = np.random.normal(100, 5, size=200).cumsum() + 1000
        high_prices = close_prices * (1 + np.random.uniform(0, 0.05, size=200))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.05, size=200))
        open_prices = close_prices * (1 + np.random.normal(0, 0.01, size=200))
        volumes = np.random.randint(1000, 10000, size=200)
        
        # 데이터프레임 생성
        self.test_data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        # 필요한 속성 추가 (process_data 함수 모방)
        from src.utils.indicators import calculate_technical_indicators
        indicators = calculate_technical_indicators(self.test_data)
        self.processed_data = pd.concat([self.test_data, indicators], axis=1)
        
        # 트레이닝 및 테스트 데이터 분할
        self.train_data = self.processed_data.iloc[:150]
        self.test_data = self.processed_data.iloc[150:]
        
        # 트레이딩 환경 생성 (훈련용)
        self.train_env = TradingEnv(
            data=self.train_data,
            initial_capital=100000,
            trading_cost=0.0005,
            slippage=0.0001,
            risk_free_rate=0.02,
            max_position_size=1.0,
            stop_loss_pct=0.02
        )
        
        # 트레이딩 환경 생성 (테스트용)
        self.test_env = TradingEnv(
            data=self.test_data,
            initial_capital=100000,
            trading_cost=0.0005,
            slippage=0.0001,
            risk_free_rate=0.02,
            max_position_size=1.0,
            stop_loss_pct=0.02
        )
        
        # GRPO 에이전트 생성
        self.state_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.n
        
        self.agent = GRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            lr=3e-4,
            gamma=0.99,
            reward_scale=1.0,
            penalty_scale=0.5,
            device='cpu'
        )
        
        # 기준 성능 파일 경로
        self.benchmark_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'benchmark_performance.json'
        )
    
    def train_agent(self, num_episodes=2):
        """에이전트 훈련 함수"""
        episode_rewards = []
        update_interval = 10
        
        for episode in range(num_episodes):
            state = self.train_env.reset()
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done:
                # 행동 선택
                action = self.agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, done, _ = self.train_env.step(action)
                
                # 경험 저장
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # 정책 업데이트
                if step_count % update_interval == 0:
                    self.agent.update()
                
                # 상태 및 보상 업데이트
                state = next_state
                episode_reward += reward
                step_count += 1
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)
    
    def evaluate_agent(self, num_episodes=5):
        """에이전트 평가 함수"""
        episode_rewards = []
        portfolio_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for episode in range(num_episodes):
            state = self.test_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 결정론적 행동 선택
                action = self.agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, done, info = self.test_env.step(action)
                
                # 보상 누적
                episode_reward += reward
                state = next_state
            
            # 에피소드 지표 계산
            episode_rewards.append(episode_reward)
            
            # 포트폴리오 수익률 계산
            initial_value = self.test_env.initial_capital
            final_value = info['portfolio_value']
            portfolio_return = (final_value / initial_value) - 1
            portfolio_returns.append(portfolio_return)
            
            # 포트폴리오 가치 배열
            portfolio_values = np.array(self.test_env.portfolio_values)
            
            # 일별 수익률 계산
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Sharpe 비율 계산 (연간화, 무위험 수익률 0.02 가정)
            if len(daily_returns) > 1:
                sharpe_ratio = np.sqrt(252) * (np.mean(daily_returns) - 0.02/252) / np.std(daily_returns)
                sharpe_ratios.append(sharpe_ratio)
            
            # 최대 손실(Max Drawdown) 계산
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)
            max_drawdowns.append(max_drawdown)
        
        # 평균 지표 반환
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'mean_return': float(np.mean(portfolio_returns)),
            'mean_sharpe': float(np.mean(sharpe_ratios)) if sharpe_ratios else 0,
            'mean_max_drawdown': float(np.mean(max_drawdowns))
        }
    
    def save_benchmark(self, performance):
        """기준 성능 저장"""
        with open(self.benchmark_file, 'w') as f:
            json.dump(performance, f)
    
    def load_benchmark(self):
        """기준 성능 로드"""
        if os.path.exists(self.benchmark_file):
            with open(self.benchmark_file, 'r') as f:
                return json.load(f)
        return None
    
    def test_performance_regression(self):
        """성능 회귀 테스트"""
        # 에이전트 훈련
        train_reward = self.train_agent(num_episodes=2)
        
        # 훈련된 에이전트 평가
        performance = self.evaluate_agent(num_episodes=3)
        
        # 기준 성능 로드
        benchmark = self.load_benchmark()
        
        if benchmark is None:
            # 기준 성능이 없는 경우 현재 성능을 저장
            self.save_benchmark(performance)
            self.skipTest("기준 성능이 없어 현재 성능을 기준으로 저장했습니다.")
        else:
            # 성능 비교
            # 보상 회귀 확인: 현재 평균 보상이 기준의 95% 이상이어야 함
            self.assertGreaterEqual(
                performance['mean_reward'],
                benchmark['mean_reward'] * 0.95,
                f"평균 보상 회귀: 현재 {performance['mean_reward']:.4f}, 기준 {benchmark['mean_reward']:.4f}"
            )
            
            # 수익률 회귀 확인: 현재 평균 수익률이 기준의 95% 이상이어야 함
            self.assertGreaterEqual(
                performance['mean_return'],
                benchmark['mean_return'] * 0.95,
                f"평균 수익률 회귀: 현재 {performance['mean_return']:.4f}, 기준 {benchmark['mean_return']:.4f}"
            )
            
            # Sharpe 비율 회귀 확인: 현재 평균 Sharpe 비율이 기준의 95% 이상이어야 함
            if benchmark['mean_sharpe'] > 0:
                self.assertGreaterEqual(
                    performance['mean_sharpe'],
                    benchmark['mean_sharpe'] * 0.95,
                    f"평균 Sharpe 비율 회귀: 현재 {performance['mean_sharpe']:.4f}, 기준 {benchmark['mean_sharpe']:.4f}"
                )
            
            # 최대 손실 회귀 확인: 현재 평균 최대 손실이 기준의 105% 이하여야 함 (낮을수록 좋음)
            self.assertLessEqual(
                performance['mean_max_drawdown'],
                benchmark['mean_max_drawdown'] * 1.05,
                f"평균 최대 손실 회귀: 현재 {performance['mean_max_drawdown']:.4f}, 기준 {benchmark['mean_max_drawdown']:.4f}"
            )
            
            # 성능이 향상된 경우 새로운 기준 저장 (선택적)
            if (performance['mean_reward'] > benchmark['mean_reward'] and
                performance['mean_return'] > benchmark['mean_return']):
                self.save_benchmark(performance)
    
    def test_model_stability(self):
        """모델 안정성 테스트"""
        # 동일한 시드로 여러 번 평가하여 일관성 확인
        num_runs = 3
        results = []
        
        for run in range(num_runs):
            # 매번 동일한 시드 설정
            torch.manual_seed(42)
            np.random.seed(42)
            
            # 에이전트 재초기화
            agent = GRPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=64,
                lr=3e-4,
                gamma=0.99,
                device='cpu'
            )
            
            # 동일 조건에서 훈련
            state = self.train_env.reset()
            for step in range(100):  # 간단한 훈련
                action = agent.select_action(state)
                next_state, reward, done, _ = self.train_env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                if done:
                    state = self.train_env.reset()
            
            agent.update()
            
            # 테스트 행동 기록
            self.test_env.reset()
            actions = []
            for _ in range(10):
                state = self.test_env.reset()
                action = agent.select_action(state)
                actions.append(action)
            
            results.append(actions)
        
        # 모든 실행에서 행동이 동일한지 확인
        for i in range(1, num_runs):
            self.assertEqual(results[0], results[i], f"실행 {i}의 행동이 실행 0과 다릅니다.")
    
    def test_hyperparameter_sensitivity(self):
        """하이퍼파라미터 민감도 테스트"""
        # 기본 에이전트 평가
        self.train_agent(num_episodes=1)
        base_performance = self.evaluate_agent(num_episodes=2)
        
        # 학습률을 변경한 에이전트 평가
        agent_lr01 = GRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            lr=1e-4,  # 학습률 감소
            gamma=0.99,
            device='cpu'
        )
        
        # 감마를 변경한 에이전트 평가
        agent_gamma95 = GRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            lr=3e-4,
            gamma=0.95,  # 감마 감소
            device='cpu'
        )
        
        # 각 에이전트 훈련 및 평가
        def train_and_evaluate(agent):
            # 훈련
            state = self.train_env.reset()
            for step in range(200):
                action = agent.select_action(state)
                next_state, reward, done, _ = self.train_env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                if step % 20 == 0:
                    agent.update()
                if done:
                    state = self.train_env.reset()
            
            # 평가
            return self.evaluate_agent(num_episodes=2)
        
        performance_lr01 = train_and_evaluate(agent_lr01)
        performance_gamma95 = train_and_evaluate(agent_gamma95)
        
        # 결과 기록 (실패하지 않음, 민감도만 확인)
        print(f"기본 성능: 보상 {base_performance['mean_reward']:.4f}, 수익률 {base_performance['mean_return']:.4f}")
        print(f"학습률 0.0001 성능: 보상 {performance_lr01['mean_reward']:.4f}, 수익률 {performance_lr01['mean_return']:.4f}")
        print(f"감마 0.95 성능: 보상 {performance_gamma95['mean_reward']:.4f}, 수익률 {performance_gamma95['mean_return']:.4f}")
        
        # 민감도 확인 (로그만 남기고 명시적 검사는 하지 않음)
        # 하이퍼파라미터 변경으로 인한 성능 변화 추세만 확인

if __name__ == '__main__':
    unittest.main()
