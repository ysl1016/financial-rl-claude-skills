import sys
import os
import unittest
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경에서 테스트를 위한 백엔드 설정

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent

class TestIntegration(unittest.TestCase):
    """트레이딩 환경과 GRPO 에이전트의 통합 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 환경 설정"""
        # 랜덤 시드 설정
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 테스트용 데이터 생성
        dates = pd.date_range(start='2020-01-01', periods=100)
        
        # 가격 데이터 생성
        close_prices = np.random.normal(100, 5, size=100).cumsum() + 1000
        high_prices = close_prices * (1 + np.random.uniform(0, 0.05, size=100))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.05, size=100))
        open_prices = close_prices * (1 + np.random.normal(0, 0.01, size=100))
        volumes = np.random.randint(1000, 10000, size=100)
        
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
        
        # 트레이딩 환경 생성
        self.env = TradingEnv(
            data=self.processed_data,
            initial_capital=100000,
            trading_cost=0.0005,
            slippage=0.0001,
            risk_free_rate=0.02,
            max_position_size=1.0,
            stop_loss_pct=0.02
        )
        
        # GRPO 에이전트 생성
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
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
    
    def test_single_episode(self):
        """단일 에피소드 실행 테스트"""
        # 에피소드 시작
        state = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 20  # 테스트를 위해 단축된 에피소드
        
        while not done and step_count < max_steps:
            # 에이전트를 통한 행동 선택
            action = self.agent.select_action(state)
            
            # 환경에서 다음 상태와 보상 얻기
            next_state, reward, done, info = self.env.step(action)
            
            # 경험 저장
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # 상태 업데이트
            state = next_state
            total_reward += reward
            step_count += 1
        
        # 에피소드 종료 후 확인
        self.assertGreaterEqual(step_count, 1)
        self.assertIsInstance(total_reward, float)
        
        # 포트폴리오 가치 확인
        self.assertGreater(len(self.env.portfolio_values), 1)
        
        # 거래 기록 확인
        self.assertIsInstance(self.env.trades, list)
    
    def test_training_loop(self):
        """트레이닝 루프 테스트"""
        # 업데이트 간격
        update_interval = 10
        
        # 에피소드 시작
        state = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 50  # 테스트를 위해 단축된 에피소드
        
        metrics = {}
        
        while not done and step_count < max_steps:
            # 에이전트를 통한 행동 선택
            action = self.agent.select_action(state)
            
            # 환경에서 다음 상태와 보상 얻기
            next_state, reward, done, info = self.env.step(action)
            
            # 경험 저장
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # 상태 업데이트
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 정책 업데이트
            if step_count % update_interval == 0:
                metrics = self.agent.update()
        
        # 업데이트가 실행되었는지 확인
        self.assertGreaterEqual(step_count / update_interval, 1)
        
        # 올바른 메트릭이 반환되었는지 확인
        if metrics:
            self.assertIn('policy_loss', metrics)
            self.assertIn('q_loss', metrics)
            self.assertIn('mean_reward', metrics)
            self.assertIn('mean_advantage', metrics)
    
    def test_model_save_load(self):
        """모델 저장 및 로드 테스트"""
        # 임시 파일 경로
        temp_path = "temp_test_model.pt"
        
        try:
            # 에이전트 학습
            state = self.env.reset()
            for _ in range(20):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                if done:
                    state = self.env.reset()
            
            # 학습 업데이트
            self.agent.update()
            
            # 원래 모델의 행동 기록
            self.env.reset()
            original_actions = []
            for _ in range(10):
                state = np.random.rand(self.state_dim).astype(np.float32)
                original_actions.append(self.agent.select_action(state))
            
            # 모델 저장
            self.agent.save(temp_path)
            
            # 새 에이전트 생성 및 로드
            new_agent = GRPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=64,
                device='cpu'
            )
            
            new_agent.load(temp_path)
            
            # 로드된 모델의 행동 기록
            loaded_actions = []
            for _ in range(10):
                state = np.random.rand(self.state_dim).astype(np.float32)
                loaded_actions.append(new_agent.select_action(state))
            
            # 두 모델의 결정이 일치하는지 확인
            # 동일한 랜덤 시드를 사용하면 결정이 일치해야 함
            torch.manual_seed(42)
            np.random.seed(42)
            for i in range(10):
                state = np.random.rand(self.state_dim).astype(np.float32)
                action1 = self.agent.select_action(state)
                action2 = new_agent.select_action(state)
                self.assertEqual(action1, action2)
                
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_evaluation(self):
        """에이전트 평가 테스트"""
        # 짧은 에피소드 수 설정
        num_episodes = 2
        rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 평가 모드에서는 탐험 없이 결정론적으로 행동 선택
                action = self.agent.select_action(state)
                
                # 환경에서 다음 상태와 보상 얻기
                next_state, reward, done, _ = self.env.step(action)
                
                # 에피소드 보상 누적
                episode_reward += reward
                
                # 상태 업데이트
                state = next_state
            
            rewards.append(episode_reward)
        
        # 평가 결과 확인
        self.assertEqual(len(rewards), num_episodes)
        
        # 평균 보상 계산
        mean_reward = np.mean(rewards)
        self.assertIsInstance(mean_reward, float)
    
    def test_integration_process_data(self):
        """data_processor 모듈과의 통합 테스트"""
        # 모의 데이터로 data_processor 모방
        from src.utils.indicators import calculate_technical_indicators
        
        # 트레이딩 환경과 데이터 처리 통합
        processed_data = self.processed_data.copy()
        env = TradingEnv(data=processed_data)
        
        # 환경 초기화 및 단계 실행
        state = env.reset()
        action = 1  # 매수 액션
        next_state, reward, done, info = env.step(action)
        
        # 상태 및 보상 확인
        self.assertEqual(len(state), 14)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn('portfolio_value', info)

if __name__ == '__main__':
    unittest.main()
