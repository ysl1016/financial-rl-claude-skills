import sys
import os
import unittest
import numpy as np
import torch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.grpo_agent import GRPOAgent, GRPONetwork

class TestGRPONetwork(unittest.TestCase):
    """GRPONetwork 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 환경 설정"""
        # 테스트 환경 설정
        self.state_dim = 14
        self.action_dim = 3
        self.hidden_dim = 64
        
        # 랜덤 시드 설정
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 테스트용 네트워크 초기화
        self.network = GRPONetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
    
    def test_init(self):
        """초기화 테스트"""
        # 네트워크 구조 확인
        self.assertIsInstance(self.network.policy, torch.nn.Sequential)
        self.assertIsInstance(self.network.q_estimator, torch.nn.Sequential)
        
        # 정책 네트워크 레이어 수 확인
        policy_layers = [module for module in self.network.policy if isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(policy_layers), 3)
        
        # Q 추정기 레이어 수 확인
        q_layers = [module for module in self.network.q_estimator if isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(q_layers), 3)
        
        # 첫 번째 레이어 입력 차원 확인
        self.assertEqual(policy_layers[0].in_features, self.state_dim)
        self.assertEqual(q_layers[0].in_features, self.state_dim + self.action_dim)
        
        # 마지막 레이어 출력 차원 확인
        self.assertEqual(policy_layers[-1].out_features, self.action_dim)
        self.assertEqual(q_layers[-1].out_features, 1)
    
    def test_forward(self):
        """순전파 테스트"""
        # 랜덤 상태 생성
        batch_size = 8
        state = torch.randn(batch_size, self.state_dim)
        
        # 순전파 실행
        action_probs = self.network(state)
        
        # 출력 형태 확인
        self.assertEqual(action_probs.shape, (batch_size, self.action_dim))
        
        # 출력이 확률 분포인지 확인 (합계가 1이고 모든 요소가 0과 1 사이)
        self.assertTrue(torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6))
        self.assertTrue((action_probs >= 0).all() and (action_probs <= 1).all())
    
    def test_estimate_q_value(self):
        """Q-값 추정 테스트"""
        # 랜덤 상태 및 행동 생성
        batch_size = 8
        state = torch.randn(batch_size, self.state_dim)
        action_onehot = torch.zeros(batch_size, self.action_dim)
        action_onehot[:, 0] = 1  # 모든 샘플에 대해 첫 번째 행동 선택
        
        # Q-값 추정 실행
        q_values = self.network.estimate_q_value(state, action_onehot)
        
        # 출력 형태 확인
        self.assertEqual(q_values.shape, (batch_size, 1))

class TestGRPOAgent(unittest.TestCase):
    """GRPOAgent 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 환경 설정"""
        # 테스트 환경 설정
        self.state_dim = 14
        self.action_dim = 3
        self.hidden_dim = 64
        self.lr = 3e-4
        self.gamma = 0.99
        self.reward_scale = 1.0
        self.penalty_scale = 0.5
        
        # 랜덤 시드 설정
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 테스트에 GPU 사용 안함
        self.device = 'cpu'
        
        # 테스트용 에이전트 초기화
        self.agent = GRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            penalty_scale=self.penalty_scale,
            device=self.device
        )
    
    def test_init(self):
        """초기화 테스트"""
        # 에이전트 속성 확인
        self.assertEqual(self.agent.device, self.device)
        self.assertEqual(self.agent.gamma, self.gamma)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.reward_scale, self.reward_scale)
        self.assertEqual(self.agent.penalty_scale, self.penalty_scale)
        
        # 네트워크 및 옵티마이저 초기화 확인
        self.assertIsInstance(self.agent.network, GRPONetwork)
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
        
        # 경험 버퍼 초기화 확인
        self.assertEqual(len(self.agent.states), 0)
        self.assertEqual(len(self.agent.actions), 0)
        self.assertEqual(len(self.agent.rewards), 0)
        self.assertEqual(len(self.agent.next_states), 0)
        self.assertEqual(len(self.agent.dones), 0)
    
    def test_select_action(self):
        """행동 선택 테스트"""
        # 랜덤 상태 생성
        state = np.random.rand(self.state_dim).astype(np.float32)
        
        # 행동 선택
        action = self.agent.select_action(state)
        
        # 행동이 유효한 범위에 있는지 확인
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
    
    def test_store_transition(self):
        """경험 저장 테스트"""
        # 경험 샘플 생성
        state = np.random.rand(self.state_dim).astype(np.float32)
        action = 1
        reward = 0.5
        next_state = np.random.rand(self.state_dim).astype(np.float32)
        done = False
        
        # 경험 저장
        self.agent.store_transition(state, action, reward, next_state, done)
        
        # 버퍼에 저장되었는지 확인
        self.assertEqual(len(self.agent.states), 1)
        self.assertEqual(len(self.agent.actions), 1)
        self.assertEqual(len(self.agent.rewards), 1)
        self.assertEqual(len(self.agent.next_states), 1)
        self.assertEqual(len(self.agent.dones), 1)
        
        np.testing.assert_array_equal(self.agent.states[0], state)
        self.assertEqual(self.agent.actions[0], action)
        self.assertEqual(self.agent.rewards[0], reward)
        np.testing.assert_array_equal(self.agent.next_states[0], next_state)
        self.assertEqual(self.agent.dones[0], done)
    
    def test_to_onehot(self):
        """원핫 인코딩 변환 테스트"""
        # 행동 리스트 생성
        actions = [0, 1, 2, 0, 1]
        
        # 원핫 인코딩 변환
        onehot = self.agent._to_onehot(actions)
        
        # 형태 확인
        self.assertEqual(onehot.shape, (len(actions), self.action_dim))
        
        # 각 행의 합이 1인지 확인
        self.assertTrue(torch.all(onehot.sum(dim=1) == 1))
        
        # 첫 번째 샘플 확인
        expected_first = torch.tensor([1, 0, 0], device=self.device)
        self.assertTrue(torch.all(onehot[0] == expected_first))
        
        # 두 번째 샘플 확인
        expected_second = torch.tensor([0, 1, 0], device=self.device)
        self.assertTrue(torch.all(onehot[1] == expected_second))
    
    def test_update_empty_buffer(self):
        """빈 버퍼에서의 업데이트 테스트"""
        # 빈 버퍼에서 업데이트 시도
        result = self.agent.update()
        
        # 결과가 빈 딕셔너리인지 확인
        self.assertEqual(result, {})
    
    def test_update(self):
        """에이전트 업데이트 테스트"""
        # 여러 개의 경험 샘플 저장
        for _ in range(10):
            state = np.random.rand(self.state_dim).astype(np.float32)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(self.state_dim).astype(np.float32)
            done = np.random.choice([True, False])
            
            self.agent.store_transition(state, action, reward, next_state, done)
        
        # 업데이트 실행
        result = self.agent.update()
        
        # 결과 확인
        self.assertIn('policy_loss', result)
        self.assertIn('q_loss', result)
        self.assertIn('mean_reward', result)
        self.assertIn('mean_advantage', result)
        
        # 버퍼가 비워졌는지 확인
        self.assertEqual(len(self.agent.states), 0)
        self.assertEqual(len(self.agent.actions), 0)
        self.assertEqual(len(self.agent.rewards), 0)
        self.assertEqual(len(self.agent.next_states), 0)
        self.assertEqual(len(self.agent.dones), 0)
    
    def test_save_load(self):
        """모델 저장 및 로드 테스트"""
        # 임시 파일 경로
        temp_path = "temp_model.pt"
        
        try:
            # 모델 저장
            self.agent.save(temp_path)
            
            # 새 에이전트 생성
            new_agent = GRPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                device=self.device
            )
            
            # 모델 로드
            new_agent.load(temp_path)
            
            # 두 에이전트의 네트워크 가중치 비교
            for param1, param2 in zip(self.agent.network.parameters(), new_agent.network.parameters()):
                self.assertTrue(torch.all(param1 == param2))
                
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
