import unittest
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.deepseek_grpo_agent import (
    DeepSeekGRPOAgent, TemporalEncoder, FeatureAttentionModule as FeatureAttention,
    DeepSeekPolicyNetwork, DistributionalValueNetwork, MetaController
)
from src.models.grpo_agent import GRPOAgent

class TestDeepSeekGRPOAgent(unittest.TestCase):
    """DeepSeekGRPOAgent 및 관련 컴포넌트에 대한 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 셋업"""
        # 테스트 디렉토리
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_models')
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # CUDA 사용 가능 여부 확인
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 테스트 하이퍼파라미터
        cls.state_dim = 20
        cls.action_dim = 3
        cls.seq_length = 10
        cls.feature_dim = 12
        cls.hidden_dim = 64
        cls.batch_size = 4
    
    def test_temporal_encoder(self):
        """TemporalEncoder 모듈 테스트"""
        # 임의의 입력 생성
        batch_size = self.batch_size
        seq_length = self.seq_length
        feature_dim = self.feature_dim
        hidden_dim = self.hidden_dim
        
        x = torch.randn(batch_size, seq_length, feature_dim)
        
        # 컴포넌트 초기화
        encoder = TemporalEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            max_seq_length=seq_length
        )
        
        # 순전파
        output = encoder(x)
        
        # 출력 형태 확인
        self.assertEqual(output.shape, (batch_size, hidden_dim))
        
        # 마스크 테스트
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
        mask[:, -3:] = True  # 마지막 3개 요소 마스킹
        
        # 마스크 적용 순전파
        masked_output = encoder(x, mask)
        
        # 출력 형태 확인
        self.assertEqual(masked_output.shape, (batch_size, hidden_dim))
    
    def test_feature_attention(self):
        """FeatureAttention 모듈 테스트"""
        # 임의의 입력 생성
        batch_size = self.batch_size
        feature_dim = self.state_dim
        hidden_dim = self.hidden_dim
        
        x = torch.randn(batch_size, feature_dim)
        
        # 컴포넌트 초기화
        attention = FeatureAttention(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # 순전파
        encoded_features, importance_weights = attention(x)
        
        # 출력 형태 확인
        self.assertEqual(encoded_features.shape, (batch_size, hidden_dim))
        self.assertEqual(importance_weights.shape, (batch_size, feature_dim))
        
        # 중요도 가중치 합 확인 (0-1 사이 값)
        self.assertTrue(torch.all(importance_weights >= 0))
        self.assertTrue(torch.all(importance_weights <= 1))
    
    def test_deepseek_policy_network(self):
        """DeepSeekPolicyNetwork 모듈 테스트"""
        # 임의의 입력 생성
        batch_size = self.batch_size
        state_dim = self.state_dim
        action_dim = self.action_dim
        hidden_dim = self.hidden_dim
        
        state = torch.randn(batch_size, state_dim)
        memory = torch.randn(batch_size, 1, hidden_dim)  # 시계열 특성
        
        # 컴포넌트 초기화
        policy_network = DeepSeekPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # 메모리 없이 순전파
        action_probs = policy_network(state)
        
        # 출력 형태 확인
        self.assertEqual(action_probs.shape, (batch_size, action_dim))
        
        # 확률 합 확인 (1)
        self.assertTrue(torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6))
        
        # 메모리와 함께 순전파
        action_probs_with_memory = policy_network(state, memory)
        
        # 출력 형태 확인
        self.assertEqual(action_probs_with_memory.shape, (batch_size, action_dim))
        
        # 확률 합 확인 (1)
        self.assertTrue(torch.allclose(action_probs_with_memory.sum(dim=1), torch.ones(batch_size), atol=1e-6))
    
    def test_distributional_value_network(self):
        """DistributionalValueNetwork 모듈 테스트"""
        # 임의의 입력 생성
        batch_size = self.batch_size
        state_dim = self.state_dim
        action_dim = self.action_dim
        hidden_dim = self.hidden_dim
        num_atoms = 51
        v_min = -10.0
        v_max = 10.0
        
        state = torch.randn(batch_size, state_dim)
        action_onehot = torch.zeros(batch_size, action_dim)
        action_onehot[:, 0] = 1.0  # 첫 번째 행동
        
        # 컴포넌트 초기화
        value_network = DistributionalValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        )
        
        # 순전파
        value_dist, expected_value = value_network(state, action_onehot)
        
        # 출력 형태 확인
        self.assertEqual(value_dist.shape, (batch_size, num_atoms))
        self.assertEqual(expected_value.shape, (batch_size, 1))
        
        # 확률 분포 합 확인 (1)
        self.assertTrue(torch.allclose(value_dist.sum(dim=1), torch.ones(batch_size), atol=1e-6))
        
        # 지원 범위 확인
        self.assertAlmostEqual(value_network.v_min, v_min)
        self.assertAlmostEqual(value_network.v_max, v_max)
        self.assertEqual(value_network.support.shape[0], num_atoms)
    
    def test_meta_controller(self):
        """MetaController 모듈 테스트"""
        # 임의의 입력 생성
        batch_size = self.batch_size
        state_dim = self.state_dim
        hidden_dim = self.hidden_dim
        num_regimes = 4
        
        state = torch.randn(batch_size, state_dim)
        
        # 컴포넌트 초기화
        meta_controller = MetaController(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_regimes=num_regimes
        )
        
        # 순전파
        regime_probs, reward_scale, penalty_scale, exploration_temp = meta_controller(state)
        
        # 출력 형태 확인
        self.assertEqual(regime_probs.shape, (batch_size, num_regimes))
        self.assertEqual(reward_scale.shape, (batch_size,))
        self.assertEqual(penalty_scale.shape, (batch_size,))
        self.assertEqual(exploration_temp.shape, (batch_size,))
        
        # 확률 합 확인 (1)
        self.assertTrue(torch.allclose(regime_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6))
        
        # 파라미터 범위 확인
        self.assertTrue(torch.all(reward_scale >= 0) and torch.all(reward_scale <= 2))
        self.assertTrue(torch.all(penalty_scale >= 0) and torch.all(penalty_scale <= 1))
        self.assertTrue(torch.all(exploration_temp >= 0) and torch.all(exploration_temp <= 5))
    
    def test_agent_initialization(self):
        """에이전트 초기화 테스트"""
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 주요 구성 요소 확인
        self.assertIsInstance(agent.temporal_encoder, TemporalEncoder)
        self.assertIsInstance(agent.feature_attention, FeatureAttention)
        self.assertIsInstance(agent.policy_network, DeepSeekPolicyNetwork)
        self.assertIsInstance(agent.value_network, DistributionalValueNetwork)
        self.assertIsInstance(agent.meta_controller, MetaController)
        self.assertIsInstance(agent.optimizer, torch.optim.Adam)
        
        # 하이퍼파라미터 확인
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertEqual(agent.seq_length, self.seq_length)
        self.assertEqual(agent.feature_dim, self.feature_dim)
        self.assertEqual(agent.hidden_dim, self.hidden_dim)
    
    def test_select_action(self):
        """행동 선택 테스트"""
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 임의의 상태 생성
        state = np.random.randn(self.state_dim)
        history = np.random.randn(self.seq_length, self.feature_dim)
        
        # 결정론적 행동 선택
        action, action_info = agent.select_action(state, history, deterministic=True)
        
        # 행동 및 정보 확인
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
        self.assertIn('action_probs', action_info)
        self.assertIn('regime_probs', action_info)
        self.assertIn('exploration_temp', action_info)
        
        # 확률적 행동 선택
        action, action_info = agent.select_action(state, history, deterministic=False)
        
        # 행동 및 정보 확인
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
        
        # 로그 확률 저장 확인
        self.assertEqual(len(agent.log_probs), 1)
    
    def test_store_transition(self):
        """경험 저장 테스트"""
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 초기 버퍼 상태 확인
        self.assertEqual(len(agent.states), 0)
        self.assertEqual(len(agent.actions), 0)
        self.assertEqual(len(agent.rewards), 0)
        self.assertEqual(len(agent.next_states), 0)
        self.assertEqual(len(agent.dones), 0)
        
        # 경험 저장
        state = np.random.randn(self.state_dim)
        action = 1
        reward = 0.5
        next_state = np.random.randn(self.state_dim)
        done = False
        history = np.random.randn(self.seq_length, self.feature_dim)
        
        agent.store_transition(state, action, reward, next_state, done, history)
        
        # 버퍼 상태 확인
        self.assertEqual(len(agent.states), 1)
        self.assertEqual(len(agent.actions), 1)
        self.assertEqual(len(agent.rewards), 1)
        self.assertEqual(len(agent.next_states), 1)
        self.assertEqual(len(agent.dones), 1)
        self.assertEqual(len(agent.histories), 1)
        
        # 저장된 값 확인
        self.assertTrue(np.array_equal(agent.states[0], state))
        self.assertEqual(agent.actions[0], action)
        self.assertEqual(agent.rewards[0], reward)
        self.assertTrue(np.array_equal(agent.next_states[0], next_state))
        self.assertEqual(agent.dones[0], done)
        self.assertTrue(np.array_equal(agent.histories[0], history))
    
    def test_save_load(self):
        """모델 저장 및 로드 테스트"""
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 저장 경로
        save_path = os.path.join(self.test_dir, 'test_agent.pt')
        
        # 모델 저장
        agent.save(save_path)
        
        # 파일 존재 확인
        self.assertTrue(os.path.exists(save_path))
        
        # 새 에이전트 초기화
        new_agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 모델 로드
        new_agent.load(save_path)
        
        # 성공적인 로드 확인 (모델 구조 비교)
        for param1, param2 in zip(agent.temporal_encoder.parameters(), new_agent.temporal_encoder.parameters()):
            self.assertTrue(torch.equal(param1, param2))
    
    def test_update_empty_buffer(self):
        """빈 버퍼로 업데이트 테스트"""
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 빈 버퍼로 업데이트
        metrics = agent.update()
        
        # 결과 확인
        self.assertEqual(metrics, {})
    
    def test_update_with_data(self):
        """데이터가 있는 버퍼로 업데이트 테스트"""
        # 간단한 테스트를 위해 작은 모델 사용
        mini_state_dim = 4
        mini_action_dim = 2
        mini_seq_length = 3
        mini_feature_dim = 4
        mini_hidden_dim = 8
        
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=mini_state_dim,
            action_dim=mini_action_dim,
            seq_length=mini_seq_length,
            feature_dim=mini_feature_dim,
            hidden_dim=mini_hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 가짜 경험 생성
        for _ in range(10):
            state = np.random.randn(mini_state_dim)
            action = np.random.randint(0, mini_action_dim)
            reward = np.random.randn() * 0.1
            next_state = np.random.randn(mini_state_dim)
            done = np.random.random() > 0.9
            history = np.random.randn(mini_seq_length, mini_feature_dim)
            
            # 행동 선택 (로그 확률 생성을 위해)
            agent.select_action(state, history, deterministic=False)
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done, history)
        
        # 업데이트 수행
        metrics = agent.update(epochs=1, batch_size=4)
        
        # 결과 확인
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy', metrics)
        self.assertIn('total_loss', metrics)
        
        # 버퍼가 비워졌는지 확인
        self.assertEqual(len(agent.states), 0)
        self.assertEqual(len(agent.actions), 0)
        self.assertEqual(len(agent.rewards), 0)
        self.assertEqual(len(agent.next_states), 0)
        self.assertEqual(len(agent.dones), 0)
        self.assertEqual(len(agent.histories), 0)
        self.assertEqual(len(agent.log_probs), 0)
    
    def test_feature_importance(self):
        """특성 중요도 추출 테스트"""
        # 에이전트 초기화
        agent = DeepSeekGRPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            device='cpu'  # CPU로 테스트
        )
        
        # 임의의 상태 생성
        state = np.random.randn(self.state_dim)
        history = np.random.randn(self.seq_length, self.feature_dim)
        
        # 특성 중요도 추출
        importance_weights = agent.get_feature_importance(state, history)

        # 결과 확인
        self.assertEqual(importance_weights.shape, (1, self.state_dim))
        self.assertTrue(np.all(importance_weights >= 0))
        self.assertTrue(np.all(importance_weights <= 1))


class TestGRPOAgentEdgeCases(unittest.TestCase):
    """GRPOAgent edge case tests for reward/penalty loss"""

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.agent = GRPOAgent(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=8, device='cpu')

        def mock_forward(x):
            return torch.ones(x.size(0), self.agent.action_dim, device='cpu', requires_grad=True) / self.agent.action_dim

        def mock_q(state, action_onehot):
            return torch.zeros(state.size(0), 1, device='cpu', requires_grad=True)

        self.agent.network.forward = MagicMock(side_effect=mock_forward)
        self.agent.network.estimate_q_value = MagicMock(side_effect=mock_q)

    def _populate(self, reward):
        for _ in range(4):
            state = np.zeros(self.state_dim)
            next_state = np.zeros(self.state_dim)
            self.agent.store_transition(state, 0, reward, next_state, False)

    def test_update_all_positive_advantages(self):
        self._populate(1.0)
        metrics = self.agent.update()
        self.assertIn('policy_loss', metrics)
        self.assertFalse(np.isnan(metrics['policy_loss']))

    def test_update_all_negative_advantages(self):
        self._populate(-1.0)
        metrics = self.agent.update()
        self.assertIn('policy_loss', metrics)
        self.assertFalse(np.isnan(metrics['policy_loss']))


if __name__ == '__main__':
    unittest.main()
