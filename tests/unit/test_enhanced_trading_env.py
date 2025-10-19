import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gym

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.enhanced_trading_env import EnhancedTradingEnv
from src.utils.indicators import calculate_technical_indicators

class TestEnhancedTradingEnv(unittest.TestCase):
    """EnhancedTradingEnv 클래스에 대한 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 셋업 - 테스트 데이터 생성"""
        # 테스트 데이터 경로
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # 가상의 주가 데이터 생성
        cls.create_mock_stock_data()
    
    @classmethod
    def create_mock_stock_data(cls):
        """가상의 주가 데이터 생성"""
        # 날짜 범위 생성
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 랜덤 시드 설정
        np.random.seed(42)
        
        # 초기 가격 설정
        initial_price = 100.0
        
        # 주가 데이터 생성
        prices = [initial_price]
        for i in range(1, len(date_range)):
            # 랜덤 일일 변동폭 (-2% ~ +2%)
            daily_return = np.random.normal(0.0005, 0.015)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # 데이터프레임 생성
        stock_data = pd.DataFrame({
            'Date': date_range,
            'Open': prices * np.random.uniform(0.99, 1.0, len(prices)),
            'High': prices * np.random.uniform(1.01, 1.03, len(prices)),
            'Low': prices * np.random.uniform(0.97, 0.99, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(prices))
        })
        
        # 인덱스 설정
        stock_data.set_index('Date', inplace=True)
        
        # 기술적 지표 계산
        indicators = calculate_technical_indicators(stock_data)
        
        # 데이터 결합
        cls.stock_data = pd.concat([stock_data, indicators], axis=1)
        
        # 정규화된 특성 생성 (테스트용)
        for col in indicators.columns:
            cls.stock_data[f'{col}_norm'] = (indicators[col] - indicators[col].mean()) / (indicators[col].std() + 1e-9)
        
        # 가상의 거시경제 특성 추가
        cls.stock_data['Macro_Interest_norm'] = np.random.normal(0, 1, len(cls.stock_data))
        cls.stock_data['Macro_Inflation_norm'] = np.random.normal(0, 1, len(cls.stock_data))
        cls.stock_data['Macro_GDP_norm'] = np.random.normal(0, 1, len(cls.stock_data))
        
        # 가상의 뉴스 감성 특성 추가
        cls.stock_data['News_Sentiment_norm'] = np.random.normal(0, 1, len(cls.stock_data))
        cls.stock_data['News_Volume_norm'] = np.random.normal(0, 1, len(cls.stock_data))
        
        # 결측치 처리
        cls.stock_data = cls.stock_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # CSV 파일로 저장
        cls.test_data_path = os.path.join(cls.test_data_dir, 'test_stock_data_with_indicators.csv')
        cls.stock_data.to_csv(cls.test_data_path)
    
    def setUp(self):
        """각 테스트 전 설정"""
        # 데이터 로드
        self.data = self.stock_data.copy()
        
        # 환경 생성
        self.env = EnhancedTradingEnv(
            data=self.data,
            initial_capital=100000,
            trading_cost=0.0005,
            slippage=0.0001,
            risk_free_rate=0.02,
            max_position_size=1.0,
            stop_loss_pct=0.02,
            window_size=10,
            reward_type='sharpe',
            action_type='discrete',
            use_risk_adjustment=True,
            use_market_indicators=True
        )
    
    def test_initialization(self):
        """초기화 테스트"""
        # 환경 속성 확인
        self.assertEqual(self.env.initial_capital, 100000)
        self.assertEqual(self.env.trading_cost, 0.0005)
        self.assertEqual(self.env.slippage, 0.0001)
        self.assertEqual(self.env.risk_free_rate, 0.02/252)  # 일별 환산
        self.assertEqual(self.env.max_position_size, 1.0)
        self.assertEqual(self.env.stop_loss_pct, 0.02)
        self.assertEqual(self.env.reward_type, 'sharpe')
        self.assertEqual(self.env.action_type, 'discrete')
        self.assertTrue(self.env.use_risk_adjustment)
        self.assertTrue(self.env.use_market_indicators)
        
        # 관찰 및 행동 공간 확인
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 3)  # 0: 보유, 1: 매수, 2: 매도
    
    def test_feature_selection(self):
        """특성 선택 테스트"""
        # 기본 특성 확인
        expected_basic = ['RSI_norm', 'ForceIndex2_norm', '%K_norm', '%D_norm', 
                          'MACD_norm', 'MACDSignal_norm', 'BBWidth_norm', 'ATR_norm',
                          'VPT_norm', 'VPT_MA_norm', 'OBV_norm', 'ROC_norm']
        
        # 사용 가능한 특성 확인
        available_basic = [col for col in expected_basic if col in self.data.columns]
        self.assertGreater(len(available_basic), 0)
        
        # 특성 컬럼 확인
        all_feature_columns = self.env.all_feature_columns
        self.assertIsInstance(all_feature_columns, list)
        self.assertGreater(len(all_feature_columns), 0)
        
        # 거시경제 특성 확인
        self.assertGreater(len(self.env.macro_columns), 0)
        
        # 뉴스 감성 특성 확인
        self.assertGreater(len(self.env.news_columns), 0)
    
    def test_reset(self):
        """환경 리셋 테스트"""
        # 환경 리셋
        observation = self.env.reset()
        
        # 관찰 확인
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (self.env.observation_space.shape[0],))
        
        # 초기 상태 확인
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.cash, self.env.initial_capital)
        self.assertEqual(self.env.holdings, 0)
        self.assertEqual(self.env.position_duration, 0)
        self.assertEqual(len(self.env.portfolio_values), 1)
        self.assertEqual(self.env.portfolio_values[0], self.env.initial_capital)
    
    def test_step_hold(self):
        """보유 행동 테스트"""
        # 환경 리셋
        self.env.reset()
        
        # 보유 행동 (0)
        observation, reward, done, info = self.env.step(0)
        
        # 관찰 및 보상 확인
        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)  # 첫 스텝에서는 done이 False여야 함
        self.assertIsInstance(info, dict)
        
        # 상태 확인
        self.assertEqual(self.env.position, 0)  # 여전히 무포지션
        self.assertEqual(self.env.holdings, 0)  # 보유량 없음
        self.assertEqual(self.env.position_duration, 0)  # 포지션 기간 0
        self.assertEqual(len(self.env.portfolio_values), 2)  # 초기값 + 새 값
    
    def test_step_buy(self):
        """매수 행동 테스트"""
        # 환경 리셋
        self.env.reset()
        
        # 매수 행동 (1)
        observation, reward, done, info = self.env.step(1)
        
        # 상태 확인
        self.assertEqual(self.env.position, 1)  # 롱 포지션
        self.assertGreater(self.env.holdings, 0)  # 보유량 있음
        self.assertEqual(self.env.position_duration, 0)  # 포지션 기간 초기화
        self.assertEqual(len(self.env.trades), 1)  # 거래 이력 1개
        self.assertEqual(self.env.trades[0][0], 'buy')  # 매수 거래
        
        # 이전 스텝
        prev_holdings = self.env.holdings
        prev_cash = self.env.cash
        
        # 한 번 더 스텝 (보유)
        observation, reward, done, info = self.env.step(0)
        
        # 포지션 유지 확인
        self.assertEqual(self.env.position, 1)  # 여전히 롱 포지션
        self.assertEqual(self.env.holdings, prev_holdings)  # 보유량 동일
        self.assertEqual(self.env.cash, prev_cash)  # 현금 동일
        self.assertEqual(self.env.position_duration, 1)  # 포지션 기간 증가
    
    def test_step_sell(self):
        """매도 행동 테스트"""
        # 환경 리셋
        self.env.reset()
        
        # 매도 행동 (2) - 무포지션에서 숏 진입
        observation, reward, done, info = self.env.step(2)
        
        # 상태 확인
        self.assertEqual(self.env.position, -1)  # 숏 포지션
        self.assertLess(self.env.holdings, 0)  # 음수 보유량 (숏)
        self.assertEqual(self.env.position_duration, 0)  # 포지션 기간 초기화
        self.assertEqual(len(self.env.trades), 1)  # 거래 이력 1개
        self.assertEqual(self.env.trades[0][0], 'short')  # 숏 거래
        
        # 롱 포지션에서 숏으로 전환 테스트
        self.env.reset()
        self.env.step(1)  # 먼저 매수
        
        # 매도 행동 (2) - 롱에서 숏으로 전환
        observation, reward, done, info = self.env.step(2)
        
        # 상태 확인
        self.assertEqual(self.env.position, -1)  # 숏 포지션
        self.assertLess(self.env.holdings, 0)  # 음수 보유량 (숏)
        self.assertEqual(len(self.env.trades), 3)  # 거래 이력 3개 (매수, 청산, 매도)
        self.assertEqual(self.env.trades[1][0], 'close_long')  # 롱 청산
        self.assertEqual(self.env.trades[2][0], 'short')  # 숏 진입
    
    def test_continuous_action(self):
        """연속적 행동 공간 테스트"""
        # 연속적 행동 환경 생성
        cont_env = EnhancedTradingEnv(
            data=self.data,
            initial_capital=100000,
            action_type='continuous'
        )
        
        # 관찰 및 행동 공간 확인
        self.assertIsInstance(cont_env.action_space, gym.spaces.Box)
        self.assertEqual(cont_env.action_space.shape, (1,))
        
        # 환경 리셋
        cont_env.reset()
        
        # 행동 수행
        action = np.array([0.5])  # 50% 롱 포지션
        observation, reward, done, info = cont_env.step(action)
        
        # 상태 확인
        self.assertEqual(cont_env.position, 1)  # 롱 포지션
        self.assertGreater(cont_env.holdings, 0)  # 보유량 있음
        
        # 음수 행동
        action = np.array([-0.3])  # 30% 숏 포지션
        observation, reward, done, info = cont_env.step(action)
        
        # 상태 확인
        self.assertEqual(cont_env.position, -1)  # 숏 포지션
        self.assertLess(cont_env.holdings, 0)  # 음수 보유량 (숏)
    
    def test_stop_loss(self):
        """스톱로스 기능 테스트"""
        # 환경 리셋 및 롱 포지션 진입
        self.env.reset()
        self.env.step(1)  # 매수
        
        # 진입 가격 확인
        entry_price = self.env.entry_price
        
        # 스톱로스 발동을 위해 가격 조작
        stop_loss_price = entry_price * (1 - self.env.stop_loss_pct - 0.001)  # 손절선보다 약간 낮게
        
        # 다음 단계의 가격을 스톱로스 이하로 설정
        self.env.data.loc[self.env.index, 'Close'] = stop_loss_price
        self.env.data.loc[self.env.index, 'Low'] = stop_loss_price * 0.99
        
        # 보유 행동 (0) - 스톱로스 확인
        observation, reward, done, info = self.env.step(0)
        
        # 상태 확인
        self.assertEqual(self.env.position, 0)  # 스톱로스로 포지션 청산
        self.assertEqual(self.env.holdings, 0)  # 보유량 없음
        self.assertTrue('stop_loss' in info)  # 스톱로스 정보 포함
        self.assertTrue(info['stop_loss'])  # 스톱로스 발동
    
    def test_get_performance_metrics(self):
        """성과 지표 계산 테스트"""
        # 환경 리셋
        self.env.reset()
        
        # 몇 가지 행동 수행
        self.env.step(1)  # 매수
        self.env.step(0)  # 보유
        self.env.step(2)  # 매도
        self.env.step(0)  # 보유
        
        # 성과 지표 계산
        metrics = self.env.get_performance_metrics()
        
        # 주요 지표 확인
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('volatility', metrics)
        
        # 지표 타입 확인
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float))
    
    def test_different_reward_types(self):
        """다양한 보상 함수 테스트"""
        # 다양한 보상 유형 테스트
        reward_types = ['simple', 'sharpe', 'sortino', 'calmar']
        
        for reward_type in reward_types:
            # 환경 생성
            env = EnhancedTradingEnv(
                data=self.data,
                initial_capital=100000,
                reward_type=reward_type
            )
            
            # 환경 리셋
            env.reset()
            
            # 행동 수행
            observation, reward, done, info = env.step(1)  # 매수
            
            # 보상 확인
            self.assertIsInstance(reward, float)
    
    def test_hybrid_action_space(self):
        """하이브리드 행동 공간 테스트"""
        # 하이브리드 행동 환경 생성
        hybrid_env = EnhancedTradingEnv(
            data=self.data,
            initial_capital=100000,
            action_type='hybrid'
        )
        
        # 관찰 및 행동 공간 확인
        self.assertIsInstance(hybrid_env.action_space, gym.spaces.Tuple)
        self.assertEqual(len(hybrid_env.action_space.spaces), 2)
        self.assertIsInstance(hybrid_env.action_space.spaces[0], gym.spaces.Discrete)
        self.assertIsInstance(hybrid_env.action_space.spaces[1], gym.spaces.Box)
        
        # 환경 리셋
        hybrid_env.reset()
        
        # 행동 수행
        action = (1, np.array([0.5]))  # 매수, 50% 크기
        observation, reward, done, info = hybrid_env.step(action)
        
        # 상태 확인
        self.assertEqual(hybrid_env.position, 1)  # 롱 포지션
        self.assertGreater(hybrid_env.holdings, 0)  # 보유량 있음


if __name__ == '__main__':
    unittest.main()
