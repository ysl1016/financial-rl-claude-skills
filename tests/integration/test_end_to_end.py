import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import json
from src.models.trading_env import TradingEnv
from src.claude_integration.claude_analyzer import ClaudeMarketAnalyzer

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'Open': np.random.rand(100) * 100,
            'High': np.random.rand(100) * 100,
            'Low': np.random.rand(100) * 100,
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.rand(100) * 1000,
            'RSI': np.random.rand(100) * 100,
            'MACD': np.random.rand(100) * 10
        }, index=dates)
        
        # Initialize environment
        self.env = TradingEnv(data=self.data)
        
        # Mock Claude Analyzer
        self.mock_analyzer = MagicMock(spec=ClaudeMarketAnalyzer)
        self.mock_analyzer.analyze_market_state.return_value = {
            "market_sentiment": "bullish",
            "confidence_level": 0.8,
            "trading_recommendation": {
                "suggested_action": "buy",
                "reasoning": "Upward trend",
                "entry_criteria": "Now",
                "exit_criteria": "Later"
            }
        }

    def test_trading_loop_with_analysis(self):
        state = self.env.reset()
        done = False
        
        while not done:
            # Simulate getting analysis
            analysis = self.mock_analyzer.analyze_market_state(
                market_data=self.data.iloc[:self.env.index + 1],
                technical_indicators={'RSI': 50},
                current_position=0,
                portfolio_value=self.env.portfolio_values[-1]
            )
            
            # Use analysis to decide action (simplified logic)
            if analysis['trading_recommendation']['suggested_action'] == 'buy':
                action = 1 # Buy
            else:
                action = 0 # Hold
            
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            
        self.assertTrue(done)
        self.assertGreater(len(self.env.trades), 0)

if __name__ == '__main__':
    unittest.main()
