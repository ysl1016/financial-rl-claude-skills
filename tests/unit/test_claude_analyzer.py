import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import json
from src.claude_integration.claude_analyzer import ClaudeMarketAnalyzer

class TestClaudeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.analyzer = ClaudeMarketAnalyzer(api_key="test_key")
        self.analyzer.client = self.mock_client

    def test_init(self):
        self.assertEqual(self.analyzer.model, "claude-3-7-sonnet-20250219")
        self.assertEqual(self.analyzer.max_tokens, 2048)
        self.assertEqual(self.analyzer.temperature, 0.7)

    def test_analyze_market_state(self):
        # Mock data
        market_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Volume': [1000, 1100, 1200]
        })
        technical_indicators = {
            'RSI_norm': 0.5,
            'MACD_norm': 0.1
        }
        
        # Mock Claude response
        mock_response_content = {
            "market_sentiment": "bullish",
            "confidence_level": 0.8,
            "trading_recommendation": {
                "suggested_action": "buy",
                "reasoning": "Upward trend",
                "entry_criteria": "Now",
                "exit_criteria": "Later"
            }
        }
        
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response_content))]
        self.mock_client.messages.create.return_value = mock_message

        # Run analysis
        result = self.analyzer.analyze_market_state(
            market_data, 
            technical_indicators,
            current_position=0,
            portfolio_value=100000
        )

        # Verify
        self.assertEqual(result['market_sentiment'], "bullish")
        self.assertEqual(result['confidence_level'], 0.8)
        self.mock_client.messages.create.assert_called_once()

    def test_analyze_market_state_parsing_error(self):
        # Mock data
        market_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Volume': [1000, 1100, 1200]
        })
        technical_indicators = {'RSI_norm': 0.5}

        # Mock invalid JSON response
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Invalid JSON")]
        self.mock_client.messages.create.return_value = mock_message

        # Run analysis
        result = self.analyzer.analyze_market_state(
            market_data, 
            technical_indicators,
            current_position=0,
            portfolio_value=100000
        )

        # Verify fallback
        self.assertTrue(result.get('error', False))
        self.assertEqual(result['market_sentiment'], 'neutral')

    def test_explain_indicator_divergence(self):
        price_data = pd.Series([100, 105, 110])
        indicator_data = pd.Series([50, 45, 40])
        
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Bearish divergence detected.")]
        self.mock_client.messages.create.return_value = mock_message

        result = self.analyzer.explain_indicator_divergence(price_data, indicator_data, "RSI")
        
        self.assertEqual(result, "Bearish divergence detected.")

    def test_interpret_pattern(self):
        recent_candles = pd.DataFrame({
            'Open': [100], 'High': [110], 'Low': [90], 'Close': [105], 'Volume': [1000]
        })
        
        mock_response = {
            "pattern_identified": "Doji",
            "reliability": "medium",
            "interpretation": "Indecision",
            "key_levels": {"support": 90, "resistance": 110},
            "confirmation_needed": ["Next candle close"]
        }
        
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response))]
        self.mock_client.messages.create.return_value = mock_message

        result = self.analyzer.interpret_pattern(recent_candles)
        
        self.assertEqual(result['pattern_identified'], "Doji")

    def test_generate_trading_narrative(self):
        trade_history = [{'action': 'buy', 'price': 100}]
        metrics = {'total_return': 0.05}
        
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Good session.")]
        self.mock_client.messages.create.return_value = mock_message

        result = self.analyzer.generate_trading_narrative(trade_history, metrics)
        
        self.assertEqual(result, "Good session.")

    def test_get_analysis_summary_empty(self):
        result = self.analyzer.get_analysis_summary()
        self.assertEqual(result['message'], 'No analysis history available')

if __name__ == '__main__':
    unittest.main()
