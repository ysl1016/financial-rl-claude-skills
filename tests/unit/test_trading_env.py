import unittest
import pandas as pd

from src.models.trading_env import TradingEnv


class TestTradingEnv(unittest.TestCase):
    """Tests for the basic TradingEnv."""

    def _create_env(self, prices):
        data = pd.DataFrame({"Close": prices})
        # Add required indicator columns with neutral values
        cols = [
            "RSI_norm",
            "ForceIndex2_norm",
            "%K_norm",
            "%D_norm",
            "MACD_norm",
            "MACDSignal_norm",
            "BBWidth_norm",
            "ATR_norm",
            "VPT_norm",
            "VPT_MA_norm",
            "OBV_norm",
            "ROC_norm",
        ]
        for col in cols:
            data[col] = 0.0

        env = TradingEnv(
            data=data,
            initial_capital=1000,
            trading_cost=0.0,
            slippage=0.0,
            risk_free_rate=0.0,
            max_position_size=1.0,
            stop_loss_pct=0.02,
        )
        env.reset()
        return env

    def test_buy_profit(self):
        env = self._create_env([100, 110])
        _, reward, done, _ = env.step(1)
        self.assertAlmostEqual(env.portfolio_values[-1], 1010.0)
        self.assertTrue(reward > 0)
        self.assertTrue(done)

    def test_buy_loss(self):
        env = self._create_env([100, 90])
        _, reward, done, _ = env.step(1)
        self.assertAlmostEqual(env.portfolio_values[-1], 990.0)
        self.assertTrue(reward < 0)
        self.assertTrue(done)

