import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import types
import importlib

# Stub utils package to avoid heavy dependencies
fake_utils = types.ModuleType("utils")
fake_indicators = types.ModuleType("indicators")

def calculate_technical_indicators(df):
    return df

fake_indicators.calculate_technical_indicators = calculate_technical_indicators
fake_utils.indicators = fake_indicators
sys.modules.setdefault("src.utils", fake_utils)
sys.modules.setdefault("src.utils.indicators", fake_indicators)

# Stub bayes_opt.logger dependency before importing project modules
fake_logger = types.ModuleType("logger")

class JSONLogger:
    pass

class ScreenLogger:
    pass

fake_logger.JSONLogger = JSONLogger
fake_logger.ScreenLogger = ScreenLogger

sys.modules.setdefault("bayes_opt.logger", fake_logger)

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Create a stub package for src.data to avoid executing its __init__
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [os.path.join(ROOT_DIR, "src")]
sys.modules.setdefault("src", src_pkg)

data_pkg = types.ModuleType("src.data")
data_pkg.__path__ = [os.path.join(ROOT_DIR, "src", "data")]
sys.modules.setdefault("src.data", data_pkg)

# Import module after stubs are in place
data_processor = importlib.import_module("src.data.data_processor")
download_stock_data = data_processor.download_stock_data
process_data = data_processor.process_data


class TestDownloadStockData(unittest.TestCase):
    def test_yfinance_failure_raises_value_error(self):
        with patch('src.data.data_processor.yf.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception('API failure')
            with self.assertRaises(ValueError):
                download_stock_data('FAIL')

    def test_empty_dataframe_raises_value_error(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch('src.data.data_processor.yf.Ticker', return_value=mock_ticker):
            with self.assertRaises(ValueError):
                download_stock_data('EMPTY')

    def test_missing_columns_raises_value_error(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({'Open': [1]})
        with patch('src.data.data_processor.yf.Ticker', return_value=mock_ticker):
            with self.assertRaises(ValueError):
                download_stock_data('MISSING')

    def test_process_data_propagates_exception(self):
        with patch('src.data.data_processor.download_stock_data', side_effect=ValueError('fail')):
            with self.assertRaises(ValueError):
                process_data('SPY')


if __name__ == '__main__':
    unittest.main()
