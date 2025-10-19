from fastapi.testclient import TestClient

from src.api.app import create_app
from src.models.grpo_agent import GRPOAgent

class DummyPackager:
    def load_model(self, model_name, version):
        state_dim = 14
        action_dim = 3
        model = GRPOAgent(state_dim, action_dim, device='cpu')
        env_config = {}
        metadata = {"model_name": model_name, "version": version}
        return model, env_config, metadata

class DummyOptimizer:
    def optimize_for_inference(self, model, optimization_level):
        return model


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr('src.api.app.ModelPackager', lambda: DummyPackager())
    monkeypatch.setattr('src.api.app.ModelOptimizer', lambda: DummyOptimizer())
    app = create_app('dummy/path')
    with TestClient(app) as client:
        features = [
            'RSI_norm', 'ForceIndex2_norm', '%K_norm', '%D_norm',
            'MACD_norm', 'MACDSignal_norm', 'BBWidth_norm', 'ATR_norm',
            'VPT_norm', 'VPT_MA_norm', 'OBV_norm', 'ROC_norm'
        ]
        market_data = {f: [0.0] for f in features}
        payload = {
            "market_data": market_data,
            "portfolio_state": {"position": 0.0, "portfolio_return": 0.0}
        }
        headers = {"X-API-KEY": "default_development_key"}

        response = client.post('/predict', json=payload, headers=headers)
        assert response.status_code == 200
        assert 'action' in response.json()
