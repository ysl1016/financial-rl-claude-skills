"""
100 Episodes Training with State-Dependent Reward Environment

This script creates a Colab notebook for training the existing CleanGRPOAgent
with the new StateDependentRewardEnv for 100 episodes.

Goal: Verify state-dependent rewards prevent extreme trading behaviors
Target: Alpha > 0%, Trades 10-20, Buy/Sell balance 30-70%
"""

import json
import os


def create_training_notebook():
    """Create Colab notebook for 100 episodes training."""
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "accelerator": "GPU"
        },
        "cells": []
    }
    
    # Cell 1: Title
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 100 Episodes Training with State-Dependent Rewards\n",
            "\n",
            "**Date**: 2025-11-23  \n",
            "**Goal**: Verify state-dependent reward effectiveness\n",
            "\n",
            "## Objectives\n",
            "\n",
            "- Alpha > 0% (vs -12.37% in Phase 3)\n",
            "- Trades: 10-20 (vs 0-1 in previous phases)\n",
            "- Buy/Sell balance: 30-70% (vs 100% extreme)\n",
            "\n",
            "## Expected Time\n",
            "\n",
            "20-30 minutes"
        ]
    })
    
    # Cell 2: Setup
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Mount Drive\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Install dependencies\n",
            "!pip install -q gym numpy pandas torch yfinance scikit-learn matplotlib seaborn tqdm\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "from torch.distributions import Categorical\n",
            "import matplotlib.pyplot as plt\n",
            "from collections import deque\n",
            "from tqdm.notebook import tqdm\n",
            "import yfinance as yf\n",
            "from datetime import datetime, timedelta\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f\"✓ Using device: {device}\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 3: Data Download
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Download real market data\n",
            "def download_and_process_data(symbol='SPY', days=730):\n",
            "    end_date = datetime.now()\n",
            "    start_date = end_date - timedelta(days=days)\n",
            "    \n",
            "    print(f\"Downloading {symbol} data...\")\n",
            "    ticker = yf.Ticker(symbol)\n",
            "    data = ticker.history(start=start_date, end=end_date)\n",
            "    \n",
            "    # Calculate technical indicators\n",
            "    data['Returns'] = data['Close'].pct_change()\n",
            "    data['Volatility'] = data['Returns'].rolling(window=20).std()\n",
            "    \n",
            "    # RSI\n",
            "    delta = data['Close'].diff()\n",
            "    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()\n",
            "    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()\n",
            "    rs = gain / loss\n",
            "    data['RSI'] = 100 - (100 / (1 + rs))\n",
            "    \n",
            "    # MACD\n",
            "    exp1 = data['Close'].ewm(span=12, adjust=False).mean()\n",
            "    exp2 = data['Close'].ewm(span=26, adjust=False).mean()\n",
            "    data['MACD'] = exp1 - exp2\n",
            "    \n",
            "    # SMA\n",
            "    data['SMA_20'] = data['Close'].rolling(window=20).mean()\n",
            "    data['SMA_50'] = data['Close'].rolling(window=50).mean()\n",
            "    \n",
            "    # Drop NaN\n",
            "    data = data.dropna()\n",
            "    \n",
            "    # Split train/test\n",
            "    split_idx = int(len(data) * 0.8)\n",
            "    train_data = data.iloc[:split_idx].copy()\n",
            "    test_data = data.iloc[split_idx:].copy()\n",
            "    \n",
            "    print(f\"✓ Train: {len(train_data)} days\")\n",
            "    print(f\"✓ Test: {len(test_data)} days\")\n",
            "    \n",
            "    return train_data, test_data\n",
            "\n",
            "train_data, test_data = download_and_process_data()\n",
            "train_data.head()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 4: Environment (same as before)
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# StateDependentRewardEnv (copy from phase1_state_dependent_test.ipynb Cell 4)\n",
            "import gym\n",
            "from gym import spaces\n",
            "\n",
            "class StateDependentRewardEnv(gym.Env):\n",
            "    # ... (same implementation as before)\n",
            "    pass  # Replace with full implementation\n",
            "\n",
            "print(\"✓ StateDependentRewardEnv defined\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 5: Agent
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# CleanGRPOAgent\n",
            "class CleanGRPOAgent:\n",
            "    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, device='cpu'):\n",
            "        self.device = device\n",
            "        self.action_dim = action_dim\n",
            "        \n",
            "        # Network\n",
            "        self.network = nn.Sequential(\n",
            "            nn.Linear(state_dim, hidden_dim),\n",
            "            nn.LayerNorm(hidden_dim),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(hidden_dim, hidden_dim),\n",
            "            nn.LayerNorm(hidden_dim),\n",
            "            nn.ReLU()\n",
            "        ).to(device)\n",
            "        \n",
            "        self.actor = nn.Linear(hidden_dim, action_dim).to(device)\n",
            "        \n",
            "        self.optimizer = optim.Adam(\n",
            "            list(self.network.parameters()) + list(self.actor.parameters()),\n",
            "            lr=lr\n",
            "        )\n",
            "        \n",
            "        # Epsilon-greedy\n",
            "        self.epsilon = 0.3\n",
            "        self.epsilon_decay = 0.99\n",
            "        self.epsilon_min = 0.01\n",
            "        \n",
            "        # Memory\n",
            "        self.memory = deque(maxlen=10000)\n",
            "        self.batch_size = 32\n",
            "    \n",
            "    def select_action(self, state, deterministic=False):\n",
            "        if not deterministic and np.random.rand() < self.epsilon:\n",
            "            return np.random.randint(self.action_dim)\n",
            "        \n",
            "        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)\n",
            "        with torch.no_grad():\n",
            "            features = self.network(state_tensor)\n",
            "            action_probs = torch.softmax(self.actor(features), dim=-1)\n",
            "            action = torch.argmax(action_probs, dim=-1)\n",
            "        \n",
            "        return action.item()\n",
            "    \n",
            "    def store_transition(self, state, action, reward, next_state, done):\n",
            "        self.memory.append((state, action, reward, next_state, done))\n",
            "    \n",
            "    def update(self):\n",
            "        if len(self.memory) < self.batch_size:\n",
            "            return\n",
            "        \n",
            "        # Sample batch\n",
            "        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)\n",
            "        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])\n",
            "        \n",
            "        states = torch.FloatTensor(np.array(states)).to(self.device)\n",
            "        actions = torch.LongTensor(actions).to(self.device)\n",
            "        rewards = torch.FloatTensor(rewards).to(self.device)\n",
            "        \n",
            "        # Policy loss\n",
            "        features = self.network(states)\n",
            "        action_probs = torch.softmax(self.actor(features), dim=-1)\n",
            "        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)\n",
            "        \n",
            "        loss = -(log_probs * rewards).mean()\n",
            "        \n",
            "        self.optimizer.zero_grad()\n",
            "        loss.backward()\n",
            "        self.optimizer.step()\n",
            "    \n",
            "    def decay_epsilon(self):\n",
            "        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)\n",
            "\n",
            "print(\"✓ CleanGRPOAgent defined\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 6: Training
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Training\n",
            "env = StateDependentRewardEnv(train_data)\n",
            "agent = CleanGRPOAgent(state_dim=10, action_dim=3, device=device)\n",
            "\n",
            "print(\"Starting 100 episodes training...\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "episode_rewards = []\n",
            "portfolio_values = []\n",
            "alphas = []\n",
            "trades_count = []\n",
            "\n",
            "for episode in tqdm(range(100), desc=\"Training\"):\n",
            "    state = env.reset()\n",
            "    done = False\n",
            "    episode_reward = 0\n",
            "    \n",
            "    while not done:\n",
            "        action = agent.select_action(state)\n",
            "        next_state, reward, done, info = env.step(action)\n",
            "        agent.store_transition(state, action, reward, next_state, done)\n",
            "        \n",
            "        if len(agent.memory) >= agent.batch_size:\n",
            "            agent.update()\n",
            "        \n",
            "        state = next_state\n",
            "        episode_reward += reward\n",
            "    \n",
            "    agent.decay_epsilon()\n",
            "    \n",
            "    episode_rewards.append(episode_reward)\n",
            "    portfolio_values.append(env.portfolio_values[-1])\n",
            "    alphas.append(info['alpha'])\n",
            "    trades_count.append(len(env.trades))\n",
            "    \n",
            "    if (episode + 1) % 20 == 0:\n",
            "        avg_pv = np.mean(portfolio_values[-20:])\n",
            "        avg_alpha = np.mean(alphas[-20:])\n",
            "        avg_trades = np.mean(trades_count[-20:])\n",
            "        print(f\"\\nEpisode {episode+1}/100:\")\n",
            "        print(f\"  Portfolio: ${avg_pv:,.0f}\")\n",
            "        print(f\"  Alpha: {avg_alpha*100:+.2f}%\")\n",
            "        print(f\"  Trades: {avg_trades:.1f}\")\n",
            "        print(f\"  Epsilon: {agent.epsilon:.3f}\")\n",
            "\n",
            "print(\"\\n✓ Training complete!\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 7: Test Evaluation
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Test evaluation\n",
            "print(\"=\"*70)\n",
            "print(\"Test Data Evaluation\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "test_env = StateDependentRewardEnv(test_data)\n",
            "state = test_env.reset()\n",
            "done = False\n",
            "test_actions = []\n",
            "\n",
            "while not done:\n",
            "    action = agent.select_action(state, deterministic=True)\n",
            "    test_actions.append(action)\n",
            "    state, _, done, info = test_env.step(action)\n",
            "\n",
            "# Results\n",
            "test_pv = np.array(test_env.portfolio_values)\n",
            "test_returns = np.diff(test_pv) / test_pv[:-1]\n",
            "test_sharpe = np.mean(test_returns) / (np.std(test_returns) + 1e-9) * np.sqrt(252)\n",
            "test_peak = np.maximum.accumulate(test_pv)\n",
            "test_dd = (test_peak - test_pv) / test_peak\n",
            "test_max_dd = np.max(test_dd)\n",
            "\n",
            "rl_return = (test_pv[-1] / test_pv[0] - 1) * 100\n",
            "bh_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100\n",
            "alpha = rl_return - bh_return\n",
            "\n",
            "action_counts = [test_actions.count(i) for i in range(3)]\n",
            "\n",
            "print(f\"\\nInitial: ${test_pv[0]:,.2f}\")\n",
            "print(f\"Final: ${test_pv[-1]:,.2f}\")\n",
            "print(f\"\\nRL Return: {rl_return:.2f}%\")\n",
            "print(f\"Buy & Hold: {bh_return:.2f}%\")\n",
            "print(f\"Alpha: {alpha:+.2f}%\")\n",
            "print(f\"Sharpe: {test_sharpe:.2f}\")\n",
            "print(f\"Max DD: {test_max_dd*100:.2f}%\")\n",
            "print(f\"Trades: {len(test_env.trades)}\")\n",
            "\n",
            "print(f\"\\nAction Distribution:\")\n",
            "print(f\"  Hold: {action_counts[0]} ({action_counts[0]/len(test_actions)*100:.1f}%)\")\n",
            "print(f\"  Buy: {action_counts[1]} ({action_counts[1]/len(test_actions)*100:.1f}%)\")\n",
            "print(f\"  Sell: {action_counts[2]} ({action_counts[2]/len(test_actions)*100:.1f}%)\")\n",
            "\n",
            "# Success criteria\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"Success Criteria\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "criteria = {\n",
            "    'Alpha > 0%': (alpha, 0.0, alpha > 0),\n",
            "    'Trades > 10': (len(test_env.trades), 10, len(test_env.trades) > 10),\n",
            "    'Buy 30-70%': (action_counts[1]/len(test_actions)*100, None, 30 < action_counts[1]/len(test_actions)*100 < 70),\n",
            "    'Sell > 0%': (action_counts[2]/len(test_actions)*100, 0, action_counts[2] > 0)\n",
            "}\n",
            "\n",
            "for name, (actual, target, passed) in criteria.items():\n",
            "    status = \"✅\" if passed else \"❌\"\n",
            "    if target is not None:\n",
            "        print(f\"{status} {name:<20} Actual: {actual:>6.2f} / Target: {target:>6.2f}\")\n",
            "    else:\n",
            "        print(f\"{status} {name:<20} Actual: {actual:>6.2f}%\")\n",
            "\n",
            "passed_count = sum(p for _, _, p in criteria.values())\n",
            "print(\"=\"*70)\n",
            "print(f\"\\nResult: {passed_count}/{len(criteria)} criteria passed\")\n",
            "\n",
            "if passed_count >= 3:\n",
            "    print(\"\\n🎉 SUCCESS! State-dependent rewards working!\")\n",
            "else:\n",
            "    print(\"\\n⚠️ Partial success - may need adjustment\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 8: Save Model
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Save model\n",
            "save_path = '/content/drive/MyDrive/financial-rl-trading/models/state_dependent_100ep.pt'\n",
            "torch.save({\n",
            "    'network_state_dict': agent.network.state_dict(),\n",
            "    'actor_state_dict': agent.actor.state_dict(),\n",
            "    'optimizer_state_dict': agent.optimizer.state_dict(),\n",
            "    'epsilon': agent.epsilon,\n",
            "    'test_results': {\n",
            "        'alpha': alpha,\n",
            "        'sharpe': test_sharpe,\n",
            "        'max_dd': test_max_dd,\n",
            "        'trades': len(test_env.trades),\n",
            "        'action_counts': action_counts\n",
            "    }\n",
            "}, save_path)\n",
            "\n",
            "print(f\"✓ Model saved: {save_path}\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    return notebook


def main():
    """Generate training notebook."""
    print("Generating 100 episodes training notebook...")
    
    notebook = create_training_notebook()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "train_100_episodes.ipynb")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Notebook created: {output_file}")
    print("\nNext steps:")
    print("  1. Upload to Colab")
    print("  2. Run all cells")
    print("  3. Wait 20-30 minutes")
    print("  4. Check results")


if __name__ == "__main__":
    main()
