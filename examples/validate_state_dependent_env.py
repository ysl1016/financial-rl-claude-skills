"""
Validation test for StateDependentRewardEnv

This script validates that state-dependent rewards work correctly:
1. Buy gets high reward in oversold (RSI < 30)
2. Sell gets high reward in overbought (RSI > 70)
3. Same action gets different rewards in different market conditions

Author: Based on architecture_modification_plan.md
Date: 2025-11-23
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.environments.state_dependent_env import StateDependentRewardEnv


def create_test_data():
    """Create synthetic test data with various market conditions."""
    n_steps = 100
    
    # Create scenarios
    data = pd.DataFrame({
        'Close': np.linspace(100, 110, n_steps),
        'Volume': np.random.randint(1000, 2000, n_steps),
        'Returns': np.random.randn(n_steps) * 0.01,
        'Volatility': np.random.rand(n_steps) * 0.02,
    })
    
    # Add technical indicators with specific scenarios
    rsi = np.ones(n_steps) * 50  # Neutral
    rsi[0:10] = 25  # Oversold (Buy opportunity)
    rsi[20:30] = 75  # Overbought (Sell opportunity)
    rsi[40:50] = 50  # Neutral
    
    macd = np.zeros(n_steps)
    macd[0:10] = 5  # Positive (bullish)
    macd[20:30] = -5  # Negative (bearish)
    
    sma_20 = np.linspace(100, 108, n_steps)
    sma_50 = np.linspace(100, 107, n_steps)
    
    # Golden cross at step 50
    sma_20[50:] = sma_50[50:] + 2
    
    data['RSI'] = rsi
    data['MACD'] = macd
    data['SMA_20'] = sma_20
    data['SMA_50'] = sma_50
    
    return data


def test_reward_differentiation():
    """Test that same action gets different rewards in different states."""
    print("="*70)
    print("Test 1: Reward Differentiation")
    print("="*70)
    
    data = create_test_data()
    env = StateDependentRewardEnv(data)
    
    # Test Buy action in different market conditions
    print("\n【Buy Action in Different Conditions】")
    
    # Scenario 1: Oversold (RSI=25) → Should get high reward
    env.reset()
    env.current_step = 5  # Oversold region
    _, reward_buy_oversold, _, _ = env.step(1)  # Buy
    print(f"  Buy in Oversold (RSI=25): Reward = {reward_buy_oversold:.4f}")
    
    # Scenario 2: Overbought (RSI=75) → Should get penalty
    env.reset()
    env.current_step = 25  # Overbought region
    _, reward_buy_overbought, _, _ = env.step(1)  # Buy
    print(f"  Buy in Overbought (RSI=75): Reward = {reward_buy_overbought:.4f}")
    
    # Scenario 3: Neutral (RSI=50) → Should get neutral reward
    env.reset()
    env.current_step = 45  # Neutral region
    _, reward_buy_neutral, _, _ = env.step(1)  # Buy
    print(f"  Buy in Neutral (RSI=50): Reward = {reward_buy_neutral:.4f}")
    
    # Validation
    print(f"\n  Validation:")
    if reward_buy_oversold > reward_buy_neutral > reward_buy_overbought:
        print(f"  ✅ Buy rewards correctly differentiated!")
        print(f"     Oversold ({reward_buy_oversold:.2f}) > Neutral ({reward_buy_neutral:.2f}) > Overbought ({reward_buy_overbought:.2f})")
    else:
        print(f"  ❌ Buy rewards NOT correctly differentiated")
    
    # Test Sell action in different market conditions
    print("\n【Sell Action in Different Conditions】")
    
    # First, buy some shares
    env.reset()
    for _ in range(3):
        env.step(1)  # Buy 3 times to have shares
    
    # Scenario 1: Overbought (RSI=75) → Should get high reward
    env.current_step = 25  # Overbought region
    _, reward_sell_overbought, _, _ = env.step(2)  # Sell
    print(f"  Sell in Overbought (RSI=75): Reward = {reward_sell_overbought:.4f}")
    
    # Reset and buy again
    env.reset()
    for _ in range(3):
        env.step(1)
    
    # Scenario 2: Oversold (RSI=25) → Should get penalty
    env.current_step = 5  # Oversold region
    _, reward_sell_oversold, _, _ = env.step(2)  # Sell
    print(f"  Sell in Oversold (RSI=25): Reward = {reward_sell_oversold:.4f}")
    
    # Reset and buy again
    env.reset()
    for _ in range(3):
        env.step(1)
    
    # Scenario 3: Neutral (RSI=50) → Should get neutral reward
    env.current_step = 45  # Neutral region
    _, reward_sell_neutral, _, _ = env.step(2)  # Sell
    print(f"  Sell in Neutral (RSI=50): Reward = {reward_sell_neutral:.4f}")
    
    # Validation
    print(f"\n  Validation:")
    if reward_sell_overbought > reward_sell_neutral > reward_sell_oversold:
        print(f"  ✅ Sell rewards correctly differentiated!")
        print(f"     Overbought ({reward_sell_overbought:.2f}) > Neutral ({reward_sell_neutral:.2f}) > Oversold ({reward_sell_oversold:.2f})")
    else:
        print(f"  ❌ Sell rewards NOT correctly differentiated")
    
    return True


def test_action_balance():
    """Test that Buy and Sell can both be optimal in different states."""
    print("\n" + "="*70)
    print("Test 2: Action Balance")
    print("="*70)
    
    data = create_test_data()
    env = StateDependentRewardEnv(data)
    
    # Test in oversold region
    print("\n【Oversold Region (RSI=25)】")
    env.reset()
    env.current_step = 5
    
    _, reward_hold, _, _ = env.step(0)
    env.current_step = 5  # Reset to same step
    _, reward_buy, _, _ = env.step(1)
    
    print(f"  Hold: {reward_hold:.4f}")
    print(f"  Buy:  {reward_buy:.4f}")
    
    if reward_buy > reward_hold:
        print(f"  ✅ Buy > Hold in oversold region")
    else:
        print(f"  ❌ Buy should be > Hold in oversold")
    
    # Test in overbought region (need shares first)
    print("\n【Overbought Region (RSI=75)】")
    env.reset()
    for _ in range(3):
        env.step(1)  # Buy shares
    
    env.current_step = 25
    _, reward_hold, _, _ = env.step(0)
    
    env.reset()
    for _ in range(3):
        env.step(1)
    env.current_step = 25
    _, reward_sell, _, _ = env.step(2)
    
    print(f"  Hold: {reward_hold:.4f}")
    print(f"  Sell: {reward_sell:.4f}")
    
    if reward_sell > reward_hold:
        print(f"  ✅ Sell > Hold in overbought region")
    else:
        print(f"  ❌ Sell should be > Hold in overbought")
    
    return True


def test_profit_realization():
    """Test profit realization bonus."""
    print("\n" + "="*70)
    print("Test 3: Profit Realization Bonus")
    print("="*70)
    
    # Create data with price increase
    data = pd.DataFrame({
        'Close': np.linspace(100, 120, 50),  # 20% increase
        'Volume': np.ones(50) * 1000,
        'RSI': np.ones(50) * 50,
        'MACD': np.zeros(50),
        'SMA_20': np.linspace(100, 120, 50),
        'SMA_50': np.linspace(100, 119, 50),
        'Returns': np.ones(50) * 0.01,
        'Volatility': np.ones(50) * 0.02,
    })
    
    env = StateDependentRewardEnv(data)
    env.reset()
    
    # Buy at step 0 (price=100)
    env.step(1)
    buy_price = env.last_buy_price
    print(f"\n  Buy at price: ${buy_price:.2f}")
    
    # Move forward
    for _ in range(10):
        env.step(0)  # Hold
    
    # Sell at step 11 (price≈104)
    sell_price = data.loc[env.current_step, 'Close']
    _, reward_with_profit, _, _ = env.step(2)  # Sell
    
    profit_pct = (sell_price - buy_price) / buy_price * 100
    print(f"  Sell at price: ${sell_price:.2f}")
    print(f"  Profit: {profit_pct:.2f}%")
    print(f"  Reward: {reward_with_profit:.4f}")
    
    # Compare with sell at loss
    env.reset()
    env.step(1)  # Buy
    # Immediately sell (no profit)
    _, reward_no_profit, _, _ = env.step(2)
    
    print(f"\n  Sell with no profit: {reward_no_profit:.4f}")
    
    if reward_with_profit > reward_no_profit:
        print(f"  ✅ Profit realization bonus working!")
    else:
        print(f"  ❌ Profit realization bonus not working")
    
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("State-Dependent Reward Environment Validation")
    print("="*70)
    
    try:
        test_reward_differentiation()
        test_action_balance()
        test_profit_realization()
        
        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run 100 episodes training")
        print("  2. Check Buy/Sell balance")
        print("  3. Evaluate Alpha performance")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
