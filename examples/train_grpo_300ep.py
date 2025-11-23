"""
GRPO Training Script - 300 Episodes

Integrates:
- TrueGRPOAgent (Phase 3) - Critic-less GRPO with group sampling
- TriAxialRewardEnv (Phase 4) - Financial/Risk/Process rewards

Based on implementation plan and architecture documents.

Author: Based on DeepSeek-R1 GRPO architecture
Date: 2025-11-23
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.true_grpo_agent import TrueGRPOAgent
from src.environments.tri_axial_reward_env import TriAxialRewardEnv


def download_and_process_data(symbol='SPY', days=730):
    """Download and process market data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Downloading {symbol} data...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Calculate technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    
    # SMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Drop NaN
    data = data.dropna()
    
    # Split train/test (80/20)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"✓ Train: {len(train_data)} days")
    print(f"✓ Test: {len(test_data)} days")
    
    return train_data, test_data


def train_grpo_agent(
    agent,
    env,
    num_episodes=300,
    update_ref_every=10,
    save_every=50
):
    """
    Train GRPO agent with group sampling.
    
    Args:
        agent: TrueGRPOAgent instance
        env: TriAxialRewardEnv instance
        num_episodes: Number of training episodes (default: 300)
        update_ref_every: Update reference network every N episodes
        save_every: Save checkpoint every N episodes
    """
    print(f"\nStarting GRPO training for {num_episodes} episodes...")
    print("="*70)
    
    # Training metrics
    episode_rewards = []
    episode_alphas = []
    episode_trades = []
    episode_portfolio_values = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Collect group samples for this episode
        states_batch = []
        actions_batch = []
        old_log_probs_batch = []
        advantages_batch = []
        
        while not done:
            # Group sampling (G=8)
            actions, log_probs = agent.get_action_group(state, G=8)
            
            # Execute each action in the group
            group_rewards = []
            for action in actions:
                # Save current env state (for rollback)
                env_state = {
                    'cash': env.cash,
                    'shares': env.shares,
                    'portfolio_value': env.portfolio_value,
                    'current_step': env.current_step
                }
                
                # Execute action
                next_state, reward, done_temp, info = env.step(action)
                group_rewards.append(reward)
                
                # Rollback environment (except for first action)
                if action != actions[0]:
                    env.cash = env_state['cash']
                    env.shares = env_state['shares']
                    env.portfolio_value = env_state['portfolio_value']
                    env.current_step = env_state['current_step']
            
            # Calculate relative advantages
            advantages = agent.calculate_relative_advantage(group_rewards)
            
            # Store transitions for batch update
            for i, (action, log_prob, advantage) in enumerate(zip(actions, log_probs, advantages)):
                states_batch.append(state)
                actions_batch.append(action)
                old_log_probs_batch.append(log_prob)
                advantages_batch.append(advantage)
            
            # Use best action for actual progression
            best_idx = np.argmax(advantages)
            best_action = actions[best_idx]
            state, reward, done, info = env.step(best_action)
            episode_reward += reward
            
            # Batch update every 32 steps
            if len(states_batch) >= 32:
                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states_batch)).to(agent.device)
                actions_tensor = torch.LongTensor(actions_batch).to(agent.device)
                old_log_probs_tensor = torch.stack(old_log_probs_batch).to(agent.device)
                advantages_tensor = torch.FloatTensor(advantages_batch).to(agent.device)
                
                # GRPO update
                loss, policy_loss, kl_div = agent.grpo_loss(
                    states_tensor,
                    actions_tensor,
                    old_log_probs_tensor,
                    advantages_tensor
                )
                
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), 0.5)
                agent.optimizer.step()
                
                agent.training_step += 1
                
                # Clear batch
                states_batch = []
                actions_batch = []
                old_log_probs_batch = []
                advantages_batch = []
        
        # Episode metrics
        episode_rewards.append(episode_reward)
        episode_alphas.append(info['alpha'])
        episode_trades.append(info['trades'])
        episode_portfolio_values.append(info['portfolio_value'])
        
        # Update reference network periodically
        if (episode + 1) % update_ref_every == 0:
            agent.update_reference_network()
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_alpha = np.mean(episode_alphas[-50:])
            avg_trades = np.mean(episode_trades[-50:])
            avg_pv = np.mean(episode_portfolio_values[-50:])
            
            print(f"\nEpisode {episode+1}/{num_episodes}:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Alpha: {avg_alpha*100:+.2f}%")
            print(f"  Avg Trades: {avg_trades:.1f}")
            print(f"  Avg Portfolio: ${avg_pv:,.0f}")
        
        # Save checkpoint
        if (episode + 1) % save_every == 0:
            checkpoint_path = f'checkpoints/grpo_ep{episode+1}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print("\n✓ Training complete!")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_alphas': episode_alphas,
        'episode_trades': episode_trades,
        'episode_portfolio_values': episode_portfolio_values
    }


def evaluate_agent(agent, env):
    """Evaluate trained agent on test data."""
    print("\n" + "="*70)
    print("Test Data Evaluation")
    print("="*70)
    
    state = env.reset()
    done = False
    test_actions = []
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        test_actions.append(action)
        state, _, done, info = env.step(action)
    
    # Calculate metrics
    test_pv = np.array(env.portfolio_values)
    test_returns = np.diff(test_pv) / test_pv[:-1]
    test_sharpe = np.mean(test_returns) / (np.std(test_returns) + 1e-9) * np.sqrt(252)
    test_peak = np.maximum.accumulate(test_pv)
    test_dd = (test_peak - test_pv) / test_peak
    test_max_dd = np.max(test_dd)
    
    rl_return = (test_pv[-1] / test_pv[0] - 1) * 100
    bh_return = (env.data['Close'].iloc[-1] / env.data['Close'].iloc[0] - 1) * 100
    alpha = rl_return - bh_return
    
    action_counts = [test_actions.count(i) for i in range(3)]
    
    print(f"\nInitial: ${test_pv[0]:,.2f}")
    print(f"Final: ${test_pv[-1]:,.2f}")
    print(f"\nRL Return: {rl_return:.2f}%")
    print(f"Buy & Hold: {bh_return:.2f}%")
    print(f"Alpha: {alpha:+.2f}%")
    print(f"Sharpe: {test_sharpe:.2f}")
    print(f"Max DD: {test_max_dd*100:.2f}%")
    print(f"Trades: {len(env.trades)}")
    
    print(f"\nAction Distribution:")
    print(f"  Hold: {action_counts[0]} ({action_counts[0]/len(test_actions)*100:.1f}%)")
    print(f"  Buy: {action_counts[1]} ({action_counts[1]/len(test_actions)*100:.1f}%)")
    print(f"  Sell: {action_counts[2]} ({action_counts[2]/len(test_actions)*100:.1f}%)")
    
    # Success criteria (from implementation plan)
    print("\n" + "="*70)
    print("Success Criteria")
    print("="*70)
    
    criteria = {
        'Alpha > +2%': (alpha, 2.0, alpha > 2.0),
        'Sharpe > 0.8': (test_sharpe, 0.8, test_sharpe > 0.8),
        'Max DD < 15%': (test_max_dd*100, 15.0, test_max_dd < 0.15),
        'Trades > 15': (len(env.trades), 15, len(env.trades) > 15),
        'Buy 30-70%': (action_counts[1]/len(test_actions)*100, None, 30 < action_counts[1]/len(test_actions)*100 < 70)
    }
    
    for name, (actual, target, passed) in criteria.items():
        status = "✅" if passed else "❌"
        if target is not None:
            print(f"{status} {name:<20} Actual: {actual:>6.2f} / Target: {target:>6.2f}")
        else:
            print(f"{status} {name:<20} Actual: {actual:>6.2f}%")
    
    passed_count = sum(p for _, _, p in criteria.values())
    print("="*70)
    print(f"\nResult: {passed_count}/{len(criteria)} criteria passed")
    
    if passed_count >= 4:
        print("\n🎉 SUCCESS! GRPO agent working!")
    else:
        print("\n⚠️ Partial success - may need adjustment")
    
    return {
        'alpha': alpha,
        'sharpe': test_sharpe,
        'max_dd': test_max_dd,
        'trades': len(env.trades),
        'action_counts': action_counts,
        'passed_count': passed_count
    }


def main():
    """Main training pipeline."""
    print("="*70)
    print("GRPO Training - 300 Episodes")
    print("="*70)
    
    # Download data
    train_data, test_data = download_and_process_data()
    
    # Initialize environment
    train_env = TriAxialRewardEnv(train_data)
    test_env = TriAxialRewardEnv(test_data)
    
    # Initialize agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = TrueGRPOAgent(
        state_dim=10,
        action_dim=3,
        hidden_dim=256,
        lr=3e-4,
        group_size=8,
        device=device
    )
    
    print(f"\n✓ Agent initialized")
    print(f"  - Device: {device}")
    print(f"  - Group size: {agent.group_size}")
    print(f"  - Parameters: {sum(p.numel() for p in agent.policy_network.parameters()):,}")
    
    # Train
    training_results = train_grpo_agent(
        agent,
        train_env,
        num_episodes=300,
        update_ref_every=10,
        save_every=50
    )
    
    # Evaluate
    test_results = evaluate_agent(agent, test_env)
    
    # Save final model
    final_path = 'models/grpo_final.pt'
    os.makedirs('models', exist_ok=True)
    agent.save(final_path)
    print(f"\n✓ Final model saved: {final_path}")
    
    # Plot training progress
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(training_results['episode_rewards'])
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(np.array(training_results['episode_alphas']) * 100)
    axes[0, 1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Training Alpha')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Alpha (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(training_results['episode_trades'])
    axes[1, 0].set_title('Training Trades')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(training_results['episode_portfolio_values'])
    axes[1, 1].axhline(100000, color='r', linestyle='--', alpha=0.5, label='Initial')
    axes[1, 1].set_title('Training Portfolio Value')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Portfolio Value ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/grpo_training_300ep.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved: results/grpo_training_300ep.png")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
