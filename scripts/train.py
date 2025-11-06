"""
Training Script
Train RL agent with optional LLM multi-agent enhancement
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_handler import DataHandler
from trading_env import TradingEnv
from multi_agent_system import AdvancedMultiAgentSystem
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_rl_only(config: dict, use_llm: bool = False):
    """
    Train RL agent (with optional LLM enhancement).
    
    Args:
        config: Configuration dictionary
        use_llm: If True, use LLM for enhanced features
    """
    logger.info("="*60)
    logger.info("TRAINING RL AGENT")
    if use_llm:
        logger.info("Mode: LLM-Enhanced")
    else:
        logger.info("Mode: Traditional RL Only")
    logger.info("="*60)
    
    # 1. Load data
    logger.info("\n📥 Step 1: Loading data...")
    handler = DataHandler(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    df = handler.fetch_and_process()
    train_df, test_df = handler.train_test_split(df, config['data']['test_split'])
    
    logger.info(f"✓ Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # 2. Create environments
    logger.info("\n🏗️  Step 2: Creating environments...")
    train_env = TradingEnv(
        df=train_df,
        initial_capital=config['trading']['initial_capital'],
        commission=config['trading']['commission'],
        slippage=config['trading']['slippage']
    )
    
    test_env = TradingEnv(
        df=test_df,
        initial_capital=config['trading']['initial_capital'],
        commission=config['trading']['commission'],
        slippage=config['trading']['slippage']
    )
    
    logger.info(f"✓ Environment created: {train_env.observation_space.shape}")
    
    # 3. Create agent
    logger.info("\n🤖 Step 3: Creating PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="logs/"
    )
    
    logger.info("✓ PPO agent created")
    
    # 4. Train
    logger.info("\n🎓 Step 4: Training...")
    timesteps = config['training']['timesteps']
    logger.info(f"Timesteps: {timesteps:,}")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path="checkpoints/",
        log_path="logs/",
        eval_freq=config['training']['eval_freq'],
        deterministic=True,
        render=False
    )
    
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        tb_log_name="ppo_trading"
    )
    
    logger.info("✓ Training completed!")
    
    # 5. Save model
    logger.info("\n💾 Step 5: Saving model...")
    model.save("checkpoints/ppo_final")
    logger.info("✓ Model saved to checkpoints/ppo_final.zip")
    
    # 6. Evaluate
    logger.info("\n📊 Step 6: Final evaluation...")
    evaluate_model(model, test_env, n_episodes=10)
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("="*60)


def train_with_llm_advisor(config: dict):
    """
    Train RL agent with LLM multi-agent advisor.
    LLM provides strategic guidance, RL does execution.
    """
    logger.info("="*60)
    logger.info("TRAINING WITH LLM MULTI-AGENT ADVISOR")
    logger.info("="*60)
    
    # 1. Setup
    logger.info("\n📥 Step 1: Setting up...")
    handler = DataHandler(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    df = handler.fetch_and_process()
    train_df, test_df = handler.train_test_split(df, config['data']['test_split'])
    
    # Initialize multi-agent system
    llm_system = AdvancedMultiAgentSystem(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        agent_config=config['agents']
    )
    
    logger.info("✓ LLM multi-agent system initialized")
    
    # 2. Generate LLM recommendations for key dates
    logger.info("\n🤖 Step 2: Generating LLM strategic recommendations...")
    
    # Sample every 5 days to avoid excessive API calls
    sample_dates = train_df['date'].unique()[::5]
    llm_recommendations = []
    
    for i, date in enumerate(sample_dates[:10]):  # Limit for demo
        for ticker in config['data']['tickers']:
            try:
                # Get market data for this date
                row = train_df[
                    (train_df['date'] == date) & 
                    (train_df['ticker'] == ticker)
                ]
                
                if len(row) == 0:
                    continue
                
                market_data = handler.get_market_summary(train_df, ticker, date)
                news = handler.generate_mock_news(ticker)
                
                # Get LLM analysis
                result = llm_system.analyze(
                    ticker=ticker,
                    date=str(date),
                    market_data=market_data,
                    news=news
                )
                
                decision = result['final_decision']
                
                # Store recommendation
                llm_recommendations.append({
                    'date': date,
                    'ticker': ticker,
                    'action': decision.get('action', 'hold'),
                    'position_size': decision.get('position_size', 0.1),
                    'confidence': decision.get('confidence', 50) / 100
                })
                
                logger.info(f"  ✓ {ticker} on {date}: {decision.get('action', 'N/A').upper()}")
                
            except Exception as e:
                logger.error(f"Error processing {ticker} on {date}: {e}")
                continue
    
    logger.info(f"✓ Generated {len(llm_recommendations)} LLM recommendations")
    
    # 3. Train RL agent
    logger.info("\n🎓 Step 3: Training RL agent with LLM guidance...")
    
    train_env = TradingEnv(
        df=train_df,
        initial_capital=config['trading']['initial_capital'],
        commission=config['trading']['commission'],
        slippage=config['trading']['slippage']
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="logs/"
    )
    
    model.learn(
        total_timesteps=config['training']['timesteps'],
        tb_log_name="ppo_llm_enhanced"
    )
    
    logger.info("✓ Training completed!")
    
    # 4. Save
    model.save("checkpoints/ppo_llm_enhanced")
    logger.info("✓ Model saved")
    
    # 5. Evaluate
    test_env = TradingEnv(df=test_df)
    evaluate_model(model, test_env)
    
    logger.info("\n" + "="*60)
    logger.info("✅ LLM-ENHANCED TRAINING COMPLETED!")
    logger.info("="*60)


def evaluate_model(model, env, n_episodes: int = 10):
    """Evaluate trained model."""
    logger.info(f"Evaluating model over {n_episodes} episodes...")
    
    episode_returns = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
        
        final_return = info['return']
        episode_returns.append(final_return)
        
        logger.info(f"  Episode {episode + 1}: Return = {final_return:.2f}%")
    
    logger.info("\n📊 Evaluation Results:")
    logger.info(f"  Mean Return: {np.mean(episode_returns):.2f}%")
    logger.info(f"  Std Return: {np.std(episode_returns):.2f}%")
    logger.info(f"  Min Return: {np.min(episode_returns):.2f}%")
    logger.info(f"  Max Return: {np.max(episode_returns):.2f}%")


def test_llm_system(config: dict):
    """Test LLM multi-agent system independently."""
    logger.info("="*60)
    logger.info("TESTING LLM MULTI-AGENT SYSTEM")
    logger.info("="*60)
    
    # Initialize
    llm_system = AdvancedMultiAgentSystem(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        agent_config=config['agents']
    )
    
    # Sample market data
    market_data = {
        'close': 180.5,
        'volume': 50000000,
        'sma_20': 178.2,
        'rsi': 65.3,
        'macd': 1.2,
        'bb_position': 0.7,
        'volume_ratio': 1.3,
        'momentum': 5.2
    }
    
    news = [
        {'title': 'Strong quarterly earnings reported', 'sentiment': 'positive'},
        {'title': 'New product launch successful', 'sentiment': 'positive'},
        {'title': 'Market volatility concerns', 'sentiment': 'negative'}
    ]
    
    # Run analysis
    result = llm_system.analyze(
        ticker="AAPL",
        date="2024-01-16",
        market_data=market_data,
        news=news
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("MULTI-AGENT ANALYSIS RESULTS")
    logger.info("="*60)
    
    logger.info("\n📊 Agent Analyses:")
    for key in ['technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'risk_analysis']:
        if key in result and result[key]:
            logger.info(f"\n{key.replace('_', ' ').title()}:")
            analysis = result[key]
            logger.info(f"  Recommendation: {analysis.get('recommendation', 'N/A')}")
            logger.info(f"  Confidence: {analysis.get('confidence', 0)}%")
            logger.info(f"  Reasoning: {analysis.get('reasoning', 'N/A')}")
    
    logger.info("\n🎯 Final Decision:")
    decision = result['final_decision']
    logger.info(f"  Action: {decision.get('action', 'N/A').upper()}")
    logger.info(f"  Conviction: {decision.get('conviction', 'N/A')}")
    logger.info(f"  Position Size: {decision.get('position_size', 0):.1%}")
    logger.info(f"  Confidence: {decision.get('confidence', 0)}%")
    logger.info(f"  Entry: {decision.get('entry_price', 'N/A')}")
    logger.info(f"  Stop Loss: {decision.get('stop_loss_price', 'N/A')}")
    logger.info(f"  Take Profit: {decision.get('take_profit_price', 'N/A')}")
    logger.info(f"  Time Horizon: {decision.get('time_horizon', 'N/A')}")
    logger.info(f"\n  Reasoning: {decision.get('reasoning', 'N/A')}")
    
    logger.info("\n" + "="*60)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train trading agent")
    parser.add_argument(
        '--mode', '-m',
        choices=['rl', 'llm', 'test-llm'],
        default='rl',
        help='Training mode: rl (traditional), llm (LLM-enhanced), test-llm (test multi-agent)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=None,
        help='Override training timesteps'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override timesteps if provided
    if args.timesteps:
        config['training']['timesteps'] = args.timesteps
    
    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run appropriate mode
    if args.mode == 'rl':
        train_rl_only(config, use_llm=False)
    elif args.mode == 'llm':
        train_with_llm_advisor(config)
    elif args.mode == 'test-llm':
        test_llm_system(config)


if __name__ == "__main__":
    main()