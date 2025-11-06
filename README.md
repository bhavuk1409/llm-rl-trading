# RL + LLM Trading System

A comprehensive trading system combining Reinforcement Learning (RL) agents with Large Language Model (LLM) capabilities for stock market prediction and trading.

## 🎯 Project Overview

This project implements a hybrid trading system with two main phases:

1. **Phase 1: Traditional RL Trading** - Pure reinforcement learning agents (PPO, DQN, DDPG, TD3) trained on market data
2. **Phase 2: LLM-Enhanced Trading** - Integration of LLM agents for sentiment analysis, strategy planning, and multi-agent decision-making using LangGraph

## 📁 Project Structure

```
rl_llm_trading/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── __init__.py
│   ├── data_fetcher.py          # Download market data (yfinance, Alpaca)
│   ├── data_processor.py        # Feature engineering & preprocessing
│   └── news_fetcher.py          # Fetch news & sentiment data
├── environments/
│   ├── __init__.py
│   ├── trading_env.py           # Gym-style trading environment
│   └── portfolio_env.py         # Portfolio management environment
├── agents/
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py         # PPO implementation
│   │   ├── dqn_agent.py         # DQN/DDQN implementation
│   │   ├── ddpg_agent.py        # DDPG implementation
│   │   └── td3_agent.py         # TD3 implementation
│   └── llm/
│       ├── __init__.py
│       ├── sentiment_analyzer.py # LLM-based sentiment analysis
│       ├── strategy_planner.py   # LLM strategy generation
│       ├── multi_agent.py        # Multi-agent LLM system
│       └── rag_memory.py         # RAG-based memory system
├── models/
│   └── networks.py              # Neural network architectures
├── utils/
│   ├── __init__.py
│   ├── config_loader.py         # Configuration utilities
│   ├── metrics.py               # Trading metrics calculation
│   ├── visualization.py         # Plotting & visualization
│   └── logger.py                # Logging utilities
├── scripts/
│   ├── train_rl.py              # Train traditional RL agents
│   ├── train_llm_rl.py          # Train LLM-enhanced agents
│   ├── backtest.py              # Backtesting script
│   └── evaluate.py              # Evaluation script
├── tests/
│   └── test_*.py                # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rl_llm_trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

1. Create a `.env` file in the root directory:

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
NEWS_API_KEY=your_news_api_key

# Optional
WANDB_API_KEY=your_wandb_key
```

2. Configure settings in `config/config.yaml`

## 📊 Phase 1: Traditional RL Trading

### Step 1: Data Preparation

```bash
# Download and preprocess market data
python scripts/prepare_data.py --tickers AAPL GOOGL MSFT --start-date 2020-01-01
```

### Step 2: Train RL Agent

```bash
# Train PPO agent
python scripts/train_rl.py --algorithm ppo --timesteps 500000

# Train DQN agent
python scripts/train_rl.py --algorithm dqn --timesteps 500000

# Train DDPG agent
python scripts/train_rl.py --algorithm ddpg --timesteps 500000
```

### Step 3: Evaluate & Backtest

```bash
# Backtest trained agent
python scripts/backtest.py --model checkpoints/ppo_best.zip --start-date 2023-01-01

# Evaluate performance
python scripts/evaluate.py --model checkpoints/ppo_best.zip
```

## 🤖 Phase 2: LLM-Enhanced Trading

### LLM Integration Patterns

1. **LLM as Feature Extractor**

   - Processes news, filings, social media
   - Generates sentiment scores and event flags
   - Feeds enriched features to RL agent

2. **LLM as Strategy Planner**

   - Generates high-level trading strategies
   - RL agent executes with optimal timing and sizing

3. **Multi-Agent System**
   - Multiple specialized LLM agents (fundamental, technical, sentiment)
   - Collaborative decision-making using LangGraph
   - RL agent or aggregator translates to actions

### Training LLM-Enhanced Agents

```bash
# Train with LLM sentiment features
python scripts/train_llm_rl.py --mode sentiment --algorithm ppo

# Train with LLM strategy planner
python scripts/train_llm_rl.py --mode planner --algorithm td3

# Train multi-agent system
python scripts/train_llm_rl.py --mode multi_agent --algorithm ppo
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- **Data sources**: Tickers, date ranges, features
- **Environment**: Initial capital, commission, slippage
- **RL algorithms**: Hyperparameters for PPO, DQN, DDPG, TD3
- **LLM settings**: Provider, model, temperature
- **Risk management**: Position limits, stop loss, max drawdown

## 📈 Key Features

### Traditional RL

- ✅ Multiple RL algorithms (PPO, DQN, DDPG, TD3, SAC)
- ✅ Realistic market simulation (commission, slippage)
- ✅ Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- ✅ Walk-forward backtesting
- ✅ Comprehensive metrics (Sharpe, Sortino, Max DD)

### LLM Integration

- ✅ Sentiment analysis from news & social media
- ✅ Strategy generation and planning
- ✅ Multi-agent debate and consensus
- ✅ RAG-based memory for historical context
- ✅ LangGraph for agent orchestration

### Risk Management

- ✅ Position sizing limits
- ✅ Stop loss and take profit
- ✅ Maximum drawdown control
- ✅ Volatility targeting

## 📊 Evaluation Metrics

- **Returns**: Total return, CAGR, excess returns
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk**: Maximum drawdown, volatility, VaR, CVaR
- **Trading**: Win rate, profit factor, average trade, turnover

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 🚨 Important Notes

### Overfitting & Data Leakage

- Use walk-forward backtesting
- Separate train/validation/test sets
- Be cautious of look-ahead bias

### Transaction Costs

- Always include realistic commission and slippage
- Model market impact for large orders

### LLM Considerations

- API costs can be significant
- Latency may affect real-time trading
- Cache LLM outputs when possible

### Regulatory & Ethical

- Algorithmic trading is heavily regulated
- Be aware of market manipulation concerns
- Test thoroughly before live deployment

## 📚 References

- FinRL: https://github.com/AI4Finance-Foundation/FinRL
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- LangGraph: https://langchain-ai.github.io/langgraph/
- Trading Agents: https://github.com/TauricResearch/TradingAgents
