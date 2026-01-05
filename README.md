# LLM-Enhanced Reinforcement Learning Trading System

A sophisticated trading system that combines deep reinforcement learning with multi-agent LLM analysis for automated stock trading decisions. The system uses stable-baselines3 for RL training and LangChain with OpenRouter for coordinated multi-agent decision making.

## Overview

This project implements a hybrid trading system that leverages both traditional reinforcement learning and modern large language models:

- **RL Agent**: PPO (Proximal Policy Optimization) agent trained on historical market data
- **Multi-Agent LLM System**: Specialized AI agents (Technical, Fundamental, Sentiment, Risk) that provide strategic guidance
- **Real-time Analysis**: Streamlit-based interface for testing and visualizing LLM decisions
- **Market Data**: Integration with Exa API for news sentiment and synthetic market data generation

## Architecture

### Core Components

1. **Trading Environment** (`src/trading_env.py`)
   - Gymnasium-compatible environment for stock trading simulation
   - Continuous action space for position sizing (-1 to 1)
   - Realistic transaction costs (commission and slippage)
   - Portfolio tracking and performance metrics

2. **Data Handler** (`src/data_handler.py`)
   - Market data fetching and preprocessing
   - Technical indicator calculation (RSI, MACD, Bollinger Bands, etc.)
   - News aggregation via Exa API
   - Synthetic data generation for testing

3. **Multi-Agent System** (`src/multi_agent_system.py`)
   - Four specialized agents with distinct roles:
     - **Technical Analyst**: Chart patterns and technical indicators
     - **Fundamental Analyst**: Valuation and market conditions
     - **Sentiment Analyst**: News and market sentiment analysis
     - **Risk Manager**: Portfolio risk assessment
   - Coordinator agent that synthesizes recommendations into actionable decisions
   - Structured output using Pydantic models for consistency

4. **Training Pipeline** (`scripts/train.py`)
   - Multiple training modes (RL-only, LLM-enhanced, testing)
   - Automated evaluation and checkpoint management
   - TensorBoard integration for monitoring

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt -c constraints.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
OPENROUTER_API_KEY=your_openrouter_key_here
EXA_API_KEY=your_exa_key_here
```

### API Keys

- **OpenRouter API**: Required for LLM multi-agent analysis. Get your key at [openrouter.ai](https://openrouter.ai)
- **Exa API**: Required for real-time news fetching. Get your key at [exa.ai](https://exa.ai)

## Configuration

Edit `config/config.yaml` to customize the system:

```yaml
data:
  tickers: ["AAPL", "GOOGL", "MSFT"]
  start_date: "2022-01-01"
  end_date: "2024-01-01"
  test_split: 0.2

trading:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  max_position: 0.3

llm:
  model: "google/gemini-2.5-flash-lite-preview-09-2025"
  temperature: 0.7
  max_tokens: 2000

agents:
  technical_analyst:
    enabled: true
    weight: 0.25
  fundamental_analyst:
    enabled: true
    weight: 0.25
  sentiment_analyst:
    enabled: true
    weight: 0.25
  risk_manager:
    enabled: true
    weight: 0.25

training:
  algorithm: "ppo"
  timesteps: 100000
  eval_freq: 10000
```

## Usage

### 1. Test Exa API Integration

Verify your Exa API setup:
```bash
python test_exa.py
```

This will test news fetching for multiple tickers and time windows.

### 2. Train Traditional RL Agent

Train a PPO agent using only reinforcement learning:
```bash
python scripts/train.py --mode rl
```

Optional arguments:
- `--config`: Path to config file (default: `config/config.yaml`)
- `--timesteps`: Override training timesteps

### 3. Train LLM-Enhanced Agent

Train with LLM multi-agent strategic guidance:
```bash
python scripts/train.py --mode llm --timesteps 50000
```

The LLM system generates strategic recommendations that guide the RL agent's learning process.

### 4. Test Multi-Agent System

Launch interactive Streamlit interface:
```bash
python scripts/train.py --mode test-llm
```

Or run directly:
```bash
streamlit run streamlit_app.py
```

The interface provides:
- Real-time multi-agent analysis
- Individual agent recommendations with confidence scores
- Final trading decision with price targets
- Market data visualization
- Recent news sentiment analysis

## Project Structure

```
.
├── config/
│   └── config.yaml           # System configuration
├── scripts/
│   └── train.py              # Training and evaluation script
├── src/
│   ├── data_handler.py       # Data fetching and processing
│   ├── trading_env.py        # Trading environment
│   └── multi_agent_system.py # LLM multi-agent coordination
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs and TensorBoard data
├── streamlit_app.py          # Interactive testing interface
├── test_exa.py              # Exa API integration tests
├── requirements.txt          # Python dependencies
├── constraints.txt           # Dependency constraints
└── README.md                # This file
```

## Features

### Technical Indicators
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Simple Moving Averages (20-day, 50-day)
- Bollinger Bands with position calculation
- Volume analysis and momentum indicators

### Multi-Agent Analysis
Each agent provides:
- **Recommendation**: Buy, sell, or hold
- **Confidence**: 0-100% confidence score
- **Reasoning**: Detailed explanation of recommendation

The coordinator synthesizes these into:
- **Action**: Final trading decision
- **Position Size**: Recommended position as decimal (0.0-1.0)
- **Price Targets**: Entry, stop-loss, and take-profit levels
- **Time Horizon**: Short, medium, or long-term
- **Conviction**: Low, medium, or high conviction level

### Trading Environment Features
- Continuous action space for flexible position sizing
- Realistic transaction costs (commission + slippage)
- Portfolio state tracking (cash, shares, total value)
- Comprehensive history recording for analysis
- Normalized observations for stable learning

## Training Modes

### Mode 1: Traditional RL
Pure reinforcement learning without LLM guidance. Best for baseline performance and faster training.

### Mode 2: LLM-Enhanced
RL agent receives strategic recommendations from the multi-agent LLM system at key decision points. Combines pattern recognition from RL with reasoning from LLMs.

### Mode 3: Test LLM
Interactive testing mode with Streamlit UI. Allows real-time analysis of individual stocks with full visibility into agent reasoning.

## Output and Evaluation

### Training Outputs
- Model checkpoints saved to `checkpoints/`
- TensorBoard logs in `logs/`
- Best model automatically saved during evaluation

### Evaluation Metrics
- Mean episode return
- Standard deviation of returns
- Min/max returns across episodes
- Final portfolio value
- Percentage return on initial capital

### Analysis Downloads
The Streamlit interface allows downloading complete analysis results as JSON, including:
- Individual agent analyses
- Final decision with all parameters
- Market data snapshot
- News articles and sentiment

## Dependencies

### Core ML/RL
- PyTorch: Deep learning framework
- stable-baselines3: RL algorithms implementation
- gymnasium: Environment interface
- NumPy/Pandas: Data manipulation

### LLM and Multi-Agent
- LangChain: LLM orchestration
- LangChain-OpenAI: OpenRouter integration
- LangGraph: Multi-agent workflow

### Data and Utilities
- exa-py: News and content search API
- yfinance: Market data (optional)
- pandas-ta: Technical analysis
- Streamlit: Interactive UI
- PyYAML: Configuration management

## Known Limitations

1. **Market Data**: Currently uses synthetic data generation. Real market data integration via yfinance or other APIs can be added.

2. **News Analysis**: Exa API primarily provides news content, not structured OHLCV data. Price data is synthetic but realistic.

3. **API Rate Limits**: Exa API has rate limits. The system samples strategically to avoid excessive calls.

4. **Transaction Costs**: Simplified model using fixed commission and slippage percentages.

## Future Enhancements

- Integration with real market data APIs (Alpha Vantage, IEX Cloud)
- Live trading execution capabilities
- Advanced risk management (portfolio optimization, diversification)
- Backtesting framework with multiple strategies
- Paper trading mode for validation
- Additional agent types (macro analyst, options specialist)
- Ensemble methods combining multiple RL algorithms

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed with constraints:
```bash
pip install -r requirements.txt -c constraints.txt
```

**API Key Errors**: Verify `.env` file exists and contains valid keys:
```bash
cat .env
```

**Exa API Issues**: Test the integration separately:
```bash
python test_exa.py
```

**Training Issues**: Check TensorBoard for insights:
```bash
tensorboard --logdir logs/
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome. Please ensure:
- Code follows existing style and structure
- New features include appropriate tests
- Documentation is updated accordingly
- Commit messages are clear and descriptive

## Acknowledgments

- stable-baselines3 for RL implementations
- LangChain for multi-agent orchestration
- Exa for news and content search API
- OpenRouter for LLM access
# rl-trading1
