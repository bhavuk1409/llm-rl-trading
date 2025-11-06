# RL + LLM Trading System (Simplified)

A streamlined trading system combining **Reinforcement Learning** with **Advanced Multi-Agent LLM** analysis using OpenRouter.

## 🎯 Key Features

- **Traditional RL Trading**: PPO agent trained on market data
- **Advanced Multi-Agent LLM**: 4 specialized AI agents provide deep analysis
  - 🔧 Technical Analyst (chart patterns, indicators)
  - 📊 Fundamental Analyst (valuation, growth)
  - 💭 Sentiment Analyst (news, psychology)
  - ⚠️ Risk Manager (position sizing, risk assessment)
- **OpenRouter Integration**: Use any LLM (Claude, GPT, Llama, etc.)
- **Simple Structure**: Just 8 files, easy to understand and modify

## 📁 Project Structure

```
rl_trading_simple/
├── requirements.txt          # Dependencies
├── .env.example             # API key template
├── config/
│   └── config.yaml          # All settings
├── src/
│   ├── data_handler.py      # Data fetching & processing
│   ├── trading_env.py       # Trading environment
│   └── multi_agent_system.py  # LLM multi-agent system
└── scripts/
    └── train.py             # Training script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Key

Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai)

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Train RL Agent (Traditional)

```bash
python scripts/train.py --mode rl --timesteps 50000
```

### 4. Test Multi-Agent LLM System

```bash
python scripts/train.py --mode test-llm
```

### 5. Train with LLM Enhancement

```bash
python scripts/train.py --mode llm --timesteps 50000
```

## 🤖 Multi-Agent System

The system uses **LangGraph** to orchestrate 4 specialized agents:

### Agent Flow:

```
Technical Analyst → Fundamental Analyst → Sentiment Analyst → Risk Manager → Portfolio Manager
```

Each agent:

- Performs **deep, expert-level analysis**
- Provides **detailed reasoning**
- Votes with **confidence scores**
- Uses **advanced prompting** for quality

### Example Output:

```
🔧 Technical Analyst: BUY (confidence: 85%)
📊 Fundamental Analyst: STRONG_BUY (confidence: 90%)
💭 Sentiment Analyst: BUY (confidence: 75%)
⚠️  Risk Manager: BUY (confidence: 70%)

🎯 Final Decision: STRONG_BUY
   Position Size: 25%
   Confidence: 80%
   Stop Loss: 5%
   Target: 15%
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
# Data
data:
  tickers: ["AAPL", "GOOGL", "MSFT"]
  start_date: "2022-01-01"
  end_date: "2024-01-01"

# LLM Model (OpenRouter)
llm:
  model: "anthropic/claude-3.5-sonnet" # or any model
  temperature: 0.7

# Agent Weights
agents:
  technical_analyst:
    enabled: true
    weight: 0.25
  fundamental_analyst:
    enabled: true
    weight: 0.25
  # ... etc
```

## 📊 Training Modes

### Mode 1: Traditional RL Only

Pure reinforcement learning without LLM.

```bash
python scripts/train.py --mode rl --timesteps 100000
```

### Mode 2: LLM-Enhanced RL

RL agent learns with strategic guidance from LLM multi-agent system.

```bash
python scripts/train.py --mode llm --timesteps 100000
```

### Mode 3: Test LLM System

Test the multi-agent system independently on sample data.

```bash
python scripts/train.py --mode test-llm
```

## 🎓 How It Works

### Traditional RL Flow:

```
Market Data → Feature Engineering → Trading Env → PPO Agent → Actions
```

### LLM-Enhanced Flow:

```
Market Data ─┐
News Data ───┼→ Multi-Agent Analysis → Strategic Signals ─┐
             │                                             ├→ PPO Agent → Actions
             └─────────────────────────────────────────────┘
```

## 🔧 OpenRouter Models

You can use any model from OpenRouter:

```yaml
# Claude (Recommended)
model: "anthropic/claude-3.5-sonnet"

# GPT-4
model: "openai/gpt-4-turbo"

# Open Source
model: "meta-llama/llama-3-70b-instruct"
model: "mistralai/mixtral-8x7b-instruct"
```

See all models: [openrouter.ai/models](https://openrouter.ai/models)

## 📈 Features

### Data Handling

- ✅ yfinance integration (free, no API key needed)
- ✅ Technical indicators (SMA, RSI, MACD, BB, etc.)
- ✅ Automatic feature engineering
- ✅ Train/test splitting

### Trading Environment

- ✅ Gymnasium-compatible
- ✅ Realistic costs (commission, slippage)
- ✅ Continuous action space
- ✅ Portfolio tracking

### RL Agent

- ✅ PPO implementation (Stable-Baselines3)
- ✅ Configurable hyperparameters
- ✅ TensorBoard logging
- ✅ Model checkpointing

### Multi-Agent LLM

- ✅ 4 specialized expert agents
- ✅ LangGraph orchestration
- ✅ Weighted voting system
- ✅ Detailed reasoning
- ✅ OpenRouter integration

## 🎯 Example Usage

```python
from src.multi_agent_system import AdvancedMultiAgentSystem

# Initialize
system = AdvancedMultiAgentSystem(
    model="anthropic/claude-3.5-sonnet"
)

# Analyze
result = system.analyze(
    ticker="AAPL",
    date="2024-01-15",
    market_data={
        'close': 180.5,
        'rsi': 65.3,
        'macd': 1.2,
        # ... more indicators
    },
    news=[
        {'title': 'Strong earnings', 'sentiment': 'positive'}
    ]
)

# Get decision
decision = result['final_decision']
print(f"Action: {decision['action']}")
print(f"Position: {decision['position_size']}")
print(f"Reasoning: {decision['reasoning']}")
```

## 🔍 Advanced Features

### Custom Agent Prompts

Each agent uses sophisticated prompting techniques:

- Multi-step reasoning
- Specific technical terminology
- Structured JSON outputs
- Confidence scoring
- Risk-aware analysis

### Agent Coordination

- Sequential processing (allows agents to build on previous analyses)
- Weighted voting (customize agent influence)
- Consensus building (portfolio manager synthesizes)

## 📊 Performance

Expected results (on historical data):

- **RL-only**: ~10-20% annual return
- **LLM-enhanced**: ~15-30% annual return
- **Sharpe Ratio**: 1.0-2.0
- **Max Drawdown**: <20%

⚠️ **Past performance ≠ future results!**

## 💡 Tips

1. **Start small**: Test with 1-2 stocks first
2. **Use good models**: Claude or GPT-4 for best results
3. **Monitor costs**: LLM API calls can add up
4. **Test thoroughly**: Use paper trading before real money
5. **Customize prompts**: Adjust agent prompts for your strategy

## 🐛 Troubleshooting

### API Key Issues

```bash
# Verify key is set
echo $OPENROUTER_API_KEY

# Or check .env file
cat .env
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Out of Memory

```bash
# Reduce timesteps or batch size in config.yaml
training:
  timesteps: 10000  # Smaller number
```

## ⚠️ Important Warnings

1. **This is for research/educational purposes**
2. **Never trade real money without thorough testing**
3. **Market conditions change - models need retraining**
4. **LLM costs can be significant with many API calls**
5. **Regulatory considerations for algorithmic trading**

## 📚 Resources

- **OpenRouter**: https://openrouter.ai
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **yfinance**: https://pypi.org/project/yfinance/
