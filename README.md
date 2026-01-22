# LLM-Enhanced Trading Analysis Platform

A professional enterprise-grade trading system that combines multi-agent LLM analysis for intelligent stock trading decisions. The system uses Groq's fast LLM inference with specialized AI agents to provide comprehensive trading analysis.

**🚀 Live Demo**: https://llm-trading-platform.streamlit.app

## Overview

This project implements an advanced multi-agent trading analysis system that leverages modern large language models:

- **Multi-Agent LLM System**: Four specialized AI agents (Technical, Fundamental, Sentiment, Risk) that provide strategic guidance
- **Professional UI**: Clean, enterprise-grade Streamlit interface for real-time analysis
- **Fast Inference**: Powered by Groq API for rapid LLM responses
- **Real-time News**: Integration with Exa API for news sentiment analysis
- **Market Analysis**: Comprehensive technical indicators and market data processing

## Live Application

**Try it now**: https://llm-trading-platform.streamlit.app

Simply select a ticker and click "Run Analysis" to get instant trading insights from our multi-agent system.

## Features

### Multi-Agent Analysis System

Four specialized AI agents analyze different aspects of trading decisions:

1. **Technical Analyst**
   - Chart patterns and price action analysis
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Moving averages and momentum signals
   - Volume analysis

2. **Fundamental Analyst**
   - Company valuation assessment
   - Market conditions evaluation
   - Financial metrics analysis
   - Industry trends

3. **Sentiment Analyst**
   - Real-time news processing
   - Market sentiment analysis
   - Social signals and trends
   - News impact assessment

4. **Risk Manager**
   - Portfolio risk assessment
   - Position sizing recommendations
   - Stop loss and take profit levels
   - Risk/reward analysis

### Professional UI

- Clean, minimal design without clutter
- Single-page workflow for ease of use
- Tab-based agent analysis view
- Real-time data visualization
- Export functionality for results

## Architecture

### Core Components

1. **Multi-Agent System** (`src/multi_agent_system.py`)
   - Groq-powered LLM agents for fast inference
   - Structured outputs using Pydantic models
   - Coordinator agent for decision synthesis
   - Configurable agent weights and parameters

2. **Data Handler** (`src/data_handler.py`)
   - Market data fetching and preprocessing
   - Technical indicator calculation
   - News aggregation via Exa API
   - Synthetic data generation for testing

3. **Trading Environment** (`src/trading_env.py`)
   - Gymnasium-compatible environment
   - Continuous action space for position sizing
   - Realistic transaction costs
   - Portfolio tracking and metrics

4. **Web Application** (`app.py`)
   - Professional Streamlit interface
   - Modular render functions
   - Clean data presentation
   - JSON export functionality

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/bhavuk1409/llm-rl-trading.git
cd llm-rl-trading
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt -c constraints.txt
```

4. **Configure API keys**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
EXA_API_KEY=your_exa_api_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### API Keys

- **Groq API**: Get your free key at [console.groq.com](https://console.groq.com)
  - Fast LLM inference
  - Multiple model options (LLaMA 3.3, Mixtral, etc.)
  - Free tier available

- **Exa API**: Get your key at [exa.ai](https://exa.ai)
  - Real-time news search
  - AI-powered content retrieval
  - Free tier available

## Configuration

Edit `config/config.yaml` to customize the system:

```yaml
# Supported tickers
data:
  tickers: ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
  start_date: "2022-01-01"
  end_date: "2024-01-01"

# Trading parameters
trading:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  max_position: 0.3

# LLM configuration (Groq)
llm:
  model: "llama-3.3-70b-versatile"
  temperature: 0.7
  max_tokens: 2000

# Agent weights
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
```

## Usage

### Web Interface

1. Open the app (live or local)
2. Select a ticker from the dropdown
3. Click "Run Analysis"
4. View results in organized sections:
   - Market Data
   - Technical Indicators
   - Recent News
   - Agent Analysis (tabbed view)
   - Final Trading Decision
5. Export results as JSON if needed

### Command Line Training

Train an RL agent (optional advanced feature):

```bash
# Train with RL only
python scripts/train.py --mode rl

# Train with LLM enhancement
python scripts/train.py --mode llm --timesteps 50000
```

## Project Structure

```
llm-rl-trading/
├── app.py                  # Main Streamlit application
├── config/
│   └── config.yaml        # System configuration
├── src/
│   ├── data_handler.py    # Data fetching and processing
│   ├── multi_agent_system.py  # LLM multi-agent logic
│   └── trading_env.py     # RL trading environment
├── scripts/
│   └── train.py          # Training scripts
├── requirements.txt       # Python dependencies
├── constraints.txt        # Version constraints
└── .env.example          # API key template
```

## Technical Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration
- **Groq API**: Fast LLM inference
- **Exa API**: Real-time news search

### ML/Data Libraries
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **pandas-ta**: Technical analysis
- **PyYAML**: Configuration management

### Optional (RL Training)
- **PyTorch**: Deep learning framework
- **stable-baselines3**: RL algorithms
- **Gymnasium**: Environment interface

## Features Highlight

### Real-time Analysis
- Instant LLM-powered insights
- Fast inference with Groq API
- Real-time news integration
- Comprehensive technical indicators

### Professional UI
- Clean, minimal design
- Enterprise-grade appearance
- No unnecessary clutter
- Mobile-responsive layout

### Multi-Agent Intelligence
- Four specialized AI agents
- Coordinated decision-making
- Confidence scoring
- Detailed reasoning

### Export Capabilities
- JSON format export
- Complete analysis data
- Timestamped results
- Easy integration

## Development

### Code Style
- Modular function design
- Clear separation of concerns
- Comprehensive docstrings
- Type hints where applicable

### Testing
```bash
# Syntax validation
python -m py_compile app.py

# Test imports
python -c "from src.multi_agent_system import AdvancedMultiAgentSystem"
```

## Deployment

This app is deployed on Streamlit Cloud:
- **Live URL**: [https://llm-rl-trading.streamlit.app](https://llm-rl-trading.streamlit.app)
- **Automatic updates** from main branch
- **Environment variables** configured in Streamlit Cloud

To deploy your own instance:
1. Fork this repository
2. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add API keys as secrets
5. Deploy!

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- **Groq** for fast LLM inference
- **Exa** for AI-powered news search
- **LangChain** for LLM orchestration
- **Streamlit** for the web framework
- **stable-baselines3** for RL algorithms

## Support

- **Issues**: [GitHub Issues](https://github.com/bhavuk1409/llm-rl-trading/issues)
- **Live Demo**: [https://llm-trading-platform.streamlit.app)

---

**Made with ❤️ using Groq, Exa, and Streamlit**
