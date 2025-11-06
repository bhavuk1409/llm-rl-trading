"""
Streamlit UI for LLM Trading System Testing
Real-time visualization of multi-agent analysis
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_handler import DataHandler
from multi_agent_system import AdvancedMultiAgentSystem
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def display_agent_analysis(analysis: dict, agent_name: str):
    """Display individual agent analysis in an expander."""
    with st.expander(f"🤖 {agent_name}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Recommendation:** `{analysis.get('recommendation', 'N/A').upper()}`")
            st.markdown(f"**Reasoning:** {analysis.get('reasoning', 'N/A')}")
        
        with col2:
            confidence = analysis.get('confidence', 0)
            st.metric("Confidence", f"{confidence}%")
            
            # Color-coded confidence bar
            if confidence >= 70:
                color = "green"
            elif confidence >= 40:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"""
            <div style="background-color: {color}; width: {confidence}%; height: 10px; border-radius: 5px;"></div>
            """, unsafe_allow_html=True)


def display_final_decision(decision: dict):
    """Display final trading decision with detailed metrics."""
    st.markdown("### 📊 Final Trading Decision")
    
    # Main action display
    action = decision.get('action', 'hold').upper()
    
    if action == 'BUY':
        st.success(f"## 🟢 {action}")
    elif action == 'SELL':
        st.error(f"## 🔴 {action}")
    else:
        st.info(f"## 🟡 {action}")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Position Size", f"{decision.get('position_size', 0):.1%}")
    
    with col2:
        st.metric("Confidence", f"{decision.get('confidence', 0)}%")
    
    with col3:
        conviction = decision.get('conviction', 'low').upper()
        st.metric("Conviction", conviction)
    
    with col4:
        st.metric("Time Horizon", decision.get('time_horizon', 'N/A').replace('-', ' ').title())
    
    # Price targets
    st.markdown("#### 🎯 Price Targets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Entry Price", f"${decision.get('entry_price', 0):.2f}")
    
    with col2:
        stop_loss = decision.get('stop_loss_price', 0)
        entry = decision.get('entry_price', 1)
        sl_pct = ((stop_loss - entry) / entry * 100) if entry else 0
        st.metric("Stop Loss", f"${stop_loss:.2f}", f"{sl_pct:.1f}%")
    
    with col3:
        take_profit = decision.get('take_profit_price', 0)
        tp_pct = ((take_profit - entry) / entry * 100) if entry else 0
        st.metric("Take Profit", f"${take_profit:.2f}", f"{tp_pct:.1f}%")
    
    # Reasoning
    st.markdown("#### 💡 Consolidated Reasoning")
    st.info(decision.get('reasoning', 'N/A'))


def display_market_data(market_data: dict):
    """Display market data in a clean layout."""
    st.markdown("### 📈 Market Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Close Price", f"${market_data.get('close', 0):.2f}")
    
    with col2:
        rsi = market_data.get('rsi', 50)
        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        st.metric("RSI", f"{rsi:.1f}", rsi_status)
    
    with col3:
        st.metric("MACD", f"{market_data.get('macd', 0):.2f}")
    
    with col4:
        volume_ratio = market_data.get('volume_ratio', 1)
        st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
    
    # Technical indicators table
    with st.expander("📊 Technical Indicators", expanded=False):
        tech_data = {
            "Indicator": ["SMA 20", "BB Position", "Momentum"],
            "Value": [
                f"${market_data.get('sma_20', 0):.2f}",
                f"{market_data.get('bb_position', 0):.2%}",
                f"{market_data.get('momentum', 0):.2f}"
            ]
        }
        st.table(pd.DataFrame(tech_data))


def display_news(news: list):
    """Display news articles."""
    st.markdown("### 📰 Recent News")
    
    if not news:
        st.warning("No news articles available")
        return
    
    for i, article in enumerate(news[:5], 1):
        with st.expander(f"📄 {article.get('title', 'Untitled')}", expanded=(i == 1)):
            st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
            st.markdown(f"**Published:** {article.get('published_date', 'Unknown')}")
            st.markdown(f"**Summary:** {article.get('summary', 'No summary available')}")
            st.markdown(f"[Read more]({article.get('url', '#')})")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="LLM Trading System Test",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM Multi-Agent Trading System")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load config
        config = load_config()
        
        ticker = st.selectbox(
            "Select Ticker",
            config['data']['tickers'],
            index=0
        )
        
        model = st.text_input(
            "LLM Model",
            value=config['llm']['model']
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config['llm']['temperature'],
            step=0.1
        )
        
        days_back = st.slider(
            "News Days Back",
            min_value=1,
            max_value=30,
            value=7
        )
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if not run_analysis:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to start")
        
        # Show sample architecture
        st.markdown("### 🏗️ System Architecture")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **🔧 Technical Analyst**
            - Chart patterns
            - Indicators (RSI, MACD)
            - Price trends
            """)
        
        with col2:
            st.markdown("""
            **💼 Fundamental Analyst**
            - Company valuation
            - Market conditions
            - Financial metrics
            """)
        
        with col3:
            st.markdown("""
            **📰 Sentiment Analyst**
            - News analysis
            - Market sentiment
            - Social signals
            """)
        
        with col4:
            st.markdown("""
            **⚠️ Risk Manager**
            - Portfolio risk
            - Position sizing
            - Stop loss levels
            """)
        
        return
    
    # Run analysis
    with st.spinner(f"🔄 Analyzing {ticker}..."):
        try:
            # Initialize system
            st.info("Initializing multi-agent system...")
            
            agent_system = AdvancedMultiAgentSystem(
                model=model,
                temperature=temperature,
                agent_config=config.get('agents', {})
            )
            
            # Fetch data
            st.info("Fetching market data and news...")
            
            handler = DataHandler(
                tickers=[ticker],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            
            df = handler.fetch_and_process()
            
            if df.empty:
                st.error("❌ No data available for this ticker")
                return
            
            # Get latest data point
            latest = df[df['ticker'] == ticker].iloc[-1]
            date_str = str(latest['date'])
            
            # Get market summary
            market_data = handler.get_market_summary(df, ticker, latest['date'])
            
            # Fetch news
            news = handler.fetch_news_exa(ticker, days_back=days_back)
            
            st.success("✅ Data loaded successfully!")
            
            # Display market data
            display_market_data(market_data)
            
            # Display news
            display_news(news)
            
            st.markdown("---")
            
            # Run agent analysis
            st.markdown("## 🤖 Agent Analysis")
            
            with st.spinner("Running multi-agent analysis..."):
                results = agent_system.analyze(
                    ticker=ticker,
                    date=date_str,
                    market_data=market_data,
                    news=news
                )
            
            # Display individual agent analyses
            agent_names = {
                'technical_analysis': '🔧 Technical Analyst',
                'fundamental_analysis': '💼 Fundamental Analyst',
                'sentiment_analysis': '📰 Sentiment Analyst',
                'risk_analysis': '⚠️ Risk Manager'
            }
            
            for key, name in agent_names.items():
                if key in results:
                    display_agent_analysis(results[key], name)
            
            st.markdown("---")
            
            # Display final decision
            if 'final_decision' in results:
                display_final_decision(results['final_decision'])
            
            # Download results as JSON
            st.markdown("---")
            st.download_button(
                label="📥 Download Full Analysis (JSON)",
                data=json.dumps(results, indent=2),
                file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()