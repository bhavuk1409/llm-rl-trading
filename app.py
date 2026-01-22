"""
Enterprise Trading Analysis Platform
Multi-Agent LLM Trading System
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
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Custom CSS for professional styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    h1 {
        font-weight: 300;
        letter-spacing: -0.5px;
        color: #1f2937;
    }
    
    h2, h3 {
        font-weight: 400;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Section dividers */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: none;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 4px;
        height: 3rem;
        font-weight: 500;
        background-color: #3b82f6;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        border: none;
    }
    
    /* Card-like containers */
    .analysis-card {
        background-color: #f9fafb;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Clean expander styling */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-buy {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .status-sell {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .status-hold {
        background-color: #fef3c7;
        color: #92400e;
    }
</style>
"""



def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load system configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def render_market_data(market_data: dict):
    """Render market data section."""
    st.subheader("Market Data")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Close Price", f"${market_data.get('close', 0):.2f}")
    
    with cols[1]:
        rsi = market_data.get('rsi', 50)
        st.metric("RSI", f"{rsi:.1f}")
    
    with cols[2]:
        st.metric("MACD", f"{market_data.get('macd', 0):.2f}")
    
    with cols[3]:
        volume_ratio = market_data.get('volume_ratio', 1)
        st.metric("Volume Ratio", f"{volume_ratio:.2f}x")


def render_technical_indicators(market_data: dict):
    """Render technical indicators section."""
    st.subheader("Technical Indicators")
    
    with st.container():
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("SMA 20", f"${market_data.get('sma_20', 0):.2f}")
            st.metric("SMA 50", f"${market_data.get('sma_50', 0):.2f}")
        
        with cols[1]:
            bb_pos = market_data.get('bb_position', 0)
            st.metric("Bollinger Position", f"{bb_pos:.2%}")
            st.metric("Momentum", f"{market_data.get('momentum', 0):.2f}")
        
        with cols[2]:
            st.metric("Volume", f"{market_data.get('volume', 0):,.0f}")
            st.metric("Volume SMA", f"{market_data.get('volume_sma', 0):,.0f}")


def render_news_section(news: list):
    """Render recent news section."""
    st.subheader("Recent News")
    
    if not news or len(news) == 0:
        st.info("No recent news articles available")
        return
    
    for article in news[:5]:
        with st.expander(article.get('title', 'Untitled')):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.caption(f"Source: {article.get('source', 'Unknown')}")
                st.write(article.get('summary', 'No summary available'))
            
            with col2:
                st.caption(article.get('published_date', 'Unknown'))
                if article.get('url'):
                    st.link_button("Read More", article['url'])


def render_agent_analysis(results: dict):
    """Render individual agent analysis section."""
    st.subheader("Agent Analysis")
    
    agents = [
        ('technical_analysis', 'Technical Analyst'),
        ('fundamental_analysis', 'Fundamental Analyst'),
        ('sentiment_analysis', 'Sentiment Analyst'),
        ('risk_analysis', 'Risk Manager')
    ]
    
    tabs = st.tabs([name for _, name in agents])
    
    for idx, (key, name) in enumerate(agents):
        with tabs[idx]:
            if key in results and results[key]:
                analysis = results[key]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    recommendation = analysis.get('recommendation', 'N/A').upper()
                    
                    if recommendation == 'BUY':
                        st.markdown(f'<span class="status-badge status-buy">{recommendation}</span>', unsafe_allow_html=True)
                    elif recommendation == 'SELL':
                        st.markdown(f'<span class="status-badge status-sell">{recommendation}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="status-badge status-hold">{recommendation}</span>', unsafe_allow_html=True)
                    
                    st.write("")
                    st.write(analysis.get('reasoning', 'No reasoning provided'))
                
                with col2:
                    confidence = analysis.get('confidence', 0)
                    st.metric("Confidence", f"{confidence}%")
            else:
                st.info(f"No analysis available from {name}")


def render_final_decision(decision: dict):
    """Render final trading decision section."""
    st.subheader("Final Trading Decision")
    
    # Main action and key metrics
    with st.container():
        cols = st.columns([1, 1, 1, 1])
        
        action = decision.get('action', 'hold').upper()
        
        with cols[0]:
            st.markdown("**Action**")
            if action == 'BUY':
                st.markdown(f'<div style="font-size: 2rem; color: #059669; font-weight: 600;">{action}</div>', unsafe_allow_html=True)
            elif action == 'SELL':
                st.markdown(f'<div style="font-size: 2rem; color: #dc2626; font-weight: 600;">{action}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size: 2rem; color: #d97706; font-weight: 600;">{action}</div>', unsafe_allow_html=True)
        
        with cols[1]:
            position_size = decision.get('position_size', 0)
            st.metric("Position Size", f"{position_size:.1%}")
        
        with cols[2]:
            confidence = decision.get('confidence', 0)
            st.metric("Confidence", f"{confidence}%")
        
        with cols[3]:
            conviction = decision.get('conviction', 'low').title()
            st.metric("Conviction", conviction)
    
    st.write("")
    
    # Price targets
    with st.container():
        cols = st.columns(4)
        
        entry = decision.get('entry_price', 0) or 0
        stop_loss = decision.get('stop_loss_price', 0) or 0
        take_profit = decision.get('take_profit_price', 0) or 0
        
        with cols[0]:
            time_horizon = decision.get('time_horizon', 'N/A').replace('-', ' ').title()
            st.metric("Time Horizon", time_horizon)
        
        with cols[1]:
            st.metric("Entry Price", f"${entry:.2f}")
        
        with cols[2]:
            sl_pct = ((stop_loss - entry) / entry * 100) if entry and stop_loss else 0
            st.metric("Stop Loss", f"${stop_loss:.2f}", f"{sl_pct:.1f}%")
        
        with cols[3]:
            tp_pct = ((take_profit - entry) / entry * 100) if entry and take_profit else 0
            st.metric("Take Profit", f"${take_profit:.2f}", f"{tp_pct:.1f}%")
    
    st.write("")
    
    # Reasoning
    st.markdown("**Analysis Summary**")
    st.info(decision.get('reasoning', 'No reasoning provided'))


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Trading Analysis Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.title("Trading Analysis Platform")
    st.caption("Multi-Agent LLM Trading System")
    
    st.write("")
    
    # Load configuration
    config = load_config()
    
    # Ticker selection (centered at top)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        ticker = st.selectbox(
            "Select Ticker Symbol",
            options=config['data']['tickers'],
            index=0,
            label_visibility="visible"
        )
        
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
    
    st.divider()
    
    # Show results or placeholder
    if not run_analysis:
        st.info("Select a ticker symbol and click 'Run Analysis' to begin")
        
        with st.container():
            st.subheader("System Overview")
            
            cols = st.columns(4)
            
            with cols[0]:
                st.markdown("**Technical Analyst**")
                st.caption("Analyzes chart patterns, technical indicators, and price trends")
            
            with cols[1]:
                st.markdown("**Fundamental Analyst**")
                st.caption("Evaluates company valuation and market conditions")
            
            with cols[2]:
                st.markdown("**Sentiment Analyst**")
                st.caption("Processes news and market sentiment signals")
            
            with cols[3]:
                st.markdown("**Risk Manager**")
                st.caption("Assesses portfolio risk and position sizing")
        
        return
    
    # Run analysis
    with st.spinner("Running analysis..."):
        try:
            # Initialize systems
            agent_system = AdvancedMultiAgentSystem(
                model=config['llm']['model'],
                temperature=config['llm']['temperature'],
                agent_config=config.get('agents', {})
            )
            
            handler = DataHandler(
                tickers=[ticker],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            
            # Fetch and process data
            df = handler.fetch_and_process()
            
            if df.empty:
                st.error("No data available for the selected ticker")
                return
            
            # Get latest data
            latest = df[df['ticker'] == ticker].iloc[-1]
            date_str = str(latest['date'])
            market_data = handler.get_market_summary(df, ticker, latest['date'])
            news = handler.fetch_news_exa(ticker, days_back=7)
            
            # Run multi-agent analysis
            results = agent_system.analyze(
                ticker=ticker,
                date=date_str,
                market_data=market_data,
                news=news
            )
            
            # Render results
            render_market_data(market_data)
            
            st.divider()
            
            render_technical_indicators(market_data)
            
            st.divider()
            
            render_news_section(news)
            
            st.divider()
            
            render_agent_analysis(results)
            
            st.divider()
            
            if 'final_decision' in results:
                render_final_decision(results['final_decision'])
            
            st.divider()
            
            # Export option
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="Export Analysis (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            logger.exception("Analysis failed")


if __name__ == "__main__":
    main()