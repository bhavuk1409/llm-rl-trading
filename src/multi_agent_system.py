"""
Advanced Multi-Agent Trading System
Uses LangGraph for sophisticated agent coordination with OpenRouter LLMs
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import TypedDict, List, Dict, Annotated
import operator
import logging
import json
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TradingState(TypedDict):
    """State shared across all agents."""
    ticker: str
    date: str
    market_data: Dict
    news: List[Dict]
    
    # Agent analyses
    technical_analysis: Dict
    fundamental_analysis: Dict
    sentiment_analysis: Dict
    risk_analysis: Dict
    
    # Agent votes with confidence
    agent_votes: Annotated[List[Dict], operator.add]
    
    # Final decision
    final_decision: Dict


class AdvancedMultiAgentSystem:
    """
    Sophisticated multi-agent trading system using LangGraph.
    Each agent is an expert that performs deep analysis.
    """
    
    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.7,
        agent_config: Dict = None
    ):
        """Initialize multi-agent system with OpenRouter."""
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
            max_tokens=2000
        )
        
        self.agent_config = agent_config or {
            'technical_analyst': {'enabled': True, 'weight': 0.25},
            'fundamental_analyst': {'enabled': True, 'weight': 0.25},
            'sentiment_analyst': {'enabled': True, 'weight': 0.25},
            'risk_manager': {'enabled': True, 'weight': 0.25}
        }
        
        # Build agent graph
        self.graph = self._build_graph()
        
        logger.info(f"Multi-agent system initialized with model: {model}")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow with parallel agent execution."""
        workflow = StateGraph(TradingState)
        
        # Add all agent nodes
        workflow.add_node("technical_analyst", self.technical_analyst)
        workflow.add_node("fundamental_analyst", self.fundamental_analyst)
        workflow.add_node("sentiment_analyst", self.sentiment_analyst)
        workflow.add_node("risk_manager", self.risk_manager)
        workflow.add_node("portfolio_manager", self.portfolio_manager)
        
        # Set entry point
        workflow.set_entry_point("technical_analyst")
        
        # Create execution flow
        workflow.add_edge("technical_analyst", "fundamental_analyst")
        workflow.add_edge("fundamental_analyst", "sentiment_analyst")
        workflow.add_edge("sentiment_analyst", "risk_manager")
        workflow.add_edge("risk_manager", "portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        
        return workflow.compile()
    
    def technical_analyst(self, state: TradingState) -> TradingState:
        """
        Deep technical analysis using price action, indicators, and patterns.
        """
        if not self.agent_config['technical_analyst']['enabled']:
            return state
        
        logger.info(f"🔧 Technical Analyst analyzing {state['ticker']}...")
        
        market = state['market_data']
        
        prompt = f"""You are an expert Technical Analyst with 20 years of experience in quantitative trading.

Analyze {state['ticker']} for {state['date']}:

Market Data:
- Price: ${market.get('close', 'N/A')}
- RSI: {market.get('rsi', 'N/A')}
- MACD: {market.get('macd', 'N/A')}
- BB Position: {market.get('bb_position', 'N/A')}
- Volume Ratio: {market.get('volume_ratio', 'N/A')}
- Momentum: {market.get('momentum', 'N/A')}
- SMA20: {market.get('sma_20', 'N/A')}

Perform DEEP technical analysis:

1. **Trend Analysis**: Multi-timeframe trend direction and strength
2. **Momentum**: RSI, MACD, momentum indicators interpretation
3. **Support/Resistance**: Key price levels based on data
4. **Volume Analysis**: Volume patterns and implications
5. **Pattern Recognition**: Chart patterns or setups
6. **Entry/Exit**: Optimal entry price and stop loss levels

Provide your analysis as JSON:
{{
  "trend": "strong_uptrend|uptrend|neutral|downtrend|strong_downtrend",
  "momentum": "very_strong|strong|neutral|weak|very_weak",
  "key_levels": {{"support": <price>, "resistance": <price>}},
  "volume_signal": "accumulation|distribution|neutral",
  "pattern": "description of any pattern",
  "recommendation": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": <0-100>,
  "target_price": <price>,
  "stop_loss": <price>,
  "reasoning": "detailed 2-3 sentence explanation"
}}

Be specific and actionable. Use precise technical terminology."""

        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_json(response.content)
            
            state['technical_analysis'] = analysis
            state['agent_votes'].append({
                'agent': 'technical_analyst',
                'recommendation': analysis.get('recommendation', 'hold'),
                'confidence': analysis.get('confidence', 50) / 100.0,
                'weight': self.agent_config['technical_analyst']['weight']
            })
            
            logger.info(f"✓ Technical: {analysis.get('recommendation', 'N/A')} "
                       f"(confidence: {analysis.get('confidence', 0)}%)")
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            state['technical_analysis'] = {'error': str(e)}
        
        return state
    
    def fundamental_analyst(self, state: TradingState) -> TradingState:
        """
        Fundamental analysis considering valuation, growth, and macro factors.
        """
        if not self.agent_config['fundamental_analyst']['enabled']:
            return state
        
        logger.info(f"📊 Fundamental Analyst analyzing {state['ticker']}...")
        
        prompt = f"""You are an expert Fundamental Analyst specializing in equity valuation and corporate finance.

Analyze {state['ticker']} for {state['date']}:

Technical Context: {state.get('technical_analysis', {}).get('trend', 'N/A')}

Perform comprehensive fundamental analysis:

1. **Valuation**: Assess if stock is undervalued, fairly valued, or overvalued
2. **Growth Prospects**: Analyze growth trajectory and sustainability
3. **Competitive Position**: Market position and competitive advantages
4. **Financial Health**: Balance sheet strength, cash flow quality
5. **Industry Trends**: Sector dynamics and positioning
6. **Catalyst Analysis**: Upcoming events or catalysts
7. **Risk Factors**: Key business and financial risks

Provide analysis as JSON:
{{
  "valuation": "deeply_undervalued|undervalued|fair|overvalued|severely_overvalued",
  "growth_outlook": "exceptional|strong|moderate|weak|declining",
  "competitive_strength": "dominant|strong|average|weak|struggling",
  "financial_health": "excellent|good|fair|concerning|poor",
  "catalysts": ["catalyst1", "catalyst2"],
  "risks": ["risk1", "risk2"],
  "recommendation": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": <0-100>,
  "fair_value_estimate": <price or null>,
  "time_horizon": "short|medium|long",
  "reasoning": "detailed 2-3 sentence explanation"
}}

Base your analysis on logical reasoning about the company's fundamentals."""

        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_json(response.content)
            
            state['fundamental_analysis'] = analysis
            state['agent_votes'].append({
                'agent': 'fundamental_analyst',
                'recommendation': analysis.get('recommendation', 'hold'),
                'confidence': analysis.get('confidence', 50) / 100.0,
                'weight': self.agent_config['fundamental_analyst']['weight']
            })
            
            logger.info(f"✓ Fundamental: {analysis.get('recommendation', 'N/A')} "
                       f"(confidence: {analysis.get('confidence', 0)}%)")
            
        except Exception as e:
            logger.error(f"Fundamental analysis error: {e}")
            state['fundamental_analysis'] = {'error': str(e)}
        
        return state
    
    def sentiment_analyst(self, state: TradingState) -> TradingState:
        """
        Sentiment analysis from news, social media, and market psychology.
        """
        if not self.agent_config['sentiment_analyst']['enabled']:
            return state
        
        logger.info(f"💭 Sentiment Analyst analyzing {state['ticker']}...")
        
        news_summary = "\n".join([
            f"- {item.get('title', 'N/A')} (sentiment: {item.get('sentiment', 'neutral')})"
            for item in state.get('news', [])
        ]) or "No recent news"
        
        prompt = f"""You are an expert Sentiment Analyst specializing in behavioral finance and market psychology.

Analyze sentiment for {state['ticker']} on {state['date']}:

Recent News:
{news_summary}

Technical Context: {state.get('technical_analysis', {}).get('trend', 'N/A')}
Fundamental Context: {state.get('fundamental_analysis', {}).get('growth_outlook', 'N/A')}

Perform deep sentiment analysis:

1. **News Sentiment**: Analyze tone, implications, and market reaction
2. **Market Psychology**: Fear/greed indicators, sentiment extremes
3. **Social Sentiment**: Retail vs institutional sentiment
4. **Sentiment Momentum**: Trend in sentiment (improving/deteriorating)
5. **Contrarian Indicators**: Identify potential sentiment extremes
6. **Short-term Impact**: Expected price impact from sentiment

Provide analysis as JSON:
{{
  "overall_sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
  "news_sentiment_score": <-1.0 to 1.0>,
  "social_sentiment": "euphoric|optimistic|neutral|pessimistic|panic",
  "sentiment_trend": "rapidly_improving|improving|stable|deteriorating|rapidly_deteriorating",
  "fear_greed_index": <0-100>,
  "contrarian_signal": "extreme_optimism|none|extreme_pessimism",
  "recommendation": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": <0-100>,
  "short_term_impact": "very_positive|positive|neutral|negative|very_negative",
  "reasoning": "detailed 2-3 sentence explanation"
}}

Consider both the news content and market psychology."""

        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_json(response.content)
            
            state['sentiment_analysis'] = analysis
            state['agent_votes'].append({
                'agent': 'sentiment_analyst',
                'recommendation': analysis.get('recommendation', 'hold'),
                'confidence': analysis.get('confidence', 50) / 100.0,
                'weight': self.agent_config['sentiment_analyst']['weight']
            })
            
            logger.info(f"✓ Sentiment: {analysis.get('recommendation', 'N/A')} "
                       f"(confidence: {analysis.get('confidence', 0)}%)")
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            state['sentiment_analysis'] = {'error': str(e)}
        
        return state
    
    def risk_manager(self, state: TradingState) -> TradingState:
        """
        Comprehensive risk assessment and position sizing.
        """
        if not self.agent_config['risk_manager']['enabled']:
            return state
        
        logger.info(f"⚠️  Risk Manager assessing {state['ticker']}...")
        
        votes = state.get('agent_votes', [])
        votes_summary = "\n".join([
            f"- {v['agent']}: {v['recommendation']} (conf: {v['confidence']:.2f})"
            for v in votes
        ])
        
        prompt = f"""You are an expert Risk Manager with expertise in portfolio risk management and capital preservation.

Assess risk for {state['ticker']} on {state['date']}:

Agent Recommendations:
{votes_summary}

Technical Analysis: {state.get('technical_analysis', {}).get('trend', 'N/A')}
Market Data: RSI={state['market_data'].get('rsi', 'N/A')}, 
             Volatility indicators available

Perform comprehensive risk assessment:

1. **Market Risk**: Volatility, beta, correlation risks
2. **Liquidity Risk**: Ability to enter/exit positions
3. **Event Risk**: Upcoming earnings, news, macro events
4. **Drawdown Risk**: Potential maximum loss scenarios
5. **Position Sizing**: Appropriate position size given risk
6. **Stop Loss**: Optimal stop loss level
7. **Risk/Reward**: Assess risk-adjusted return potential

Provide analysis as JSON:
{{
  "overall_risk": "very_low|low|moderate|high|very_high",
  "market_risk": "low|moderate|high",
  "liquidity_risk": "low|moderate|high",
  "event_risk": "low|moderate|high",
  "max_position_size": <0.0-1.0 as fraction>,
  "recommended_stop_loss": <percentage like 0.05 for 5%>,
  "recommended_take_profit": <percentage>,
  "risk_reward_ratio": <ratio like 3.0>,
  "max_drawdown_risk": <percentage>,
  "recommendation": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": <0-100>,
  "risk_factors": ["factor1", "factor2"],
  "reasoning": "detailed 2-3 sentence explanation"
}}

Prioritize capital preservation and proper risk management."""

        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_json(response.content)
            
            state['risk_analysis'] = analysis
            state['agent_votes'].append({
                'agent': 'risk_manager',
                'recommendation': analysis.get('recommendation', 'hold'),
                'confidence': analysis.get('confidence', 50) / 100.0,
                'weight': self.agent_config['risk_manager']['weight']
            })
            
            logger.info(f"✓ Risk: {analysis.get('recommendation', 'N/A')} "
                       f"(confidence: {analysis.get('confidence', 0)}%)")
            
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            state['risk_analysis'] = {'error': str(e)}
        
        return state
    
    def portfolio_manager(self, state: TradingState) -> TradingState:
        """
        Final decision synthesis from all agents using weighted voting.
        """
        logger.info(f"🎯 Portfolio Manager making final decision for {state['ticker']}...")
        
        votes = state.get('agent_votes', [])
        
        # Synthesize all analyses
        analyses_summary = f"""
Technical Analysis: {json.dumps(state.get('technical_analysis', {}), indent=2)}

Fundamental Analysis: {json.dumps(state.get('fundamental_analysis', {}), indent=2)}

Sentiment Analysis: {json.dumps(state.get('sentiment_analysis', {}), indent=2)}

Risk Analysis: {json.dumps(state.get('risk_analysis', {}), indent=2)}

Agent Votes:
{json.dumps(votes, indent=2)}
"""
        
        prompt = f"""You are an expert Portfolio Manager responsible for final trading decisions.

Synthesize all analyses for {state['ticker']} on {state['date']}:

{analyses_summary}

Make a comprehensive final decision considering:

1. **Consensus Analysis**: Synthesize all agent recommendations
2. **Conviction Level**: Assess agreement/disagreement among agents
3. **Position Sizing**: Appropriate size given risk and conviction
4. **Entry Strategy**: Market order, limit order, or staged entry
5. **Exit Strategy**: Stop loss, take profit, time horizon
6. **Alternative Scenarios**: What could go wrong/right

Provide final decision as JSON:
{{
  "action": "strong_buy|buy|hold|sell|strong_sell",
  "conviction": "very_high|high|moderate|low|very_low",
  "position_size": <0.0-1.0 as fraction of portfolio>,
  "entry_price": <target price or "market">,
  "stop_loss_price": <price>,
  "take_profit_price": <price>,
  "time_horizon": "1-3 days|1-2 weeks|1-3 months|long-term",
  "confidence": <0-100>,
  "key_factors": ["most important factor 1", "factor 2", "factor 3"],
  "risks": ["key risk 1", "key risk 2"],
  "alternative_scenario": "what could invalidate this decision",
  "reasoning": "comprehensive 3-5 sentence synthesis of all analyses"
}}

Make a decisive, well-reasoned final call."""

        try:
            response = self.llm.invoke(prompt)
            decision = self._parse_json(response.content)
            
            state['final_decision'] = decision
            
            logger.info(f"✓ Final Decision: {decision.get('action', 'N/A').upper()} "
                       f"(conviction: {decision.get('conviction', 'N/A')}, "
                       f"confidence: {decision.get('confidence', 0)}%)")
            
        except Exception as e:
            logger.error(f"Portfolio manager error: {e}")
            state['final_decision'] = {
                'action': 'hold',
                'reasoning': f'Error in decision making: {str(e)}'
            }
        
        return state
    
    def analyze(
        self,
        ticker: str,
        date: str,
        market_data: Dict,
        news: List[Dict] = None
    ) -> Dict:
        """
        Run complete multi-agent analysis.
        
        Args:
            ticker: Stock ticker
            date: Analysis date
            market_data: Market data and indicators
            news: List of news items
            
        Returns:
            Complete analysis with final decision
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Multi-Agent Analysis: {ticker} on {date}")
        logger.info(f"{'='*60}")
        
        initial_state = TradingState(
            ticker=ticker,
            date=date,
            market_data=market_data,
            news=news or [],
            technical_analysis={},
            fundamental_analysis={},
            sentiment_analysis={},
            risk_analysis={},
            agent_votes=[],
            final_decision={}
        )
        
        try:
            result = self.graph.invoke(initial_state)
            logger.info(f"{'='*60}")
            logger.info("✅ Multi-Agent Analysis Complete!")
            logger.info(f"{'='*60}\n")
            return result
        except Exception as e:
            logger.error(f"Multi-agent analysis failed: {e}")
            return {
                'final_decision': {
                    'action': 'hold',
                    'reasoning': f'Analysis error: {str(e)}'
                }
            }
    
    def _parse_json(self, text: str) -> Dict:
        """Extract and parse JSON from LLM response."""
        try:
            # Find JSON in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test multi-agent system
    system = AdvancedMultiAgentSystem()
    
    # Sample data
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
        {'title': 'Company reports strong earnings', 'sentiment': 'positive'},
        {'title': 'New product launch announced', 'sentiment': 'positive'}
    ]
    
    # Run analysis
    result = system.analyze(
        ticker="AAPL",
        date="2024-01-16",
        market_data=market_data,
        news=news
    )
    
    print("\n" + "="*60)
    print("FINAL DECISION:")
    print("="*60)
    decision = result['final_decision']
    print(f"Action: {decision.get('action', 'N/A')}")
    print(f"Position Size: {decision.get('position_size', 0):.1%}")
    print(f"Confidence: {decision.get('confidence', 0)}%")
    print(f"Reasoning: {decision.get('reasoning', 'N/A')}")