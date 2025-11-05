# RL + LLM Hybrid Agents for Market/Stock Prediction & Trading

## Overview

This project explores combining Reinforcement Learning (RL) and Large Language Models (LLMs) to build automated trading agents. The system integrates numeric market data and unstructured textual information (news, filings, social media) to generate informed trading policies.

Use this repo as a scaffold for experimentation, not as a plug-and-play trading bot.

---

## Architecture

### 1) Core Roles

* **LLM → Feature Extractor**: Converts text sources (news, filings, tweets) into structured signals (sentiment scores, event flags). These become part of the RL agent's state.
* **LLM → Strategy Planner**: Produces high-level trade intents (e.g., sector rotation, hedge intent). RL controls execution (sizing, timing).
* **Hybrid Multi-Agent**: Multiple LLMs (fundamental, sentiment, technical) debate; RL synthesizes into positions.

### 2) Environment

* Gym-style environment with:

  * Price history
  * Transaction costs + slippage
  * Discrete or continuous actions
  * Support for daily → intraday later
* Based on `FinRL` or custom Gym.

### 3) RL Algorithms

Use moderately sized nets (overfitting in finance is easy):

* PPO / A2C
* DDPG / TD3 (continuous sizing)
* DQN / DDQN (discrete)

### 4) LLM Integration

* **Offline Feature Generation:** Precompute text→vector features; store to DB.
* **On-Policy Planner:** LLM assists in action proposals; RL uses as soft constraints.
* **RAG + Memory:** Vector DB stores historical events; LLM is context-aware.

---

## Data

* OHLCV from public sources
* Text: News feeds, Twitter/X, filings
* Optional: Intraday / tick data

---

## Evaluation

Backtests must include:

* Walk-forward splits
* Transaction costs
* Out-of-sample periods (crashes)

Metrics:

* Sharpe
* Sortino
* Max drawdown
* Turnover

Beware of leakage.

---

## Risks

* Agents may behave unpredictably (e.g., collusion-like strategies under MARL).
* High sensitivity to noise.
* Regulator scrutiny.
* Not suitable for live deployment without serious controls.

---

## References

* [https://arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138)
* [https://github.com/TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
* [https://github.com/AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
* finrl.readthedocs.io

---


---
