RL + LLM Trading Research
Overview
This project explores whether LLM-derived textual signals (news, filings, social data) can improve the performance and robustness of reinforcement-learning (RL)–based trading agents beyond traditional price-only systems.
The goal is to determine whether structured + unstructured information jointly provide more stable, risk-aware decision policies.
Our Approach
We build a pipeline that:
Extracts text-based features using LLMs
(sentiment scores, event tags, embeddings)
Combines them with market/technical features
Trains an RL trading agent (PPO baseline) within a FinRL/Gym environment
Backtests performance against simple baselines such as buy-and-hold and price-only PPO models
The LLM is used offline as a feature generator, not for real-time decision making.
RL handles position sizing and execution simulation under transaction-cost constraints.
