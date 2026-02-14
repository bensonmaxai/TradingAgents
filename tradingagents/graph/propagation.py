# TradingAgents/graph/propagation.py

from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str,
        market_type: str = "crypto",
        pre_reports: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        Args:
            company_name: Ticker symbol
            trade_date: Trading date string
            market_type: Market type ('crypto', 'us', or 'tw')
            pre_reports: Optional dict with pre-fetched report strings
                         (market_report, fundamentals_report, sentiment_report, news_report).
                         When provided, analysts can be skipped (fast mode).
        """
        state = {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "market_type": market_type,
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }
        if pre_reports:
            for key in ("market_report", "fundamentals_report", "sentiment_report", "news_report"):
                if key in pre_reports:
                    state[key] = pre_reports[key]
        return state

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
