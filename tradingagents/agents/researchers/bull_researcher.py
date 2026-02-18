from langchain_core.messages import AIMessage
from datetime import date
import time
import json

from tradingagents.agents.utils.agent_states import get_market_context


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2,
                                              reference_date=date.today())

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        memory_block = ""
        if past_memory_str.strip():
            memory_block = f"""
⚠️ CRITICAL — Past Trading Lessons (from actual P&L outcomes):
{past_memory_str}
You MUST factor these lessons into your bull case. If a past lesson contradicts your current argument, explicitly address why this time is different. Do NOT repeat mistakes identified above.
---
"""

        market_type = state.get("market_type", "crypto")
        market_context = get_market_context(market_type)

        prompt = f"""You are the Bull Analyst. Make the strongest case FOR investing, using specific data from the reports below. Counter the bear's key arguments directly.
{market_context}
Structure your response:
1. **Strongest bull case** (2-3 points with specific numbers from reports)
2. **Bear rebuttal** (address bear's weakest argument with data)
3. **Catalyst** (what will drive the stock higher, with timeline)
{memory_block}
Market data: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}
Debate so far: {history}
Last bear argument: {current_response}

Be direct and data-driven. No repetition of what the bear said. No filler.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
