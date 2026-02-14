import functools
import time
import json

from tradingagents.agents.utils.agent_states import get_signal_constraints


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        market_type = state.get("market_type", "crypto")
        signal_constraints = get_signal_constraints(market_type)

        memory_instruction = ""
        if past_memory_str.strip() and past_memory_str != "No past memories found.":
            memory_instruction = f"""

⚠️ CRITICAL — Past Trading Lessons (from actual P&L outcomes):
{past_memory_str}
You MUST adjust your entry/stop-loss/target based on these lessons. If a past lesson suggests tighter stops or different position sizing for similar setups, APPLY it. State which lesson influenced your plan."""

        messages = [
            {
                "role": "system",
                "content": f"""You are a trader. Based on the investment plan, output a concrete trading plan.

{signal_constraints}

**Action**: [your decision]
**Entry price**: [specific price or tight range]
**Stop-loss**: [specific price] (mandatory)
**Target**: [specific price, 1-3 month horizon]
**Confidence**: [High/Medium/Low with one-line reason]

Then conclude with: FINAL TRANSACTION PROPOSAL: **[your decision]**
{memory_instruction}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
