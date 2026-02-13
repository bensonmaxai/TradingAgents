import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        memory_instruction = ""
        if past_memory_str.strip():
            memory_instruction = f"""
⚠️ CRITICAL — Past mistakes to avoid (from actual P&L outcomes):
{past_memory_str}
You MUST check if the current trade setup resembles any past mistake above. If it does, either REJECT the trade or explicitly adjust position size/stop-loss/target to compensate. State which lesson you applied.
"""

        prompt = f"""You are the Risk Management Judge. Make the FINAL trading decision.

Output in this exact structure:
**DECISION**: BUY / SELL / HOLD
**Confidence**: [High/Medium/Low]
**Position**: [Specific % of portfolio]
**Entry**: [Price or range]
**Stop-loss**: [Price] (max acceptable loss)
**Target 1**: [Price] (partial take-profit)
**Target 2**: [Price, timeframe] (full exit)
**Key risk**: [Single biggest threat to this trade]
**Lessons applied**: Which past lesson(s) influenced this decision
**Risk-adjusted rationale**: [3-4 sentences — which analyst was most right and why, adjusted from trader's plan: {trader_plan}]
{memory_instruction}
Risk debate:
{history}

Be decisive. Do NOT default to HOLD as compromise. Every field must have a specific value."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
