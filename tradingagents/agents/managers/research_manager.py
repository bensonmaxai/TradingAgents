import time
import json

from tradingagents.agents.utils.agent_states import get_signal_constraints


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        market_type = state.get("market_type", "crypto")
        signal_constraints = get_signal_constraints(market_type)

        memory_instruction = ""
        if past_memory_str.strip():
            memory_instruction = f"""
⚠️ CRITICAL — Past mistakes to avoid (from actual P&L outcomes):
{past_memory_str}
Before deciding, CHECK if your current call repeats any mistake above. If so, explicitly state how you are adjusting to avoid repeating it.
"""

        prompt = f"""You are the Research Manager. Evaluate the bull/bear debate and make a DECISIVE call.

{signal_constraints}

Output in this exact structure:
**Decision**: [your decision]
**Winner**: Bull or Bear (who had the stronger data-backed argument)
**Key reason**: The single most important factor driving your decision
**Risk**: The biggest risk to your call
**Action plan**: 2-3 concrete steps for the trader (specific entry price, stop-loss price, target price)
**Lessons applied**: Which past lesson(s) influenced this decision, if any

Do NOT default to HOLD as a compromise. Pick a side based on evidence strength.
{memory_instruction}
Debate:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
