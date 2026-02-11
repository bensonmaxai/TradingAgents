from langchain_core.messages import AIMessage
import time
import json


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""Conservative Risk Analyst: Argue for CAPITAL PROTECTION from the trader's plan.

Trader's plan: {trader_decision}

Structure:
1. **Downside risk**: The biggest threat the aggressive analyst ignores (cite specific data)
2. **Counter to aggressive**: Their most reckless assumption (one specific rebuttal)
3. **Recommended adjustment**: How to reduce exposure, add hedges, or tighten stops

Data: Market={market_research_report[:500]} | Fundamentals={fundamentals_report[:500]}
History: {history}
Aggressive said: {current_aggressive_response[:300]}
Neutral said: {current_neutral_response[:300]}

Be direct. If no prior responses exist, just present your case. No filler."""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
