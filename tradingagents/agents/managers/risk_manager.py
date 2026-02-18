import re
import sys
import time
import json

from tradingagents.agents.utils.agent_states import get_signal_constraints, get_market_context

_PRICE_RE = re.compile(r'[\d,]+\.?\d*')


def _parse_decision_fields(text):
    """Extract structured fields from Risk Manager output via regex."""
    fields = {}
    patterns = {
        "decision": r'\*\*DECISION\*\*[:\s]+(.+)',
        "confidence": r'\*\*Confidence\*\*[:\s]+(\d{1,3}|\w+)',
        "entry": r'\*\*Entry\*\*[:\s]+\$?([\d,\.]+)',
        "stop_loss": r'\*\*Stop-loss\*\*[:\s]+\$?([\d,\.]+)',
        "target1": r'\*\*Target 1\*\*[:\s]+\$?([\d,\.]+)',
        "target2": r'\*\*Target 2\*\*[:\s]+\$?([\d,\.]+)',
        "lessons": r'\*\*Lessons applied\*\*[:\s]+(.+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            fields[key] = m.group(1).strip().replace(",", "")
    return fields


def _validate_decision(fields, market_type, suggested_direction, has_memory):
    """Run programmatic validation. Returns list of issue strings."""
    issues = []
    if not fields:
        return issues  # can't validate if parsing failed

    try:
        entry = float(fields.get("entry", 0))
        sl = float(fields.get("stop_loss", 0))
        tp1 = float(fields.get("target1", 0))
        tp2 = float(fields.get("target2", 0))
    except (ValueError, TypeError):
        issues.append("Entry/SL/Target prices are not valid numbers.")
        return issues

    decision = fields.get("decision", "").upper()
    max_sl = 8.0 if market_type in ("us", "tw") else 20.0

    # SL distance check
    if entry > 0 and sl > 0:
        sl_pct = abs(entry - sl) / entry * 100
        if sl_pct > max_sl:
            issues.append(f"Stop-loss distance {sl_pct:.1f}% exceeds max {max_sl}%.")

    # Direction alignment
    if suggested_direction == "long" and "SELL" in decision:
        issues.append(f"Decision '{decision}' contradicts locked LONG direction.")
    elif suggested_direction == "short" and "BUY" in decision:
        issues.append(f"Decision '{decision}' contradicts locked SHORT direction.")

    # Price logic for BUY
    if "BUY" in decision and entry > 0:
        if sl > 0 and sl > entry:
            issues.append(f"For BUY, Stop-loss ({sl}) should be below Entry ({entry}).")
        if tp1 > 0 and tp1 < entry:
            issues.append(f"For BUY, Target 1 ({tp1}) should be above Entry ({entry}).")

    # Price logic for SELL
    if "SELL" in decision and entry > 0:
        if sl > 0 and sl < entry:
            issues.append(f"For SELL, Stop-loss ({sl}) should be above Entry ({entry}).")
        if tp1 > 0 and tp1 > entry:
            issues.append(f"For SELL, Target 1 ({tp1}) should be below Entry ({entry}).")

    # Confidence must be 0-100 integer
    conf = fields.get("confidence", "")
    try:
        conf_int = int(conf)
        if not (0 <= conf_int <= 100):
            issues.append(f"Confidence {conf_int} must be 0-100.")
    except (ValueError, TypeError):
        if conf.lower() not in ("high", "medium", "low"):
            issues.append(f"Confidence must be integer 0-100, got '{conf}'")

    # Lessons check
    if has_memory:
        lessons = fields.get("lessons", "")
        if not lessons or lessons.lower() in ("none", "n/a", ""):
            issues.append("Past lessons were provided but not referenced.")

    return issues


def _self_refine(response_text, llm, market_type, suggested_direction, has_memory, sl_max):
    """Validate and optionally refine Risk Manager decision. Max 1 refinement."""
    fields = _parse_decision_fields(response_text)
    issues = _validate_decision(fields, market_type, suggested_direction, has_memory)

    if not issues:
        return response_text

    issues_str = "\n".join(f"- {issue}" for issue in issues)
    print(f"[self-refine] {len(issues)} issue(s) found, refining: {issues_str}", file=sys.stderr)

    refine_prompt = f"""You are the Risk Management Judge. Review your previous decision and fix the issues below.

YOUR PREVIOUS DECISION:
{response_text}

ISSUES FOUND:
{issues_str}

RULES:
- Stop-loss must be within {sl_max} of entry price.
- If direction is locked, do not contradict it.
- Reference past lessons if they were provided.
- Entry/SL/Target must be logically consistent with the decision direction.

Output your CORRECTED decision in the same exact format. Every field must have a specific value."""

    refined = llm.invoke(refine_prompt)
    return refined.content


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        market_type = state.get("market_type", "crypto")
        suggested_direction = state.get("suggested_direction", "")
        screener_score = state.get("screener_score", 0)
        signal_constraints = get_signal_constraints(market_type, suggested_direction, screener_score)
        market_context = get_market_context(market_type)

        memory_instruction = ""
        if past_memory_str.strip():
            memory_instruction = f"""
⚠️ CRITICAL — Past mistakes to avoid (from actual P&L outcomes):
{past_memory_str}
You MUST check if the current trade setup resembles any past mistake above. If it does, either REJECT the trade or explicitly adjust position size/stop-loss/target to compensate. State which lesson you applied.
"""

        # Devil's Advocate: challenge consensus when all analysts agree
        devils_advocate = ""
        agg = risk_debate_state.get("current_aggressive_response", "")
        cons = risk_debate_state.get("current_conservative_response", "")
        neut = risk_debate_state.get("current_neutral_response", "")
        all_responses = f"{agg} {cons} {neut}".lower()
        buy_mentions = all_responses.count("buy") + all_responses.count("long") + all_responses.count("upside")
        sell_mentions = all_responses.count("sell") + all_responses.count("short") + all_responses.count("downside")
        if buy_mentions > sell_mentions * 2:
            devils_advocate = "\n🔴 DEVIL'S ADVOCATE: All analysts lean bullish. Before deciding BUY, seriously consider: What if this is a bull trap? What data would DISPROVE the bull case? If you still choose BUY, your confidence should be Medium at most unless you can refute this concern with specific data.\n"
        elif sell_mentions > buy_mentions * 2:
            devils_advocate = "\n🔴 DEVIL'S ADVOCATE: All analysts lean bearish. Before deciding SELL, seriously consider: What if this is a capitulation bottom? What data would DISPROVE the bear case? If you still choose SELL, your confidence should be Medium at most unless you can refute this concern with specific data.\n"

        sl_max = "8%" if market_type in ("us", "tw") else "20%"

        prompt = f"""You are the Risk Management Judge. Make the FINAL trading decision.

{signal_constraints}
{market_context}
CONSTRAINTS:
- Entry/Stop-loss/Targets must be specific prices, not ranges wider than 2%.

Output in this exact structure (every field mandatory):
**DECISION**: [your decision]
**Confidence**: [0-100 integer]
  Scoring guide: 80-100=strong multi-signal alignment, 60-79=moderate, 40-59=mixed/uncertain, 20-39=weak, 0-19=no case
**Entry**: [specific price]
**Stop-loss**: [Price] (mandatory, max {sl_max} from entry)
**Target 1**: [Price] (partial take-profit ~50%)
**Target 2**: [Price] (full exit)
**Key risk**: [Single biggest threat to this trade]
**Data quality note**: [which data sources were missing or weak, and how it affected your analysis]
**Lessons applied**: Which past lesson(s) influenced this decision
**Risk-adjusted rationale**: [3-4 sentences — which analyst was most right and why, adjusted from trader's plan: {trader_plan}]
{memory_instruction}{devils_advocate}
Risk debate:
{history}

Be decisive. Do NOT default to HOLD as compromise. Every field must have a specific value."""

        response = llm.invoke(prompt)

        # Self-Refine: validate and optionally correct the decision
        has_memory = bool(past_memory_str.strip())
        refined_content = _self_refine(
            response.content, llm, market_type,
            suggested_direction, has_memory, sl_max,
        )

        new_risk_debate_state = {
            "judge_decision": refined_content,
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
            "final_trade_decision": refined_content,
        }

    return risk_manager_node
