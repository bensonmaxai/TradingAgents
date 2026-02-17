# TradingAgents/graph/signal_processing.py

import re
from langchain_openai import ChatOpenAI

_CRYPTO_SIGNALS = {"BUY", "SELL", "HOLD", "CLOSE_LONG", "CLOSE_SHORT", "ADD"}
_STOCK_SIGNALS = {"BUY", "SELL", "HOLD"}
_SIGNAL_RE = re.compile(r'\b(ADD|BUY|SELL|HOLD|CLOSE_LONG|CLOSE_SHORT)\b')
# Match patterns like "CLOSE 60%" or "CLOSE 60% / HOLD 40%"
_PARTIAL_CLOSE_RE = re.compile(r'CLOSE\s+(\d+)%', re.IGNORECASE)


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str, market_type: str = "crypto", suggested_direction: str = "") -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text
            market_type: Market type ('crypto', 'us', or 'tw')
            suggested_direction: Direction lock ('long', 'short', or '' for free mode)

        Returns:
            Extracted decision (BUY, SELL, HOLD, CLOSE_LONG, or CLOSE_SHORT)
        """
        # Direction-locked mode: only allow confirm or reject
        if suggested_direction == "pyramid_long":
            valid = "ADD, HOLD, or CLOSE_LONG"
            allowed = {"ADD", "HOLD", "CLOSE_LONG"}
        elif suggested_direction == "pyramid_short":
            valid = "ADD, HOLD, or CLOSE_SHORT"
            allowed = {"ADD", "HOLD", "CLOSE_SHORT"}
        elif suggested_direction == "long":
            valid = "BUY or HOLD"
            allowed = {"BUY", "HOLD"}
        elif suggested_direction == "short":
            valid = "SELL or HOLD"
            allowed = {"SELL", "HOLD"}
        elif market_type == "crypto":
            valid = "BUY, SELL, HOLD, CLOSE_LONG, or CLOSE_SHORT"
            allowed = _CRYPTO_SIGNALS
        else:
            valid = "BUY, SELL, or HOLD"
            allowed = _STOCK_SIGNALS

        # Detect partial close patterns like "CLOSE 60% / HOLD 40%"
        partial_hint = ""
        if market_type == "crypto" and _PARTIAL_CLOSE_RE.search(full_signal):
            partial_hint = (
                " When the text describes closing a percentage of a position "
                "(e.g., 'CLOSE 60%'), classify it as CLOSE_LONG if the existing "
                "position is long/buy, or CLOSE_SHORT if the existing position is "
                "short/sell. Partial closes are still CLOSE_LONG or CLOSE_SHORT."
            )

        messages = [
            (
                "system",
                f"You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: {valid}. Provide only the extracted decision ({valid}) as your output, without adding any additional text or information.{partial_hint}",
            ),
            ("human", full_signal),
        ]

        raw = self.quick_thinking_llm.invoke(messages).content

        # Validate: extract first valid signal token from LLM response
        match = _SIGNAL_RE.search(raw)
        if match and match.group(1) in allowed:
            return match.group(1)

        # Fallback: safe default
        return "HOLD"
