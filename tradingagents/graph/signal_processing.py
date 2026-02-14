# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str, market_type: str = "crypto") -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text
            market_type: Market type ('crypto', 'us', or 'tw')

        Returns:
            Extracted decision (BUY, SELL, HOLD, CLOSE_LONG, or CLOSE_SHORT)
        """
        if market_type == "crypto":
            valid = "BUY, SELL, HOLD, CLOSE_LONG, or CLOSE_SHORT"
        else:
            valid = "BUY, SELL, or HOLD"

        messages = [
            (
                "system",
                f"You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: {valid}. Provide only the extracted decision ({valid}) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content
