"""x.ai Grok-based news data fetching using web search + X search.

Uses Grok's Responses API with web_search and x_search tools
for real-time news and social sentiment data.
"""

import os
import json
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta


XAI_API_URL = "https://api.x.ai/v1/responses"
XAI_MODEL = "grok-4-1-fast"


def _get_xai_key():
    """Get x.ai API key from environment or openclaw secrets."""
    key = os.environ.get("XAI_API_KEY")
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.openclaw/secrets.sh")) as f:
            for line in f:
                if line.startswith("export XAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    raise RuntimeError("XAI_API_KEY not set")


def _call_responses_api(query: str, tools: list, max_tokens: int = 2000) -> str:
    """Call x.ai Responses API with specified tools."""
    api_key = _get_xai_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": XAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are a financial news researcher. "
                    "Return factual, concise news summaries. "
                    "Focus on market-moving events, earnings, analyst upgrades/downgrades, "
                    "insider trading, and sentiment shifts. "
                    "Format each item as: ### Title (source: Publisher)\\nSummary\\n"
                ),
            },
            {"role": "user", "content": query},
        ],
        "tools": tools,
        "max_output_tokens": max_tokens,
    }

    resp = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Extract text from response output
    output = data.get("output", [])
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")

    return ""


def get_news_xai_grok(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using Grok web + X search.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    try:
        # Use both web search and X search for comprehensive coverage
        tools = [
            {"type": "web_search"},
            {
                "type": "x_search",
                "from_date": start_date,
                "to_date": end_date,
            },
        ]

        query = (
            f"Latest news and analysis for ${ticker} stock from {start_date} to {end_date}. "
            f"Include: earnings reports, analyst upgrades/downgrades, insider activity, "
            f"regulatory news, product announcements, and X/Twitter sentiment."
        )

        result = _call_responses_api(query, tools)

        if not result or len(result.strip()) < 20:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        return f"## {ticker} News (Grok Web+X Search), from {start_date} to {end_date}:\n\n{result}"

    except Exception as e:
        return f"Error fetching Grok news for {ticker}: {str(e)}"


def get_global_news_xai_grok(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """
    Retrieve global/macro economic news using Grok web + X search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        Formatted string containing global news articles
    """
    try:
        start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        tools = [
            {"type": "web_search"},
            {
                "type": "x_search",
                "from_date": start_date,
                "to_date": curr_date,
            },
        ]

        query = (
            f"Top {limit} market-moving financial news from {start_date} to {curr_date}. "
            f"Cover: Fed/central bank decisions, inflation data, GDP reports, "
            f"major earnings surprises, geopolitical events affecting markets, "
            f"and notable financial sentiment trends on X/Twitter."
        )

        result = _call_responses_api(query, tools, max_tokens=3000)

        if not result or len(result.strip()) < 20:
            return f"No global news found for {curr_date}"

        return f"## Global Market News (Grok Web+X Search), from {start_date} to {curr_date}:\n\n{result}"

    except Exception as e:
        return f"Error fetching global news via Grok: {str(e)}"
