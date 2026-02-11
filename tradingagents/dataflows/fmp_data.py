"""Financial Modeling Prep (FMP) data provider.

Provides earnings calendar, treasury rates, and real-time quotes.
Free tier: 250 calls/day. Used to supplement yfinance + Grok news.
"""

import os
import json
import requests
from datetime import datetime, timedelta

FMP_BASE = "https://financialmodelingprep.com/stable"


def _get_fmp_key():
    """Get FMP API key from environment or secrets file."""
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.openclaw/secrets.sh")) as f:
            for line in f:
                if line.startswith("export FMP_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _fmp_get(endpoint, params=None):
    """Make FMP API request."""
    key = _get_fmp_key()
    if not key:
        return None

    url = f"{FMP_BASE}/{endpoint}"
    p = {"apikey": key}
    if params:
        p.update(params)

    resp = requests.get(url, params=p, timeout=10)
    if resp.status_code == 200:
        try:
            return resp.json()
        except Exception:
            return None
    return None


def get_earnings_calendar(from_date: str, to_date: str) -> str:
    """Get upcoming and recent earnings reports.

    Args:
        from_date: Start date YYYY-MM-DD
        to_date: End date YYYY-MM-DD

    Returns:
        Formatted earnings calendar string
    """
    data = _fmp_get("earnings-calendar", {"from": from_date, "to": to_date})
    if not data:
        return "Earnings calendar unavailable"

    lines = [f"## Earnings Calendar ({from_date} to {to_date}):\n"]
    for item in data[:30]:
        symbol = item.get("symbol", "?")
        date = item.get("date", "?")
        eps_est = item.get("epsEstimated")
        eps_act = item.get("epsActual")
        rev_est = item.get("revenueEstimated")
        rev_act = item.get("revenueActual")

        eps_str = f"EPS est:{eps_est}"
        if eps_act is not None:
            surprise = ((eps_act - eps_est) / abs(eps_est) * 100) if eps_est else 0
            eps_str += f" → act:{eps_act} ({surprise:+.1f}%)"

        rev_str = ""
        if rev_est:
            rev_str = f" | Rev est:${rev_est/1e9:.1f}B"
            if rev_act:
                rev_str += f" → act:${rev_act/1e9:.1f}B"

        lines.append(f"- **{date} {symbol}**: {eps_str}{rev_str}")

    return "\n".join(lines)


def get_treasury_rates(lookback_days: int = 14) -> str:
    """Get recent treasury rates for macro context.

    Returns:
        Formatted treasury rates string
    """
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    data = _fmp_get("treasury-rates", {"from": from_date, "to": to_date})
    if not data:
        return "Treasury rates unavailable"

    lines = ["## US Treasury Rates:\n"]
    for item in data[:5]:
        date = item.get("date", "?")
        y2 = item.get("year2", "?")
        y5 = item.get("year5", "?")
        y10 = item.get("year10", "?")
        y30 = item.get("year30", "?")
        lines.append(f"- {date}: 2Y={y2}% 5Y={y5}% 10Y={y10}% 30Y={y30}%")

    # Calculate yield curve spread
    if len(data) >= 1:
        latest = data[0]
        spread_2_10 = float(latest.get("year10", 0)) - float(latest.get("year2", 0))
        lines.append(f"\n**2Y-10Y spread: {spread_2_10:+.2f}%** {'(inverted - recession signal)' if spread_2_10 < 0 else '(normal)'}")

    return "\n".join(lines)


def get_quote(symbol: str) -> dict:
    """Get real-time quote for a symbol.

    Returns:
        Quote dict or None
    """
    data = _fmp_get("quote", {"symbol": symbol})
    if data and len(data) > 0:
        return data[0]
    return None


def get_upcoming_earnings_for_ticker(ticker: str, days_ahead: int = 30) -> str:
    """Check if a specific ticker has upcoming earnings.

    Returns:
        String describing earnings date or empty
    """
    to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    from_date = datetime.now().strftime("%Y-%m-%d")

    data = _fmp_get("earnings-calendar", {"from": from_date, "to": to_date})
    if not data:
        return ""

    # Clean ticker for comparison (remove .TW etc)
    clean_ticker = ticker.replace(".TW", "").upper()

    for item in data:
        if item.get("symbol", "").upper() == clean_ticker:
            date = item.get("date", "?")
            eps_est = item.get("epsEstimated", "?")
            return f"⚠ EARNINGS ALERT: {ticker} reports on {date} (EPS est: {eps_est})"

    return ""


def get_macro_context(curr_date: str) -> str:
    """Build macro economic context from FMP data.

    Combines treasury rates + this week's major earnings for broader market context.
    """
    parts = []

    # Treasury rates
    rates = get_treasury_rates(14)
    if "unavailable" not in rates:
        parts.append(rates)

    # This week's earnings
    from_date = curr_date
    to_dt = datetime.strptime(curr_date, "%Y-%m-%d") + timedelta(days=7)
    to_date = to_dt.strftime("%Y-%m-%d")
    earnings = get_earnings_calendar(from_date, to_date)
    if "unavailable" not in earnings:
        parts.append(earnings)

    if not parts:
        return "Macro context: FMP API unavailable"

    return "\n\n".join(parts)
