"""FRED (Federal Reserve Economic Data) macro indicators.

Provides key macroeconomic context for trading analysis:
- Fed funds rate & rate trend
- CPI inflation
- GDP growth
- Unemployment
- VIX (fear index)
- USD strength
- Yield curve (2Y/10Y)

FRED API: free, no daily limit (but be reasonable).
"""

import os
import json
import requests
from datetime import datetime, timedelta

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Key macro series IDs
SERIES = {
    "fed_funds": {"id": "FEDFUNDS", "name": "Fed 基準利率", "unit": "%", "freq": "monthly"},
    "cpi": {"id": "CPIAUCSL", "name": "CPI 通膨率", "unit": "% YoY", "freq": "monthly", "transform": "pc1"},
    "gdp": {"id": "A191RL1Q225SBEA", "name": "GDP 成長率", "unit": "%", "freq": "quarterly"},
    "unemployment": {"id": "UNRATE", "name": "失業率", "unit": "%", "freq": "monthly"},
    "vix": {"id": "VIXCLS", "name": "VIX 恐慌指數", "unit": "", "freq": "daily"},
    "usd_index": {"id": "DTWEXBGS", "name": "美元指數", "unit": "", "freq": "daily"},
    "yield_10y": {"id": "DGS10", "name": "10年公債殖利率", "unit": "%", "freq": "daily"},
    "yield_2y": {"id": "DGS2", "name": "2年公債殖利率", "unit": "%", "freq": "daily"},
}


def _get_fred_key():
    """Get FRED API key."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.openclaw/secrets.sh")) as f:
            for line in f:
                if line.startswith("export FRED_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _fetch_series(series_id, limit=5, units="lin", sort_order="desc"):
    """Fetch observations from FRED."""
    key = _get_fred_key()
    if not key:
        return []

    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "sort_order": sort_order,
        "limit": limit,
        "units": units,
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            obs = data.get("observations", [])
            # Filter out missing values
            return [o for o in obs if o.get("value", ".") not in (".", "")]
        return []
    except Exception:
        return []


def get_macro_dashboard() -> str:
    """Build a comprehensive macro economic dashboard.

    Returns:
        Formatted string with all key macro indicators in Traditional Chinese.
    """
    key = _get_fred_key()
    if not key:
        return ""

    lines = ["## 宏觀經濟儀表板 (FRED)\n"]

    # Fed Funds Rate (with trend)
    fed = _fetch_series("FEDFUNDS", limit=6)
    if fed:
        current = fed[0]
        prev = fed[1] if len(fed) > 1 else None
        change = ""
        if prev:
            diff = float(current["value"]) - float(prev["value"])
            if diff < -0.1:
                change = " ↓ 降息趨勢"
            elif diff > 0.1:
                change = " ↑ 升息趨勢"
            else:
                change = " → 維持不變"
        lines.append(f"**Fed 基準利率: {current['value']}%** ({current['date']}){change}")
        # Show recent trend
        trend = " → ".join([f"{o['value']}%" for o in reversed(fed[:4])])
        lines.append(f"  趨勢: {trend}")

    # CPI Inflation
    cpi = _fetch_series("CPIAUCSL", limit=3, units="pc1")
    if cpi:
        lines.append(f"\n**CPI 通膨: {cpi[0]['value'][:4]}% YoY** ({cpi[0]['date']})")
        if float(cpi[0]["value"]) > 3:
            lines.append("  ⚠ 通膨偏高，Fed 可能鷹派")
        elif float(cpi[0]["value"]) < 2:
            lines.append("  ✓ 通膨受控，有降息空間")

    # GDP Growth
    gdp = _fetch_series("A191RL1Q225SBEA", limit=4)
    if gdp:
        lines.append(f"\n**GDP 成長: {gdp[0]['value']}%** ({gdp[0]['date']})")
        if float(gdp[0]["value"]) < 0:
            lines.append("  ⚠ 經濟衰退")

    # Unemployment
    unemp = _fetch_series("UNRATE", limit=3)
    if unemp:
        lines.append(f"\n**失業率: {unemp[0]['value']}%** ({unemp[0]['date']})")

    # VIX
    vix = _fetch_series("VIXCLS", limit=5)
    if vix:
        vix_val = float(vix[0]["value"])
        level = "低波動" if vix_val < 15 else "正常" if vix_val < 20 else "偏高" if vix_val < 30 else "恐慌"
        lines.append(f"\n**VIX 恐慌指數: {vix[0]['value']}** ({vix[0]['date']}) — {level}")

    # USD Index
    usd = _fetch_series("DTWEXBGS", limit=3)
    if usd:
        lines.append(f"\n**美元指數: {usd[0]['value']}** ({usd[0]['date']})")

    # Yield Curve
    y10 = _fetch_series("DGS10", limit=1)
    y2 = _fetch_series("DGS2", limit=1)
    if y10 and y2:
        spread = float(y10[0]["value"]) - float(y2[0]["value"])
        curve_status = "正常" if spread > 0 else "倒掛 ⚠ 衰退警告"
        lines.append(f"\n**殖利率曲線: 10Y={y10[0]['value']}% - 2Y={y2[0]['value']}% = {spread:+.2f}%** ({curve_status})")

    return "\n".join(lines)


def get_fed_context() -> str:
    """Get focused Fed/monetary policy context.

    Returns:
        Concise string about current Fed stance.
    """
    fed = _fetch_series("FEDFUNDS", limit=6)
    cpi = _fetch_series("CPIAUCSL", limit=2, units="pc1")
    vix = _fetch_series("VIXCLS", limit=1)

    parts = []

    if fed:
        current = float(fed[0]["value"])
        # Calculate rate change over last 3 months
        if len(fed) >= 4:
            three_months_ago = float(fed[3]["value"])
            change = current - three_months_ago
            if change < -0.3:
                parts.append(f"Fed 降息中 ({fed[3]['value']}% → {fed[0]['value']}%)")
            elif change > 0.3:
                parts.append(f"Fed 升息中 ({fed[3]['value']}% → {fed[0]['value']}%)")
            else:
                parts.append(f"Fed 利率維持 {fed[0]['value']}%")

    if cpi:
        cpi_val = float(cpi[0]["value"])
        if cpi_val > 3:
            parts.append(f"通膨偏高 {cpi_val:.1f}%")
        elif cpi_val < 2:
            parts.append(f"通膨受控 {cpi_val:.1f}%")
        else:
            parts.append(f"通膨 {cpi_val:.1f}%")

    if vix:
        vix_val = float(vix[0]["value"])
        if vix_val > 25:
            parts.append(f"VIX={vix_val:.0f} 市場恐慌")
        elif vix_val > 20:
            parts.append(f"VIX={vix_val:.0f} 波動偏高")

    return " | ".join(parts) if parts else ""
