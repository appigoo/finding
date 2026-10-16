
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime

st.set_page_config(
    page_title="Strict Earnings Growth Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Strict Earnings Growth Screener")
st.caption("硬性條件篩選｜市值 ≥ $10B｜三年盈利增長｜最近季度改善")

st.warning(
    "只要任何一項硬性條件未通過，便不列入合格名單。"
    "缺少數據不代表合格；第三方財務數據仍需核實。"
)

# -----------------------------
# Strict thresholds
# -----------------------------
with st.sidebar:
    st.header("硬性門檻")

    min_market_cap = st.number_input(
        "最低市值（十億美元）",
        min_value=10,
        value=10,
        step=5
    ) * 1e9

    min_3y_growth = st.number_input(
        "三年淨利潤累計增長（%）",
        min_value=100,
        value=100,
        step=25
    ) / 100

    min_revenue_yoy = st.number_input(
        "最近一季營收同比增長（%）",
        min_value=15,
        value=15,
        step=5
    ) / 100

    min_eps_yoy = st.number_input(
        "最近一季 EPS 同比增長（%）",
        min_value=20,
        value=20,
        step=5
    ) / 100

    min_net_income_yoy = st.number_input(
        "最近一季淨利潤同比增長（%）",
        min_value=20,
        value=20,
        step=5
    ) / 100

    st.caption(
        "門檻只可維持或提高，不可降低。"
        "這些是初始研究規則，尚非經回測優化的參數。"
    )

# -----------------------------
# Universe
# -----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_universe():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100"
    ]

    symbols = set()

    for url in urls:
        try:
            tables = pd.read_html(
                requests.get(
                    url, headers=headers, timeout=20
                ).text
            )

            for table in tables:
                for col in table.columns:
                    if str(col).lower() in [
                        "symbol", "ticker", "ticker symbol"
                    ]:
                        symbols.update(
                            table[col]
                            .dropna()
                            .astype(str)
                            .str.replace(".", "-", regex=False)
                            .tolist()
                        )
                        break
        except Exception:
            continue

    return sorted(
        s for s in symbols
        if s and s.isascii() and " " not in s
    )

# -----------------------------
# Financial data helpers
# -----------------------------
def find_row(df, candidates):
    """Return the first matching financial statement row."""
    if df is None or df.empty:
        return None

    normalized = {
        str(idx).lower().replace("_", " ").strip(): idx
        for idx in df.index
    }

    for candidate in candidates:
        key = candidate.lower().replace("_", " ").strip()
        if key in normalized:
            return normalized[key]

    return None


def series_for(df, candidates):
    row = find_row(df, candidates)

    if row is None:
        return pd.Series(dtype=float)

    s = pd.to_numeric(df.loc[row], errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")

    return (
        s[~s.index.isna()]
        .dropna()
        .sort_index()
    )


def annual_values(df, candidates):
    """Extract the latest annual values, newest first."""
    s = series_for(df, candidates)

    if s.empty:
        return s

    # Yahoo annual statements should contain annual periods.
    # Reject obvious duplicates and retain the latest observation
    # per fiscal year.
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index(ascending=False)


def pct_growth(current, previous):
    """Only calculate growth from positive, valid base values."""
    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous <= 0
        or current <= 0
    ):
        return np.nan

    return current / previous - 1


def quarterly_yoy(series):
    """
    Match a quarter with a quarter ending approximately one year
    earlier. Reject missing or non-positive comparison bases.
    """
    if len(series) < 2:
        return None

    series = series.sort_index()
    latest_date = series.index[-1]
    latest_value = float(series.iloc[-1])

    target_days = 365
    candidates = []

    for date, value in series.iloc[:-1].items():
        days = (latest_date - date).days

        if 330 <= days <= 400:
            candidates.append(
                (abs(days - target_days), float(value), date)
            )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, previous_value, previous_date = candidates[0]

    growth = pct_growth(latest_value, previous_value)

    return {
        "latest_date": latest_date,
        "previous_date": previous_date,
        "latest": latest_value,
        "previous": previous_value,
        "growth": growth
    }


# -----------------------------
# Single-stock strict screening
# -----------------------------
def screen_stock(symbol):
    result = {
        "Ticker": symbol,
        "Status": "FAIL",
        "Reasons": []
    }

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()

        market_cap = info.get("marketCap")

        result["Company"] = info.get(
            "longName", symbol
        )
        result["Market Cap ($B)"] = (
            market_cap / 1e9
            if market_cap is not None else np.nan
        )

        # Gate 1: market capitalization
        if (
            market_cap is None
            or not np.isfinite(market_cap)
            or market_cap < min_market_cap
        ):
            result["Reasons"].append(
                "市值未達標或數據缺失"
            )
            return result

        annual = ticker.income_stmt
        quarterly = ticker.quarterly_income_stmt

        if annual is None or annual.empty:
            result["Reasons"].append("缺少年度財報")
            return result

        if quarterly is None or quarterly.empty:
            result["Reasons"].append("缺少季度財報")
            return result

        net_candidates = [
            "Net Income",
            "Net Income Common Stockholders"
        ]
        revenue_candidates = [
            "Total Revenue",
            "Operating Revenue"
        ]
        eps_candidates = [
            "Diluted EPS",
            "Diluted Earnings Per Share"
        ]
        operating_income_candidates = [
            "Operating Income"
        ]

        # Gate 2: three complete annual fiscal years
        annual_net = annual_values(
            annual, net_candidates
        )

        if len(annual_net) < 3:
            result["Reasons"].append(
                "年度淨利潤數據不足三年"
            )
            return result

        # Newest 3 fiscal-year observations
        n0 = float(annual_net.iloc[0])
        n1 = float(annual_net.iloc[1])
        n2 = float(annual_net.iloc[2])

        # All three years must be profitable and strictly increasing
        if not (n0 > n1 > n2 > 0):
            result["Reasons"].append(
                "三年年度淨利潤沒有連續正增長"
            )
            return result

        three_year_growth = pct_growth(n0, n2)

        if (
            pd.isna(three_year_growth)
            or three_year_growth < min_3y_growth
        ):
            result["Reasons"].append(
                "三年淨利潤累計增長未達標"
            )
            return result

        result["3Y Net Income Growth (%)"] = (
            three_year_growth * 100
        )

        # Gate 3: latest quarter vs comparable year-ago quarter
        q_rev = quarterly_yoy(
            series_for(quarterly, revenue_candidates)
        )
        q_net = quarterly_yoy(
            series_for(quarterly, net_candidates)
        )
        q_eps = quarterly_yoy(
            series_for(quarterly, eps_candidates)
        )
        q_op = quarterly_yoy(
            series_for(
                quarterly, operating_income_candidates
            )
        )

        if any(x is None for x in [q_rev, q_net, q_eps, q_op]):
            result["Reasons"].append(
                "最近季度缺少可比較的營收、EPS、淨利潤或營業利潤數據"
            )
            return result

        if (
            pd.isna(q_rev["growth"])
            or q_rev["growth"] < min_revenue_yoy
        ):
            result["Reasons"].append("最近一季營收增長未達標")
            return result

        if (
            pd.isna(q_eps["growth"])
            or q_eps["growth"] < min_eps_yoy
        ):
            result["Reasons"].append("最近一季 EPS 增長未達標")
            return result

        if (
            pd.isna(q_net["growth"])
            or q_net["growth"] < min_net_income_yoy
        ):
            result["Reasons"].append("最近一季淨利潤增長未達標")
            return result

        # Gate 4: operating margin must improve year over year.
        # Both revenue and operating income must be positive.
        if (
            q_rev["latest"] <= 0
            or q_rev["previous"] <= 0
            or q_op["latest"] <= 0
            or q_op["previous"] <= 0
        ):
            result["Reasons"].append(
                "營業利潤或營收基礎不合格"
            )
            return result

        margin_now = q_op["latest"] / q_rev["latest"]
        margin_prev = q_op["previous"] / q_rev["previous"]

        margin_change_pp = (margin_now - margin_prev) * 100

        if margin_change_pp <= 0:
            result["Reasons"].append(
                "營業利潤率沒有同比改善"
            )
            return result

        # Passed every implemented gate
        result.update({
            "Status": "PASS",
            "Reasons": ["所有已實施的硬性條件通過"],
            "Latest Quarter": str(
                q_rev["latest_date"].date()
            ),
            "Revenue YoY (%)": q_rev["growth"] * 100,
            "EPS YoY (%)": q_eps["growth"] * 100,
            "Net Income YoY (%)": q_net["growth"] * 100,
            "Operating Margin Change (pp)": margin_change_pp
        })

        return result

    except Exception as e:
        result["Reasons"].append(
            f"資料擷取或計算失敗：{type(e).__name__}"
        )
        return result


# -----------------------------
# User interface
# -----------------------------
mode = st.radio(
    "選擇股票範圍",
    [
        "S&P 500 + Nasdaq-100",
        "自行輸入股票"
    ],
    horizontal=True
)

symbols = []

if mode == "S&P 500 + Nasdaq-100":
    if st.button("載入股票名單"):
        universe = get_universe()

        if universe:
            st.session_state["universe"] = universe
        else:
            st.error(
                "無法載入成分股名單。請檢查網絡，"
                "或改用自行輸入股票模式。"
            )

    symbols = st.session_state.get("universe", [])

    if symbols:
        st.write(f"名單包含 {len(symbols)} 個股票代號")
else:
    raw = st.text_area(
        "輸入股票代號，以逗號、空格或換行分隔",
        value="NVDA, MSFT, META, AMZN, GOOGL, AVGO",
        height=120
    )

    symbols = sorted(set(
        s.strip().upper()
        for s in raw.replace(",", " ").split()
        if s.strip()
    ))

if st.button(
    "開始嚴格篩選",
    type="primary",
    disabled=not symbols
):
    rows = []

    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        status.write(
            f"檢查 {symbol}（{i + 1}/{len(symbols)}）"
        )

        rows.append(screen_stock(symbol))
        progress.progress((i + 1) / len(symbols))

    df = pd.DataFrame(rows)
    st.session_state["screen_results"] = df

    status.write("掃描完成。")

if "screen_results" in st.session_state:
    df = st.session_state["screen_results"]

    passed = df[df["Status"] == "PASS"].copy()
    failed = df[df["Status"] != "PASS"].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("掃描股票", len(df))
    c2.metric("全部條件通過", len(passed))
    c3.metric(
        "淘汰比例",
        f"{(len(failed) / len(df) * 100):.1f}%"
        if len(df) else "—"
    )

    st.subheader("✅ 全部條件通過的股票")

    if passed.empty:
        st.info(
            "目前沒有股票通過所有已實施條件。"
            "不會降低門檻或推薦不合格股票。"
        )
    else:
        display_cols = [
            "Ticker", "Company", "Market Cap ($B)",
            "3Y Net Income Growth (%)",
            "Revenue YoY (%)", "EPS YoY (%)",
            "Net Income YoY (%)",
            "Operating Margin Change (pp)",
            "Latest Quarter"
        ]

        available_cols = [
            c for c in display_cols if c in passed.columns
        ]

        st.dataframe(
            passed[available_cols],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "下載合格股票 CSV",
            data=passed.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="strict_earnings_growth_pass.csv",
            mime="text/csv"
        )

    with st.expander("查看淘汰股票及原因"):
        st.dataframe(
            failed,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "下載完整掃描報告",
            data=df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="strict_earnings_growth_all.csv",
            mime="text/csv"
        )

st.caption(
    "研究用途，不構成投資建議。PASS 只表示通過本程式已實施的規則，"
    "不代表已完成 SEC 原始財報核對、一次性收益審查或審計。"
)
