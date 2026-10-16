
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

# -----------------------------
# Automatic universe + screening
# -----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_large_cap_universe():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100"
    ]

    symbols = set()
    errors = []

    for url in urls:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )
            response.raise_for_status()

            tables = pd.read_html(StringIO(response.text))
            found = False

            for table in tables:
                for col in table.columns:
                    col_name = str(col).strip().lower()

                    if col_name in [
                        "symbol",
                        "ticker",
                        "ticker symbol"
                    ]:
                        symbols.update(
                            table[col]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .str.replace(".", "-", regex=False)
                        )
                        found = True
                        break

                if found:
                    break

            if not found:
                errors.append(f"無法解析成分股：{url}")

        except Exception as exc:
            errors.append(
                f"{url}: {type(exc).__name__}"
            )

    symbols = sorted(
        s for s in symbols
        if s
        and s.isascii()
        and " " not in s
        and "/" not in s
    )

    return symbols, errors


st.subheader("大型股自動掃描")

st.write(
    "股票池：S&P 500 + Nasdaq-100。"
    "自動合併重複成分股，再以市值及全部盈利條件嚴格淘汰。"
)

if st.button("🔄 更新成分股名單"):
    get_large_cap_universe.clear()

symbols, universe_errors = get_large_cap_universe()

if universe_errors:
    st.warning(
        "部分成分股來源載入失敗。"
        "請檢查網絡及來源頁面，避免把不完整名單當成完整名單。"
    )
    for err in universe_errors:
        st.caption(err)

if not symbols:
    st.error(
        "無法取得有效股票名單，已停止掃描。"
        "不會以不完整名單冒充完整結果。"
    )
    st.stop()

st.metric("已載入的去重股票數", len(symbols))

st.info(
    "成分股名單每日快取 24 小時。"
    "掃描時仍會獨立核實每家公司市值是否達到 100 億美元。"
)

if st.button("▶ 開始全自動嚴格篩選", type="primary"):
    rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        status.write(
            f"正在檢查 {symbol}（{i + 1}/{len(symbols)}）"
        )

        rows.append(screen_stock(symbol))
        progress.progress((i + 1) / len(symbols))

    df = pd.DataFrame(rows)
    st.session_state["screen_results"] = df

    status.success("全股票池掃描完成。")

# Display results
if "screen_results" in st.session_state:
    df = st.session_state["screen_results"]

    passed = df[df["Status"] == "PASS"].copy()
    failed = df[df["Status"] != "PASS"].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("已掃描", len(df))
    c2.metric("全部條件通過", len(passed))
    c3.metric(
        "未通過",
        len(failed)
    )

    st.subheader("符合所有硬性條件的股票")

    if passed.empty:
        st.warning(
            "沒有股票通過全部條件。"
            "維持原有標準，不降低門檻。"
        )
    else:
        st.dataframe(
            passed,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "下載合格股票 CSV",
            passed.to_csv(index=False).encode("utf-8-sig"),
            "strict_pass.csv",
            "text/csv"
        )

    with st.expander("查看所有淘汰原因"):
        st.dataframe(
            failed,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "下載完整掃描報告",
            df.to_csv(index=False).encode("utf-8-sig"),
            "strict_all_results.csv",
            "text/csv"
        )
