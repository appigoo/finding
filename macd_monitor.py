"""
╔══════════════════════════════════════════════════════════════╗
║         MACD 瀑布動能傳導系統  v2.0                          ║
║                                                              ║
║  設計原理：                                                   ║
║  1. 時框瀑布傳導：1m→5m→15m→30m→1h→1d→1w                   ║
║     小時框動能積累，逐層推動大時框翻轉                         ║
║                                                              ║
║  2. 大時框定方向（1d/1w）確認主趨勢                           ║
║     小時框找入場點（30m Histogram D+1 預測翻正）              ║
║                                                              ║
║  3. D+1/D+2/D+3 = 提前預判翻轉時間點                        ║
║     不是等信號發生才通知，而是提前告知「即將翻正」             ║
║     讓用戶有時間準備，而不是追漲                              ║
║                                                              ║
║  4. 觸發邏輯：                                               ║
║     確認時框(1d/1h) Histogram > 0  ← 大方向多頭              ║
║     觸發時框(30m)  當前 Hist < 0   ← 尚未入場                ║
║     觸發時框(30m)  D+1 預測 > 0    ← 預計翻正 → 預警         ║
║                                                              ║
║  ── v2.1 新增（2026-07-03，因應 07-02 TSLA 暴跌）──          ║
║  5. 空頭側對稱警報：                                         ║
║     a) Hist > 0 且 D+1 < 0        → 預警：預測翻負           ║
║     b) 前根 ≥ 0 → 當根 < 0        → 警報：實際翻負           ║
║     c) Hist < 0 且 D+1 < Hist                                 ║
║        且大時框半數同向轉空       → 警報：瀑布下行           ║
║  6. 「接近反轉」閾值改用 Hist 滾動σ歸一化（0.15σ）           ║
║  7. Telegram 警報自動推送（MD5 去重，每訊號每小時一次）      ║
║  8. 空頭警報：紅色橫幅 + 反向 ATR 止損/目標                  ║
║                                                              ║
║  ── v2.2/v2.3 新增（量能整合 + 計分制 + 交易建議）──         ║
║  9. RVOL 相對量能：日內用「過去10天同時段」基準，            ║
║     解決日內 U 形量能造成的開盤誤報                          ║
║  10. VWAP 當日錨定：價格在 VWAP 上/下方作方向確認            ║
║  11. 開盤缺口偵測：跳空 ≥1.5% 直接高分警報                   ║
║  12. 計分制警報（多空對稱）：                                ║
║      翻轉 +2｜D+1 預測翻轉 +1｜加速>1.5σ +2                  ║
║      RVOL ≥2.5 +3｜1.5–2.5 +1｜<0.8 −2（記到陰/陽方向）     ║
║      VWAP 方向 +1｜大時框同向每個 +1｜缺口 +3                ║
║      ≥5 分 🔴 警報｜3–4 分 🟠 觀察｜<3 分靜默                ║
║  13. 訊號判定一律用「已收盤」bar，杜絕未收盤 bar 重繪誤報    ║
║  14. 時間戳修正為真正的美東時間；休市/非交易時段標註         ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime
import requests
import time
import hashlib
from zoneinfo import ZoneInfo

# ══════════════════════════════════════════════════════════════
# 頁面設定
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MACD 瀑布動能系統",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 設計系統 CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans TC', sans-serif;
    background-color: #f5f2ed;
    color: #2c2c2c;
}
.stApp { background-color: #f5f2ed; }
[data-testid="stSidebar"] {
    background-color: #ede9e2;
    border-right: 1px solid #d4cfc6;
}
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; color: #2c2c2c; }

/* ── 指標卡片 ── */
.metric-card {
    background: #fff8f0;
    border: 1px solid #d4cfc6;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    height: 100%;
    box-sizing: border-box;
}
.metric-card .label {
    font-size: 10px; color: #999;
    text-transform: uppercase; letter-spacing: 1px;
}
.metric-card .value {
    font-size: 22px; font-weight: 700; margin: 4px 0 2px;
}
.metric-card .sub { font-size: 12px; color: #777; }

/* ── 顏色 ── */
.pos { color: #3d8b5e; }
.neg { color: #c0392b; }
.neu { color: #7a7a7a; }

/* ── Badge ── */
.badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-size: 11px; font-weight: 700;
}
.badge-bull { background:#d4edda; color:#1e6b3d; }
.badge-bear { background:#fde8e8; color:#9b2335; }
.badge-warn { background:#fff3cd; color:#7d5a00; }
.badge-neu  { background:#e8e8e8; color:#555; }

/* ── MACD 表格 ── */
.macd-table {
    width:100%; border-collapse:collapse;
    font-family:'IBM Plex Mono',monospace; font-size:12px;
    background:#fff8f0; border-radius:10px;
    overflow:hidden; border:1px solid #d4cfc6;
}
.macd-table th {
    background:#e8e3da; color:#555; padding:9px 10px;
    text-align:center; font-weight:700; font-size:10px;
    letter-spacing:0.5px; white-space:nowrap;
}
.macd-table td {
    padding:8px 10px; text-align:center;
    border-bottom:1px solid #ede9e2; white-space:nowrap;
}
.macd-table tr:last-child td { border-bottom:none; }
.macd-table tr:hover td { background:#f0ece5; }
.cell-pos { color:#3d8b5e; font-weight:700; }
.cell-neg { color:#c0392b; font-weight:700; }

/* ── 預警卡片 ── */
.alert-card {
    background:linear-gradient(135deg,#fffbf0,#fff3d4);
    border:2px solid #f0c040; border-radius:12px;
    padding:16px 20px; margin:8px 0;
}
.alert-card-bear {
    background:linear-gradient(135deg,#fff5f4,#fde8e8);
    border:2px solid #c0392b; border-radius:12px;
    padding:16px 20px; margin:8px 0;
}
.alert-body {
    font-size:13px; color:#444; line-height:2;
    font-family:'IBM Plex Mono',monospace;
}

/* ── 共振評分 ── */
.resonance-score {
    display:inline-flex; align-items:center; gap:4px;
    background:#fff8f0; border:1px solid #d4cfc6;
    border-radius:20px; padding:4px 14px;
    font-family:'IBM Plex Mono',monospace; font-size:12px;
}
.star-on  { color:#f0a500; font-size:15px; }
.star-off { color:#ddd;    font-size:15px; }

/* ── Telegram 預覽 ── */
.tg-box {
    background:#fff8f0; border:1px solid #d4cfc6;
    border-left:4px solid #4a8c6f; border-radius:8px;
    padding:14px 18px; font-size:12px;
    white-space:pre-wrap; font-family:'IBM Plex Mono',monospace;
    line-height:1.8;
}

/* ── 趨勢 pill ── */
.pill-bull { background:#d4edda; color:#1e6b3d; padding:3px 10px; border-radius:20px; font-weight:700; font-size:11px; display:inline-block; }
.pill-bear { background:#fde8e8; color:#9b2335; padding:3px 10px; border-radius:20px; font-weight:700; font-size:11px; display:inline-block; }
.pill-neu  { background:#e8e8e8; color:#555;    padding:3px 10px; border-radius:20px; font-weight:700; font-size:11px; display:inline-block; }
.pill-warn { background:#fff3cd; color:#7d5a00; padding:3px 10px; border-radius:20px; font-weight:700; font-size:11px; display:inline-block; }

hr { border-color:#d4cfc6; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 常數
# ══════════════════════════════════════════════════════════════
CASCADE_CHAIN = ["1m","5m","15m","30m","1h","1d","1w"]

TIMEFRAME_MAP = {
    "1m":  {"period":"1d",   "interval":"1m"},
    "5m":  {"period":"5d",   "interval":"5m"},
    "15m": {"period":"10d",  "interval":"15m"},
    "30m": {"period":"30d",  "interval":"30m"},
    "1h":  {"period":"60d",  "interval":"1h"},
    "1d":  {"period":"180d", "interval":"1d"},
    "1w":  {"period":"2y",   "interval":"1wk"},
    "1mo": {"period":"5y",   "interval":"1mo"},
}

DEFAULT_SYMBOLS = ["QQQ", "TSLA", "TSLL", "SPCX", "AAPL", "GOOGL", "XPEV", "NIO", "META", "MSFT", "NVDA", "AMD", "INTC", "TSM", "XOM" , "VST", "RKLB", "ARM", "SNDK","MU","NKE","AVGO"]

# ── v2.7：大時框「形成中 bar」偵測 ──────────────────────────────
# 日內時框（1m–1h）在 analyze_cascade() 已用 hist_c/hist_c_prev（永遠丟最後一根未收盤
# bar）處理過；但主時框頂部卡片（收盤/Hist/趨勢…）與 build_macd_table 一直是用
# df 最後一列，日/週/月線在「當天/當週/當月尚未收盤」時會把 yfinance 回傳的
# 即時/預覽價格當成「收盤」顯示 —— 這正是 AAPL 卡片數字與實際昨收不符的原因。
def is_bar_forming(bar_ts, interval):
    """判斷 df 最後一根 K 線是否仍在形成中（尚未真正收盤）。僅處理 1d/1wk/1mo；
    分鐘級時框由 analyze_cascade 的已收盤邏輯另外處理，這裡一律回傳 False。"""
    try:
        ny = datetime.now(ZoneInfo("America/New_York"))
        bts = bar_ts.tz_convert("America/New_York") if bar_ts.tzinfo else bar_ts
    except Exception:
        return False
    if interval == "1d":
        return bts.date() == ny.date() and ny.time() < dtime(16, 0)
    if interval == "1wk":
        bw = bts.isocalendar()[:2]; nw = ny.isocalendar()[:2]
        return bw == nw and not (ny.weekday() == 4 and ny.time() >= dtime(16, 0))
    if interval == "1mo":
        return (bts.year, bts.month) == (ny.year, ny.month)
    return False

def drop_forming_bar(df, interval):
    """若最後一根尚未收盤，丟棄它，改用「上一根已收盤」的數據做顯示與計算。"""
    if len(df) > 1 and is_bar_forming(df.index[-1], interval):
        return df.iloc[:-1]
    return df

STATUS_NEXT_DAY = {
    "空頭動能強":    ("跌勢延續",    "bear"),
    "空頭減弱":      ("跌速放慢",    "warn"),
    "跌勢放緩":      ("接近底部",    "warn"),
    "空頭衰退":      ("技術反彈",    "warn"),
    "接近反轉":      ("金叉概率提升","neu"),
    "多頭開始回補":  ("動能轉正",    "bull"),
    "Histogram翻正": ("短線突破",    "bull"),
    "Histogram翻負": ("跌勢展開",    "bear"),
    "MACD金叉確認":  ("多頭加速",    "bull"),
    "多頭加速":      ("趨勢延續",    "bull"),
    "強勢多頭":      ("趨勢延續",    "bull"),
}


# ══════════════════════════════════════════════════════════════
# 核心計算
# ══════════════════════════════════════════════════════════════
def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ml  = calc_ema(close, fast) - calc_ema(close, slow)
    sl  = calc_ema(ml, signal)
    return ml, sl, ml - sl

def calc_atr(df, period=14):
    h, l, cp = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().iloc[-1]

# ── v2.2 量能指標 ────────────────────────────────────────────
INTRADAY_IVS = {"1m","5m","15m","30m","1h","60m","90m"}

def calc_rvol(df):
    """
    RVOL（日內版）：當前 bar 成交量 ÷ 過去10天「同一時段」平均量。
    日內量能天然呈 U 形（開盤/尾盤爆量、午盤清淡），
    用同時段基準才不會在開盤時段永遠誤報放量。
    """
    try:
        tod  = df.index.strftime("%H:%M")
        base = df["Volume"].groupby(tod).transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        return df["Volume"] / base
    except Exception:
        return pd.Series(np.nan, index=df.index)

def calc_rvol_simple(df, n=20):
    """RVOL（日/週線版）：當前量 ÷ 前 n 根平均量"""
    base = df["Volume"].shift(1).rolling(n, min_periods=5).mean()
    return df["Volume"] / base

def calc_vwap(df):
    """當日錨定 VWAP（僅日內時框有意義）"""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.normalize()
    cum_pv = (tp * df["Volume"]).groupby(day).cumsum()
    cum_v  = df["Volume"].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)

FIB_RETR = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT  = [1.272, 1.618, 2.0, 2.618]

def calc_fib_levels(df, lookback=60):
    """
    斐波那契回撤／延伸位。
    取最近 lookback 根「已收盤」K 線的最高高點與最低低點，
    用高低點出現的先後順序判斷目前擺動方向：
      低點在後（低→高後才拉回）→ 上升段回撤：100%=高點、0%=低點，
                                   回撤位由高往低算，是「拉回找支撐」。
      高點在後（高→低後才反彈）→ 下降段反彈：100%=低點、0%=高點，
                                   回撤位由低往高算，是「反彈找壓力」。
    延伸位（127.2%…）沿原擺動方向越過終點：上升段在高點上方、下降段在低點下方。
    回傳最近 K 線收盤價相對各回撤/延伸位的位置，及「最近關鍵位」。
    """
    if len(df) < 10:
        return None
    win   = df.iloc[-lookback:] if len(df) > lookback else df
    hi_i  = win["High"].idxmax(); lo_i = win["Low"].idxmin()
    hi    = float(win["High"].loc[hi_i]); lo = float(win["Low"].loc[lo_i])
    if hi <= lo:
        return None
    rng   = hi - lo
    swing = "up" if lo_i < hi_i else "down"   # 低點先出現＝上升段；反之為下降段

    levels = {}
    for r in FIB_RETR:
        levels[r] = (hi - rng*r) if swing == "up" else (lo + rng*r)
    for r in FIB_EXT:
        # 延伸位（v2.14 修正）：沿「原擺動方向」越過終點的量測目標。
        #   上升段：由低點往上量 → 高點上方（多方目標）
        #   下降段：由高點往下量 → 低點下方（空方目標）
        # 舊版把上升段的延伸位算到低點下方，那是「跌破後的下檔」而非買方目標。
        levels[r] = (lo + rng*r) if swing == "up" else (hi - rng*r)

    px = float(df["Close"].iloc[-1])
    nearest_r = min(levels, key=lambda r: abs(levels[r] - px))
    nearest_v = levels[nearest_r]
    dist_pct  = (px/nearest_v - 1) * 100 if nearest_v else 0.0
    # 上升段：價格在關鍵位之上＝該位是支撐；下降段：價格在關鍵位之下＝該位是壓力
    if swing == "up":
        role = "支撐" if px >= nearest_v else "壓力"
    else:
        role = "壓力" if px <= nearest_v else "支撐"

    return {
        "swing": swing, "high": hi, "low": lo,
        "levels": levels, "nearest_ratio": nearest_r,
        "nearest_level": nearest_v, "dist_pct": dist_pct, "role": role,
    }

def calc_sr_levels(df, window=5, tol_pct=0.006, max_each=3):
    """
    支撐／壓力線（擺動高低點聚類）。
    1) 左右各 window 根都不比它更高/更低的 K 線，記為擺動高點/低點（pivot）；
       最後 window 根還無法確認，不納入（不會因新 K 線而重繪）。
    2) 價差 ≤ tol_pct 的 pivot 合併為同一價位，觸及次數越多越強。
    3) 現價下方取最強 max_each 個為「支撐」、上方取最強 max_each 個為「壓力」；
       至少觸及 2 次才算可靠，若一個都沒有則退而取最近的單點，避免整張圖空白。
    """
    n = len(df)
    if n < window * 2 + 3:
        return []
    hi = df["High"].values; lo = df["Low"].values
    px = float(df["Close"].iloc[-1])
    piv = []
    for i in range(window, n - window):
        if hi[i] >= hi[i-window:i+window+1].max(): piv.append((float(hi[i]), i))
        if lo[i] <= lo[i-window:i+window+1].min(): piv.append((float(lo[i]), i))
    if not piv:
        return []
    piv.sort()
    clusters = []                       # [價格清單, 位置清單]
    for p, i in piv:
        if clusters and abs(p / (sum(clusters[-1][0]) / len(clusters[-1][0])) - 1) <= tol_pct:
            clusters[-1][0].append(p); clusters[-1][1].append(i)
        else:
            clusters.append([[p], [i]])
    lv = [{"price": sum(ps)/len(ps), "touches": len(ps), "last_i": max(ids)}
          for ps, ids in clusters]

    def pick(lst):
        strong = sorted([c for c in lst if c["touches"] >= 2],
                        key=lambda c: (-c["touches"], -c["last_i"]))[:max_each]
        if not strong and lst:
            strong = sorted(lst, key=lambda c: abs(c["price"] - px))[:1]
        return strong

    out = [dict(c, kind="支撐") for c in pick([c for c in lv if c["price"] < px])]
    out += [dict(c, kind="壓力") for c in pick([c for c in lv if c["price"] > px])]
    return out

# ══════════════════════════════════════════════════════════════
# v2.14 斐波那契買點掃描：週線定方向 → 日線找位置 → 4H 確認入場
#   規則 A：週線走弱 → 不因日線 61.8% 就判買入
#   規則 B：日線用 ZigZag（ATR 過濾）自動識別波段 A（低）→ B（高）
#   規則 C：回調到目標區後，須 4H 轉強訊號 + 成交量 + EMA20 + 前期支撐，缺則續等
# ══════════════════════════════════════════════════════════════
_NY = "America/New_York"

def atr_series(df, period=14):
    h, l, cp = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def zigzag_pivots(df, reversal_atr=2.5, atr_period=14):
    """
    ATR 過濾的 ZigZag：價格自極值反向走了 ≥ reversal_atr × ATR 才確認一個擺動點，
    小於這個幅度的波動一律忽略。
    回傳 (已確認 pivots, tail)；tail＝目前進行中、尚未被反向確認的極值。
    pivot / tail 皆為 {"i": 位置, "price": 價格, "kind": "H"|"L"}。
    """
    n = len(df)
    if n < atr_period + 5:
        return [], None
    h = df["High"].values; l = df["Low"].values
    atr = atr_series(df, atr_period).values
    pivots, direction = [], 0
    hi_p, hi_i, lo_p, lo_i = h[0], 0, l[0], 0
    for i in range(1, n):
        thr = reversal_atr * atr[i]
        if direction == 0:
            if h[i] > hi_p: hi_p, hi_i = h[i], i
            if l[i] < lo_p: lo_p, lo_i = l[i], i
            if hi_p - lo_p >= thr:
                if hi_i > lo_i:
                    pivots.append({"i": lo_i, "price": float(lo_p), "kind": "L"}); direction = 1
                elif lo_i > hi_i:
                    pivots.append({"i": hi_i, "price": float(hi_p), "kind": "H"}); direction = -1
        elif direction == 1:                       # 上升段：追蹤最高點
            if h[i] > hi_p:
                hi_p, hi_i = h[i], i
            elif hi_p - l[i] >= thr:
                pivots.append({"i": hi_i, "price": float(hi_p), "kind": "H"})
                direction, lo_p, lo_i = -1, l[i], i
        else:                                      # 下降段：追蹤最低點
            if l[i] < lo_p:
                lo_p, lo_i = l[i], i
            elif h[i] - lo_p >= thr:
                pivots.append({"i": lo_i, "price": float(lo_p), "kind": "L"})
                direction, hi_p, hi_i = 1, h[i], i
    tail = None
    if direction == 1:    tail = {"i": hi_i, "price": float(hi_p), "kind": "H"}
    elif direction == -1: tail = {"i": lo_i, "price": float(lo_p), "kind": "L"}
    return pivots, tail

def pick_up_leg(pivots, tail, atr_d, min_leg_atr=4.0, max_back=4):
    """最近一個「主要」上升波段 A（低）→ B（高）；幅度須 ≥ min_leg_atr × 日線 ATR。"""
    seq = list(pivots) + ([tail] if tail else [])
    tried = 0
    for k in range(len(seq) - 1, 0, -1):
        if seq[k]["kind"] == "H" and seq[k-1]["kind"] == "L":
            a, b = seq[k-1], seq[k]
            tried += 1
            if b["price"] - a["price"] >= min_leg_atr * atr_d:
                return {"A": a, "B": b, "b_confirmed": b is not tail}
            if tried >= max_back:
                break
    return None

def weekly_context(dfw, reversal_atr=2.0):
    """
    規則 A：週線背景。5 項計分：收盤>20週EMA、10週EMA>20週EMA、20週EMA 向上、
    週 MACD Hist>0、擺動結構 HH+HL（+1）／LH+LL（−1）。
    ≥4 分＝上升；≤1 分＝走弱；其餘＝盤整。
    """
    if dfw is None or len(dfw) < 30:
        return None
    c = dfw["Close"]
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    _, _, hist = calc_macd(c)
    piv, tail = zigzag_pivots(dfw, reversal_atr=reversal_atr)
    pts = piv + ([tail] if tail else [])
    Hs = [p for p in pts if p["kind"] == "H"]; Ls = [p for p in pts if p["kind"] == "L"]
    structure, struct_src = "mixed", "擺動點"
    if len(Hs) >= 2 and len(Ls) >= 2:
        if Hs[-1]["price"] > Hs[-2]["price"] and Ls[-1]["price"] > Ls[-2]["price"]:
            structure = "HH_HL"
        elif Hs[-1]["price"] < Hs[-2]["price"] and Ls[-1]["price"] < Ls[-2]["price"]:
            structure = "LH_LL"
    elif len(dfw) >= 52:
        # 備援：趨勢平穩、回調淺於 ZigZag 門檻時找不到足夠擺動點，
        # 改比較「近 26 週」與「前 26 週」的高低點（半年窗口的 HH/HL、LH/LL）
        r_, p_ = dfw.iloc[-26:], dfw.iloc[-52:-26]
        if r_["High"].max() > p_["High"].max() and r_["Low"].min() > p_["Low"].min():
            structure, struct_src = "HH_HL", "半年窗口"
        elif r_["High"].max() < p_["High"].max() and r_["Low"].min() < p_["Low"].min():
            structure, struct_src = "LH_LL", "半年窗口"
    px = float(c.iloc[-1])
    conds = [
        (px > ema20.iloc[-1],                         "收盤>20週EMA"),
        (ema10.iloc[-1] > ema20.iloc[-1],             "10週EMA>20週EMA"),
        (len(ema20) > 5 and ema20.iloc[-1] > ema20.iloc[-5], "20週EMA 向上"),
        (hist.iloc[-1] > 0,                           "週MACD Hist>0"),
    ]
    score = sum(1 for ok, _ in conds if ok)
    score += 1 if structure == "HH_HL" else (-1 if structure == "LH_LL" else 0)
    label = "上升" if score >= 4 else ("走弱" if score <= 1 else "盤整")
    ds = lambda p: dfw.index[p["i"]].strftime("%Y-%m-%d")
    if Hs:   last_high = (Hs[-1]["price"], ds(Hs[-1]))
    else:    r_ = dfw.iloc[-26:]; last_high = (float(r_["High"].max()), r_["High"].idxmax().strftime("%Y-%m-%d"))
    if Ls:   last_low = (Ls[-1]["price"], ds(Ls[-1]))
    else:    r_ = dfw.iloc[-26:]; last_low = (float(r_["Low"].min()), r_["Low"].idxmin().strftime("%Y-%m-%d"))
    return {
        "label": label, "score": score, "structure": structure, "struct_src": struct_src,
        "checks": [(ok, t) for ok, t in conds],
        "ema10": float(ema10.iloc[-1]), "ema20": float(ema20.iloc[-1]), "close": px,
        "last_high": last_high, "last_low": last_low,
    }

def build_4h(df1h, now_ny=None):
    """
    1h → 美股 4H：每個交易日 2 根（09:30–13:30、13:30–16:00），
    並丟掉「還在走」的最後一根（與全站「只用已收盤 bar」同一原則）。
    """
    if df1h is None or len(df1h) < 8:
        return pd.DataFrame()
    d = df1h.copy()
    idx = d.index
    idx = idx.tz_localize(_NY) if idx.tz is None else idx.tz_convert(_NY)
    d.index = idx
    d["_day"] = idx.date
    d["_blk"] = ((idx.hour * 60 + idx.minute) >= 13*60 + 30).astype(int)
    d["_ts"]  = idx
    g = d.groupby(["_day", "_blk"], sort=True).agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum"), ts=("_ts", "first"))
    g = g.reset_index()
    out = g[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.DatetimeIndex(g["ts"])
    now = now_ny or datetime.now(ZoneInfo(_NY))
    last = g.iloc[-1]
    end_h, end_m = (13, 30) if last["_blk"] == 0 else (16, 0)
    end_dt = datetime(last["_day"].year, last["_day"].month, last["_day"].day,
                      end_h, end_m, tzinfo=ZoneInfo(_NY))
    if now < end_dt:
        out = out.iloc[:-1]
    return out

def _scan_breakout(dfd, atr_d, lookback=20):
    """爆升前／突破：20 日高點（Donchian）突破與蓄勢（整理收斂）偵測，只看已收盤日線。"""
    n = len(dfd)
    if n < lookback + 10:
        return None
    hi = dfd["High"].values; lo = dfd["Low"].values
    cl = dfd["Close"].values; vol = dfd["Volume"].values
    fresh = None
    for k in (2, 1, 0):                         # 最近 3 根內，取最早一根有效突破
        j = n - 1 - k
        lvl = hi[j-lookback:j].max()
        vma = vol[j-lookback:j].mean()
        rv  = vol[j] / vma if vma > 0 else float("nan")
        if cl[j] > lvl and rv >= 1.5 and cl[-1] >= lvl - 0.5 * atr_d:
            fresh = {"level": float(lvl), "rvol": float(rv), "days_ago": k,
                     "date": dfd.index[j].strftime("%Y-%m-%d")}
            break
    res    = float(hi[-lookback:].max())
    near   = (res - cl[-1] <= 1.0 * atr_d) and cl[-1] <= res
    rng8   = float(hi[-8:].max() - lo[-8:].min())
    tight  = rng8 <= 2.5 * atr_d
    hl     = bool(lo[-4:].min() >= lo[-8:-4].min())
    dry    = bool(vol[-5:].mean() <= 0.85 * vol[-20:].mean())
    return {"fresh": fresh, "res": res, "near": bool(near), "tight": bool(tight),
            "higher_lows": hl, "vol_dry": dry, "rng8_atr": rng8 / atr_d,
            "base_low": float(lo[-8:].min()),
            "coil": bool(near and tight and (hl or dry))}

def _targets_above(ref, cands, atr_d):
    """從斐波那契候選價挑出高於 ref 的兩個目標；不足時以 ATR 倍數補。"""
    ups = sorted(set(round(c, 4) for c in cands if c is not None and c > ref + 0.5 * atr_d))
    src = "斐波那契"
    if len(ups) >= 2:
        return ups[0], ups[1], src
    if len(ups) == 1:
        return ups[0], ref + 3.5 * atr_d, "斐波那契+ATR"
    return ref + 2.0 * atr_d, ref + 3.5 * atr_d, "ATR"

def _rr(entry, stop, tgt):
    risk = entry - stop
    return (tgt - entry) / risk if (risk > 0 and tgt is not None) else None

SETUP_PRIO = {"✅": 1, "🚀": 2, "🟡": 3, "👀": 4, "🎯": 5, "⏳回": 6, "⏳等": 7,
              "⚠️": 8, "⛔": 9, "❌": 10, "—": 11}

def calc_fib_setup(dfw, dfd, df4, reversal_atr=2.5, min_leg_atr=4.0):
    """
    斐波那契買點掃描（週線定方向｜日線找位置｜4H 確認）。
    回傳 dict：status / prio / grade / weekly / A / B / levels / zone / checks / plan / breakout …
    """
    out = {"status": "— 資料不足", "prio": 11, "grade": None, "weekly": None,
           "A": None, "B": None, "levels": None, "checks": None, "plan": None,
           "breakout": None, "notes": []}
    if dfd is None or len(dfd) < 60:
        return out

    atr_d = float(atr_series(dfd).iloc[-1]); out["atr_d"] = atr_d
    wk = weekly_context(dfw); out["weekly"] = wk
    w_label = wk["label"] if wk else None
    has4 = df4 is not None and len(df4) >= 25
    px = float(df4["Close"].iloc[-1]) if has4 else float(dfd["Close"].iloc[-1])
    out["px"] = px; out["px_src"] = "4H 已收盤" if has4 else "日線已收盤"
    out["ema20_d"] = float(dfd["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    out["ema50_d"] = float(dfd["Close"].ewm(span=50, adjust=False).mean().iloc[-1])

    def done(status, prio, **kw):
        out["status"], out["prio"] = status, prio; out.update(kw); return out

    piv, tail = zigzag_pivots(dfd, reversal_atr)
    leg = pick_up_leg(piv, tail, atr_d, min_leg_atr)
    bo  = _scan_breakout(dfd, atr_d); out["breakout"] = bo
    if wk is None:
        out["notes"].append("週線資料不足，無法判斷大方向")
    if leg is None:
        return done("— 無有效波段", 11,
                    notes=out["notes"] + [f"近期無 ≥{min_leg_atr:g}×ATR 的主要上升波段"])

    A, B = leg["A"], leg["B"]
    a, b = A["price"], B["price"]; rng = b - a
    levels = {r: b - rng * r for r in (0.236, 0.382, 0.5, 0.618, 0.786)}
    ext    = {1.272: a + 1.272 * rng, 1.618: a + 1.618 * rng}
    ds = lambda p: dfd.index[p["i"]].strftime("%Y-%m-%d")
    out.update({"A": {"price": a, "date": ds(A)}, "B": {"price": b, "date": ds(B),
                "confirmed": leg["b_confirmed"]},
                "levels": levels, "ext": ext, "leg_atr": rng / atr_d,
                "zone": (levels[0.618], levels[0.382]),
                "golden": (levels[0.618], levels[0.5])})
    ret = (b - px) / rng; out["ret"] = ret

    # 回調低點（B 之後的日線低點 + 之後新增的 4H 低點）
    after = dfd["Low"].iloc[B["i"] + 1:]
    pb_low = float(after.min()) if len(after) else px
    if has4:
        last_d = dfd.index[-1].date()
        m = [t.date() > last_d for t in df4.index]
        if any(m):
            pb_low = min(pb_low, float(df4["Low"][m].min()))
    out["pb_low"] = pb_low
    closes_after = dfd["Close"].iloc[B["i"] + 1:]
    broken = (px < a) or (len(closes_after) and float(closes_after.min()) < a)

    # 回調量能：回調段平均量 ÷ 上升段平均量（<1 為健康的量縮回調）
    up_vol = dfd["Volume"].iloc[A["i"]:B["i"] + 1].mean()
    pb_vol = dfd["Volume"].iloc[B["i"] + 1:].mean() if len(after) else float("nan")
    out["pb_vol_ratio"] = float(pb_vol / up_vol) if up_vol and not np.isnan(pb_vol) else None

    # ── 規則 C：4H 確認（4 項）─────────────────────────────
    checks = None
    if has4:
        last, prev = df4.iloc[-1], df4.iloc[-2]
        rng4 = float(last["High"] - last["Low"])
        body_up = last["Close"] > last["Open"]
        lw = float(min(last["Open"], last["Close"]) - last["Low"])
        brk   = last["Close"] > prev["High"]
        engulf = (prev["Close"] < prev["Open"] and last["Close"] > prev["Open"]
                  and last["Open"] <= prev["Close"])
        hammer = rng4 > 0 and lw >= 0.5 * rng4 and last["Close"] >= last["Low"] + 0.6 * rng4
        candle = bool(body_up and (brk or engulf or hammer))
        c_txt = ("收陽並突破前根高點" if (body_up and brk) else
                 "陽包陰" if (body_up and engulf) else
                 "長下影錘子線" if (body_up and hammer) else "未出現轉強 K 線")
        blk = (df4.index.hour * 60 + df4.index.minute) >= 13*60 + 30
        same = df4["Volume"][blk == blk[-1]].iloc[:-1].tail(10).mean()
        vr = float(last["Volume"] / same) if same and same > 0 else None
        vol_ok = bool(body_up and vr is not None and vr >= 1.2)
        ema4 = df4["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_ok = bool(last["Close"] > ema4)
        zlo, zhi = levels[0.786] - 0.5 * atr_d, levels[0.382] + 0.5 * atr_d
        sr = [l for l in calc_sr_levels(dfd) if l["kind"] == "支撐"]
        hold = [l for l in sr if zlo <= l["price"] <= zhi and pb_low >= l["price"] - 0.5 * atr_d
                and px >= l["price"]]
        sup_ok = bool(hold)
        s_txt = (f"日線支撐 {hold[0]['price']:.2f}（觸及 {hold[0]['touches']} 次）守住"
                 if hold else "回調區內無已守住的前期支撐")
        checks = {
            "candle":  {"ok": candle,  "text": c_txt},
            "volume":  {"ok": vol_ok,  "text": (f"同類 4H 均量 {vr:.2f}×" if vr is not None else "量能資料不足")
                                                + ("（收陽放量）" if vol_ok else "（未達 1.2× 或非陽線）")},
            "ema20":   {"ok": ema_ok,  "text": f"4H 收 {last['Close']:.2f} {'>' if ema_ok else '≤'} EMA20 {ema4:.2f}"},
            "support": {"ok": sup_ok,  "text": s_txt},
        }
        out["checks"] = checks
        out["n_ok"] = sum(1 for v in checks.values() if v["ok"])
        out["bar4_time"] = df4.index[-1].strftime("%m-%d %H:%M")
    eps = 0.03
    n_ok = out.get("n_ok", 0)

    def plan_pullback():
        entry = px; stop = pb_low - 0.5 * atr_d
        t1, t2, src = b, ext[1.272], "斐波那契"
        t3 = ext[1.618]
        return {"kind": "回調買入", "entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": t3,
                "rr1": _rr(entry, stop, t1), "rr2": _rr(entry, stop, t2), "src": src}

    def plan_wait():
        entry = (levels[0.5] + levels[0.618]) / 2; stop = levels[0.786] - 0.5 * atr_d
        return {"kind": "等待回調（預估）", "entry": entry, "stop": stop, "t1": b, "t2": ext[1.272],
                "t3": ext[1.618], "rr1": _rr(entry, stop, b), "rr2": _rr(entry, stop, ext[1.272]),
                "src": "斐波那契"}

    def plan_break(entry, stop, ref):
        t1, t2, src = _targets_above(ref, [b, ext[1.272], ext[1.618], a + 2.0 * rng], atr_d)
        return {"kind": "突破買入", "entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": None,
                "rr1": _rr(entry, stop, t1), "rr2": _rr(entry, stop, t2), "src": src}

    weekly_ok  = (w_label == "上升")
    weekly_mid = (w_label == "盤整")

    # ── 狀態判定（規則 A 最優先）────────────────────────────
    if broken:
        return done("❌ 跌破 A 點，波段失效", 10, notes=out["notes"] + ["收盤已跌破 A，不使用此波段的斐波那契"])
    if w_label == "走弱":
        out["plan"] = plan_wait()
        return done("⛔ 週線走弱，不買", 9, grade=None,
                    notes=out["notes"] + ["規則 A：週線走弱，日線即使到 61.8% 也不判買入"])

    if wk is None:                                  # 沒有週線就不能判斷大方向（規則 A 無法執行）
        out["plan"] = plan_wait()
        return done("— 週線資料不足，無法判斷大方向", 11,
                    notes=out["notes"] + ["規則 A 需要至少 30 根週線"])

    grade = "A" if weekly_ok else ("B" if weekly_mid else None)
    out["grade"] = grade
    wk_tag = "" if weekly_ok else "（週線盤整，輕倉）"

    # 突破（20 日新高 + 放量）
    if bo and bo["fresh"]:
        f = bo["fresh"]
        out["plan"] = plan_break(px, f["level"] - 0.5 * atr_d, px)
        st_txt = ("🚀 突破成立" if weekly_ok else "🟡 突破成立，週線盤整（輕倉）")
        return done(st_txt, 2 if weekly_ok else 3, notes=out["notes"] + [
            f"{f['date']} 收盤突破前 20 日高 {f['level']:.2f}，RVOL {f['rvol']:.2f}×"])

    if ret < 0:                                     # 現價在 B 之上
        out["plan"] = plan_break(b + 0.1 * atr_d, b - 1.0 * atr_d, b + 0.1 * atr_d)
        out["plan"]["kind"] = "回測 B 買入（預估）"
        return done("⏳ 已突破 B，等回測", 6, notes=out["notes"] + ["現價高於波段高點 B，追高風險大，等回測 B 附近"])

    if ret < 0.382 - eps:                           # 淺回調，尚未進目標區
        if bo and bo["coil"]:
            entry = bo["res"] + 0.1 * atr_d; stop = bo["base_low"] - 0.3 * atr_d
            out["plan"] = plan_break(entry, stop, entry)
            st_txt = "🎯 蓄勢待突破" if weekly_ok else "🟡 蓄勢待突破，週線盤整（輕倉）"
            return done(st_txt, 5 if weekly_ok else 3, notes=out["notes"] + [
                f"近 8 日區間僅 {bo['rng8_atr']:.1f}×ATR，貼近 20 日高 {bo['res']:.2f}；"
                f"突破 {entry:.2f} 才進場"])
        out["plan"] = plan_wait()
        return done("⏳ 等待回調", 7, notes=out["notes"] + [
            f"目標區 {levels[0.618]:.2f}～{levels[0.382]:.2f}（38.2%～61.8%）"])

    in_main = ret <= 0.618 + eps
    in_deep = (not in_main) and ret <= 0.786 + eps
    if not (in_main or in_deep):
        return done("⚠️ 回調過深（>78.6%）", 8, notes=out["notes"] + ["回調超過 78.6%，波段偏弱，不追"])

    out["plan"] = plan_pullback()
    if checks is None:
        return done("👀 到位待確認（缺 4H 資料）", 4, notes=out["notes"])
    need_all = in_deep
    confirmed = checks["candle"]["ok"] and (n_ok == 4 if need_all else n_ok >= 3)
    missing = [k for k, v in checks.items() if not v["ok"]]
    name = {"candle": "轉強K線", "volume": "放量", "ema20": "站上EMA20", "support": "前期支撐"}
    zone_txt = "深回調區 61.8%～78.6%" if in_deep else "回調區 38.2%～61.8%"
    if confirmed:
        st_txt = "✅ 入場確認" if weekly_ok else "🟡 到位且確認，週線盤整（輕倉）"
        return done(st_txt, 1 if weekly_ok else 3, notes=out["notes"] + [
            f"價格在{zone_txt}，4H 確認 {n_ok}/4"])
    return done("👀 到位待確認" + ("（深回調需 4/4）" if need_all else ""), 4, notes=out["notes"] + [
        f"價格在{zone_txt}，4H 確認 {n_ok}/4，缺：" + "、".join(name[k] for k in missing)])

def fetch_setup_data(symbol):
    """週線（2y）＋日線（1y）＋ 4H（由 1h 合成）；週/日線已排除形成中的最後一根。"""
    dfw = drop_forming_bar(fetch_data(symbol, TIMEFRAME_MAP["1w"]["period"], "1wk"), "1wk")
    dfd = drop_forming_bar(fetch_data(symbol, "1y", "1d"), "1d")
    d1h = fetch_data(symbol, TIMEFRAME_MAP["1h"]["period"], "1h")
    return dfw, dfd, build_4h(d1h)

def scan_fib_setup(symbol, reversal_atr=2.5, min_leg_atr=4.0):
    dfw, dfd, df4 = fetch_setup_data(symbol)
    return calc_fib_setup(dfw, dfd, df4, reversal_atr, min_leg_atr)


# ── v2.14 買點掃描：顯示輔助（純函數）────────────────────────────
_STRUCT_TXT = {"HH_HL": "高點與低點同步抬高（HH+HL）", "LH_LL": "高點與低點同步下移（LH+LL）",
               "mixed": "高低點結構不明確"}

def setup_color(prio):
    return {1: "#2e7d4f", 2: "#2e7d4f", 3: "#c27a00", 4: "#c27a00", 5: "#2f6fb0",
            6: "#6b6b6b", 7: "#6b6b6b", 8: "#b8651b", 9: "#c0392b", 10: "#c0392b"}.get(prio, "#6b6b6b")

def _f2(v):
    return "—" if v is None else f"{v:.2f}"

def setup_weekly_text(wk):
    if not wk:
        return "週線資料不足"
    ok = "、".join(t for o, t in wk["checks"] if o) or "無"
    bad = "、".join(t for o, t in wk["checks"] if not o)
    s = (f"{wk['label']}（{wk['score']} 分｜結構：{_STRUCT_TXT.get(wk['structure'], wk['structure'])}"
         f"〔{wk.get('struct_src', '')}〕）｜通過：{ok}")
    if bad:
        s += f"｜未通過：{bad}"
    return s

def setup_checks_text(checks):
    if not checks:
        return None
    name = {"candle": "轉強K線", "volume": "成交量", "ema20": "站上4H EMA20", "support": "前期支撐"}
    return [(("✓" if v["ok"] else "✗"), name[k], v["text"]) for k, v in checks.items()]

def setup_plan_text(plan):
    if not plan:
        return None
    rr = lambda x: "—" if x is None else f"{x:.2f}"
    t3 = f"｜T3 {plan['t3']:.2f}" if plan.get("t3") else ""
    return (f"{plan['kind']}：進場 {plan['entry']:.2f}｜止損 {plan['stop']:.2f}｜"
            f"T1 {plan['t1']:.2f}｜T2 {plan['t2']:.2f}{t3}｜R:R {rr(plan.get('rr1'))} / {rr(plan.get('rr2'))}"
            f"（目標來源：{plan.get('src', '—')}）")

def setup_card_html(symbol, s):
    col = setup_color(s["prio"])
    L = [f"<b>① 週線定方向</b>　{setup_weekly_text(s.get('weekly'))}"]
    if s.get("A"):
        zl, zh = s["zone"]; gl, gh = s["golden"]
        bconf = "" if s["B"].get("confirmed") else "（進行中）"
        L.append(f"<b>② 日線找位置</b>　A {s['A']['price']:.2f}（{s['A']['date']}）→ "
                 f"B {s['B']['price']:.2f}（{s['B']['date']}）{bconf}｜幅度 {s['leg_atr']:.0f}×ATR"
                 f"｜現價 {s['px']:.2f}（{s['px_src']}）已回撤 <b>{s['ret']*100:.1f}%</b>")
        L.append(f"　　回調區 38.2%–61.8%：<b>{zl:.2f} ～ {zh:.2f}</b>｜核心 50%–61.8%：{gl:.2f} ～ {gh:.2f}"
                 f"｜78.6% = {s['levels'][0.786]:.2f}｜跌破 A {s['A']['price']:.2f} 波段失效")
        if s.get("pb_vol_ratio") is not None:
            L.append(f"　　回調量能 {s['pb_vol_ratio']:.2f}×（相對上升段；<1 為量縮的健康回調）"
                     f"｜日線 EMA20 {s['ema20_d']:.2f}／EMA50 {s['ema50_d']:.2f}")
    rows = setup_checks_text(s.get("checks"))
    if rows:
        L.append(f"<b>③ 4H 確認</b>（{s.get('bar4_time', '')}｜{s.get('n_ok', 0)}/4）")
        for mk, nm, tx in rows:
            c = "#2e7d4f" if mk == "✓" else "#c0392b"
            L.append(f"　　<span style='color:{c};font-weight:700'>{mk}</span> {nm}：{tx}")
    elif s.get("A"):
        L.append("<b>③ 4H 確認</b>　4H 資料不足")
    bo = s.get("breakout")
    if bo:
        flag = []
        flag.append(f"貼近20日高 {bo['res']:.2f}" if bo["near"] else "未貼近20日高")
        flag.append(f"近8日區間 {bo['rng8_atr']:.1f}×ATR{'（收斂）' if bo['tight'] else ''}")
        if bo["higher_lows"]: flag.append("低點抬高")
        if bo["vol_dry"]:     flag.append("量縮")
        L.append("<b>爆升前</b>　" + "｜".join(flag))
    pt = setup_plan_text(s.get("plan"))
    if pt:
        L.append(f"<b>計畫</b>　{pt}")
    for n in s.get("notes", []):
        L.append(f"<span style='color:#888'>· {n}</span>")
    return (f'<div class="metric-card" style="margin-top:8px">'
            f'<div class="label">🎯 斐波那契買點｜{symbol}｜週線定方向 → 日線找位置 → 4H 確認入場</div>'
            f'<div style="font-size:19px;font-weight:700;color:{col};margin:4px 0 6px">{s["status"]}</div>'
            f'<div class="sub" style="line-height:1.85;font-family:\'IBM Plex Mono\',monospace">'
            + "<br>".join(L) + '</div></div>')

def setup_tg_lines(s):
    if not s or s["prio"] == 11 and not s.get("A"):
        return []
    L = ["\n【斐波那契買點】" + s["status"]]
    wk = s.get("weekly")
    if wk:
        L.append(f"  週線 {wk['label']}（{wk['score']}分）")
    if s.get("A"):
        zl, zh = s["zone"]
        L.append(f"  波段 A {s['A']['price']:.2f} → B {s['B']['price']:.2f}（{s['leg_atr']:.0f}×ATR）")
        L.append(f"  現價 {s['px']:.2f} 回撤 {s['ret']*100:.1f}%｜回調區 {zl:.2f}～{zh:.2f}")
    rows = setup_checks_text(s.get("checks"))
    if rows:
        L.append(f"  4H 確認 {s.get('n_ok', 0)}/4：" + " ".join(f"{m}{n}" for m, n, _ in rows))
    pt = setup_plan_text(s.get("plan"))
    if pt:
        L.append("  " + pt)
    return L

def setup_prompt_lines(s):
    if not s:
        return []
    L = [f"- 斐波那契買點掃描（週線定方向｜日線找位置｜4H 確認）：{s['status']}"]
    L.append(f"  - 週線：{setup_weekly_text(s.get('weekly'))}")
    if s.get("A"):
        zl, zh = s["zone"]; gl, gh = s["golden"]
        lv = s["levels"]
        L.append(f"  - 日線波段（ZigZag，ATR 過濾）：A {s['A']['price']:.2f}（{s['A']['date']}）→ "
                 f"B {s['B']['price']:.2f}（{s['B']['date']}），幅度 {s['leg_atr']:.1f}×日線ATR（{s['atr_d']:.3f}）")
        L.append(f"  - 回撤位：23.6%={lv[0.236]:.2f}、38.2%={lv[0.382]:.2f}、50%={lv[0.5]:.2f}、"
                 f"61.8%={lv[0.618]:.2f}、78.6%={lv[0.786]:.2f}；延伸 127.2%={s['ext'][1.272]:.2f}、"
                 f"161.8%={s['ext'][1.618]:.2f}")
        L.append(f"  - 現價 {s['px']:.2f}（{s['px_src']}）已回撤 {s['ret']*100:.1f}%；回調區 {zl:.2f}～{zh:.2f}；"
                 f"回調低點 {s['pb_low']:.2f}")
        if s.get("pb_vol_ratio") is not None:
            L.append(f"  - 回調量能 {s['pb_vol_ratio']:.2f}×（相對上升段）；日線 EMA20 {s['ema20_d']:.2f}、EMA50 {s['ema50_d']:.2f}")
    rows = setup_checks_text(s.get("checks"))
    if rows:
        L.append(f"  - 4H 確認 {s.get('n_ok', 0)}/4（{s.get('bar4_time', '')}）：")
        for mk, nm, tx in rows:
            L.append(f"    - {mk} {nm}：{tx}")
    bo = s.get("breakout")
    if bo:
        L.append(f"  - 爆升前：20日高 {bo['res']:.2f}｜貼近={'是' if bo['near'] else '否'}｜近8日區間 "
                 f"{bo['rng8_atr']:.1f}×ATR｜收斂={'是' if bo['tight'] else '否'}｜低點抬高={'是' if bo['higher_lows'] else '否'}"
                 f"｜量縮={'是' if bo['vol_dry'] else '否'}｜蓄勢={'是' if bo['coil'] else '否'}"
                 + (f"｜突破成立：{bo['fresh']['date']} 收破 {bo['fresh']['level']:.2f}（RVOL {bo['fresh']['rvol']:.2f}×）"
                    if bo.get("fresh") else ""))
    pt = setup_plan_text(s.get("plan"))
    if pt:
        L.append(f"  - {pt}")
    for n in s.get("notes", []):
        L.append(f"  - 備註：{n}")
    return L

def setup_table_rows(results):
    """跨股票掃描表：依狀態優先序（✅ > 🚀 > 🟡 > 👀 > 🎯 > ⏳ …），同級以 R:R 高者優先。"""
    items = sorted(results.items(),
                   key=lambda kv: (kv[1]["prio"], -(((kv[1].get("plan") or {}).get("rr1")) or 0)))
    rows = []
    for i, (sym, s) in enumerate(items, 1):
        p = s.get("plan") or {}
        wk = s.get("weekly")
        rows.append({
            "排名": i, "代碼": sym, "狀態": s["status"],
            "週線": (f"{wk['label']}({wk['score']})" if wk else "—"),
            "波段 A→B": (f"{s['A']['price']:.2f}→{s['B']['price']:.2f}" if s.get("A") else "—"),
            "現價": _f2(s.get("px")),
            "回撤": (f"{s['ret']*100:.1f}%" if s.get("ret") is not None else "—"),
            "回調區(61.8–38.2%)": (f"{s['zone'][0]:.2f}～{s['zone'][1]:.2f}" if s.get("zone") else "—"),
            "4H確認": (f"{s['n_ok']}/4" if s.get("checks") else "—"),
            "進場": _f2(p.get("entry")), "止損": _f2(p.get("stop")),
            "T1": _f2(p.get("t1")), "T2": _f2(p.get("t2")),
            "R:R(T1)": ("—" if p.get("rr1") is None else f"{p['rr1']:.2f}"),
        })
    return rows


def calc_rolling_vwap(df, n=20):
    """
    滾動 VWAP（日線／週線用）。
    日線一根 K = 一天，用「當日錨定」會退化成該根自己的典型價
    （價格跟自己比恆等於零，毫無資訊量），故改用 n 根滾動的
    成交量加權均價，作為中期持倉成本錨。
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = (tp * df["Volume"]).rolling(n, min_periods=max(3, n//4)).sum()
    v  = df["Volume"].rolling(n, min_periods=max(3, n//4)).sum()
    return pv / v.replace(0, np.nan)

def predict_next3(hist):
    """
    D+1: 線性外推（延續當前動能斜率）
    D+2: 動能開始鈍化（斜率減半）
    D+3: 慣性衰減（大概率回落）
    """
    if len(hist) < 3:
        return 0.0, 0.0, 0.0
    v = hist.iloc[-3:].values
    slope = (v[-1] - v[0]) / 2
    d1 = v[-1] + slope
    d2 = d1 + slope * 0.5
    d3 = d2 - abs(slope) * 0.3
    return d1, d2, d3

def classify_status(hist, macd, signal):
    out = []
    # 反轉閾值用 Histogram 滾動標準差歸一化（取代固定 0.005，
    # 否則 TSLA 5m 級別的 Hist 永遠不會命中「接近反轉」）
    sig_roll = hist.rolling(40, min_periods=5).std()
    for i in range(len(hist)):
        h  = hist.iloc[i]
        hp = hist.iloc[i-1]  if i>0 else h
        m  = macd.iloc[i]
        s  = signal.iloc[i]
        mp = macd.iloc[i-1]  if i>0 else m
        sp = signal.iloc[i-1] if i>0 else s
        sg = sig_roll.iloc[i]
        thr = max(0.005, 0.15 * sg) if pd.notna(sg) else 0.005
        if   abs(h) < thr:                        st = "接近反轉"
        elif i>0 and hp<0 and h>=0:               st = "Histogram翻正"
        elif i>0 and hp>=0 and h<0:               st = "Histogram翻負"
        elif i>0 and m>s and mp<=sp:              st = "MACD金叉確認"
        elif h>0 and m>0:                          st = "強勢多頭"
        elif h>0 and i>0 and h>hp:                st = "多頭加速"
        elif h<0 and i>0 and h<hp:                st = "空頭動能強"
        elif h<0 and i>0 and abs(h)<abs(hp):      st = "空頭減弱" if h<=hp else "跌勢放緩"
        elif h<0:                                  st = "空頭動能強"
        else:                                      st = "多頭加速"
        out.append(st)
    return out

def get_trend(macd_val, hist_val, status):
    if macd_val>0 and hist_val>0:   return "強勢多頭"
    elif hist_val>0:                 return "多頭趨勢"
    elif hist_val<0 and macd_val<0: return "空頭趨勢"
    elif any(k in status for k in ["接近反轉","翻正","金叉"]): return "趨勢反轉中"
    else:                            return "震盪觀望"

def trend_pill(tr):
    m = {"強勢多頭":"bull","多頭趨勢":"bull","空頭趨勢":"bear","趨勢反轉中":"warn"}
    c = m.get(tr,"neu")
    i = {"bull":"▲","bear":"▼","warn":"◆","neu":"●"}[c]
    return f'<span class="pill-{c}">{i} {tr}</span>'

def badge_html(st):
    if any(k in st for k in ["多頭","翻正","金叉","放緩"]):
        return f'<span class="badge badge-bull">▲ {st}</span>'
    elif any(k in st for k in ["空頭","動能強","翻負"]):
        return f'<span class="badge badge-bear">▼ {st}</span>'
    elif any(k in st for k in ["接近反轉","減弱"]):
        return f'<span class="badge badge-warn">◆ {st}</span>'
    return f'<span class="badge badge-neu">● {st}</span>'

def nd_badge(status):
    txt, typ = STATUS_NEXT_DAY.get(status, ("待觀察","neu"))
    i = {"bull":"▲","bear":"▼","warn":"◆","neu":"●"}.get(typ,"●")
    return f'<span class="badge badge-{typ}">{i} {txt}</span>'

def fmt(v, d=3):
    return f"{v:+.{d}f}"

def pc(v):
    if v == "—": return "<td>—</td>"
    fv = float(v.replace("+",""))
    c  = "cell-pos" if fv>=0 else "cell-neg"
    return f'<td class="{c}">{v}</td>'


# ══════════════════════════════════════════════════════════════
# 資料獲取
# ══════════════════════════════════════════════════════════════
def fetch_data(symbol, period, interval):
    """
    Session-state 快取版本：
    - 每次頁面載入檢查快取是否過期
    - 過期（超過 refresh_interval 秒）→ 清除快取，重新拉取
    - 未過期 → 直接返回 session_state 中的 DataFrame
    - 完全不依賴 st.cache_data，確保自動刷新時數據一定更新
    """
    import time as _t
    ttl      = st.session_state.get("refresh_interval", 60)
    cache_key = f"df_{symbol}_{period}_{interval}"
    ts_key    = f"ts_{symbol}_{period}_{interval}"
    now       = _t.time()

    # 檢查是否需要更新
    last_fetch = st.session_state.get(ts_key, 0)
    if now - last_fetch >= ttl or cache_key not in st.session_state:
        # 過期或從未拉取 → 重新取數據
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df.empty:
                df = pd.DataFrame()
            else:
                df = df[["Open","High","Low","Close","Volume"]].dropna()
        except:
            df = pd.DataFrame()
        st.session_state[cache_key] = df
        st.session_state[ts_key]    = now
        st.session_state["_last_any_fetch_ts"] = now

    return st.session_state[cache_key]


# ══════════════════════════════════════════════════════════════
# 瀑布傳導分析
# ══════════════════════════════════════════════════════════════
def analyze_cascade(symbol, chain):
    results = []
    for tf in chain:
        cfg = TIMEFRAME_MAP.get(tf)
        if not cfg:
            continue
        df = fetch_data(symbol, cfg["period"], cfg["interval"])
        df = drop_forming_bar(df, cfg["interval"])
        if df.empty or len(df) < 30:
            results.append({"tf":tf, "valid":False})
            continue
        intraday = cfg["interval"] in INTRADAY_IVS
        macd, sig, hist = calc_macd(df["Close"])
        sts  = classify_status(hist, macd, sig)
        hv   = hist.iloc[-1]     # 最後一根（可能未收盤，僅顯示用）
        mv   = macd.iloc[-1]
        sv   = sts[-1]

        # ── v2.2：訊號判定一律用「已收盤」bar，避免未收盤 bar 重繪 ──
        hist_closed = hist.iloc[:-1] if len(hist) > 1 else hist
        hc  = hist_closed.iloc[-1]
        hcp = hist_closed.iloc[-2] if len(hist_closed) > 1 else hc
        # D+1/D+2/D+3 由已收盤序列外推（= 對「當前形成中 bar」的預測，不重繪）
        d1, d2, d3 = predict_next3(hist_closed)
        sigma = hist_closed.iloc[-40:].std() if len(hist_closed) >= 10 else max(abs(hc), 1e-9)

        # 動能加速（已收盤最後 3 根，>1.5σ 才算，過濾雜訊）
        accel_dn = accel_up = False
        if len(hist_closed) >= 3 and pd.notna(sigma) and sigma > 0:
            h3 = hist_closed.iloc[-3:].values
            accel_dn = bool(hc < 0 and h3[0] > h3[1] > h3[2] and (h3[0]-h3[2]) > 1.5*sigma)
            accel_up = bool(hc > 0 and h3[0] < h3[1] < h3[2] and (h3[2]-h3[0]) > 1.5*sigma)

        # ── 量能：RVOL（已收盤 bar）──
        rv_series = calc_rvol(df) if intraday else calc_rvol_simple(df)
        rvol = rv_series.iloc[-2] if len(rv_series) > 1 else np.nan

        # ── VWAP（已收盤 bar 位置）──
        #    日內：當日錨定 VWAP｜日/週線：20 根滾動 VWAP
        vwap_val, below_vwap = None, None
        vwap_kind = "當日VWAP" if intraday else "20期滾動VWAP"
        vw = calc_vwap(df) if intraday else calc_rolling_vwap(df)
        if len(vw) > 1 and pd.notna(vw.iloc[-2]):
            vwap_val   = float(vw.iloc[-2])
            below_vwap = bool(df["Close"].iloc[-2] < vwap_val)

        # ── 缺口（日內：當日開盤跳空｜日/週線：該根開盤 vs 前一根收盤）──
        gap_pct = None
        if intraday:
            days = df.index.normalize()
            uday = days.unique()
            if len(uday) >= 2:
                prev_close = df["Close"][days == uday[-2]].iloc[-1]
                gap_pct    = float(df["Open"][days == uday[-1]].iloc[0] / prev_close - 1)
        elif len(df) >= 3:
            prev_close = df["Close"].iloc[-3]
            if prev_close:
                gap_pct = float(df["Open"].iloc[-2] / prev_close - 1)

        bar_dn = bool(df["Close"].iloc[-2] < df["Open"].iloc[-2]) if len(df) > 1 else False

        # ── 斐波那契：日內丟最後一根形成中 bar（與 hist_c 同一慣例）；
        #    日/週/月線的 df 在抓取時已經過 drop_forming_bar，本身就是已收盤的 ──
        fib_df = df.iloc[:-1] if (intraday and len(df) > 1) else df
        fib    = calc_fib_levels(fib_df)

        results.append({
            "tf":tf, "valid":True,
            "hist":hv, "macd":mv, "status":sv,
            "hist_prev": hist.iloc[-2] if len(hist) > 1 else hv,
            "hist_c":hc, "hist_c_prev":hcp,
            "d1":d1, "d2":d2, "d3":d3,
            "sigma":sigma, "accel_dn":accel_dn, "accel_up":accel_up,
            "rvol": float(rvol) if pd.notna(rvol) else None,
            "vwap":vwap_val, "below_vwap":below_vwap, "vwap_kind":vwap_kind,
            "gap_pct":gap_pct, "bar_dn":bar_dn,
            "last_bar_day": str(df.index[-1].date()),
            "trend":get_trend(mv,hv,sv),
            "atr":calc_atr(df),
            "close":df["Close"].iloc[-1],
            # v2.9：已收盤價──與 hist_c / vwap_val / below_vwap / rvol / bar_dn
            # 同一根 K 棒，VWAP 上下方判定與距離%務必用這個，不要混用形成中價格
            "close_c": df["Close"].iloc[-2] if len(df) > 1 else df["Close"].iloc[-1],
            "fib": fib,
        })
    return results


ALERT_RED    = 5   # >= 5 分 → 紅色警報（立即行動級）
ALERT_YELLOW = 3   # 3–4 分  → 橙色觀察（提高注意）

def calc_resonance(cascade, confirm_tfs, trigger_tf):
    """
    v2.2 計分制警報引擎（多空完全對稱，全部用已收盤 bar 判定）：

      事件分：
        Histogram 翻轉          +2
        D+1 預測翻轉            +1
        動能加速（3根遞減>1.5σ） +2
        開盤缺口 |gap| >= 1.5%   +3
        大時框同向               每個 +1
      量能分（記到已收盤 K 線的陰/陽方向）：
        RVOL >= 2.5  +3（機構級）
        RVOL 1.5–2.5 +1（顯著放量）
        RVOL <  0.8  −2（縮量 → 大概率洗盤，壓制警報）
      位置分：
        價格 < VWAP → 空方 +1；價格 > VWAP → 多方 +1

      判定：事件錨定 — 必須至少有一個「事件分」訊號才能構成警報，
        同向/量能/VWAP 分只能加成、不能單獨發起（防止全天亂響）；
        取多/空較高分方向（同分空方優先＝風險優先）
        >= 5 分 → 紅色警報（推送）
        3–4 分 → 橙色觀察（推送）
        <  3 分 → 靜默
    """
    valid = [r for r in cascade if r.get("valid")]
    confirm_score, bear_score, confirm_max = 0, 0, 0
    for r in valid:
        if r["tf"] in confirm_tfs:
            confirm_max += 1
            h_ref = r.get("hist_c", r["hist"])
            if   h_ref > 0: confirm_score += 1
            elif h_ref < 0: bear_score    += 1

    trig      = next((r for r in valid if r["tf"] == trigger_tf), None)
    alert     = False
    atype     = None
    direction = None
    level     = None
    pts, detail = 0, []
    bear_pts, bull_pts = 0, 0
    bear_dt,  bull_dt  = [], []

    if trig:
        hc   = trig.get("hist_c", 0)
        hcp  = trig.get("hist_c_prev", hc)
        d1   = trig.get("d1", 0)
        rvol = trig.get("rvol")
        bvw  = trig.get("below_vwap")
        gap  = trig.get("gap_pct")

        # ── 空頭事件 ──────────────────────────────────────
        bear_evt = bull_evt = False
        if hcp >= 0 and hc < 0:
            bear_pts += 2; bear_dt.append("Histogram 翻負 +2"); bear_evt = True
        if hc > 0 and d1 < 0:
            bear_pts += 1; bear_dt.append("D+1 預測翻負 +1"); bear_evt = True
        if trig.get("accel_dn"):
            bear_pts += 2; bear_dt.append("空頭加速 >1.5σ +2"); bear_evt = True
        if gap is not None and gap <= -0.015:
            bear_pts += 3; bear_dt.append(f"開盤缺口 {gap*100:+.1f}% +3"); bear_evt = True
        if bear_score:
            bear_pts += bear_score
            bear_dt.append(f"大時框同向 ×{bear_score} +{bear_score}")

        # ── 多頭事件（鏡像）──────────────────────────────
        if hcp <= 0 and hc > 0:
            bull_pts += 2; bull_dt.append("Histogram 翻正 +2"); bull_evt = True
        if hc < 0 and d1 > 0:
            bull_pts += 1; bull_dt.append("D+1 預測翻正 +1"); bull_evt = True
        if trig.get("accel_up"):
            bull_pts += 2; bull_dt.append("多頭加速 >1.5σ +2"); bull_evt = True
        if gap is not None and gap >= 0.015:
            bull_pts += 3; bull_dt.append(f"開盤缺口 {gap*100:+.1f}% +3"); bull_evt = True
        if confirm_score:
            bull_pts += confirm_score
            bull_dt.append(f"大時框同向 ×{confirm_score} +{confirm_score}")

        # ── 量能分（依已收盤 K 線陰陽記方向）─────────────
        if rvol is not None and not np.isnan(rvol):
            vp, vtxt = 0, None
            if   rvol >= 2.5: vp, vtxt =  3, f"RVOL {rvol:.1f}× 機構級 +3"
            elif rvol >= 1.5: vp, vtxt =  1, f"RVOL {rvol:.1f}× 放量 +1"
            elif rvol <  0.8: vp, vtxt = -2, f"RVOL {rvol:.1f}× 縮量 −2"
            if vtxt:
                if trig.get("bar_dn"):
                    bear_pts += vp; bear_dt.append(vtxt)
                else:
                    bull_pts += vp; bull_dt.append(vtxt)

        # ── VWAP 位置分 ──────────────────────────────────
        if bvw is True:
            bear_pts += 1; bear_dt.append("價格 < VWAP +1")
        elif bvw is False:
            bull_pts += 1; bull_dt.append("價格 > VWAP +1")

        # ── 斐波那契共振分（只加成，不能單獨發起警報；仍受事件錨定約束）──
        #    價格距 0.382/0.5/0.618/0.786 任一回撤位 ≤ 0.5×ATR 才算「貼近關鍵位」；
        #    該位是支撐 → 多方加分；是壓力 → 空方加分；0.618 黃金分割位權重 +2
        fib = trig.get("fib")
        if fib and trig.get("atr"):
            r_  = fib["nearest_ratio"]
            px_ = trig.get("close_c", trig.get("close"))
            if r_ in (0.382, 0.5, 0.618, 0.786) and px_ is not None \
                    and abs(px_ - fib["nearest_level"]) <= 0.5 * trig["atr"]:
                w = 2 if r_ == 0.618 else 1
                lbl = f"斐波那契 {r_*100:.1f}% {fib['role']} +{w}"
                if fib["role"] == "支撐":
                    bull_pts += w; bull_dt.append(lbl)
                else:
                    bear_pts += w; bear_dt.append(lbl)

        # ── 判定（事件錨定：無觸發框事件則不構成警報，
        #    同向分/量能分/VWAP分只能「加成」不能「發起」）──
        cands = []
        if bear_evt and bear_pts >= ALERT_YELLOW:
            cands.append(("bear", bear_pts, bear_dt))
        if bull_evt and bull_pts >= ALERT_YELLOW:
            cands.append(("bull", bull_pts, bull_dt))
        if cands:
            # 同分時取列表首位 = 空方優先（風險優先原則）
            direction, pts, detail = max(cands, key=lambda c: c[1])
            alert = True
            level = "red" if pts >= ALERT_RED else "yellow"
            if direction == "bear":
                atype = (f"警報：空頭 {pts} 分，建議減倉/離場" if level == "red"
                         else f"觀察：空頭訊號累積 {pts} 分")
            else:
                atype = (f"預警：多頭 {pts} 分，準備做多" if level == "red"
                         else f"觀察：多頭訊號累積 {pts} 分")

    # 星級共振分（沿用原顯示邏輯，按警報方向計對齊數）
    base  = bear_score if direction == "bear" else confirm_score
    total = base + (1 if alert else 0)
    return {
        "score":     total,
        "max":       confirm_max + 1,
        "alert":     alert,
        "atype":     atype,
        "direction": direction,
        "level":     level,
        "pts":       pts,
        "bear_pts":  bear_pts,
        "bull_pts":  bull_pts,
        "detail":    detail,
        "trigger":   trig,
    }


def stars_html(score, max_score):
    n      = 5
    filled = round(score / max_score * n) if max_score > 0 else 0
    s = "".join([
        '<span class="star-on">★</span>' if i < filled
        else '<span class="star-off">★</span>'
        for i in range(n)
    ])
    return f'<span class="resonance-score">{s} &nbsp;共振 {score}/{max_score}</span>'


# ══════════════════════════════════════════════════════════════
# 圖表
# ══════════════════════════════════════════════════════════════
def fmt_labels(index, interval):
    intraday = interval in ["1m","5m","15m","30m","1h","60m","90m"]
    out = []
    for ts in index:
        dt = pd.Timestamp(ts)
        try: dt = dt.tz_convert("America/New_York") if dt.tzinfo else dt
        except: pass
        out.append(dt.strftime("%m/%d %H:%M") if intraday else dt.strftime("%m/%d"))
    return out

def make_ticks(labels, interval):
    intraday = interval in ["1m","5m","15m","30m","1h","60m","90m"]
    step     = max(1, len(labels)//12)
    sel      = labels[::step]
    if not intraday:
        return sel, sel
    tt, prev = [], None
    for lbl in sel:
        parts = lbl.split(" ")
        d, t  = parts[0], (parts[1] if len(parts)>1 else lbl)
        tt.append(lbl if d != prev else t)
        prev = d
    return sel, tt

def build_macd_chart(df, symbol, macd, signal, hist, interval="1d", overlays=None,
                    show_volume=False, rvol=None):
    xl = fmt_labels(df.index,     interval)
    xm = fmt_labels(macd.index,   interval)
    xh = fmt_labels(hist.index,   interval)
    xs = fmt_labels(signal.index, interval)

    # v2.13：成交量副圖（價格 / 成交量 / MACD 三格）；無量能數據時退回原本兩格
    has_vol = bool(show_volume and "Volume" in df.columns
                   and float(df["Volume"].fillna(0).sum()) > 0)
    mr = 3 if has_vol else 2                    # MACD 所在的 row
    if has_vol:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.46, 0.18, 0.36], vertical_spacing=0.045,
                            subplot_titles=[f"{symbol} 收盤價",
                                            "成交量（粗框＝RVOL≥2.5×｜淡色＝縮量 <0.8×）",
                                            "MACD (12,26,9)"])
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.52,0.48], vertical_spacing=0.05,
                            subplot_titles=[f"{symbol} 收盤價", "MACD (12,26,9)"])

    fig.add_trace(go.Scatter(x=xl, y=df["Close"].values, mode="lines",
        name="收盤價", line=dict(color="#5a7fa8",width=2)), row=1,col=1)

    # ── v2.12 價格圖疊加：主力成本線 / 斐波那契 / 支撐壓力 ──────────────
    if overlays:
        _c  = df["Close"].dropna()
        _lo, _hi = float(_c.min()), float(_c.max())
        _pad = (_hi - _lo) * 0.08 if _hi > _lo else _hi * 0.01
        _vis = lambda v: (_lo - _pad) <= v <= (_hi + _pad)   # 只畫落在可視範圍內的線，避免價格被壓扁

        vw = overlays.get("vwap")
        if vw is not None and len(vw) == len(df) and vw.notna().any():
            fig.add_trace(go.Scatter(x=xl, y=vw.values, mode="lines",
                name=overlays.get("vwap_label", "主力成本線（VWAP）"),
                line=dict(color="#8e44ad", width=2.4)), row=1, col=1)

        fb = overlays.get("fib")
        if fb:
            _sty = {0.0:("#6b6b6b","solid",1.0), 1.0:("#6b6b6b","solid",1.0),
                    0.236:("#b0b0b0","dash",1.0), 0.382:("#8c8c8c","dash",1.2),
                    0.5:("#8e6bbf","dash",1.2), 0.618:("#c9a227","dash",2.0),
                    0.786:("#8c8c8c","dash",1.2),
                    1.272:("#b07a3a","dot",1.2), 1.618:("#b07a3a","dot",1.2)}
            for r_, (col, dsh, wd) in _sty.items():
                v_ = fb["levels"].get(r_)
                if v_ is None or not _vis(v_):
                    continue
                fig.add_hline(y=v_, line_color=col, line_dash=dsh, line_width=wd,
                    annotation_text=f"Fib {r_*100:.1f}%  {v_:.2f}",
                    annotation_position="top right",
                    annotation_font=dict(size=9, color=col), row=1, col=1)

        stp = overlays.get("setup")
        if stp and stp.get("A"):
            zl, zh = stp["zone"]
            if _vis(zl) or _vis(zh):
                fig.add_hrect(y0=max(zl, _lo - _pad), y1=min(zh, _hi + _pad),
                              fillcolor="rgba(61,139,94,0.12)", line_width=0, row=1, col=1)
                fig.add_annotation(xref="x domain", yref="y", x=0.0, y=(zl + zh) / 2,
                                   text="回調買入區 38.2–61.8%", showarrow=False,
                                   xanchor="left", font=dict(size=9, color="#2e7d4f"))
            for nm, pv, cl in (("B 波段高", stp["B"]["price"], "#2e7d4f"),
                               ("A 波段低（跌破失效）", stp["A"]["price"], "#c0392b")):
                if _vis(pv):
                    fig.add_hline(y=pv, line_color=cl, line_dash="solid", line_width=1.4,
                                  annotation_text=f"{nm} {pv:.2f}", annotation_position="bottom right",
                                  annotation_font=dict(size=9, color=cl), row=1, col=1)

        for lv in (overlays.get("sr") or []):
            if not _vis(lv["price"]):
                continue
            col = "#3d8b5e" if lv["kind"] == "支撐" else "#c0392b"
            fig.add_hline(y=lv["price"], line_color=col, line_dash="dashdot", line_width=1.6,
                annotation_text=f"{lv['kind']} {lv['price']:.2f} ×{lv['touches']}",
                annotation_position="top left",
                annotation_font=dict(size=9, color=col), row=1, col=1)

    if has_vol:
        vol  = df["Volume"].fillna(0).values
        up   = (df["Close"].values >= df["Open"].values)
        vcol = ["#3d8b5e" if u else "#c0392b" for u in up]
        n_   = len(vol)
        op, lw = [0.70] * n_, [0.0] * n_
        if rvol is not None and len(rvol) == n_:
            for i, v in enumerate(rvol.values):
                if pd.isna(v):
                    continue
                if   v >= 2.5: op[i], lw[i] = 1.00, 1.8      # 機構級放量：不透明 + 粗框
                elif v >= 1.5: op[i]        = 0.90           # 顯著放量
                elif v <  0.8: op[i]        = 0.35           # 縮量：淡化
        fig.add_trace(go.Bar(x=xl, y=vol, name="成交量",
            marker=dict(color=vcol, opacity=op,
                        line=dict(color="#222", width=lw))), row=2, col=1)
        vma = pd.Series(vol).rolling(20, min_periods=5).mean()
        fig.add_trace(go.Scatter(x=xl, y=vma.values, mode="lines", name="均量(20)",
            line=dict(color="#e07b39", width=1.4)), row=2, col=1)
        fig.update_yaxes(tickformat=".2s", row=2, col=1)

    colors = ["#3d8b5e" if v>=0 else "#c0392b" for v in hist.values]
    fig.add_trace(go.Bar(x=xh, y=hist.values, name="Histogram",
        marker_color=colors, opacity=0.85), row=mr,col=1)
    fig.add_trace(go.Scatter(x=xm, y=macd.values, mode="lines", name="MACD",
        line=dict(color="#5a7fa8",width=1.5)), row=mr,col=1)
    fig.add_trace(go.Scatter(x=xs, y=signal.values, mode="lines", name="Signal",
        line=dict(color="#e07b39",width=1.5,dash="dot")), row=mr,col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#bbb", row=mr, col=1)

    tv, tt = make_ticks(xl, interval)
    xcfg = dict(type="category", tickvals=tv, ticktext=tt,
                tickangle=-35, gridcolor="#e8e3da", showgrid=True)
    fig.update_layout(
        paper_bgcolor="#fff8f0", plot_bgcolor="#fff8f0",
        font=dict(family="IBM Plex Mono, Noto Sans TC", color="#2c2c2c", size=11),
        margin=dict(l=10,r=10,t=36,b=20),
        legend=dict(orientation="h", y=1.02, x=0),
        height=620 if has_vol else 480, xaxis_rangeslider_visible=False,
        xaxis=xcfg, xaxis2=xcfg,
    )
    if has_vol:
        fig.update_layout(xaxis3=xcfg)
    fig.update_yaxes(gridcolor="#e8e3da", zeroline=True, zerolinecolor="#c0bbb2")
    return fig


def build_cascade_chart(cascade):
    """
    瀑布傳導總覽圖：
    - 實色柱 = 當前 Histogram
    - 半透明柱 = D+1 預測
    一眼看清傳導方向是否對齊
    """
    valid  = [r for r in cascade if r.get("valid")]
    tfs    = [r["tf"]   for r in valid]
    hists  = [r["hist"] for r in valid]
    d1s    = [r["d1"]   for r in valid]

    c_now  = ["#3d8b5e" if h>=0 else "#c0392b" for h in hists]
    c_pred = ["rgba(61,139,94,0.35)" if d>=0 else "rgba(192,57,43,0.35)" for d in d1s]
    b_pred = ["#3d8b5e" if d>=0 else "#c0392b" for d in d1s]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tfs, y=hists, name="當前 Histogram",
        marker_color=c_now,
        text=[fmt(h,3) for h in hists],
        textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono"),
    ))
    fig.add_trace(go.Bar(
        x=tfs, y=d1s, name="D+1 預測",
        marker_color=c_pred,
        marker_line=dict(width=2, color=b_pred),
        text=[fmt(d,3) for d in d1s],
        textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa")
    fig.update_layout(
        paper_bgcolor="#fff8f0", plot_bgcolor="#fff8f0",
        font=dict(family="IBM Plex Mono", color="#2c2c2c", size=11),
        margin=dict(l=10,r=10,t=10,b=10),
        height=230, barmode="group",
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(gridcolor="#e8e3da", type="category"),
        yaxis=dict(gridcolor="#e8e3da", zeroline=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# MACD 表格
# ══════════════════════════════════════════════════════════════
def build_macd_table(df, n=10):
    macd, signal, hist = calc_macd(df["Close"])
    sts = classify_status(hist, macd, signal)
    d1,d2,d3 = predict_next3(hist)
    tail = df.tail(n)
    rows = []
    for i in range(len(tail)):
        h   = hist.tail(n).iloc[i]
        m   = macd.tail(n).iloc[i]
        st  = sts[-n:][i]
        last= i == len(tail)-1
        rows.append({
            "日線":  tail.index[i].strftime("%m/%d"),
            "收盤":  f"{tail['Close'].iloc[i]:.2f}",
            "MACD":  fmt(m,3),
            "Hist":  fmt(h,3),
            "狀態":  st,
            "下一交易日": st,
            "D+1":   fmt(d1,3) if last else "—",
            "D+2":   fmt(d2,3) if last else "—",
            "D+3":   fmt(d3,3) if last else "—",
            "_h":h, "_m":m, "_st":st,
        })
    return pd.DataFrame(rows), macd, signal, hist

def render_table(df_t):
    cols = ["日線","收盤","MACD","Hist","狀態","下一交易日","D+1","D+2","D+3"]
    hdr  = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, r in df_t.iterrows():
        hc = "cell-pos" if r["_h"]>=0 else "cell-neg"
        mc = "cell-pos" if float(r["MACD"].replace("+",""))>=0 else "cell-neg"
        body += f"""<tr>
            <td>{r['日線']}</td><td>{r['收盤']}</td>
            <td class="{mc}">{r['MACD']}</td>
            <td class="{hc}">{r['Hist']}</td>
            <td>{badge_html(r['狀態'])}</td>
            <td>{nd_badge(r['下一交易日'])}</td>
            {pc(r['D+1'])}{pc(r['D+2'])}{pc(r['D+3'])}
        </tr>"""
    return f'<table class="macd-table"><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table>'


# ══════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════
# ── v2.4 排序比較 + 深度分析 Prompt 產生器 ──────────────────────
_LEVEL_RANK = {"red": 2, "yellow": 1, None: 0}

def leaning(r):
    """未達警報門檻時的潛在傾向：比較底層多/空分數（非正式訊號，僅供排序參考）"""
    bp, sp = r.get("bull_pts", 0), r.get("bear_pts", 0)
    if bp == 0 and sp == 0:
        return None
    return "bear" if sp >= bp else "bull"

def rank_key(r):
    """
    排序鍵（由粗到細）：
      1. 正式警報等級（紅>橙>無）
      2. 正式警報分數
      3. 底層潛在傾向分（即使未觸發警報，分數越高代表方向性越強）
         → 這一層修正「大時框很少翻轉 → 全部靜默 → 排序退化成輸入順序」的問題
      4. 同分時空頭優先（風險優先顯示）
    """
    lvl     = _LEVEL_RANK.get(r["level"], 0)
    pts     = r["pts"]
    sub     = max(r.get("bull_pts", 0), r.get("bear_pts", 0))
    eff_dir = r["direction"] or leaning(r)
    return (lvl, pts, sub, 1 if eff_dir == "bear" else 0)

def build_analysis_prompt(rank_data, main_tf, trigger_tf,
                          chain_tfs=None, confirm_tfs=None, plan_style=None):
    """
    產生可直接貼給任意 AI（ChatGPT / Claude / Gemini 等）的深度分析 Prompt。
    本系統只做「技術面 + 量能」的程式化判定；Prompt 的用途是請另一個 AI
    補上本系統做不到的部分：基本面、新聞事件、產業比較、情緒面交叉驗證。

    v2.11：內容為歷代版本的「超集」──舊版所有段落/欄位原樣保留，另補上
    系統設定、評分規則、各時框完整明細、量能/VWAP、斐波那契全部位、
    交易建議與 ATR 參考位、主時框近 10 根 MACD 表、輸出格式要求。
    """
    if not rank_data:
        return ""
    ranked = sorted(rank_data, key=rank_key, reverse=True)
    ny   = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    syms = "、".join(r["symbol"] for r in ranked)

    def sg(v, d=3):          # 帶正負號的數字，None 顯示為 —
        return "—" if v is None else f"{v:+.{d}f}"

    L = []
    L.append(f"# 股票技術面排行分析請求（數據時間：{ny}）")
    L.append("")
    L.append("以下是我用自建的 MACD 瀑布傳導 + RVOL/VWAP/斐波那契 量能計分系統，")
    L.append(f"對多支股票在 {main_tf} 主時框、{trigger_tf} 觸發時框下跑出的技術面排行結果。")
    _vk = next((r.get("vwap_kind") for r in rank_data if r.get("vwap_kind")), None)
    if _vk:
        L.append(f"（VWAP 定義：{_vk}；日內為當日錨定，日/週線為 20 期滾動成交量加權均價）")

    # ── 系統設定 ───────────────────────────────────────────
    L.append("")
    L.append("## 系統設定")
    L.append(f"- 標的（{len(ranked)} 支）：{syms}")
    L.append(f"- 主時框：{main_tf}｜觸發時框：{trigger_tf}")
    if chain_tfs:
        L.append("- 傳導鏈（由小到大）：" + " → ".join(chain_tfs))
    if confirm_tfs:
        L.append("- 確認時框：" + "、".join(confirm_tfs))
    if plan_style:
        L.append(f"- 交易建議風格：{plan_style}"
                 "（穩健＝等收盤確認、1.5×ATR 止損；激進＝訊號 bar 即進場、1.0×ATR 止損）")
    if rank_data and any(r.get("setup") for r in rank_data):
        L.append("- 買點掃描資料：週線（2 年）、日線（1 年）、4H（由 1h 合成，美股每日 2 根）")
    L.append("- 所有訊號判定只用「已收盤」K 線；標示「形成中」的數值是尚未收盤的預覽，請勿當成確認訊號")
    L.append("- Hist＝MACD 柱狀圖；D+1/D+2/D+3＝以最近 3 根線性外推的未來 3 根預測值")

    # ── 評分規則 ───────────────────────────────────────────
    L.append("")
    L.append("## 評分規則（請據此解讀分數）")
    L.append("- 事件分：Histogram 翻轉 +2｜D+1 預測翻轉 +1｜動能加速（連 3 根遞減/遞增且幅度 >1.5σ）+2｜開盤缺口 ≥1.5% +3")
    L.append("- 確認時框同向：每個 +1")
    L.append("- 量能（依已收盤 K 線陰/陽記方向）：RVOL ≥2.5 +3｜1.5–2.5 +1｜<0.8 −2（縮量壓制訊號）")
    L.append("- VWAP 位置：價格在 VWAP 下方 → 空方 +1；上方 → 多方 +1")
    L.append("- 斐波那契：價格距 38.2/50/61.8/78.6% 任一回撤位 ≤0.5×ATR → 該位為支撐＝多方加分、為壓力＝空方加分；61.8% +2、其餘 +1")
    L.append(f"- 判定：必須先有至少一個「事件分」才能發警報（確認同向/量能/VWAP/斐波那契只能加成、不能單獨發起）；"
             f"≥{ALERT_RED} 分＝🔴 紅色警報、{ALERT_YELLOW}–{ALERT_RED-1} 分＝🟠 橙色觀察、<{ALERT_YELLOW} 分＝靜默；"
             "無正式警報時改用「潛在傾向分」（底層多/空分數）排序")

    L.append("")
    L.append("## 斐波那契買點掃描規則（週線定方向｜日線找位置｜4H 確認入場）")
    L.append("- 規則 A（週線）：5 項計分——收盤>20週EMA、10週EMA>20週EMA、20週EMA 向上、週MACD Hist>0、"
             "高低點結構 HH+HL（+1）/ LH+LL（−1）；≥4 分＝上升、≤1 分＝走弱、其餘＝盤整。"
             "週線走弱時，日線即使回到 61.8% 也不判買入；週線盤整只給「輕倉」等級。")
    L.append("- 規則 B（日線）：用 ZigZag（反轉門檻＝N×日線ATR）自動識別擺動高低點，取最近一個幅度 ≥ M×ATR 的"
             "上升波段 A（低）→ B（高）；回調區＝38.2%～61.8%，深回調區＝61.8%～78.6%（需 4/4 確認），"
             ">78.6% 視為過深，收盤跌破 A 波段失效。延伸目標：127.2% / 161.8%（由 A 往上量）。")
    L.append("- 規則 C（4H 確認，只用已收盤 4H）：①轉強 K 線（收陽並突破前根高點／陽包陰／長下影錘子線）"
             "②成交量 ≥ 同類 4H 均量 1.2× 且收陽 ③收盤站上 4H EMA20 ④回調區內有已守住的日線前期支撐。"
             "回調區須 4 項中至少 3 項（且必含轉強 K 線）才判「入場確認」，否則續等。")
    L.append("- 爆升前：以 20 日高（Donchian）判斷——收盤突破 20 日高且 RVOL≥1.5＝突破成立；"
             "貼近 20 日高、近 8 日區間 ≤2.5×ATR、且低點抬高或量縮＝蓄勢待突破（觸發價＝20 日高＋0.1×ATR）。")
    L.append("- 止損＝回調低點 −0.5×日線ATR；T1＝波段高點 B，T2/T3＝127.2%/161.8% 延伸。")

    # ── 分析任務（原 1–4 項原樣保留，新增 5–7 項）──────────
    L.append("")
    L.append("請你在此技術面基礎上，補充做以下深度分析：")
    L.append("1. 逐一檢查近期基本面／財報／新聞事件，是否與技術面訊號吻合或衝突")
    L.append("2. 若有多支同向訊號，比較哪一支的風險報酬比最佳，並說明原因")
    L.append("3. 指出技術面看似強烈、但基本面有潛在風險的標的（反之亦然）")
    L.append("4. 給出這批股票的優先順序建議（若只能選 1–2 支操作）")
    L.append("5. 驗證各標的的斐波那契關鍵位、VWAP、ATR 止損/目標是否合理，必要時給出你認為更好的進場、止損、目標價位")
    L.append("6. 檢查各時框 Histogram 是否有方向衝突或背離（例如大時框多、小時框空），並說明應以哪個時框為主")
    L.append("7. 針對下一個交易日，給出情境推演（上漲／盤整／下跌各自的觸發條件與關鍵價位）")
    L.append("")
    L.append("輸出要求：請用繁體中文；先給一張總結表"
             "（代碼｜方向｜進場價｜止損價｜目標價｜風險報酬比｜信心）再逐檔說明；"
             "請給出具體價位與明確方向，避免模稜兩可的描述。")

    # ── 排行結果 ───────────────────────────────────────────
    L.append("")
    L.append("## 排行結果（依警報等級與評分排序；無正式警報時退回「潛在傾向分」排序）")
    all_silent = all(r["pts"] == 0 for r in ranked)
    if all_silent:
        L.append("")
        L.append("⚠️ 注意：這批標的目前都沒有觸發正式警報（大時框較少翻轉是正常現象），"
                  "下表的「方向/評分」是尚未達門檻的潛在傾向分，僅供參考，"
                  "請勿當作已確認的技術訊號來解讀。")
    L.append("")
    L.append("| 排名 | 代碼 | 方向 | 等級 | 評分 | 多/空潛在分 | 收盤 | ATR% | 趨勢 | RVOL | VWAP位置 | 斐波那契 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        eff_dir = r["direction"] or leaning(r)
        sub     = max(r.get("bull_pts", 0), r.get("bear_pts", 0))
        dirn  = {"bear":"🔴空頭","bull":"🟢多頭",None:"⚪中性"}.get(eff_dir,"⚪中性")
        if not r["direction"] and eff_dir:
            dirn += "(潛在)"
        lvl   = {"red":"紅色警報","yellow":"橙色觀察",None:"靜默"}.get(r["level"],"靜默")
        score_disp = r["pts"] if r["pts"] else f"潛在{sub}"
        rvol  = f"{r['rvol']:.2f}×" if r.get("rvol") is not None else "—"
        if r.get("vwap") and r.get("below_vwap") is not None:
            vwapp = (f"{'下方' if r['below_vwap'] else '上方'} "
                     f"{abs(r.get('close_c', r['close'])/r['vwap']-1)*100:.2f}%（{r['vwap']:.2f}）")
        else:
            vwapp = "—"
        fibp = (f"{r['fib']['nearest_ratio']*100:.1f}%{r['fib']['role']}"
                f"({r['fib']['nearest_level']:.2f}，{r['fib']['dist_pct']:+.2f}%)"
                if r.get("fib") else "—")
        L.append(f"| {i} | {r['symbol']} | {dirn} | {lvl} | {score_disp} | "
                  f"{r.get('bull_pts',0)}/{r.get('bear_pts',0)} | "
                  f"{r['close']:.2f} | {r['atr_pct']:.2f}% | {r['trend']} | {rvol} | {vwapp} | {fibp} |")

    # ── 斐波那契買點掃描總表 ───────────────────────────────
    _sres = {r["symbol"]: r.get("setup") for r in ranked if r.get("setup")}
    if _sres:
        L.append("")
        L.append("## 斐波那契買點掃描（依狀態優先序；週線定方向｜日線找位置｜4H 確認）")
        L.append("")
        _rows = setup_table_rows(_sres)
        _cols = list(_rows[0].keys())
        L.append("| " + " | ".join(_cols) + " |")
        L.append("|" + "|".join(["---"] * len(_cols)) + "|")
        for _r in _rows:
            L.append("| " + " | ".join(str(_r[c]) for c in _cols) + " |")

    # ── 各標的技術細節 ─────────────────────────────────────
    L.append("")
    L.append("## 各標的技術細節")
    for r in ranked:
        trig = r.get("trig") or {}
        L.append(f"\n### {r['symbol']}")

        # 價格 / ATR
        pp = [f"主時框({main_tf})最新收盤 {r['close']:.2f}"]
        if r.get("close_c") is not None:
            pp.append(f"觸發時框已收盤價 {r['close_c']:.2f}")
        if trig.get("close") is not None:
            pp.append(f"觸發時框即時價 {trig['close']:.2f}")
        pp.append(f"主時框 ATR {r['atr']:.3f}（{r['atr_pct']:.2f}%）")
        if trig.get("atr"):
            pp.append(f"觸發時框 ATR {trig['atr']:.3f}")
        L.append("- 價格與波動：" + "｜".join(pp))
        L.append(f"- 多空潛在分：多 {r.get('bull_pts',0)}｜空 {r.get('bear_pts',0)}"
                 f"｜確認時框同向 {r['score']}/{r['max']}")

        # 舊版既有段落（原樣保留）
        if r["atype"]:
            L.append(f"- 判定：{r['atype']}")
        if r.get("detail"):
            L.append(f"- 評分依據：{'；'.join(r['detail'])}")
        if r.get("gap_pct") is not None and abs(r["gap_pct"]) >= 0.003:
            L.append(f"- 開盤缺口：{r['gap_pct']*100:+.2f}%")

        # 量能與位置
        vp = []
        if r.get("rvol") is not None:
            rv  = r["rvol"]
            tag = "機構級" if rv >= 2.5 else ("放量" if rv >= 1.5 else ("縮量" if rv < 0.8 else "正常"))
            vp.append(f"RVOL {rv:.2f}×（{tag}）")
        if r.get("vwap") and r.get("below_vwap") is not None:
            cc = r.get("close_c", r["close"])
            vp.append(f"{r.get('vwap_kind') or 'VWAP'} {r['vwap']:.2f}，已收盤價 {cc:.2f} 在"
                      f"{'下方' if r['below_vwap'] else '上方'} {abs(cc/r['vwap']-1)*100:.2f}%")
        if vp:
            L.append("- 量能與位置：" + "｜".join(vp))

        # 觸發時框明細
        if trig.get("valid"):
            L.append(f"- 觸發時框 {trigger_tf} Histogram：已收盤 {sg(trig.get('hist_c'))}"
                     f"（前根 {sg(trig.get('hist_c_prev'))}）｜形成中 {sg(trig.get('hist'))}"
                     f"｜D+1 {sg(trig.get('d1'))} / D+2 {sg(trig.get('d2'))} / D+3 {sg(trig.get('d3'))}")
            fl = []
            if trig.get("accel_dn"): fl.append("空頭加速（>1.5σ）")
            if trig.get("accel_up"): fl.append("多頭加速（>1.5σ）")
            if trig.get("sigma"):    fl.append(f"Hist 滾動 σ={trig['sigma']:.3f}")
            if fl:
                L.append("- 觸發時框動能：" + "｜".join(fl))

        # 斐波那契（摘要原樣保留 + 全部位）
        fb = r.get("fib")
        if fb:
            sw = "上升段回撤" if fb["swing"] == "up" else "下降段反彈"
            L.append(f"- 斐波那契（{sw}）：區間 {fb['low']:.2f}～{fb['high']:.2f}；"
                     f"38.2%={fb['levels'][0.382]:.2f}、50%={fb['levels'][0.5]:.2f}、"
                     f"61.8%={fb['levels'][0.618]:.2f}；延伸 127.2%={fb['levels'][1.272]:.2f}、"
                     f"161.8%={fb['levels'][1.618]:.2f}")
            L.append("- 斐波那契全部位：" + "、".join(
                f"{x*100:.1f}%={fb['levels'][x]:.2f}" for x in (FIB_RETR + FIB_EXT)))
            L.append(f"- 斐波那契最近位：{fb['nearest_ratio']*100:.1f}%＝{fb['nearest_level']:.2f}"
                     f"（{fb['role']}，價格偏離 {fb['dist_pct']:+.2f}%）")

        # 斐波那契買點掃描（週線/日線/4H）
        L.extend(setup_prompt_lines(r.get("setup")))

        # 各時框（舊版單行原樣保留 + 完整明細表）
        casc = r.get("cascade") or []
        if casc:
            tf_line = "、".join(
                f"{c['tf']}:{c['hist']:+.3f}(D+1 {c['d1']:+.3f})"
                for c in casc if c.get("hist") is not None
            )
            L.append(f"- 各時框 Histogram：{tf_line}")
            L.append("")
            L.append("| 時框 | 角色 | Hist(已收盤) | Hist(形成中) | D+1 | D+2 | D+3 | 狀態 | 趨勢 | ATR |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for c in casc:
                L.append(f"| {c['tf']} | {c.get('role') or '—'} | {sg(c.get('hist'))} | "
                         f"{sg(c.get('hist_live'))} | {sg(c.get('d1'))} | {sg(c.get('d2'))} | "
                         f"{sg(c.get('d3'))} | {c.get('status') or '—'} | {c.get('trend') or '—'} | "
                         f"{('%.3f' % c['atr']) if c.get('atr') else '—'} |")
            L.append("")

        # 交易建議（有警報時）
        if r["alert"] and trig.get("valid"):
            style = plan_style or "穩健"
            plan  = build_trade_plan(r["direction"], r["level"], r["pts"], trig, style)
            if plan:
                L.append(f"- 交易建議（{style}）：")
                for p in plan:
                    if p.strip().startswith("【"):
                        continue
                    L.append(f"  - {p.strip()}")

        # ATR 參考位（無論有無警報，給具體價位）
        ap, px = trig.get("atr"), trig.get("close")
        if ap and px:
            L.append(f"- 日內參考位（觸發時框 ATR {ap:.3f}，以即時價 {px:.2f} 為基準）："
                     f"多單 止損 {px-1.5*ap:.2f}｜目標 {px+2.0*ap:.2f} / {px+3.5*ap:.2f}；"
                     f"空單 止損 {px+1.5*ap:.2f}｜目標 {px-2.0*ap:.2f} / {px-3.5*ap:.2f}")
        am, cm = r.get("atr"), r.get("close")
        if am and cm:
            L.append(f"- 波段參考位（主時框 {main_tf} ATR {am:.3f}，以收盤 {cm:.2f} 為基準）："
                     f"多單 止損 {cm-1.5*am:.2f}｜目標 {cm+2.0*am:.2f} / {cm+3.5*am:.2f}；"
                     f"空單 止損 {cm+1.5*am:.2f}｜目標 {cm-2.0*am:.2f} / {cm-3.5*am:.2f}")

        # 主時框近 10 根 K 線 MACD
        mrows = r.get("macd_rows") or []
        if mrows:
            L.append(f"- 主時框 {main_tf} 近 {len(mrows)} 根 K 線 MACD：")
            L.append("")
            L.append("| 日期 | 收盤 | MACD | Hist | 狀態 |")
            L.append("|---|---|---|---|---|")
            for m in mrows:
                L.append(f"| {m['date']} | {m['close']} | {m['macd']} | {m['hist']} | {m['status']} |")
            L.append("")

    L.append("")
    L.append("---")
    L.append("以上數據為程式化技術指標計算結果，非投資建議；"
              "請結合你查得到的最新公開資訊進行判斷。最后將交易計劃表生成可以下載的JSON格式:啟用,代碼,方向(做多,做空),入場下限,入場上限,止損,止損上限(可空),目標1,目標2,目標3,R:R 風險回報比備註。")
    return "\n".join(L)


def build_trade_plan(direction, level, pts, trig, style):
    """
    依警報方向/等級/風格產生交易建議。
    激進：訊號 bar 即進場、1.0×ATR 緊止損、允許加倉、含持倉處置
    穩健：等下一根收盤確認、1.5×ATR 止損
    ATR 一律用「觸發時框」的 ATR（日內訊號配日內止損距離）。
    """
    if not trig or not trig.get("valid"):
        return []
    px = trig.get("close")
    a  = trig.get("atr", 0)
    if not px or not a or a <= 0:
        return []
    L = [f"\n【交易建議｜{style}】"]

    if style == "激進":
        if direction == "bear":
            if level == "red":
                L += [
                    f"  空單進場  市價 {px:.2f}（訊號 bar 即進場，不等確認）",
                    f"  止損      {px + 1.0*a:.2f}（+1.0×ATR 觸發時框）",
                    f"  目標1     {px - 1.5*a:.2f}（-1.5×ATR，到手止損移保本）",
                    f"  目標2     {px - 3.0*a:.2f}（-3.0×ATR，追蹤出場）",
                    f"  倉位      首倉全額；評分 ≥7 可於反抽 VWAP 加倉一次",
                    f"  持多倉    立即減半；跌破 VWAP 全部離場",
                ]
            else:
                L += [
                    f"  試探空單  ≤1/3 倉位 @ {px:.2f}",
                    f"  止損      {px + 0.75*a:.2f}（+0.75×ATR 收緊）",
                    f"  升級      評分升至 ≥5 補至全倉；退回 <3 分立即平倉",
                ]
        else:  # bull
            if level == "red":
                L += [
                    f"  多單進場  市價 {px:.2f}（訊號 bar 即進場，不等確認）",
                    f"  止損      {px - 1.0*a:.2f}（-1.0×ATR 觸發時框）",
                    f"  目標1     {px + 1.5*a:.2f}（+1.5×ATR，到手止損移保本）",
                    f"  目標2     {px + 3.0*a:.2f}（+3.0×ATR，追蹤出場）",
                    f"  倉位      首倉全額；評分 ≥7 可於回踩 VWAP 加倉一次",
                    f"  持空倉    立即減半；站回 VWAP 全部回補",
                ]
            else:
                L += [
                    f"  試探多單  ≤1/3 倉位 @ {px:.2f}",
                    f"  止損      {px - 0.75*a:.2f}（-0.75×ATR 收緊）",
                    f"  升級      評分升至 ≥5 補至全倉；退回 <3 分立即平倉",
                ]
    else:  # 穩健
        if level == "red":
            if direction == "bear":
                L += [
                    f"  空單進場  下一根 K 線收盤確認後 @ 市價",
                    f"  止損      {px + 1.5*a:.2f}（+1.5×ATR 觸發時框）",
                    f"  目標      {px - 2.0*a:.2f} / {px - 3.5*a:.2f}",
                    f"  持多倉    減倉並收緊止損至 {px + 1.0*a:.2f}",
                ]
            else:
                L += [
                    f"  多單進場  下一根 K 線收盤確認後 @ 市價",
                    f"  止損      {px - 1.5*a:.2f}（-1.5×ATR 觸發時框）",
                    f"  目標      {px + 2.0*a:.2f} / {px + 3.5*a:.2f}",
                ]
        else:
            L += ["  僅觀察，暫不建倉；等評分升級或翻轉確認"]

    L.append("  ⚠️ 程式化參考，非投資建議")
    return L


def build_tg_msg(symbol, cascade, resonance, confirm_tfs, trigger_tf, close_price, atr_1d, plan_style="穩健", setup=None):
    atype     = resonance.get("atype","")
    trig      = resonance.get("trigger") or {}
    direction = resonance.get("direction")
    level     = resonance.get("level")
    pts       = resonance.get("pts", 0)

    if resonance["alert"] and direction == "bear" and level == "red":
        header = f"🔴 {symbol} 風險警報（空頭 {pts} 分）"
    elif resonance["alert"] and direction == "bull" and level == "red":
        header = f"⚡ {symbol} 進場預警（多頭 {pts} 分）"
    elif resonance["alert"] and direction == "bear":
        header = f"🟠 {symbol} 觀察訊號（空頭 {pts} 分）"
    elif resonance["alert"]:
        header = f"👀 {symbol} 觀察訊號（多頭 {pts} 分）"
    elif resonance["score"] >= resonance["max"] * 0.7:
        header = f"📈 {symbol} 多頭共振"
    else:
        header = f"📊 {symbol} 市場分析"

    lines = [header, "━"*32, "【大時框方向確認】"]
    for r in cascade:
        if not r.get("valid") or r["tf"] not in confirm_tfs:
            continue
        aligned = (r["hist"] < 0) if direction == "bear" else (r["hist"] > 0)
        icon = "✅" if aligned else "❌"
        lines.append(f"  {r['tf']:>3s}  {icon}  Hist {fmt(r['hist'],3)}  D+1 {fmt(r['d1'],3)}")

    if trig and trig.get("valid"):
        h, d1 = trig["hist"], trig["d1"]
        hc    = trig.get("hist_c", h)
        if   hc < 0 and d1 > 0: warn = " ← 預計翻正 🚨"
        elif hc > 0 and d1 < 0: warn = " ← 預計翻負 ⚠️"
        else:                   warn = ""
        lines += [
            f"\n【觸發時框 {trigger_tf}】",
            f"  已收盤 Hist {fmt(hc,3)}  ← 訊號判定用",
            f"  形成中 Hist {fmt(h,3)}（未收盤，僅預覽）",
            f"  D+1        {fmt(d1,3)}{warn}",
            f"  D+2        {fmt(trig['d2'],3)}",
            f"  D+3        {fmt(trig['d3'],3)}",
        ]

        # ── v2.2 量能確認 ──────────────────────────────────
        vparts = []
        rv = trig.get("rvol")
        if rv is not None:
            tag = "機構級 🔥" if rv >= 2.5 else ("放量" if rv >= 1.5 else ("縮量" if rv < 0.8 else "正常"))
            vparts.append(f"RVOL  {rv:.2f}×（{tag}）")
        vw = trig.get("vwap")
        if vw:
            pos  = "下方" if trig.get("below_vwap") else "上方"
            dist = abs(trig.get("close_c", trig["close"])/vw - 1) * 100
            kind = trig.get("vwap_kind", "VWAP")
            vparts.append(f"{kind}  {vw:.2f}（價格在{pos} {dist:.2f}%）")
        gp = trig.get("gap_pct")
        if gp is not None and abs(gp) >= 0.003:
            vparts.append(f"缺口  {gp*100:+.2f}%")
        if vparts:
            lines += ["\n【量能確認】"] + [f"  {p}" for p in vparts]

        # ── v2.10 斐波那契位置 ───────────────────────────────
        fb = trig.get("fib")
        if fb:
            sw = "上升段回撤" if fb["swing"] == "up" else "下降段反彈"
            lines += [
                f"\n【斐波那契｜{sw}】",
                f"  區間  {fb['low']:.2f} ～ {fb['high']:.2f}",
                f"  最近位 {fb['nearest_ratio']*100:.1f}% = {fb['nearest_level']:.2f}"
                f"（{fb['role']}，價格偏離 {fb['dist_pct']:+.2f}%）",
            ]
            key = [(r, fb["levels"][r]) for r in (0.382, 0.5, 0.618)]
            lines.append("  關鍵位 " + "｜".join(f"{r*100:.1f}%:{v:.2f}" for r, v in key))
            ext = [(r, fb["levels"][r]) for r in (1.272, 1.618)]
            lines.append("  延伸目標 " + "｜".join(f"{r*100:.1f}%:{v:.2f}" for r, v in ext))

    # ── v2.2 警報評分明細 ─────────────────────────────────
    if resonance.get("detail"):
        badge = "🔴 警報" if level == "red" else "🟠 觀察"
        lines += [f"\n【警報評分 {pts} 分｜{badge}】"]
        lines += [f"  • {d}" for d in resonance["detail"]]

    # ── v2.3 交易建議 ─────────────────────────────────────
    if resonance["alert"]:
        lines += build_trade_plan(direction, level, pts, trig, plan_style)

    lines += [f"\n【共振強度】 {resonance['score']}/{resonance['max']}"]
    if atype:
        lines.append(f"【信號類型】 {atype}")

    if atr_1d > 0:
        if direction == "bear":
            stop = close_price + 1.5 * atr_1d
            tg1  = close_price - 2.0 * atr_1d
            tg2  = close_price - 3.5 * atr_1d
            lines += [
                f"\n【參考位置｜空頭】 收盤 {close_price:.2f}",
                f"  止損  {stop:.2f}  (+1.5×ATR)",
                f"  目標1 {tg1:.2f}  (-2.0×ATR)",
                f"  目標2 {tg2:.2f}  (-3.5×ATR)",
                f"  ATR   {atr_1d:.3f}",
            ]
        else:
            stop = close_price - 1.5 * atr_1d
            tg1  = close_price + 2.0 * atr_1d
            tg2  = close_price + 3.5 * atr_1d
            lines += [
                f"\n【參考位置】 收盤 {close_price:.2f}",
                f"  止損  {stop:.2f}  (-1.5×ATR)",
                f"  目標1 {tg1:.2f}  (+2.0×ATR)",
                f"  目標2 {tg2:.2f}  (+3.5×ATR)",
                f"  ATR   {atr_1d:.3f}",
            ]

    # ── v2.14 斐波那契買點掃描 ────────────────────────────
    lines += setup_tg_lines(setup)

    # ── v2.2：真正的美東時間 + 休市標註 ────────────────────
    ny = datetime.now(ZoneInfo("America/New_York"))
    stale = (trig.get("last_bar_day") and ny.strftime("%Y-%m-%d") != trig["last_bar_day"])
    if ny.weekday() >= 5 or stale:
        lines.insert(2, "⏸ 非交易時段／休市，以下為最近收盤數據")
    lines.append(f"\n📅 {ny.strftime('%Y-%m-%d %H:%M')} ET")
    return "\n".join(lines)

def send_telegram(token, chat_id, text):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"Markdown"},
            timeout=10,
        )
        return r.status_code==200, r.text
    except Exception as e:
        return False, str(e)



# ══════════════════════════════════════════════════════════════
# 回測引擎  ── 忠實還原三步入場邏輯
# ══════════════════════════════════════════════════════════════
#
#  入場三步驟：
#  Step 1  1d + 1h Histogram > 0          ← 大方向多頭確認
#  Step 2  傳導鏈預警：
#           1m D+1 > 0 → 5m D+1 > 0 → 15m D+1 > 0 → 30m D+1 > 0
#           （小時框 D+1 依序預測翻正 → 預警發出）
#  Step 3  30m Histogram 實際翻正          ← 執行入場
#
#  回測資料：最近 60 天日內 1m/5m/15m/30m/1h + 日線 1d
#  出場：持倉 N 根 30m K 線 或 ATR 止損/止盈
# ══════════════════════════════════════════════════════════════

BACKTEST_PERIODS = {
    "1年":  "1y",
    "2年":  "2y",
    "5年":  "5y",
    "10年": "10y",
}

# 回測固定用 60 天日內數據
BT_INTRADAY_CONFIGS = {
    "1m":  {"period": "7d",  "interval": "1m"},
    "5m":  {"period": "60d", "interval": "5m"},
    "15m": {"period": "60d", "interval": "15m"},
    "30m": {"period": "60d", "interval": "30m"},
    "1h":  {"period": "60d", "interval": "1h"},
    "1d":  {"period": "180d","interval": "1d"},
}


def fetch_bt_tf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """回測數據：session_state 快取，TTL = 300 秒（回測不需要即時數據）"""
    import time as _t
    cache_key = f"bt_{symbol}_{period}_{interval}"
    ts_key    = f"bt_ts_{symbol}_{period}_{interval}"
    now       = _t.time()
    if now - st.session_state.get(ts_key, 0) >= 300 or cache_key not in st.session_state:
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df.empty:
                df = pd.DataFrame()
            else:
                df = df[["Open","High","Low","Close","Volume"]].dropna()
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        except:
            df = pd.DataFrame()
        st.session_state[cache_key] = df
        st.session_state[ts_key]    = now
    return st.session_state[cache_key]


def get_d1_series(df: pd.DataFrame) -> pd.Series:
    """
    為 df 每一根 K 線計算當時的 D+1 預測值。
    使用 rolling 方式，每個時間點只用歷史數據（無未來偷看）。
    """
    _, _, hist = calc_macd(df["Close"])
    d1_vals = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(hist)):
        v     = hist.iloc[i-2:i+1].values          # 最近3根
        slope = (v[-1] - v[0]) / 2
        d1_vals.iloc[i] = v[-1] + slope
    return d1_vals, hist


def _latest_before(series: pd.Series, t) -> float:
    """取時間點 t 之前（含）最新的有效值，用於跨時框對齊"""
    sub = series[series.index <= t].dropna()
    return float(sub.iloc[-1]) if len(sub) > 0 else np.nan


def align_to_30m(df_30m: pd.DataFrame,
                 df_5m:  pd.DataFrame,
                 df_15m: pd.DataFrame,
                 df_1h:  pd.DataFrame,
                 df_1d:  pd.DataFrame) -> pd.DataFrame:
    """
    以 30m K 線為基準時間軸，對齊各時框最新狀態。
    每根 30m K 線結束時，查詢當時各時框的最新值（無未來偷看）。
    """
    # 計算各時框的 D+1 預測序列 和 Histogram 序列
    # get_d1_series 回傳 (d1_series, hist_series)
    d1_5m,  hist_5m  = get_d1_series(df_5m)
    d1_15m, hist_15m = get_d1_series(df_15m)
    d1_30m, hist_30m = get_d1_series(df_30m)
    d1_1h,  hist_1h  = get_d1_series(df_1h)
    # 1d 只需要 Histogram（不需要 D+1 預測），calc_macd 回傳3個值
    _, _, hist_1d = calc_macd(df_1d["Close"])

    rows = []
    for i in range(10, len(df_30m)):
        t30 = df_30m.index[i]   # 當前 30m K 線結束時間

        # 30m 本身（直接用 index，不需要 latest_before）
        h30   = float(hist_30m.iloc[i])
        d1_30 = float(d1_30m.iloc[i]) if not pd.isna(d1_30m.iloc[i]) else np.nan

        # 5m / 15m / 1h：t30 之前最新值
        h5    = _latest_before(hist_5m,  t30)
        d5    = _latest_before(d1_5m,    t30)
        h15   = _latest_before(hist_15m, t30)
        d15   = _latest_before(d1_15m,   t30)
        h1h   = _latest_before(hist_1h,  t30)
        d1h   = _latest_before(d1_1h,    t30)

        # 1d：取 t30 日期當天或之前最新日線
        t30_date = pd.Timestamp(t30).date()
        sub_1d   = hist_1d[pd.to_datetime(hist_1d.index).date <= t30_date]
        h1d      = float(sub_1d.iloc[-1]) if len(sub_1d) > 0 else np.nan

        rows.append({
            "time":  t30,
            "close": df_30m["Close"].iloc[i],
            # 30m
            "h30":   h30,
            "d1_30": d1_30,
            # 5m
            "h5":    h5,
            "d1_5":  d5,
            # 15m
            "h15":   h15,
            "d1_15": d15,
            # 1h 確認
            "h1h":   h1h,
            "d1_1h": d1h,
            # 1d 確認
            "h1d":   h1d,
        })

    return pd.DataFrame(rows).set_index("time")


def run_cascade_backtest(symbol: str, atr_sl: float = 1.5,
                          atr_tp: float = 3.0, hold_bars: int = 6,
                          exit_mode: str = "固定根數") -> dict:
    """
    三步瀑布傳導回測（最近 60 天 30m K 線）

    正確的三步時序邏輯：
    ┌─ 每根 K 線持續檢查 ──────────────────────────────────┐
    │ Step 1  h1d > 0 AND h1h > 0   → 大方向多頭確認      │
    │ Step 2  預警根：h30 < 0                              │
    │         AND d1_5>0, d1_15>0, d1_30>0               │
    │         （傳導鏈 D+1 全部預測翻正，但30m尚未翻）     │
    │ Step 3  入場根（預警根的下一根）：                   │
    │         h30_prev < 0 AND h30_now >= 0               │
    │         （30m 實際翻正 → 當根收盤入場）              │
    └──────────────────────────────────────────────────────┘
    出場模式（exit_mode）：
      "固定根數"    → 持倉 hold_bars 根 30m K 線後平倉
      "30m轉負"     → 30m Histogram 再次轉負時出場
      "1d/1h轉負"   → 1d 或 1h Histogram 轉負時出場（跟隨大趨勢）
    所有模式均疊加 ATR 止損 / ATR 止盈保護
    """
    # ── 載入所有時框 ─────────────────────────────────────
    dfs = {}
    for tf, cfg in BT_INTRADAY_CONFIGS.items():
        df = fetch_bt_tf(symbol, cfg["period"], cfg["interval"])
        if df.empty:
            return {"error": f"{tf} 數據載入失敗"}
        dfs[tf] = df

    # ── 時框對齊 ─────────────────────────────────────────
    try:
        aligned = align_to_30m(
            dfs["30m"], dfs["5m"], dfs["15m"], dfs["1h"], dfs["1d"]
        )
    except Exception as e:
        return {"error": f"時框對齊失敗: {e}"}

    if len(aligned) < 20:
        return {"error": "有效數據不足20根30m K線"}

    atr_30m = calc_atr(dfs["30m"])

    # ── 診斷計數器（找出哪一步卡住）─────────────────────
    diag = {
        "total_bars":    len(aligned),
        "step1_pass":    0,   # 1d+1h 都多頭的 bar 數
        "step2_alert":   0,   # Step2 預警觸發次數
        "step3_entry":   0,   # Step3 實際入場次數
        "skip_nan":      0,
    }

    # ── 主回測循環 ────────────────────────────────────────
    trades        = []
    in_trade      = False
    entry_price   = entry_bar = None
    prev_alerted  = False   # 上一根是否已發出 Step2 預警

    close_arr = aligned["close"].values
    h30_arr   = aligned["h30"].values
    d1_30_arr = aligned["d1_30"].values
    d1_5_arr  = aligned["d1_5"].values
    d1_15_arr = aligned["d1_15"].values
    h1h_arr   = aligned["h1h"].values
    h1d_arr   = aligned["h1d"].values
    times     = aligned.index

    for i in range(5, len(aligned)):
        # NaN 跳過
        if np.isnan(h30_arr[i]) or np.isnan(close_arr[i]):
            diag["skip_nan"] += 1
            prev_alerted = False
            continue

        if not in_trade:
            # ── Step 1：大方向確認 ─────────────────────────
            s1_1d = (not np.isnan(h1d_arr[i]) and h1d_arr[i] > 0)
            s1_1h = (not np.isnan(h1h_arr[i]) and h1h_arr[i] > 0)
            step1 = s1_1d and s1_1h
            if step1:
                diag["step1_pass"] += 1

            if not step1:
                prev_alerted = False
                continue

            # ── Step 3 優先：上一根已預警，本根是否翻正？──
            if prev_alerted:
                h30_prev = h30_arr[i-1]
                step3 = (h30_prev < 0 and h30_arr[i] >= 0)
                if step3:
                    in_trade    = True
                    entry_bar   = i
                    entry_price = close_arr[i]
                    diag["step3_entry"] += 1
                    prev_alerted = False
                    continue
                # 若上一根預警後本根仍負，繼續等（不重置 prev_alerted，
                # 但要重新判斷 Step2 是否仍然成立）

            # ── Step 2：傳導鏈 D+1 預測翻正（預警根）─────
            s2_5m  = (not np.isnan(d1_5_arr[i])  and d1_5_arr[i]  > 0)
            s2_15m = (not np.isnan(d1_15_arr[i]) and d1_15_arr[i] > 0)
            s2_30m = (not np.isnan(d1_30_arr[i]) and d1_30_arr[i] > 0)
            s2_neg = (h30_arr[i] < 0)   # 30m 當前仍為負（尚未翻正）

            step2 = s2_5m and s2_15m and s2_30m and s2_neg
            if step2:
                prev_alerted = True
                diag["step2_alert"] += 1
            else:
                # Step2 不滿足，重置預警狀態
                prev_alerted = False

        else:
            # ── 出場檢測 ──────────────────────────────────
            prev_alerted = False
            bars_held = i - entry_bar
            cur  = close_arr[i]
            sl   = entry_price - atr_sl * atr_30m
            tp   = entry_price + atr_tp * atr_30m

            reason = exit_price = None

            # ATR 止損（所有模式共用）
            if cur <= sl:
                reason, exit_price = "止損", max(sl, cur)
            # ATR 止盈（所有模式共用）
            elif cur >= tp:
                reason, exit_price = "止盈", min(tp, cur)
            # ── 出場模式 ──────────────────────────────────
            elif exit_mode == "固定根數":
                if bars_held >= hold_bars:
                    reason, exit_price = "固定根數平倉", cur

            elif exit_mode == "30m轉負":
                # 30m Histogram 從正轉負時出場
                h30_cur  = h30_arr[i]
                h30_prev_e = h30_arr[i-1] if i > 0 else h30_cur
                if h30_prev_e >= 0 and h30_cur < 0:
                    reason, exit_price = "30m轉負出場", cur
                # 保底：超過 hold_bars*3 根未出場則強制平倉
                elif bars_held >= hold_bars * 3:
                    reason, exit_price = "保底平倉", cur

            elif exit_mode == "1d/1h轉負":
                # 1d 或 1h Histogram 轉負時出場
                h1h_cur = h1h_arr[i]
                h1h_prv = h1h_arr[i-1] if i > 0 else h1h_cur
                h1d_cur = h1d_arr[i]
                h1d_prv = h1d_arr[i-1] if i > 0 else h1d_cur
                h1h_neg = (not np.isnan(h1h_cur) and not np.isnan(h1h_prv)
                           and h1h_prv >= 0 and h1h_cur < 0)
                h1d_neg = (not np.isnan(h1d_cur) and not np.isnan(h1d_prv)
                           and h1d_prv >= 0 and h1d_cur < 0)
                if h1h_neg:
                    reason, exit_price = "1h轉負出場", cur
                elif h1d_neg:
                    reason, exit_price = "1d轉負出場", cur
                # 保底：超過 hold_bars*5 根未出場則強制平倉
                elif bars_held >= hold_bars * 5:
                    reason, exit_price = "保底平倉", cur

            if reason:
                pnl = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "entry_time":  times[entry_bar],
                    "exit_time":   times[i],
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "bars_held":   bars_held,
                    "pnl_pct":     pnl,
                    "win":         pnl > 0,
                    "exit_reason": reason,
                    "date":        pd.Timestamp(times[entry_bar]).date(),
                })
                in_trade    = False
                entry_price = entry_bar = None

    if not trades:
        return {
            "trades": [], "total": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "max_consec_loss": 0,
            "total_return": 0, "aligned": aligned, "diag": diag,
        }

    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t["win"]]
    losses = df_t[~df_t["win"]]

    # 最大連虧
    mc = cur_mc = 0
    for w in df_t["win"]:
        cur_mc = 0 if w else cur_mc + 1
        mc     = max(mc, cur_mc)

    aw = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0
    al = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    pf = abs(aw * len(wins)) / abs(al * len(losses)) if len(losses) > 0 and al != 0 else 99

    return {
        "trades":          trades,
        "df_trades":       df_t,
        "total":           len(df_t),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        len(wins) / len(df_t) * 100,
        "avg_win":         aw,
        "avg_loss":        al,
        "profit_factor":   min(pf, 99),
        "max_consec_loss": mc,
        "total_return":    df_t["pnl_pct"].sum(),
        "aligned":         aligned,
        "diag":            diag,
    }


# ── 回測圖表 ──────────────────────────────────────────────────

def build_bt_entry_chart(aligned: pd.DataFrame, trades: list) -> go.Figure:
    """
    在 30m Histogram 圖上標注入場/出場點
    """
    _, _, hist_vals = calc_macd(pd.Series(aligned["close"].values,
                                           index=aligned.index))
    xl     = [str(t) for t in aligned.index]
    colors = ["#3d8b5e" if v >= 0 else "#c0392b" for v in aligned["h30"]]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06,
                        subplot_titles=["30m 收盤價  ▲入場  ▼出場", "30m Histogram + D+1 預測"])

    # 收盤線
    fig.add_trace(go.Scatter(
        x=xl, y=aligned["close"].values, mode="lines",
        name="收盤", line=dict(color="#5a7fa8", width=1.5),
    ), row=1, col=1)

    # 入場/出場標記
    if trades:
        entry_x = [str(t["entry_time"]) for t in trades]
        entry_y = [t["entry_price"]     for t in trades]
        exit_x  = [str(t["exit_time"])  for t in trades]
        exit_y  = [t["exit_price"]      for t in trades]
        win_clr = ["#3d8b5e" if t["win"] else "#c0392b" for t in trades]

        fig.add_trace(go.Scatter(
            x=entry_x, y=entry_y, mode="markers",
            name="入場", marker=dict(symbol="triangle-up", size=12,
                                     color=win_clr, line=dict(width=1, color="#fff")),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=exit_x, y=exit_y, mode="markers",
            name="出場", marker=dict(symbol="triangle-down", size=12,
                                     color=win_clr, line=dict(width=1, color="#fff")),
        ), row=1, col=1)

    # Histogram 柱
    fig.add_trace(go.Bar(
        x=xl, y=aligned["h30"].values, name="30m Hist",
        marker_color=colors, opacity=0.8,
    ), row=2, col=1)

    # D+1 預測線
    fig.add_trace(go.Scatter(
        x=xl, y=aligned["d1_30"].values, mode="lines",
        name="D+1 預測", line=dict(color="#e07b39", width=1.5, dash="dot"),
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#bbb", row=2, col=1)

    # x 軸抽稀
    step = max(1, len(xl)//14)
    tv   = xl[::step]

    fig.update_layout(
        paper_bgcolor="#fff8f0", plot_bgcolor="#fff8f0",
        font=dict(family="IBM Plex Mono", color="#2c2c2c", size=10),
        margin=dict(l=10, r=10, t=36, b=10),
        height=500,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis=dict(type="category", tickvals=tv, tickangle=-35, gridcolor="#e8e3da"),
        xaxis2=dict(type="category", tickvals=tv, tickangle=-35, gridcolor="#e8e3da"),
    )
    fig.update_yaxes(gridcolor="#e8e3da", zeroline=True, zerolinecolor="#c0bbb2")
    return fig


def build_equity_curve(trades: list) -> go.Figure:
    if not trades:
        return go.Figure()
    df_t    = pd.DataFrame(trades)
    cum     = df_t["pnl_pct"].cumsum().values
    colors  = ["#3d8b5e" if p > 0 else "#c0392b" for p in df_t["pnl_pct"]]
    xlabels = [str(t["entry_time"])[:16] for t in trades]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06,
                        subplot_titles=["累計收益（%）", "每筆盈虧（%）"])
    fig.add_trace(go.Scatter(
        x=xlabels, y=cum, mode="lines+markers",
        name="累計收益", line=dict(color="#5a7fa8", width=2),
        fill="tozeroy", fillcolor="rgba(90,127,168,0.08)",
        marker=dict(size=5, color=colors),
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=1, col=1)
    fig.add_trace(go.Bar(
        x=xlabels, y=df_t["pnl_pct"].values,
        name="單筆", marker_color=colors, opacity=0.85,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=2, col=1)
    step = max(1, len(xlabels)//10)
    tv   = xlabels[::step]
    fig.update_layout(
        paper_bgcolor="#fff8f0", plot_bgcolor="#fff8f0",
        font=dict(family="IBM Plex Mono", color="#2c2c2c", size=10),
        margin=dict(l=10, r=10, t=36, b=10), height=380, showlegend=False,
        xaxis=dict(type="category", tickvals=tv, tickangle=-35, gridcolor="#e8e3da"),
        xaxis2=dict(type="category", tickvals=tv, tickangle=-35, gridcolor="#e8e3da"),
    )
    fig.update_yaxes(gridcolor="#e8e3da", zeroline=True, zerolinecolor="#c0bbb2")
    return fig


def render_bt_kpi(result: dict, symbol: str, hold_bars: int,
                   atr_sl: float, atr_tp: float) -> str:
    wr  = result["win_rate"]
    pf  = result["profit_factor"]
    tr  = result["total_return"]
    wc  = "#3d8b5e" if wr >= 55 else ("#e07b39" if wr >= 45 else "#c0392b")
    pc_ = "#3d8b5e" if pf >= 1.5 else ("#e07b39" if pf >= 1.0 else "#c0392b")
    tc  = "#3d8b5e" if tr >= 0   else "#c0392b"
    return f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0;">
        <div class="metric-card">
            <div class="label">總信號</div>
            <div class="value" style="font-size:30px;">{result['total']}</div>
            <div class="sub">最近60天 30m</div>
        </div>
        <div class="metric-card">
            <div class="label">整體勝率</div>
            <div class="value" style="color:{wc};font-size:30px;">{wr:.1f}%</div>
            <div class="sub">{result['wins']}勝 {result['losses']}負</div>
        </div>
        <div class="metric-card">
            <div class="label">盈虧比 PF</div>
            <div class="value" style="color:{pc_};font-size:30px;">{pf:.2f}</div>
            <div class="sub">勝 {result['avg_win']:.2f}% / 負 {result['avg_loss']:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">最大連虧</div>
            <div class="value" style="color:#c0392b;font-size:30px;">{result['max_consec_loss']}</div>
            <div class="sub">連續次數</div>
        </div>
        <div class="metric-card">
            <div class="label">累計收益</div>
            <div class="value" style="color:{tc};font-size:30px;">{tr:+.1f}%</div>
            <div class="sub">持{hold_bars}根·SL {atr_sl}×·TP {atr_tp}×ATR</div>
        </div>
    </div>"""


def render_trades_table(trades: list, max_rows: int = 25) -> str:
    cols = ["入場時間","出場時間","入場價","出場價","持倉根","盈虧%","結果","原因"]
    hdr  = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    reason_cls = {"止損":"badge-bear","止盈":"badge-bull","到期平倉":"badge-neu"}
    for t in trades[-max_rows:]:
        pnl = t["pnl_pct"]
        pc_ = "cell-pos" if pnl >= 0 else "cell-neg"
        rslt= '<span class="badge badge-bull">▲ 盈</span>' if t["win"]               else '<span class="badge badge-bear">▼ 虧</span>'
        rc  = reason_cls.get(t["exit_reason"], "badge-neu")
        rb  = f'<span class="badge {rc}">{t["exit_reason"]}</span>'
        et  = str(t["entry_time"])[:16]
        xt  = str(t["exit_time"])[:16]
        body += f"""<tr>
            <td>{et}</td><td>{xt}</td>
            <td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td>
            <td>{t['bars_held']}</td>
            <td class="{pc_}">{pnl:+.2f}%</td>
            <td>{rslt}</td><td>{rb}</td>
        </tr>"""
    return f'<table class="macd-table"><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table>'


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 系統設定")
    st.markdown("---")

    raw     = st.text_area("股票代碼（逗號分隔）", value=",".join(DEFAULT_SYMBOLS), height=75)
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    st.markdown("**主時框（K線分析）**")
    main_tf = st.selectbox("時間框架", list(TIMEFRAME_MAP.keys()), index=5)

    st.markdown("**🌊 瀑布傳導鏈**")
    chain_tfs = st.multiselect(
        "選擇時框（系統自動由小到大排序）",
        CASCADE_CHAIN,
        default=["30m","1h","1d","1w"],
    )
    chain_tfs = [tf for tf in CASCADE_CHAIN if tf in chain_tfs]

    if len(chain_tfs) >= 2:
        trigger_tf  = st.selectbox("⚡ 入場觸發時框（最小）", chain_tfs, index=0)
        confirm_tfs = [tf for tf in chain_tfs if tf != trigger_tf]
    else:
        trigger_tf  = chain_tfs[0] if chain_tfs else "30m"
        confirm_tfs = []

    st.markdown("**🎯 交易建議風格**")
    plan_style = st.radio(
        "風格", ["穩健", "激進"], horizontal=True, index=0,
        label_visibility="collapsed",
        help="穩健：等下一根收盤確認、1.5×ATR 止損｜"
             "激進：訊號 bar 即進場、1.0×ATR 緊止損、允許加倉、含持倉處置",
    )

    st.markdown("**📊 圖表疊加**")
    show_fib  = st.checkbox("斐波那契線",         value=True, key="ov_fib")
    show_sr   = st.checkbox("支撐／壓力線",        value=True, key="ov_sr")
    show_vwap = st.checkbox("主力成本線（VWAP）",  value=True, key="ov_vwap")
    show_vol  = st.checkbox("成交量副圖",          value=True, key="ov_vol")

    st.markdown("**🎯 斐波那契買點掃描**")
    zz_rev = st.slider("ZigZag 反轉門檻（×日線ATR）", 1.5, 4.0, 2.5, 0.5, key="zz_rev",
                       help="價格自極值反向走超過 N×ATR 才確認一個擺動點；越大越只看大波段。")
    zz_leg = st.slider("最小主要波段（×日線ATR）", 2.0, 8.0, 4.0, 0.5, key="zz_leg",
                       help="A→B 的漲幅須 ≥ N×日線 ATR 才算主要波段，過濾掉小波動。")

    st.markdown("---")
    st.markdown("**⏱ 自動刷新**")
    auto_refresh     = st.checkbox("啟用", value=True, key="auto_refresh_enabled",
                                    help="Streamlit 只有在互動或此開關開啟時才會重新執行整頁；"
                                         "關閉此開關且分頁長時間沒有點擊，畫面會停在上次載入時的舊資料。")
    refresh_interval = st.selectbox("間隔（秒）", [60,120,180,300], index=0)
    # 存入 session_state 供 fetch_data 動態快取使用
    st.session_state["refresh_interval"] = refresh_interval

    st.markdown("---")
    st.markdown("**📡 Telegram**")
    tg_token = st.text_input("Bot Token", type="password", placeholder="xxxxx:ABC...")
    tg_chat  = st.text_input("Chat ID",  placeholder="-100xxxxxxxxx")
    tg_auto  = st.checkbox("🔔 警報自動推送", value=False,
                            help="偵測到預警/警報時自動推送 Telegram（同一訊號每小時最多一次），"
                                 "需配合「自動刷新」使用")
    tg_send  = st.button("📤 發送所有信號")

    st.markdown("---")
    if st.button("🔄 立即清除快取", help="強制清除所有數據快取，下次載入重新拉取"):
        # 清除所有數據快取（含刷新計時器，確保立即重新拉取）
        keys_to_del = [k for k in st.session_state.keys()
                       if k.startswith(("df_","ts_","bt_","bt_ts_","_last_page"))]
        for k in keys_to_del:
            del st.session_state[k]
        st.success(f"✅ 已清除 {len(keys_to_del)} 個快取，正在重新載入...")
        st.rerun()
    st.caption(f"更新：{datetime.now().strftime('%H:%M:%S')}")

if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    # st_autorefresh 用 React component 定時觸發 st.rerun()
    # 每次 rerun 前自動清除數據快取，確保數據一定更新
    _count = st_autorefresh(
        interval=refresh_interval * 1000,  # 毫秒
        limit=None,                         # 無限次
        key="autorefresh_timer",
    )
    if _count > 0:
        # 每次 autorefresh 觸發時清除數據快取
        _keys_expired = [k for k in st.session_state.keys()
                         if k.startswith(("df_","ts_"))]
        for _k in _keys_expired:
            del st.session_state[_k]
    st.info(f"⏱ 每 {refresh_interval} 秒自動更新 · 已刷新 {_count} 次")


# ══════════════════════════════════════════════════════════════
# 主頁面
# ══════════════════════════════════════════════════════════════
st.markdown("# 🌊 MACD 瀑布動能傳導系統")
import time as _time_main

# ── v2.8：資料過期警示 ──────────────────────────────────────────
# Streamlit 只在使用者互動或 st_autorefresh 觸發時才會重新執行整頁；
# 若「自動刷新」關閉且分頁長時間沒有互動，fetch_data() 的 TTL 檢查
# 根本沒機會被執行，畫面會停在「上次腳本真的跑過」那一刻的舊快照
# （例如週一開著分頁沒動，週三看到的還是週一的收盤）。
_last_fetch_ts = st.session_state.get("_last_any_fetch_ts")
if _last_fetch_ts:
    _stale_h = (_time_main.time() - _last_fetch_ts) / 3600
    if (not auto_refresh) and _stale_h >= 2:
        st.warning(
            f"⚠️ 偵測到此分頁已約 {_stale_h:.1f} 小時沒有重新整理過(「自動刷新」目前關閉,"
            f"Streamlit 只在互動或自動刷新觸發時才會重跑)。以下可能是舊快照,"
            f"建議開啟自動刷新,或點側邊欄「🔄 立即清除快取」。"
        )

_ttl_now = st.session_state.get("refresh_interval", 60)
_slot_ts = (int(_time_main.time()) // _ttl_now) * _ttl_now
_data_ts = datetime.fromtimestamp(_slot_ts).strftime("%H:%M:%S")
chain_str   = "  →  ".join(chain_tfs) if chain_tfs else "未設定"
confirm_str = ", ".join(confirm_tfs)  if confirm_tfs else "—"
st.markdown(f"""
<div style="font-size:12px;color:#888;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">
傳導鏈：{chain_str} &nbsp;|&nbsp; 觸發：<b style="color:#f0a500">{trigger_tf}</b>
&nbsp;|&nbsp; 確認：{confirm_str}
&nbsp;|&nbsp; 📡 數據批次：<b style="color:#3d8b5e">{_data_ts}</b>
&nbsp;{"&nbsp;|&nbsp; ⏱ 每 " + str(_ttl_now) + " 秒更新" if auto_refresh else ""}
</div>
""", unsafe_allow_html=True)
st.markdown("---")

tf_cfg      = TIMEFRAME_MAP[main_tf]
tg_msgs_all = []
rank_data   = []
setup_results = {}

for symbol in symbols:
    st.markdown(f"## 🔷 {symbol}")

    with st.spinner(f"載入 {symbol}..."):
        df_main = fetch_data(symbol, tf_cfg["period"], tf_cfg["interval"])
        df_main = drop_forming_bar(df_main, tf_cfg["interval"])

    if df_main.empty or len(df_main) < 30:
        st.warning(f"⚠️ {symbol} 數據不足，跳過")
        st.markdown("---")
        continue

    df_t, macd_s, sig_s, hist_s = build_macd_table(df_main, n=10)
    last      = df_t.iloc[-1]
    macd_val  = float(last["MACD"].replace("+",""))
    hist_val  = last["_h"]
    status_v  = last["_st"]
    close_val = float(last["收盤"])
    trend     = get_trend(macd_val, hist_val, status_v)
    d1,d2,d3  = predict_next3(hist_s)
    atr_main  = calc_atr(df_main)

    # ── 頂部四格指標 ────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    prev  = df_main["Close"].iloc[-2]
    chg   = close_val - prev
    pct   = chg / prev * 100
    with c1:
        dc,ds = ("pos","▲") if chg>=0 else ("neg","▼")
        st.markdown(f"""<div class="metric-card">
            <div class="label">收盤 ({main_tf})</div>
            <div class="value">{close_val:.2f}</div>
            <div class="sub {dc}">{ds} {chg:+.2f} ({pct:+.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        hc = "pos" if hist_val>=0 else "neg"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Histogram</div>
            <div class="value {hc}">{hist_val:+.3f}</div>
            <div class="sub">{status_v}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        d1c = "pos" if d1>=0 else "neg"
        st.markdown(f"""<div class="metric-card">
            <div class="label">三日推演</div>
            <div class="value {d1c}">{d1:+.3f}</div>
            <div class="sub">D+2 {d2:+.3f} &nbsp;|&nbsp; D+3 {d3:+.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">趨勢 / ATR</div>
            <div style="margin:8px 0 4px">{trend_pill(trend)}</div>
            <div class="sub">ATR {atr_main:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── v2.10 斐波那契面板（主時框，已收盤 K 線）────────────────
    fib_main = calc_fib_levels(df_main.iloc[:-1] if (tf_cfg["interval"] in INTRADAY_IVS and len(df_main) > 1) else df_main)
    if fib_main:
        sw_txt = "上升段回撤（找支撐）" if fib_main["swing"] == "up" else "下降段反彈（找壓力）"
        rows_html = ""
        for r_ in FIB_RETR + FIB_EXT:
            v_   = fib_main["levels"][r_]
            here = (r_ == fib_main["nearest_ratio"])
            mark = " ◀ 最近" if here else ""
            wt   = "700" if here else "400"
            rows_html += (f"<span style='display:inline-block;min-width:150px;"
                          f"font-weight:{wt}'>{r_*100:.1f}% &nbsp;{v_:.2f}{mark}</span>")
        st.markdown(f"""<div class="metric-card" style="margin-top:8px">
            <div class="label">📐 斐波那契（{main_tf}｜{sw_txt}｜{fib_main['low']:.2f} ～ {fib_main['high']:.2f}）</div>
            <div class="sub" style="line-height:2;font-family:'IBM Plex Mono',monospace">
                最近位 <b>{fib_main['nearest_ratio']*100:.1f}%</b> = {fib_main['nearest_level']:.2f}
                （{fib_main['role']}，價格偏離 {fib_main['dist_pct']:+.2f}%）<br>{rows_html}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── v2.14 斐波那契買點掃描：週線定方向 → 日線找位置 → 4H 確認 ──
    try:
        with st.spinner(f"掃描 {symbol} 買點（週線/日線/4H）..."):
            setup = scan_fib_setup(symbol, reversal_atr=zz_rev, min_leg_atr=zz_leg)
    except Exception as _e:
        setup = {"status": "— 掃描失敗", "prio": 11, "grade": None, "weekly": None, "A": None,
                 "B": None, "checks": None, "plan": None, "breakout": None,
                 "notes": [f"{type(_e).__name__}: {_e}"]}
    setup_results[symbol] = setup
    st.markdown(setup_card_html(symbol, setup), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 瀑布傳導核心面板 ────────────────────────────────────
    if chain_tfs:
        with st.spinner("計算傳導鏈..."):
            cascade   = analyze_cascade(symbol, chain_tfs)
            resonance = calc_resonance(cascade, confirm_tfs, trigger_tf)

        # 標題 + 共振評分
        col_t, col_s = st.columns([3,2])
        with col_t:
            st.markdown("#### 🌊 瀑布動能傳導鏈")
        with col_s:
            st.markdown(
                f"<div style='text-align:right;margin-top:10px'>{stars_html(resonance['score'],resonance['max'])}</div>",
                unsafe_allow_html=True)

        # ── 預警橫幅（v2.2 計分制：紅色警報卡 / 橙色觀察條）──
        if resonance["alert"]:
            trig_d  = resonance.get("trigger") or {}
            atype   = resonance.get("atype","")
            lvl     = resonance.get("level")
            pts_v   = resonance.get("pts",0)
            det_ls  = resonance.get("detail",[])
            det_html = "".join(f"&nbsp;&nbsp;• {d}<br>" for d in det_ls)
            plan_ls  = build_trade_plan(resonance.get("direction"), lvl, pts_v,
                                        trig_d, plan_style)
            plan_html = "".join(
                f"&nbsp;&nbsp;{p.strip()}<br>" for p in plan_ls
                if not p.startswith("\n【"))
            hc_v  = trig_d.get("hist_c",0)
            hcp_v = trig_d.get("hist_c_prev",0)
            d1_v  = trig_d.get("d1",0)
            rv_v  = trig_d.get("rvol")
            rv_txt = f"｜RVOL {rv_v:.2f}×" if rv_v is not None else ""

            if resonance.get("direction") == "bear" and lvl == "red":
                st.markdown(f"""<div class="alert-card-bear">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;
                                font-weight:700;margin-bottom:10px;color:#9b2335;">
                        🔴 風險警報 {pts_v} 分｜{symbol} [{trigger_tf}]{rv_txt}
                    </div>
                    <div class="alert-body">
                    {atype}<br>
                    {trigger_tf} Histogram（已收盤）前根 <b>{hcp_v:+.3f}</b> → 現在
                    <b style="color:#c0392b">{hc_v:+.3f}</b>｜D+1 預測
                    <b style="color:#c0392b">{d1_v:+.3f}</b><br>
                    <br><b>評分明細：</b><br>{det_html}
                    <br><b>交易建議（{plan_style}）：</b><br>{plan_html}
                    </div>
                </div>""", unsafe_allow_html=True)
            elif resonance.get("direction") == "bull" and lvl == "red":
                st.markdown(f"""<div class="alert-card">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;
                                font-weight:700;margin-bottom:10px;">
                        ⚡ 進場預警 {pts_v} 分｜{symbol} [{trigger_tf}]{rv_txt}
                    </div>
                    <div class="alert-body">
                    {atype}<br>
                    {trigger_tf} Histogram（已收盤）現在 <b>{hc_v:+.3f}</b>｜D+1 預測
                    <b style="color:#3d8b5e">{d1_v:+.3f}</b><br>
                    <br><b>評分明細：</b><br>{det_html}
                    <br><b>交易建議（{plan_style}）：</b><br>{plan_html}
                    </div>
                </div>""", unsafe_allow_html=True)
            elif resonance.get("direction") == "bear":
                plan_txt = "  \n".join(p.strip() for p in plan_ls if not p.startswith("\n【"))
                st.warning(f"🟠 {atype}｜" + "、".join(det_ls) + "  \n" + plan_txt)
            else:
                plan_txt = "  \n".join(p.strip() for p in plan_ls if not p.startswith("\n【"))
                st.info(f"👀 {atype}｜" + "、".join(det_ls) + "  \n" + plan_txt)

        # ── 傳導鏈總覽圖 ────────────────────────────────────
        st.plotly_chart(build_cascade_chart(cascade), use_container_width=True, key=f"cascade_{symbol}_{id(cascade)}")

        # ── 傳導鏈各時框卡片 ────────────────────────────────
        valid_c = [r for r in cascade if r.get("valid")]
        if valid_c:
            cols_c = st.columns(len(valid_c))
            for idx, r in enumerate(valid_c):
                hc  = "pos" if r["hist"]>=0 else "neg"
                is_trig = r["tf"] == trigger_tf
                border  = "border:2px solid #f0a500;" if is_trig else ""
                role    = "⚡ 觸發時框" if is_trig else ("✅ 確認時框" if r["tf"] in confirm_tfs else "")

                # D+1/D+2/D+3 顏色
                def dspan(v):
                    c = "pos" if v>=0 else "neg"
                    return f'<span class="{c}" style="font-weight:700;">{v:+.3f}</span>'

                # 如果是觸發時框且即將翻正，高亮 D+1
                d1_extra = ""
                if is_trig and r["hist"]<0 and r["d1"]>0:
                    d1_extra = ' style="background:#fff3cd;border-radius:4px;padding:0 4px;"'

                with cols_c[idx]:
                    st.markdown(f"""<div class="metric-card" style="{border}">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div class="label">{r['tf'].upper()}</div>
                            <div style="font-size:10px;background:#e8e3da;color:#555;
                                        padding:1px 7px;border-radius:3px;font-weight:700;
                                        font-family:'IBM Plex Mono',monospace;">
                                ATR {r['atr']:.3f}
                            </div>
                        </div>
                        <div class="value {hc}" style="font-size:20px;margin:6px 0 2px;">
                            {r['hist']:+.3f}
                        </div>
                        <div style="margin-bottom:8px;">{trend_pill(r['trend'])}</div>
                        <div style="border-top:1px solid #e8e3da;padding-top:8px;
                                    font-family:'IBM Plex Mono',monospace;font-size:11px;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                                <span style="color:#aaa;">D+1</span>
                                <span{d1_extra}>{dspan(r['d1'])}</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                                <span style="color:#aaa;">D+2</span>{dspan(r['d2'])}
                            </div>
                            <div style="display:flex;justify-content:space-between;">
                                <span style="color:#aaa;">D+3</span>{dspan(r['d3'])}
                            </div>
                        </div>
                        {f'<div style="margin-top:7px;font-size:10px;color:#f0a500;font-weight:700;font-family:IBM Plex Mono,monospace;">{role}</div>' if role else ''}
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    else:
        cascade   = []
        resonance = {"score":0,"max":0,"alert":False,"atype":None,"trigger":None}

    # ── MACD 表格 ────────────────────────────────────────────
    st.markdown("#### 📋 近 10 根 K 線 MACD 分析")
    st.markdown(render_table(df_t), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── MACD 圖表 ────────────────────────────────────────────
    with st.expander(f"📈 {symbol} MACD 圖表", expanded=True):
        cdf = df_main.tail(60)
        # 疊加線：日內主時框丟掉形成中的最後一根（與其他訊號同一慣例）
        _intr = tf_cfg["interval"] in INTRADAY_IVS
        _ref  = df_main.iloc[:-1] if (_intr and len(df_main) > 1) else df_main
        overlays = {}
        if show_fib:
            overlays["fib"] = calc_fib_levels(_ref)
        if show_sr:
            overlays["sr"]  = calc_sr_levels(_ref)
        if show_vwap:
            _vw = calc_vwap(df_main) if _intr else calc_rolling_vwap(df_main)
            overlays["vwap"]       = _vw.tail(60)
            overlays["vwap_label"] = ("主力成本線（當日VWAP）" if _intr
                                      else "主力成本線（20期滾動VWAP）")
        if show_fib and setup.get("A"):
            overlays["setup"] = setup
        _rv = (calc_rvol(df_main) if _intr else calc_rvol_simple(df_main)).tail(60)
        fig = build_macd_chart(cdf, symbol,
                                macd_s.tail(60), sig_s.tail(60), hist_s.tail(60),
                                interval=tf_cfg["interval"], overlays=overlays,
                                show_volume=show_vol, rvol=_rv)
        st.plotly_chart(fig, use_container_width=True, key=f"macd_chart_{symbol}")

    # ── Telegram 信號 ─────────────────────────────────────────
    if chain_tfs:
        tg_msg = build_tg_msg(symbol, cascade, resonance,
                               confirm_tfs, trigger_tf, close_val, atr_main,
                               plan_style=plan_style, setup=setup_results.get(symbol))
        tg_msgs_all.append(tg_msg)

        # ── v2.4：蒐集排序比較數據（供排行榜與深度分析 Prompt 使用）──
        trig_d = resonance.get("trigger") or {}
        rank_data.append({
            "symbol": symbol, "close": close_val, "atr": atr_main,
            "atr_pct": (atr_main/close_val*100) if close_val else 0,
            "trend": trend,
            "alert": resonance["alert"], "direction": resonance.get("direction"),
            "level": resonance.get("level"), "pts": resonance.get("pts", 0),
            "bull_pts": resonance.get("bull_pts", 0), "bear_pts": resonance.get("bear_pts", 0),
            "atype": resonance.get("atype"),
            "score": resonance["score"], "max": resonance["max"],
            "detail": resonance.get("detail", []),
            "rvol": trig_d.get("rvol"), "below_vwap": trig_d.get("below_vwap"),
            "close_c": trig_d.get("close_c"),
            "fib": trig_d.get("fib"),
            "setup": setup_results.get(symbol),
            "vwap": trig_d.get("vwap"), "vwap_kind": trig_d.get("vwap_kind"),
            "gap_pct": trig_d.get("gap_pct"),
            "trigger_tf": trigger_tf,
            "cascade": [
                {"tf": r["tf"], "hist": r.get("hist_c", r.get("hist")), "d1": r.get("d1"),
                 "hist_live": r.get("hist"), "d2": r.get("d2"), "d3": r.get("d3"),
                 "status": r.get("status"), "trend": r.get("trend"), "atr": r.get("atr"),
                 "role": ("觸發" if r["tf"] == trigger_tf
                          else ("確認" if r["tf"] in confirm_tfs else None))}
                for r in cascade if r.get("valid")
            ],
            "trig": trig_d,
            "macd_rows": [
                {"date": x["日線"], "close": x["收盤"], "macd": x["MACD"],
                 "hist": x["Hist"], "status": x["狀態"]}
                for _, x in df_t.iterrows()
            ],
        })
    else:
        tg_msg = ""

    # ── 警報自動推送（去重：symbol+方向+等級 每小時最多一次）──
    if chain_tfs and tg_auto and resonance["alert"] and tg_token and tg_chat:
        _dk = hashlib.md5(
            f"{symbol}|{resonance.get('direction')}|{resonance.get('level')}|{trigger_tf}|"
            f"{datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H')}".encode()
        ).hexdigest()
        _sent = st.session_state.setdefault("tg_sent_hashes", set())
        if _dk not in _sent:
            ok_auto, _ = send_telegram(tg_token, tg_chat, tg_msg)
            if ok_auto:
                _sent.add(_dk)
                st.toast(f"🔔 已自動推送 {symbol} 警報", icon="📡")
    if chain_tfs:
        with st.expander(f"📡 Telegram 信號 — {symbol}"):
            st.markdown(f'<div class="tg-box">{tg_msg}</div>', unsafe_allow_html=True)

    st.markdown("---")


# ── v2.4 排序比較 + 深度分析 Prompt ─────────────────────────────
# ── v2.14 斐波那契買點掃描總表 ───────────────────────────────────
if setup_results:
    st.markdown("## 🎯 斐波那契買點掃描")
    st.caption("週線定方向 → 日線 ZigZag 找 A→B 波段與回調區 → 4H 已收盤 K 線確認入場。"
               "排序：✅入場確認 > 🚀突破成立 > 🟡週線盤整輕倉 > 👀到位待確認 > 🎯蓄勢 > ⏳等待 > ⚠️過深 > ⛔週線走弱 > ❌波段失效。")
    st.dataframe(pd.DataFrame(setup_table_rows(setup_results)), hide_index=True,
                 use_container_width=True)
    _hits = [s for s, v in setup_results.items() if v["prio"] <= 2]
    if _hits:
        st.success("目前符合條件：" + "、".join(
            f"{s}（{setup_results[s]['status']}）" for s in _hits))
    else:
        st.info("目前沒有「入場確認」或「突破成立」的標的——沒有轉強訊號就繼續等待，不為買而買。")

if rank_data:
    st.markdown("## 🏆 排序比較")
    ranked = sorted(rank_data, key=rank_key, reverse=True)
    disp_rows = []
    for i, r in enumerate(ranked, 1):
        eff_dir = r["direction"] or leaning(r)
        sub     = max(r.get("bull_pts", 0), r.get("bear_pts", 0))
        if r["direction"]:
            dirn = {"bear":"🔴 空頭","bull":"🟢 多頭"}[r["direction"]]
        elif eff_dir:
            dirn = {"bear":f"🔴 偏空（未達門檻，潛在 {sub} 分）",
                    "bull":f"🟢 偏多（未達門檻，潛在 {sub} 分）"}[eff_dir]
        else:
            dirn = "⚪ 中性"
        lvl  = {"red":"🔴 紅色警報","yellow":"🟠 橙色觀察",None:"⚪ 靜默"}.get(r["level"],"⚪ 靜默")
        disp_rows.append({
            "排名": i, "代碼": r["symbol"], "方向": dirn, "等級": lvl,
            "評分": r["pts"] if r["pts"] else f"(潛在{sub})",
            "收盤": f"{r['close']:.2f}",
            "ATR%": f"{r['atr_pct']:.2f}%", "趨勢": r["trend"],
            "RVOL": f"{r['rvol']:.2f}×" if r.get("rvol") is not None else "—",
            "Fib": (f"{r['fib']['nearest_ratio']*100:.1f}%{r['fib']['role']}"
                    f"（{r['fib']['dist_pct']:+.2f}%）") if r.get("fib") else "—",
            "VWAP位置": (
                f"{'下方' if r['below_vwap'] else '上方'} "
                f"{abs(r.get('close_c', r['close'])/r['vwap']-1)*100:.2f}%"
                if r.get("vwap") and r.get("below_vwap") is not None else "—"
            ),
        })
    st.dataframe(pd.DataFrame(disp_rows), hide_index=True, use_container_width=True)
    if all(r["pts"] == 0 for r in rank_data):
        st.caption("⚠️ 這批標的目前都沒有正式警報（大時框本來就少見翻轉屬正常），"
                   "排序改用「潛在傾向分」——僅供參考，不代表已觸發訊號。")

    st.markdown("#### 📋 最佳股票分析 Prompt · 複製後貼入任意 AI 進行深度分析")
    st.caption("本系統只做技術面 + 量能的程式化判定；把下方內容貼給 ChatGPT / Claude / "
               "Gemini 等，可補上基本面、新聞、產業比較等本系統覆蓋不到的部分。")
    prompt_text = build_analysis_prompt(rank_data, main_tf, trigger_tf,
                                        chain_tfs=chain_tfs, confirm_tfs=confirm_tfs,
                                        plan_style=plan_style)
    st.code(prompt_text, language="markdown")

st.markdown("---")

# ── 批量發送 Telegram ─────────────────────────────────────────
if tg_send:
    if not tg_token or not tg_chat:
        st.error("請填寫 Bot Token 和 Chat ID")
    else:
        ok = sum(1 for msg in tg_msgs_all
                 if (send_telegram(tg_token, tg_chat, msg)[0] or time.sleep(0.5) or False))
        st.success(f"✅ 已發送 {ok}/{len(tg_msgs_all)} 個信號")



# ══════════════════════════════════════════════════════════════
# 回測區塊（主頁面底部）
# ══════════════════════════════════════════════════════════════
st.markdown("# 📊 三步瀑布傳導回測")
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#888;
            background:#fff8f0;border:1px solid #d4cfc6;border-left:4px solid #5a7fa8;
            border-radius:8px;padding:12px 16px;margin-bottom:12px;line-height:1.9;">
<b>回測邏輯（忠實還原三步入場）：</b><br>
Step 1 &nbsp;→&nbsp; 1d + 1h Histogram > 0 &nbsp;（大方向多頭確認）<br>
Step 2 &nbsp;→&nbsp; 5m D+1 > 0 &nbsp;+&nbsp; 15m D+1 > 0 &nbsp;+&nbsp; 30m D+1 > 0 &nbsp;（傳導鏈預警）<br>
Step 3 &nbsp;→&nbsp; 30m Histogram 前根 &lt; 0 → 當根 ≥ 0 &nbsp;（實際翻正，執行入場）<br>
<b>數據範圍：最近 60 天日內數據（5m / 15m / 30m / 1h / 1d）</b>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── 回測參數設定 ──────────────────────────────────────────────
bc1, bc2, bc3, bc4 = st.columns(4)
with bc1:
    bt_symbols = st.multiselect(
        "回測股票",
        symbols if symbols else DEFAULT_SYMBOLS,
        default=symbols[:1] if symbols else ["TSLA"],
    )
with bc2:
    bt_exit_mode = st.radio(
        "📤 出場模式",
        ["固定根數", "30m轉負", "1d/1h轉負"],
        index=0,
        help=(
            "固定根數：持倉滿 N 根 30m K 線後平倉\n"
            "30m轉負：30m Histogram 再次轉負時出場\n"
            "1d/1h轉負：1h 或 1d Histogram 轉負時出場（跟隨大趨勢）"
        ),
    )
    bt_hold = st.slider(
        "固定根數（30m K線）" if bt_exit_mode == "固定根數" else "保底根數倍率參考",
        3, 48, 6,
        disabled=(bt_exit_mode != "固定根數"),
        help="固定根數模式：持倉 N 根後平倉\n動態模式：作為保底上限倍率的基礎",
    )
with bc3:
    bt_atr_sl = st.slider("止損 (×ATR 30m)", 0.5, 3.0, 1.5, 0.5)
    bt_atr_tp = st.slider("止盈 (×ATR 30m)", 1.0, 6.0, 3.0, 0.5)
with bc4:
    st.markdown("**出場模式說明**")
    if bt_exit_mode == "固定根數":
        _md_title = "🕐 固定根數"
        _md_info  = f"持倉 <b>{bt_hold}</b> 根 30m K線 ≈ {bt_hold//2}小時<br>止損/盈：30m ATR<br><br>適合：短線、震盪市"
    elif bt_exit_mode == "30m轉負":
        _md_title = "🔄 30m 轉負出場"
        _md_info  = "30m Histogram 正→負 時出場<br>止損/盈：30m ATR<br><br>適合：日內趨勢跟蹤"
    else:
        _md_title = "📅 1d/1h 轉負出場"
        _md_info  = "1h 或 1d Histogram 轉負時出場<br>止損/盈：30m ATR<br><br>適合：波段持倉數天"
    st.markdown(
        '<div class="metric-card" style="font-size:11px;line-height:1.9;">'
        + '<div class="label">' + _md_title + '</div>'
        + '<div style="margin-top:8px;">' + _md_info + '</div></div>',
        unsafe_allow_html=True,
    )
    run_bt = st.button("🚀 開始回測", type="primary", use_container_width=True)

if run_bt and bt_symbols:
    for bt_sym in bt_symbols:
        mode_label = {
            "固定根數":  f"固定 {bt_hold} 根 30m",
            "30m轉負":   "30m Histogram 轉負",
            "1d/1h轉負": "1d/1h Histogram 轉負",
        }[bt_exit_mode]
        st.markdown(f"## 📈 {bt_sym}  —  最近60天 · 出場：{mode_label}")

        with st.spinner(f"載入 {bt_sym} 多時框數據並計算..."):
            result = run_cascade_backtest(
                bt_sym, atr_sl=bt_atr_sl, atr_tp=bt_atr_tp,
                hold_bars=bt_hold, exit_mode=bt_exit_mode,
            )

        # 錯誤處理
        if "error" in result:
            st.error(f"⚠️ {bt_sym}：{result['error']}")
            continue

        if result.get("total", 0) == 0:
            diag = result.get("diag", {})
            total_bars  = diag.get("total_bars",  "?")
            step1_pass  = diag.get("step1_pass",  0)
            step2_alert = diag.get("step2_alert", 0)
            step3_entry = diag.get("step3_entry", 0)

            # 找出卡在哪一步
            if step1_pass == 0:
                bottleneck = "❌ Step 1 從未通過：1d / 1h Histogram 未同時 > 0（大方向未確認多頭）"
                tip = "目前市場可能處於空頭或震盪。建議等待 1d 和 1h 同時翻正後再回測。"
            elif step2_alert == 0:
                bottleneck = "❌ Step 2 從未觸發：Step1 通過但傳導鏈 D+1 未同時預測翻正"
                tip = "5m / 15m / 30m 的 D+1 預測未能同時 > 0。可能是震盪市，或者 5m 數據 yfinance 只能取 7天，可能不足。"
            elif step3_entry == 0:
                bottleneck = "❌ Step 3 從未入場：預警發出但 30m Histogram 翻正時 Step1 已失效"
                tip = "預警發出的下一根 K 線，1d 或 1h 已轉負。市場變化快，大方向短暫失效。可嘗試放寬 Step1 條件。"
            else:
                bottleneck = "⚠️ 信號找到但未生成交易記錄，請檢查數據"
                tip = ""

            st.warning(f"⚠️ {bt_sym}：60天內未找到完整三步入場信號")
            st.markdown(f"""
            <div style="background:#fff8f0;border:1px solid #d4cfc6;border-radius:10px;
                        padding:16px 20px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                        line-height:2;">
            <b>📊 診斷報告</b><br>
            ─────────────────────────────────────<br>
            總 30m K 線數：<b>{total_bars}</b> 根<br>
            Step 1 通過（1d+1h 同時多頭）：<b>{step1_pass}</b> 根
            &nbsp;{"✅" if step1_pass > 0 else "❌"}<br>
            Step 2 預警觸發（傳導鏈D+1翻正）：<b>{step2_alert}</b> 次
            &nbsp;{"✅" if step2_alert > 0 else "❌"}<br>
            Step 3 實際入場（30m翻正）：<b>{step3_entry}</b> 次
            &nbsp;{"✅" if step3_entry > 0 else "❌"}<br>
            ─────────────────────────────────────<br>
            <b>瓶頸：</b>{bottleneck}<br>
            <b>建議：</b>{tip}
            </div>
            """, unsafe_allow_html=True)
            continue

        # ── KPI 卡片 ──────────────────────────────────────
        st.markdown(render_bt_kpi(result, bt_sym, bt_hold, bt_atr_sl, bt_atr_tp),
                    unsafe_allow_html=True)

        # ── 主圖：入場標注在 30m 圖上 ───────────────────────
        st.markdown("##### 🎯 30m K線入場/出場標注圖")
        st.markdown("""
        <div style="font-size:11px;color:#aaa;font-family:'IBM Plex Mono',monospace;margin-bottom:6px;">
        ▲ 綠色三角 = 盈利入場 &nbsp;|&nbsp; ▲ 紅色三角 = 虧損入場 &nbsp;|&nbsp;
        橙色虛線 = D+1預測（翻正即觸發Step2）
        </div>""", unsafe_allow_html=True)
        fig_entry = build_bt_entry_chart(result["aligned"], result["trades"])
        st.plotly_chart(fig_entry, use_container_width=True, key=f"bt_entry_{bt_sym}")

        # ── 副圖：權益曲線 ────────────────────────────────
        st.markdown("##### 📉 累計收益曲線")
        fig_eq = build_equity_curve(result["trades"])
        st.plotly_chart(fig_eq, use_container_width=True, key=f"bt_eq_{bt_sym}")

        # ── 出場原因分析 ──────────────────────────────────
        col_pie, col_reason, col_extreme = st.columns(3)
        df_t_bt = pd.DataFrame(result["trades"])

        with col_pie:
            st.markdown("##### 出場原因分佈")
            rc_counts = df_t_bt["exit_reason"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=rc_counts.index.tolist(),
                values=rc_counts.values.tolist(),
                marker=dict(colors=["#c0392b","#3d8b5e","#5a7fa8"]),
                textinfo="label+percent", hole=0.4,
            ))
            fig_pie.update_layout(
                paper_bgcolor="#fff8f0",
                font=dict(family="IBM Plex Mono", size=11),
                margin=dict(l=0,r=0,t=10,b=0), height=220, showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key=f"bt_pie_{bt_sym}")

        with col_reason:
            st.markdown("##### 各出場方式勝率")
            for reason in ["止損","止盈","到期平倉"]:
                sub = df_t_bt[df_t_bt["exit_reason"]==reason]
                if len(sub) == 0: continue
                wr_r = sub["win"].mean()*100
                wc_r = "#3d8b5e" if wr_r>=55 else ("#e07b39" if wr_r>=45 else "#c0392b")
                st.markdown(f"""<div class="metric-card" style="margin-bottom:8px;">
                    <div class="label">{reason}</div>
                    <div class="value" style="color:{wc_r};font-size:20px;">{wr_r:.1f}%</div>
                    <div class="sub">{len(sub)}筆 · 均 {sub['pnl_pct'].mean():+.2f}%</div>
                </div>""", unsafe_allow_html=True)

        with col_extreme:
            st.markdown("##### 最佳 / 最差")
            best  = df_t_bt.loc[df_t_bt["pnl_pct"].idxmax()]
            worst = df_t_bt.loc[df_t_bt["pnl_pct"].idxmin()]
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:8px;border:1.5px solid #3d8b5e;">
                <div class="label">🏆 最佳交易</div>
                <div class="value pos">{best['pnl_pct']:+.2f}%</div>
                <div class="sub">{str(best['entry_time'])[:16]}</div>
                <div class="sub">{best['exit_reason']}</div>
            </div>
            <div class="metric-card" style="border:1.5px solid #c0392b;">
                <div class="label">💥 最差交易</div>
                <div class="value neg">{worst['pnl_pct']:+.2f}%</div>
                <div class="sub">{str(worst['entry_time'])[:16]}</div>
                <div class="sub">{worst['exit_reason']}</div>
            </div>""", unsafe_allow_html=True)

        # ── 逐筆明細 ──────────────────────────────────────
        with st.expander("🔍 逐筆交易明細（最近25筆）"):
            st.markdown(render_trades_table(result["trades"], max_rows=25),
                        unsafe_allow_html=True)

        st.markdown("---")

elif not run_bt:
    st.markdown("""
    <div style="text-align:center;padding:40px;color:#aaa;
                font-family:'IBM Plex Mono',monospace;font-size:13px;">
    👆 選擇股票，設定參數，點擊「開始回測」<br><br>
    系統將嚴格按照三步入場邏輯，在最近60天30m K線上<br>
    模擬每一個信號的入場和出場，統計真實成功率
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#bbb;font-size:11px;margin-top:20px;
            font-family:IBM Plex Mono,monospace;'>
MACD 瀑布動能傳導系統 v2.0 ｜ 數據：Yahoo Finance ｜ 僅供參考，不構成投資建議
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#bbb;font-size:11px;margin-top:20px;
            font-family:IBM Plex Mono,monospace;'>
MACD 瀑布動能傳導系統 v2.0 ｜ 數據：Yahoo Finance ｜ 僅供參考，不構成投資建議
</div>
""", unsafe_allow_html=True)
