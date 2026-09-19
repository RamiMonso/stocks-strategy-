"""
סימולטור אסטרטגיית פולבק RSI - Streamlit
הרצה מקומית:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from simulator import data as D
from simulator.engine import Dataset, Params, run_backtest
from simulator.portfolio import cash_rate_series, curve_metrics, simulate_portfolio, trade_stats, yearly_table

st.set_page_config(page_title="סימולטור פולבק RSI", page_icon="📈", layout="wide")
st.markdown(
    """
<style>
.stMarkdown, .stAlert, label, [data-testid="stMetric"], .stCaption, summary { direction: rtl; text-align: right; }
[data-testid="stSidebar"] label { direction: rtl; text-align: right; }
.stTabs [data-baseweb="tab-list"] { direction: rtl; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Defaults, presets and widget-state handling
# --------------------------------------------------------------------------------------
DEFAULTS = Params().to_dict()
DEFAULTS.update(
    start_d=dt.date(2021, 9, 20),
    end_d=dt.date(2026, 9, 18),
    pool_rank=90,
    custom_tickers="AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, JNJ, V",
    use_split=False,
    split_d=dt.date(2024, 1, 2),
    preset="— בחר —",
)

T2 = dict(use_rvol=False, earnings_days=30, stop_mode="none", tp1_rsi=50.0, tp2_rsi=60.0)
PRESETS = {
    "האסטרטגיה המקורית (כפי שהוגדרה)": {},
    "T2 – יציאות RSI 50/60, בלי סטופ ובלי RVOL, טופ 50": dict(T2, top_n=50),
    "T2 טופ 100, 10% לפוזיציה (כמו בסיכום)": dict(T2, top_n=100, position_pct=10.0, max_positions=10, cost_bps=5.0),
    "T1 – יציאות RSI 60/70, בלי סטופ ובלי RVOL": dict(use_rvol=False, earnings_days=30, stop_mode="none", top_n=50),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)


def apply_preset():
    name = st.session_state.get("preset")
    if name in PRESETS:
        st.session_state.update({**{k: v for k, v in DEFAULTS.items() if k in Params().to_dict()}, **PRESETS[name]})


UNIVERSE_LABELS = {
    "pit_top_n": "טופ N לפי שווי שוק בכל יום (נקודתי, מומלץ)",
    "static_top_n": "טופ N לפי שווי שוק היום (קבוע; הטיית הסתכלות קדימה)",
    "custom": "רשימת מניות שאבחר",
}


def params_from_state() -> Params:
    s = st.session_state
    d = {k: s[k] for k in Params().to_dict().keys()}
    d["start"] = s["start_d"].isoformat()
    d["end"] = s["end_d"].isoformat()
    return Params(**d)


# --------------------------------------------------------------------------------------
# Cached data loaders (shared across reruns; heavy on first run only)
# --------------------------------------------------------------------------------------
END_STR = (dt.date.today() + dt.timedelta(days=1)).isoformat()
CACHE_TTL = 12 * 3600


@st.cache_resource(show_spinner=False)
def _store() -> dict:
    """Process-wide memo table. (We avoid @st.cache_* on the loaders because they call st.progress via callbacks,
    and Streamlit would try to replay those UI elements on a cache hit.)"""
    return {}


def memo(key, fn):
    store = _store()
    hit = store.get(key)
    if hit is not None and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    val = fn()
    store[key] = (time.time(), val)
    return val


def cached_bench():
    return memo(("bench", END_STR), lambda: D.download_benchmarks(D.DATA_START, END_STR))


def cached_sp500():
    return memo(("sp500",), D.get_sp500_tickers)


def cached_prices(tickers: tuple, start: str, progress=None):
    return memo(("prices", tickers, start), lambda: D.download_prices(tickers, start, END_STR, progress=progress))


def cached_meta(tickers: tuple, progress=None):
    return memo(("meta", tickers), lambda: D.download_meta(tickers, progress=progress))


def cached_peg(tickers: tuple):
    return memo(("peg", tickers), lambda: D.download_peg(tickers))


def cached_universe(pool_rank: int, progress=None):
    """S&P 500 -> candidate pool -> point-in-time market-cap ranks."""
    def build():
        tickers = tuple(cached_sp500())
        prices_all = cached_prices(tickers, D.DATA_START, progress=progress)
        pool = tuple(D.select_pool(prices_all, pool_rank))
        meta = cached_meta(pool, progress=progress)
        prices = {t: prices_all[t] for t in pool if t in prices_all}
        rank = D.build_marketcap_rank(prices, meta)
        return prices, meta, rank
    return memo(("universe", int(pool_rank), END_STR), build)


def load_dataset(p: Params, pool_rank: int, custom: str, box) -> Dataset:
    bar = box.progress(0.0, text="טוען נתונים…")

    def prog(x, msg):
        bar.progress(min(max(x, 0.0), 1.0), text=msg)

    bench = cached_bench()
    if bench.get("spx") is None:
        raise RuntimeError("לא ניתן היה להוריד את מדד ה-S&P 500 מ-Yahoo. נסה שוב בעוד רגע.")
    if p.universe_mode == "custom":
        tickers = tuple(sorted({t.strip().upper().replace(".", "-") for t in custom.replace("\n", ",").split(",") if t.strip()}))
        if not tickers:
            raise RuntimeError("לא הוזנו מניות.")
        start = (pd.Timestamp(p.start) - pd.Timedelta(days=430)).date().isoformat()
        prices = cached_prices(tickers, min(start, D.DATA_START), progress=prog)
        meta = cached_meta(tuple(prices.keys()), progress=prog) if p.use_earnings else None
        rank = None
    else:
        prices, meta, rank = cached_universe(int(pool_rank), progress=prog)
    peg = cached_peg(tuple(prices.keys())) if p.use_peg else {}
    bar.empty()
    return D.assemble_dataset(prices, bench, meta, rank, peg)


# --------------------------------------------------------------------------------------
# Sidebar: parameters
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ פרמטרים")
st.sidebar.selectbox("טען פריסט", ["— בחר —"] + list(PRESETS.keys()), key="preset", on_change=apply_preset)

with st.sidebar.form("params"):
    with st.expander("📅 תקופה ומאגר מניות", expanded=True):
        st.date_input("תחילת סימולציה", key="start_d", min_value=dt.date(2016, 1, 4))
        st.date_input("סוף סימולציה", key="end_d")
        st.selectbox("מאגר", list(UNIVERSE_LABELS.keys()), key="universe_mode", format_func=UNIVERSE_LABELS.get)
        st.number_input("N – כמה מהגדולות (לפי שווי שוק)", 5, 500, key="top_n", step=5)
        st.number_input("גודל מאגר מועמדים (לפי מחזור מסחר)", 60, 200, key="pool_rank", step=10,
                        help="מניות שהיו אי-פעם בין X הנסחרות ביותר. גדול יותר = טעינה איטית יותר, כיסוי טוב יותר.")
        st.text_area("רשימת מניות (למצב 'רשימה שאבחר')", key="custom_tickers", height=70)
        st.checkbox("הצג גם פיצול אימון/בדיקה", key="use_split")
        st.date_input("תאריך פיצול (אימון עד, בדיקה אחריו)", key="split_d")

    with st.expander("🔎 תנאי כניסה", expanded=False):
        st.checkbox("מדד S&P מעל SMA", key="use_index_filter")
        st.slider("אורך SMA של המדד", 20, 250, key="index_sma")
        st.checkbox("המניה מעל SMA", key="use_stock_sma")
        st.slider("אורך SMA של המניה", 20, 250, key="stock_sma")
        st.slider("תקופת RSI", 2, 30, key="rsi_period")
        st.slider("RSI לכניסה – מתחת ל-", 10.0, 60.0, key="rsi_entry", step=1.0)
        st.checkbox("נר פריקה = יום החצייה מתחת לסף (ולא כל יום מתחתיו)", key="require_cross")
        st.checkbox("סינון PEG (משתמש בערך נוכחי – הטיית הסתכלות קדימה!)", key="use_peg")
        c1, c2 = st.columns(2)
        c1.number_input("PEG מינימום", -5.0, 10.0, key="peg_min", step=0.1)
        c2.number_input("PEG מקסימום", -5.0, 10.0, key="peg_max", step=0.1)
        st.checkbox("ללא דוחות בימים הקרובים", key="use_earnings")
        st.slider("ימים קדימה (ימי לוח)", 1, 90, key="earnings_days")

    with st.expander("🛒 פקודת כניסה", expanded=False):
        st.radio("אופן כניסה", ["breakout", "next_open"], key="entry_mode", horizontal=True,
                 format_func={"breakout": "Buy-stop מעל ה-high", "next_open": "פתיחת היום הבא"}.get)
        st.number_input("באפר מעל ה-high ($)", 0.0, 5.0, key="buffer_usd", step=0.01, format="%.2f")
        st.checkbox("תקרת קנייה מעל מחיר הכניסה", key="use_limit_cap")
        st.number_input("תקרה (%)", 0.1, 10.0, key="limit_cap_pct", step=0.1)
        st.slider("תוקף הפקודה (ימי מסחר)", 1, 10, key="order_days")
        st.checkbox("להוריד את ה-high מדי יום אם ירד", key="reeval_lower_high")
        st.checkbox("לבטל פקודה אם התנאים כבר לא מתקיימים", key="cancel_if_invalid")

    with st.expander("🛡️ סיכון", expanded=False):
        st.radio("סוג סטופ", ["fixed", "atr", "none"], key="stop_mode", horizontal=True,
                 format_func={"fixed": "אחוז קבוע", "atr": "ATR", "none": "ללא"}.get)
        st.number_input("סטופ קבוע (%)", 0.5, 50.0, key="stop_pct", step=0.5)
        st.slider("תקופת ATR", 5, 30, key="atr_period")
        st.number_input("מכפיל ATR", 0.5, 10.0, key="stop_atr_mult", step=0.5)
        st.checkbox("בדיקת RVOL (פרוקסי: נפח יומי מלא)", key="use_rvol")
        st.number_input("RVOL מינימום", 0.1, 3.0, key="rvol_min", step=0.1)
        st.slider("חלון ממוצע נפח (ימים)", 5, 60, key="rvol_window")
        st.checkbox("לבדוק RVOL בכל יום (ולא רק ביום הכניסה)", key="rvol_every_day")
        st.number_input("סגירה בזמן: מקס' ימי החזקה (0 = כבוי)", 0, 250, key="max_hold_days", step=5)

    with st.expander("🎯 יציאות", expanded=False):
        st.slider("מימוש חלקי ב-RSI", 30.0, 90.0, key="tp1_rsi", step=1.0)
        st.slider("חלק שנמכר במימוש הראשון", 0.1, 0.9, key="tp1_fraction", step=0.05)
        st.checkbox("Break-even לחלק הנותר", key="use_breakeven")
        st.slider("מימוש יתרה ב-RSI", 40.0, 95.0, key="tp2_rsi", step=1.0)
        st.checkbox("יציאה בסגירה מתחת ל-EMA (ליתרה)", key="use_ema_exit")
        st.slider("אורך EMA", 5, 100, key="ema_period")
        st.radio("ביצוע יציאות לפי RSI", ["next_open", "same_close"], key="exec_mode", horizontal=True,
                 format_func={"next_open": "פתיחת היום הבא", "same_close": "סגירה באותו יום"}.get)
        st.number_input("עמלה+החלקה לצד (נקודות בסיס)", 0.0, 100.0, key="cost_bps", step=1.0)

    with st.expander("💼 ניהול תיק", expanded=False):
        st.number_input("הון התחלתי ($)", 1000.0, 1e9, key="initial_capital", step=10000.0)
        st.number_input("גודל פוזיציה (% מההון)", 0.5, 100.0, key="position_pct", step=0.5)
        st.number_input("מקסימום פוזיציות פתוחות", 1, 100, key="max_positions")
        st.number_input("תקרת חשיפה ברוטו (% מההון; מעל 100 = מינוף)", 10.0, 400.0, key="max_gross_pct", step=10.0)
        st.radio("ריבית על מזומן", ["tbill", "fixed", "zero"], key="cash_rate_mode", horizontal=True,
                 format_func={"tbill": "מק\"מ אמיתי", "fixed": "קבועה", "zero": "אפס"}.get)
        st.number_input("ריבית קבועה (% שנתי)", 0.0, 15.0, key="cash_rate_fixed", step=0.25)
        st.number_input("מרווח ריבית על הלוואה (% שנתי)", 0.0, 10.0, key="borrow_spread_pct", step=0.25)

    submitted = st.form_submit_button("▶️ הרץ סימולציה", width="stretch")


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def run_simulation():
    p = params_from_state()
    problems = []
    if p.end <= p.start:
        problems.append("תאריך הסיום חייב להיות אחרי ההתחלה.")
    if p.tp1_rsi >= p.tp2_rsi:
        problems.append("RSI של המימוש הראשון צריך להיות נמוך מזה של המימוש השני.")
    if p.universe_mode != "custom" and p.start < "2016-01-04":
        problems.append("במאגר לפי שווי שוק נקודתי אפשר להתחיל מ-4.1.2016 (Yahoo מחזיק היסטוריית מספר מניות מ-2015).")
    if problems:
        for m in problems:
            st.error(m)
        return

    box = st.empty()
    try:
        with st.spinner("טוען נתונים (בפעם הראשונה זה יכול לקחת כמה דקות; אחר כך הכול נשמר בזיכרון)…"):
            ds = load_dataset(p, st.session_state["pool_rank"], st.session_state["custom_tickers"], box)
        with st.spinner("מריץ סימולציה…"):
            trades, counters = run_backtest(ds, p)
            bench = cached_bench()
            spx = bench["spx"]
            days = spx.loc[p.start:p.end].index
            if len(days) < 5:
                st.error("אין מספיק ימי מסחר בתקופה שנבחרה.")
                return
            prev = spx.index[spx.index < days[0]]
            base_day = prev[-1] if len(prev) else days[0]
            rf = cash_rate_series(p, spx.loc[base_day:p.end].index, bench.get("irx"))
            eq, expo, info = simulate_portfolio(trades, p, days, rf)
            eq = pd.concat([pd.Series({base_day: p.initial_capital}), eq])
            eq = eq[~eq.index.duplicated(keep="last")].sort_index()
            btr = (bench.get("spxtr") if bench.get("spxtr") is not None else spx).loc[base_day:p.end]
            btr = btr / btr.iloc[0] * p.initial_capital
    except Exception as e:  # noqa: BLE001
        box.empty()
        st.error(f"שגיאה בטעינת נתונים או בהרצה: {e}")
        return

    if len(trades):
        trades["mae"] = [min(v for _, v in pth) for pth in trades["path"]]
    res = dict(p=p, trades=trades, counters=counters, equity=eq, exposure=expo, info=info, bench=btr, rf=rf,
               stats=trade_stats(trades), m=curve_metrics(eq, rf), mb=curve_metrics(btr, rf),
               ts=dt.datetime.now().strftime("%H:%M:%S"))
    st.session_state["result"] = res
    if len(trades):
        s, m = res["stats"], res["m"]
        st.session_state["history"].append({
            "שעה": res["ts"], "תקופה": f"{p.start}→{p.end}", "מאגר": f"{p.universe_mode}:{p.top_n}",
            "RSI כניסה": p.rsi_entry, "סטופ": p.stop_mode if p.stop_mode == "none" else (f"{p.stop_pct}%" if p.stop_mode == "fixed" else f"ATR×{p.stop_atr_mult}"),
            "יציאות RSI": f"{p.tp1_rsi:.0f}/{p.tp2_rsi:.0f}", "RVOL": p.rvol_min if p.use_rvol else "כבוי",
            "עסקאות": s["n"], "הצלחה%": round(100 * s["win_rate"], 1), "PF": round(s["profit_factor"], 2),
            "CAGR%": round(100 * m.get("cagr", np.nan), 1), "MDD%": round(100 * m.get("mdd", np.nan), 1),
            "Sharpe": round(m.get("sharpe", np.nan), 2),
        })


if submitted:
    run_simulation()

# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
st.title("📈 סימולטור אסטרטגיית פולבק RSI")

res = st.session_state["result"]
if res is None:
    st.info("בחר פרמטרים בסרגל הצד (או טען פריסט) ולחץ **הרץ סימולציה**. הטעינה הראשונה מורידה נתוני אמת מ-Yahoo Finance ועשויה לקחת כמה דקות.")
    st.stop()

p: Params = res["p"]
trades: pd.DataFrame = res["trades"]
if len(trades) == 0:
    st.warning("לא נמצאו עסקאות עם הפרמטרים האלה. נסה להרחיב תנאים (למשל RSI גבוה יותר, מאגר גדול יותר).")
    st.stop()
s, m, mb = res["stats"], res["m"], res["mb"]

with st.expander("⚠️ מגבלות ואזהרות (חשוב לקרוא)", expanded=False):
    w = [
        "המאגר הוא חברות שנמצאות **היום** ב-S&P 500 (הטיית הישרדות): מניות שנפלו ויצאו מהמדד לא נכללות. הטיה זו משפרת במיוחד אסטרטגיות בלי סטופ.",
        "אין נתוני PEG היסטוריים חינמיים. סינון PEG משתמש בערך של היום ולכן **אינו תקף לבק-טסט**.",
        "אין נתוני דקה היסטוריים; RVOL מחושב מנפח יומי מלא מול ממוצע ימים קודמים (פרוקסי).",
        "בנרות יומיים לא ידוע סדר האירועים בתוך היום. ההנחה: אם המחיר נגע בסטופ ביום הכניסה, העסקה נעצרה (שמרני). "
        "אם הפתיחה מעל התקרה, אין מילוי באותו יום.",
        "כל הרצה נוספת של פרמטרים היא ניסיון נוסף. **כוונון עד שהתוצאה נראית טוב מנפח את הביצועים**. השתמש בפיצול אימון/בדיקה ובהיסטוריית ההרצות.",
        "מחירים מותאמים לפיצולים בלבד; דיבידנדים של המניות המוחזקות לא נכללים. המדד להשוואה הוא S&P 500 Total Return.",
        "אין כאן ייעוץ השקעות.",
    ]
    if p.use_peg:
        w.insert(0, "**סינון PEG פעיל** – התוצאות מוטות בגלל הסתכלות קדימה.")
    if p.stop_mode == "none":
        w.insert(0, "**אין סטופ**: אחוז ההצלחה וה-PF נראים גבוהים כי הפסדים לא ממומשים; בדוק את ה-MAE והעסקה הגרועה.")
    if p.max_gross_pct > 100:
        w.insert(0, "תקרת חשיפה מעל 100% = מינוף. לא מדומים מרג'ין קול או תנועות תוך-יומיות.")
    if len(st.session_state["history"]) >= 20:
        w.insert(0, f"הרצת {len(st.session_state['history'])} וריאציות – הסיכון ל-overfitting גבוה.")
    for line in w:
        st.markdown(f"- {line}")

# ---- KPIs
k = st.columns(4)
k[0].metric("תשואה שנתית", pct(m["cagr"]), delta=f"מדד: {pct(mb['cagr'])}", delta_color="off")
k[1].metric("תשואה מצטברת", pct(m["cumulative"], 0), delta=f"מדד: {pct(mb['cumulative'], 0)}", delta_color="off")
k[2].metric("ירידה מקסימלית", pct(m["mdd"]), delta=f"מדד: {pct(mb['mdd'])}", delta_color="off")
k[3].metric("Sharpe", f"{m['sharpe']:.2f}", delta=f"מדד: {mb['sharpe']:.2f}", delta_color="off")
k = st.columns(4)
k[0].metric("עסקאות", f"{s['n']}")
k[1].metric("אחוז הצלחה", pct(s["win_rate"]))
k[2].metric("Profit Factor", f"{s['profit_factor']:.2f}")
k[3].metric("חשיפה ממוצעת", pct(res["exposure"].mean(), 0))

tab_eq, tab_tr, tab_break, tab_split, tab_hist = st.tabs(["עקומת הון", "עסקאות", "פילוחים", "אימון/בדיקה", "היסטוריית הרצות"])

# ---- equity curve
with tab_eq:
    logy = st.checkbox("סקאלה לוגריתמית", value=False)
    eq, btr = res["equity"], res["bench"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="אסטרטגיה", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=btr.index, y=btr.values, name="S&P 500 (Total Return)", line=dict(width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=(eq / eq.cummax() - 1) * 100, name="ירידה – אסטרטגיה", fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Scatter(x=btr.index, y=(btr / btr.cummax() - 1) * 100, name="ירידה – מדד", line=dict(dash="dot")), row=2, col=1)
    fig.update_yaxes(type="log" if logy else "linear", row=1, col=1, title="שווי תיק ($)")
    fig.update_yaxes(title="ירידה (%)", row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    mt = pd.DataFrame({
        "אסטרטגיה": [pct(m["cagr"]), pct(m["cumulative"], 0), pct(m["mdd"]), f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}", f"{m['calmar']:.2f}", pct(m["vol"])],
        "S&P 500 TR": [pct(mb["cagr"]), pct(mb["cumulative"], 0), pct(mb["mdd"]), f"{mb['sharpe']:.2f}", f"{mb['sortino']:.2f}", f"{mb['calmar']:.2f}", pct(mb["vol"])],
    }, index=["תשואה שנתית", "מצטבר", "ירידה מקס'", "Sharpe (עודף על מק\"מ)", "Sortino", "Calmar", "תנודתיות שנתית"])
    st.dataframe(mt, width="stretch")
    yt = yearly_table(eq, btr)
    st.markdown("**תשואה לפי שנה קלנדרית** (שנה ראשונה/אחרונה חלקיות)")
    st.dataframe(yt.style.format("{:.1%}"), width="stretch")
    st.caption(f"עסקאות שנכנסו לתיק: {res['info']['taken']} | דולגו בגלל מגבלות תיק (מס' פוזיציות/חשיפה): {res['info']['skipped']}")

# ---- trades
with tab_tr:
    c = res["counters"]
    st.markdown(
        f"**מחזור חיי הפקודות:** נוצרו {c['created']} | מולאו {c['filled']} | פגו אחרי {p.order_days} ימים {c['expired']} | בוטלו (תנאים לא מתקיימים) {c['cancelled']} "
        f"| ימים שדולגו כי הפתיחה מעל התקרה {c['nofill_gap']}"
    )
    a, b = st.columns(2)
    stat_rows = {
        "ממוצע לעסקה": pct(s["avg"], 2), "חציון": pct(s["median"], 2), "רווח ממוצע במנצחות": pct(s["avg_win"], 2),
        "הפסד ממוצע במפסידות": pct(s["avg_loss"], 2), "העסקה הטובה / הגרועה": f"{pct(s['best'])} / {pct(s['worst'])}",
        "החזקה ממוצעת / מקס' (ימים)": f"{s['avg_hold']:.1f} / {s['max_hold']}", "הגיעו למימוש חלקי": pct(s["reached_tp1"], 0),
        "t-stat של הממוצע": f"{s['t_stat']:.2f}", "עברו דוחות בזמן ההחזקה": pct(s["through_earnings"], 0),
        "MAE: ירדו >10% מתחת לכניסה": pct((trades["mae"] < -0.10).mean(), 0), "MAE: ירדו >20%": pct((trades["mae"] < -0.20).mean(), 0),
    }
    a.dataframe(pd.DataFrame({"ערך": stat_rows}), width="stretch")
    lab = {"tp2_ema": "יעד RSI/EMA", "stop": "סטופ", "stop_gap": "פער מעל הסטופ", "breakeven": "Break-even", "rvol": "RVOL נמוך", "time": "זמן", "open_end": "פתוחה בסוף"}
    rs = pd.Series(s["reasons"]).rename(index=lab)
    b.dataframe(pd.DataFrame({"אחוז מהעסקאות": rs.map(lambda x: f"{100 * x:.1f}%")}), width="stretch")
    h = go.Figure(go.Histogram(x=trades["ret"] * 100, nbinsx=50))
    h.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="התפלגות תשואה לעסקה (%)")
    st.plotly_chart(h, width="stretch")
    show = trades.drop(columns=["path"]).copy()
    show["ret"] = (show["ret"] * 100).round(2)
    show["mae"] = (show["mae"] * 100).round(2)
    st.dataframe(show, width="stretch", height=360)
    st.download_button("⬇️ הורד עסקאות (CSV)", show.to_csv(index=False).encode("utf-8-sig"), "trades.csv", "text/csv")
    st.download_button("⬇️ הורד פרמטרים (JSON)", json.dumps(p.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"), "params.json", "application/json")

# ---- breakdowns
with tab_break:
    t2 = trades.copy()
    t2["year"] = pd.to_datetime(t2["entry"]).dt.year

    def agg(g):
        r = g["ret"]
        pf = r[r > 0].sum() / abs(r[r <= 0].sum()) if (r <= 0).any() and r[r <= 0].sum() != 0 else np.inf
        return pd.Series({"עסקאות": len(r), "הצלחה": f"{100 * (r > 0).mean():.1f}%", "ממוצע": f"{100 * r.mean():.2f}%", "PF": round(pf, 2), "הגרועה": f"{100 * r.min():.1f}%"})

    st.markdown("**לפי שנת כניסה**")
    st.dataframe(t2.groupby("year").apply(agg), width="stretch")
    st.markdown("**ריכוז לפי מניה (חמש המובילות ברווח המצטבר)**")
    cc = t2.groupby("ticker")["ret"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    top5 = cc.head(5)
    rest = t2[~t2["ticker"].isin(top5.index)]
    top5 = top5.assign(**{"חלק מהרווח": (top5["sum"] / cc["sum"].sum()).map("{:.0%}".format)})
    st.dataframe(top5.rename(columns={"count": "עסקאות", "sum": "סכום תשואות"}), width="stretch")
    if len(rest):
        rr = rest["ret"]
        st.caption(f"בלי חמש אלה: {len(rr)} עסקאות, הצלחה {100 * (rr > 0).mean():.1f}%, ממוצע {100 * rr.mean():.2f}%, PF {rr[rr > 0].sum() / max(abs(rr[rr <= 0].sum()), 1e-9):.2f}")
    st.markdown("**לפי אופן היציאה**")
    st.dataframe(t2.groupby("reason").agg(עסקאות=("ret", "size"), ממוצע=("ret", lambda x: f"{100 * x.mean():.2f}%"), החזקה=("hold", "mean")).round(1), width="stretch")

# ---- split
with tab_split:
    if not st.session_state["use_split"]:
        st.info("סמן 'הצג גם פיצול אימון/בדיקה' בסרגל הצד והרץ שוב. כוונן פרמטרים רק על תקופת האימון, ובדוק פעם אחת על תקופת הבדיקה.")
    else:
        sd = pd.Timestamp(st.session_state["split_d"])
        rows = {}
        for nm, msk, sl in (("אימון", pd.to_datetime(trades["entry"]) <= sd, slice(None, sd)), ("בדיקה", pd.to_datetime(trades["entry"]) > sd, slice(sd, None))):
            g = trades[msk]
            e = res["equity"].loc[sl]
            b_ = res["bench"].loc[sl]
            if len(g) < 2 or len(e) < 5:
                rows[nm] = {"הערה": "אין מספיק נתונים"}
                continue
            ss, mm, bb = trade_stats(g), curve_metrics(e, res["rf"]), curve_metrics(b_, res["rf"])
            rows[nm] = {"עסקאות": ss["n"], "הצלחה": pct(ss["win_rate"]), "PF": f"{ss['profit_factor']:.2f}", "ממוצע לעסקה": pct(ss["avg"], 2),
                        "העסקה הגרועה": pct(ss["worst"]), "CAGR אסטרטגיה": pct(mm["cagr"]), "CAGR מדד": pct(bb["cagr"]),
                        "MDD אסטרטגיה": pct(mm["mdd"]), "MDD מדד": pct(bb["mdd"]), "Sharpe אסטרטגיה": f"{mm['sharpe']:.2f}", "Sharpe מדד": f"{bb['sharpe']:.2f}"}
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption("שים לב: אם הפרמטרים כוונו תוך כדי הסתכלות על תקופת הבדיקה, היא כבר לא בדיקה חיצונית.")

# ---- history
with tab_hist:
    hist = pd.DataFrame(st.session_state["history"])
    st.markdown(f"**מספר ההרצות בסשן זה: {len(hist)}.** ככל שהוא גדל, ההסתברות למצוא תוצאה יפה במקרה גדלה.")
    if len(hist):
        st.dataframe(hist, width="stretch")
        st.download_button("⬇️ הורד היסטוריה (CSV)", hist.to_csv(index=False).encode("utf-8-sig"), "runs.csv", "text/csv")
        if st.button("נקה היסטוריה"):
            st.session_state["history"] = []
            st.rerun()
