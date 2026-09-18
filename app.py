import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# הגדרות עמוד ראשי
st.set_page_config(
    page_title="Institutional Swing Engine - Simulator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. פונקציות עזר וטעינת נתונים
# ==========================================

TOP_50_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "WMT", "AMD", "PEP", "KO", "NFLX", "TMO",
    "LIN", "ADBE", "WFC", "DIS", "QCOM", "CSCO", "INTU", "GE", "AMAT", "TXN",
    "CAT", "VZ", "PFE", "PM", "IBM", "CMCSA", "NOW", "INTC", "SPGI", "HON"
]

@st.cache_data(ttl=3600*12)
def load_historical_data(tickers, start_date, end_date):
    """משיכת נתוני OHLCV מותאמים עבור המניות ומדד SPY"""
    data = {}
    all_symbols = list(set(tickers + ["SPY"]))
    raw = yf.download(all_symbols, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=False)
    
    for sym in all_symbols:
        try:
            if len(all_symbols) > 1:
                df = raw[sym].dropna().copy()
            else:
                df = raw.dropna().copy()
            if not df.empty and len(df) > 200:
                data[sym] = df
        except Exception:
            continue
    return data

def compute_indicators(df, rsi_len=14, sma_len=200, ema_len=20, vol_ma_len=20):
    """חישוב כל האינדיקטורים הטכניים על בסיס נרות יומיים"""
    df = df.copy()
    
    # SMA 200
    df['SMA200'] = df['Close'].rolling(window=sma_len).mean()
    
    # EMA 20
    df['EMA20'] = df['Close'].ewm(span=ema_len, adjust=False).mean()
    
    # Volume SMA 20 & RVOL
    df['Vol_SMA20'] = df['Volume'].rolling(window=vol_ma_len).mean()
    df['RVOL'] = np.where(df['Vol_SMA20'] > 0, df['Volume'] / df['Vol_SMA20'], 0.0)
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_len).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    
    return df

# ==========================================
# 2. מנוע הסימולציה המוסדי (Event-Driven Engine)
# ==========================================

def run_simulation(stock_data, params):
    spy_df = compute_indicators(stock_data["SPY"], sma_len=params["sma_len"])
    spy_df['Macro_OK'] = spy_df['Close'] > spy_df['SMA200']
    
    all_trades = []
    
    for ticker, df_raw in stock_data.items():
        if ticker == "SPY":
            continue
            
        df = compute_indicators(
            df_raw, 
            rsi_len=params["rsi_len"], 
            sma_len=params["sma_len"], 
            ema_len=params["exit_ema"], 
            vol_ma_len=20
        )
        
        # סנכרון עם מאקרו SPY לפי תאריך
        df = df.join(spy_df[['Macro_OK']], how='inner')
        
        in_trade = False
        trade_stage = 0  # 1: 100% פוזיציה, 2: 50% פוזיציה לאחר מימוש שלב א'
        entry_price = 0.0
        stop_price = 0.0
        position_shares = 0
        current_stop_loss = 0.0
        trade_entry_date = None
        
        # מעקב חלון 3 הימים
        window_active = False
        bars_since_trigger = 0
        tracked_high = 0.0
        pending_stop = 0.0
        pending_limit = 0.0
        pending_sl = 0.0
        
        dates = df.index
        n_bars = len(df)
        
        for i in range(201, n_bars):
            curr_date = dates[i]
            prev_date = dates[i-1]
            
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # ---------------------------------------------
            # א. ניהול פוזיציה פתוחה (Exits & Stops)
            # ---------------------------------------------
            if in_trade:
                # 1. בדיקת פגיעה בסטופ תוך-יומית
                if row['Low'] <= current_stop_loss:
                    exit_price = current_stop_loss - (params['slippage_cents'] / 100.0)
                    pnl = (exit_price - entry_price) * position_shares - (params['commission_per_share'] * position_shares * 2)
                    ret_pct = (exit_price / entry_price - 1.0) * 100.0
                    all_trades.append({
                        "Ticker": ticker,
                        "Entry_Date": trade_entry_date,
                        "Exit_Date": curr_date,
                        "Entry_Price": entry_price,
                        "Exit_Price": exit_price,
                        "Return_Pct": ret_pct,
                        "PnL": pnl,
                        "Exit_Reason": "Stop Loss / Breakeven",
                        "Holding_Days": (curr_date - trade_entry_date).days
                    })
                    in_trade = False
                    trade_stage = 0
                    continue
                
                # 2. ניהול שלב א' - מימוש 50% ב-RSI >= 60 והעלאה ל-Breakeven
                if trade_stage == 1 and row['RSI'] >= params['tp1_rsi']:
                    trade_stage = 2
                    current_stop_loss = entry_price  # Breakeven Stop
                    sold_shares = position_shares // 2
                    position_shares -= sold_shares
                    exit_price = row['Close'] - (params['slippage_cents'] / 100.0)
                    pnl_stage1 = (exit_price - entry_price) * sold_shares - (params['commission_per_share'] * sold_shares * 2)
                    ret_pct1 = (exit_price / entry_price - 1.0) * 100.0
                    all_trades.append({
                        "Ticker": ticker,
                        "Entry_Date": trade_entry_date,
                        "Exit_Date": curr_date,
                        "Entry_Price": entry_price,
                        "Exit_Price": exit_price,
                        "Return_Pct": ret_pct1,
                        "PnL": pnl_stage1,
                        "Exit_Reason": f"Stage 1 TP (RSI>={params['tp1_rsi']})",
                        "Holding_Days": (curr_date - trade_entry_date).days
                    })
                
                # 3. ניהול שלב ב' - יציאה משארית 50% ב-RSI >= 70 או Close < EMA 20
                if trade_stage == 2:
                    exit_cond = (row['RSI'] >= params['tp2_rsi']) or (row['Close'] < row['EMA20'])
                    if exit_cond:
                        exit_price = row['Close'] - (params['slippage_cents'] / 100.0)
                        pnl_stage2 = (exit_price - entry_price) * position_shares - (params['commission_per_share'] * position_shares * 2)
                        ret_pct2 = (exit_price / entry_price - 1.0) * 100.0
                        reason = f"Stage 2 (RSI>={params['tp2_rsi']})" if row['RSI'] >= params['tp2_rsi'] else "Stage 2 (Close < EMA20)"
                        all_trades.append({
                            "Ticker": ticker,
                            "Entry_Date": trade_entry_date,
                            "Exit_Date": curr_date,
                            "Entry_Price": entry_price,
                            "Exit_Price": exit_price,
                            "Return_Pct": ret_pct2,
                            "PnL": pnl_stage2,
                            "Exit_Reason": reason,
                            "Holding_Days": (curr_date - trade_entry_date).days
                        })
                        in_trade = False
                        trade_stage = 0
                        continue
            
            # ---------------------------------------------
            # ב. בדיקת ביצוע פקודת עבודה ממתינה (Execution)
            # ---------------------------------------------
            if not in_trade and window_active and bars_since_trigger in [1, 2, 3]:
                # בדיקה האם המחיר פרץ את מחיר ה-Stop
                if row['High'] >= pending_stop:
                    # בדיקת הגנת תקרת Limit (Cap)
                    if row['Open'] <= pending_limit:
                        # ביצוע: מחיר הפתיחה או מחיר ה-Stop (הגבוה מביניהם) + החלקה
                        fill_price = max(row['Open'], pending_stop) + (params['slippage_cents'] / 100.0)
                        if fill_price <= pending_limit:
                            in_trade = True
                            trade_stage = 1
                            entry_price = fill_price
                            current_stop_loss = pending_sl
                            trade_entry_date = curr_date
                            
                            # חישוב גודל פוזיציה לפי הקצאת הון קבועה ($10,000 לפוזיציה)
                            trade_capital = 10000.0
                            position_shares = int(trade_capital // entry_price)
                            window_active = False
                            continue
            
            # ---------------------------------------------
            # ג. עדכון מכונת המצבים בסגירת היום (EOD Evaluation)
            # ---------------------------------------------
            if not in_trade:
                # 1. פסילת סט-אפ פעיל (Invalidation)
                if window_active:
                    if (row['RSI'] >= params['rsi_invalidate']) or (row['Close'] <= row['SMA200']) or (not row['Macro_OK']):
                        window_active = False
                        bars_since_trigger = 99
                    else:
                        bars_since_trigger += 1
                        if bars_since_trigger > 3:
                            # פקיעה לאחר 3 ימים ללא מילוי
                            window_active = False
                
                # 2. זיהוי טריגר חדש (DAY 0) או איפוס מלא (Reset Engine)
                core_trigger = (
                    row['Macro_OK'] and
                    (row['Close'] > row['SMA200']) and
                    (row['RSI'] < params['rsi_trigger'])
                )
                
                if core_trigger:
                    # מקרה א': כניסה ראשונית ל-DAY 0
                    if not window_active and bars_since_trigger >= 4:
                        # בדיקת איפוס: ה-High היומי חייב להיות נמוך או שווה לשיא הקודם
                        if (tracked_high == 0.0) or (row['High'] <= tracked_high):
                            window_active = True
                            bars_since_trigger = 0
                            tracked_high = row['High']
                    elif not window_active:
                        window_active = True
                        bars_since_trigger = 0
                        tracked_high = row['High']
                
                # 3. חישוב פרמטרי פקודה ליום המסחר הבא במידה והחלון פעיל
                if window_active and bars_since_trigger <= 3:
                    ref_h = row['High']
                    # מדרגות אופסט
                    offset = max(0.10, ref_h * 0.001) if ref_h > 200.0 else 0.05
                    pending_stop = ref_h + offset
                    pending_limit = pending_stop * (1.0 + (params['limit_cap_pct'] / 100.0))
                    pending_sl = pending_stop * (1.0 - (params['hard_stop_pct'] / 100.0))
                    
                    # פילטר כרית ביטחון מגמתית (Trend Cushion Filter)
                    if params['use_trend_cushion'] and (pending_sl < row['SMA200']):
                        # הסטופ נופל מתחת ל-SMA200 -> לא מזינים פקודה
                        pass
    
    return pd.DataFrame(all_trades)

# ==========================================
# 3. סרגל צד - איקולייזר פרמטרים (UI Controls)
# ==========================================

st.sidebar.header("🎛️ איקולייזר פרמטרים מוסדי")

st.sidebar.subheader("1. שער כניסה וטכני")
p_rsi_trigger = st.sidebar.slider("טריגר RSI לפריקה (<)", min_value=25.0, max_value=45.0, value=40.0, step=0.5)
p_rsi_invalidate = st.sidebar.slider("שריפת מומנטום RSI (>=)", min_value=44.0, max_value=55.0, value=48.0, step=0.5)
p_sma_len = st.sidebar.number_input("ממוצע נע מגמתי (SMA)", value=200, step=10)

st.sidebar.subheader("2. ביצוע פקודות והגנות מחיר")
p_limit_cap = st.sidebar.select_slider(
    "תקרת Buy Stop-Limit Cap (%)",
    options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
    value=1.0,
    help="מגביל את מחיר המילוי המרבי מעל מחיר הטריגר. 1.0% הוא התקן ההדוק המקורי."
)
p_hard_stop = st.sidebar.slider("סטופ-לוס קשיח (%)", min_value=4.0, max_value=10.0, value=6.5, step=0.1)
p_trend_cushion = st.sidebar.checkbox("אכיפת 'כרית ביטחון' (SL > SMA200)", value=False, help="חוסם כניסה אם הסטופ נופל מתחת ל-SMA200")

st.sidebar.subheader("3. ניהול מימושים ויציאות (Two-Stage)")
p_tp1_rsi = st.sidebar.slider("יעד שלב א' (50% + Breakeven)", min_value=50.0, max_value=68.0, value=60.0, step=1.0)
p_tp2_rsi = st.sidebar.slider("יעד שלב ב' (שארית 50%)", min_value=65.0, max_value=85.0, value=70.0, step=1.0)
p_exit_ema = st.sidebar.number_input("ממוצע נע ליציאת שלב ב' (EMA)", value=20, step=5)

st.sidebar.subheader("4. מודל חיכוך שוק (Friction Engine)")
p_slippage_cents = st.sidebar.number_input("החלקה לעסקה (Cents/Share)", value=3.0, step=0.5)
p_comm_per_share = st.sidebar.number_input("עמלת ברוקר (USD/Share)", value=0.005, step=0.001, format="%.3f")

params = {
    "rsi_len": 14,
    "rsi_trigger": p_rsi_trigger,
    "rsi_invalidate": p_rsi_invalidate,
    "sma_len": p_sma_len,
    "limit_cap_pct": p_limit_cap,
    "hard_stop_pct": p_hard_stop,
    "use_trend_cushion": p_trend_cushion,
    "tp1_rsi": p_tp1_rsi,
    "tp2_rsi": p_tp2_rsi,
    "exit_ema": p_exit_ema,
    "slippage_cents": p_slippage_cents,
    "commission_per_share": p_comm_per_share
}

# ==========================================
# 4. גוף האפליקציה ותצוגת תוצאות
# ==========================================

st.title("🏛️ סימולטור מוסדי אינטראקטיבי - Strategy 2.1")
st.markdown("""
מערכת זו מבצעת סימולציה כמותית מלאה של אסטרטגיית הסווינג המוסדית על מניות **S&P 500 Top 50** ומדד **SPY** (2022–2026).
המנוע מיישם את **שעון 3 הימים**, מנגנון **איפוס DAY 0**, מדרגות ביצוע והגנת תקרת Limit, חיכוך מלא ויציאה דו-שלבית.
""")

col_date1, col_date2, col_btn = st.columns([2, 2, 2])
with col_date1:
    start_d = st.date_input("תאריך התחלה", datetime(2022, 1, 1))
with col_date2:
    end_d = st.date_input("תאריך סיום", datetime(2026, 9, 18))
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 הרץ סימולציה מלאה", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("טוען נתוני עומק מותאמים ומריץ מנוע סימולציה..."):
        stock_data = load_historical_data(TOP_50_TICKERS, start_d - timedelta(days=350), end_d)
        trades_df = run_simulation(stock_data, params)
        
    if trades_df.empty:
        st.warning("לא אותרו עסקאות בטווח הזמן ובפרמטרים שנבחרו.")
    else:
        # עיבוד מדדים מקצועיים
        total_trades = len(trades_df)
        wins = trades_df[trades_df['PnL'] > 0]
        losses = trades_df[trades_df['PnL'] <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
        
        trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
        equity_curve = trades_df['Cumulative_PnL']
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak)
        max_drawdown_usd = drawdown.min()
        
        avg_holding = trades_df['Holding_Days'].mean()
        avg_win_pct = wins['Return_Pct'].mean() if len(wins) > 0 else 0.0
        avg_loss_pct = losses['Return_Pct'].mean() if len(losses) > 0 else 0.0
        
        # תצוגת מדדי ביצוע בכירים
        st.markdown("### 📊 תוצאות מנוע הסימולציה")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)} מתוך {total_trades}")
        kpi2.metric("Profit Factor", f"{profit_factor:.2f}", "יעד > 3.5")
        kpi3.metric("סה\"כ רווח נקי (USD)", f"${trades_df['PnL'].sum():,.2f}")
        kpi4.metric("Max Drawdown (USD)", f"${max_drawdown_usd:,.2f}")
        kpi5.metric("זמן החזקה ממוצע", f"{avg_holding:.1f} ימים")
        
        st.divider()
        
        # גרף עקומת הון (Equity Curve)
        st.markdown("### 📈 עקומת צמיחת ההון (Cumulative Equity Curve)")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=trades_df['Exit_Date'], 
            y=trades_df['Cumulative_PnL'], 
            mode='lines',
            name='Net Portfolio PnL',
            line=dict(color='#00FFA3', width=2.5)
        ))
        fig_equity.update_layout(
            template="plotly_dark",
            xaxis_title="תאריך סגירת עסקה",
            yaxis_title="רווח מצטבר ($)",
            height=450,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # טבלאות פילוח ופרטי עסקאות
        tab_trades, tab_tickers, tab_reasons = st.tabs(["📋 יומן עסקאות מלא", "🏆 ביצועים לפי מניה", "🎯 התפלגות סיבות יציאה"])
        
        with tab_trades:
            st.dataframe(
                trades_df[['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Return_Pct', 'PnL', 'Exit_Reason', 'Holding_Days']].sort_values(by='Exit_Date', ascending=False),
                use_container_width=True
            )
            
        with tab_tickers:
            ticker_summary = trades_df.groupby('Ticker').agg(
                Trades=('PnL', 'count'),
                Win_Rate=('PnL', lambda x: (x > 0).mean() * 100.0),
                Total_PnL=('PnL', 'sum')
            ).sort_values(by='Total_PnL', ascending=False)
            st.dataframe(ticker_summary.style.format({"Win_Rate": "{:.1f}%", "Total_PnL": "${:,.2f}"}), use_container_width=True)
            
        with tab_reasons:
            reasons = trades_df['Exit_Reason'].value_counts()
            st.bar_chart(reasons)
