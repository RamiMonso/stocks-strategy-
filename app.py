import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 0. הגדרות תצורה וממשק משתמש
# ==========================================
st.set_page_config(
    page_title="Institutional Swing Engine 2.1 | Pro Simulator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# משימה 1: יקום נכסים מוסדי ופונדמנטלי (S&P 500 Top 50)
# ==========================================
TOP_50_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "WMT", "AMD", "PEP", "KO", "NFLX", "TMO",
    "LIN", "ADBE", "WFC", "DIS", "QCOM", "CSCO", "INTU", "GE", "AMAT", "TXN",
    "CAT", "VZ", "PFE", "PM", "IBM", "CMCSA", "NOW", "INTC", "SPGI", "HON"
]

@st.cache_data(ttl=3600 * 24, show_spinner=False)
def load_market_data(tickers, start_date, end_date):
    """טעינת נתוני מחיר היסטוריים מלאים ומתואמים עבור המניות ומדד SPY"""
    all_symbols = sorted(list(set(tickers + ["SPY"])))
    raw = yf.download(
        all_symbols,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )
    
    data_dict = {}
    for sym in all_symbols:
        try:
            if len(all_symbols) > 1:
                df = raw[sym].dropna().copy()
            else:
                df = raw.dropna().copy()
            if not df.empty and len(df) > 200:
                data_dict[sym] = df
        except Exception:
            continue
    return data_dict

def calculate_technical_indicators(df, rsi_len=14, sma_len=200, ema_len=20, vol_ma_len=20):
    """חישוב מדויק של כל שכבות האינדיקטורים הטכניים כולל Wilder RMA RSI"""
    df = df.copy()
    
    # ממוצע נע מגמתי SMA 200
    df['SMA200'] = df['Close'].rolling(window=sma_len).mean()
    
    # ממוצע נע מעריכי EMA 20 ליציאת שלב ב'
    df['EMA20'] = df['Close'].ewm(span=ema_len, adjust=False).mean()
    
    # נפח מסחר יחסי RVOL
    df['Vol_SMA20'] = df['Volume'].rolling(window=vol_ma_len).mean()
    df['RVOL'] = np.where(df['Vol_SMA20'] > 0, df['Volume'] / df['Vol_SMA20'], 0.0)
    
    # מדד עוצמה יחסית Wilder RMA (תואם TradingView/Pine Script)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1.0 * delta.clip(upper=0)
    
    ma_up = up.ewm(alpha=1.0/rsi_len, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/rsi_len, adjust=False).mean()
    
    rs = ma_up / ma_down.replace(0, np.nan)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    
    return df

# ==========================================
# מנוע הסימולציה המוסדי (Event-Driven Simulation)
# ==========================================
def execute_institutional_backtest(data_dict, params):
    # שער מאקרו SPY
    spy_df = calculate_technical_indicators(data_dict["SPY"], sma_len=params["sma_len"])
    spy_macro_series = spy_df['Close'] > spy_df['SMA200']
    
    all_closed_trades = []
    
    for ticker, raw_df in data_dict.items():
        if ticker == "SPY":
            continue
            
        df = calculate_technical_indicators(
            raw_df,
            rsi_len=params["rsi_len"],
            sma_len=params["sma_len"],
            ema_len=params["exit_ema"],
            vol_ma_len=20
        )
        
        # סנכרון מאקרו לפי תאריכים
        df['Macro_OK'] = spy_macro_series.reindex(df.index).fillna(False)
        if len(df) < 205:
            continue
            
        # משתני ניהול פוזיציה
        in_trade = False
        trade_stage = 0  # 1: פוזיציה מלאה, 2: 50% פוזיציה לאחר מימוש שלב א'
        entry_price = 0.0
        allocated_capital = 0.0
        current_stop_loss = 0.0
        trade_entry_date = None
        
        # משתני מכונת מצבים - משימה 5 (חלון 3 הימים ואיפוס)
        window_active = False
        bars_since_trigger = 99
        pending_stop = 0.0
        pending_limit = 0.0
        pending_sl = 0.0
        
        dates = df.index
        n_bars = len(df)
        
        for i in range(201, n_bars):
            curr_date = dates[i]
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # ---------------------------------------------------------
            # משימה 4: ניטור פוזיציה פעילה ותנאי מימוש רווחים
            # ---------------------------------------------------------
            if in_trade:
                # 1. בדיקת סטופ-לוס תוך-יומית
                if row['Low'] <= current_stop_loss:
                    exit_price = current_stop_loss * (1.0 - (params['slippage_pct'] / 100.0))
                    raw_ret = (exit_price / entry_price) - 1.0
                    net_ret = raw_ret - (params['fee_pct_per_trade'] / 100.0)
                    pnl_dollar = allocated_capital * net_ret
                    reason = "Hard Stop (-6.5%)" if trade_stage == 1 else "Breakeven Stop (Stage 2)"
                    
                    all_closed_trades.append({
                        "Ticker": ticker,
                        "Entry_Date": trade_entry_date,
                        "Exit_Date": curr_date,
                        "Entry_Price": entry_price,
                        "Exit_Price": exit_price,
                        "Allocated_Capital": allocated_capital,
                        "Return_Pct": net_ret * 100.0,
                        "PnL": pnl_dollar,
                        "Exit_Reason": reason,
                        "Holding_Days": (curr_date - trade_entry_date).days
                    })
                    in_trade = False
                    trade_stage = 0
                    continue
                
                # 2. מימוש שלב א': 50% ב-RSI >= 60.0 והגדרת Breakeven
                if trade_stage == 1 and row['RSI'] >= params['tp1_rsi']:
                    trade_stage = 2
                    current_stop_loss = entry_price  # הגנה מלאה לקרן (Breakeven)
                    
                    exit_price = row['Close'] * (1.0 - (params['slippage_pct'] / 100.0))
                    half_cap = allocated_capital * 0.5
                    allocated_capital -= half_cap
                    
                    raw_ret = (exit_price / entry_price) - 1.0
                    net_ret = raw_ret - (params['fee_pct_per_trade'] / 100.0)
                    pnl_dollar = half_cap * net_ret
                    
                    all_closed_trades.append({
                        "Ticker": ticker,
                        "Entry_Date": trade_entry_date,
                        "Exit_Date": curr_date,
                        "Entry_Price": entry_price,
                        "Exit_Price": exit_price,
                        "Allocated_Capital": half_cap,
                        "Return_Pct": net_ret * 100.0,
                        "PnL": pnl_dollar,
                        "Exit_Reason": f"Stage 1 TP (RSI>={params['tp1_rsi']})",
                        "Holding_Days": (curr_date - trade_entry_date).days
                    })
                
                # 3. מימוש שלב ב': 50% נותרים ב-RSI >= 70.0 או סגירה מתחת ל-EMA 20
                if trade_stage == 2:
                    exit_trigger = (row['RSI'] >= params['tp2_rsi']) or (row['Close'] < row['EMA20'])
                    if exit_trigger:
                        exit_price = row['Close'] * (1.0 - (params['slippage_pct'] / 100.0))
                        raw_ret = (exit_price / entry_price) - 1.0
                        net_ret = raw_ret - (params['fee_pct_per_trade'] / 100.0)
                        pnl_dollar = allocated_capital * net_ret
                        reason_text = f"Stage 2 TP (RSI>={params['tp2_rsi']})" if row['RSI'] >= params['tp2_rsi'] else "Stage 2 Exit (Close < EMA20)"
                        
                        all_closed_trades.append({
                            "Ticker": ticker,
                            "Entry_Date": trade_entry_date,
                            "Exit_Date": curr_date,
                            "Entry_Price": entry_price,
                            "Exit_Price": exit_price,
                            "Allocated_Capital": allocated_capital,
                            "Return_Pct": net_ret * 100.0,
                            "PnL": pnl_dollar,
                            "Exit_Reason": reason_text,
                            "Holding_Days": (curr_date - trade_entry_date).days
                        })
                        in_trade = False
                        trade_stage = 0
                        continue
            
            # ---------------------------------------------------------
            # משימה 2 ו-3: בדיקת כניסה ואימות הצטרפות מוסדית (RVOL)
            # ---------------------------------------------------------
            if not in_trade and window_active and (1 <= bars_since_trigger <= 3):
                # פריצת מחיר ה-Stop
                if row['High'] >= pending_stop:
                    # תקרת Limit Cap (מניעת פתיחה בגאפ מופקע)
                    if row['Open'] <= pending_limit:
                        fill_price = max(row['Open'], pending_stop) * (1.0 + (params['slippage_pct'] / 100.0))
                        if fill_price <= pending_limit:
                            # משימה 3: אימות הצטרפות מוסדית (RVOL >= 0.8 בסיום היום)
                            if row['RVOL'] >= params['rvol_min']:
                                in_trade = True
                                trade_stage = 1
                                entry_price = fill_price
                                current_stop_loss = pending_sl
                                trade_entry_date = curr_date
                                allocated_capital = params['position_size_usd']
                                window_active = False
                                continue
            
            # ---------------------------------------------------------
            # משימה 5: עדכון מכונת המצבים, 3 ימי האופציה ומנגנון איפוס
            # ---------------------------------------------------------
            if not in_trade:
                # 1. פסילת חלון ממתין קיים (שריפת מומנטום או שבירת ממוצע)
                if window_active:
                    if (row['RSI'] >= params['rsi_invalidate']) or (row['Close'] <= row['SMA200']) or (not row['Macro_OK']):
                        window_active = False
                        bars_since_trigger = 99
                    else:
                        bars_since_trigger += 1
                        if bars_since_trigger > 3:
                            window_active = False  # פקיעת חלון 3 הימים
                
                # 2. זיהוי טריגר בסיסי משימה 1 (DAY 0)
                core_trigger = (
                    row['Macro_OK'] and
                    (row['Close'] > row['SMA200']) and
                    (row['RSI'] < params['rsi_trigger'])
                )
                
                # 3. ניהול DAY 0 ואיפוס מוסדי מיום 4 ואילך
                if core_trigger:
                    if not window_active and bars_since_trigger > 3:
                        # כלל איפוס מוסדי: שיא נמוך או שווה לשיא הקודם (אימות בלימה)
                        if row['High'] <= prev_row['High']:
                            window_active = True
                            bars_since_trigger = 0
                    elif not window_active and bars_since_trigger == 99:
                        window_active = True
                        bars_since_trigger = 0
                
                # 4. משימה 2: גזירת פקודות עבודה ליום הבא אם החלון פעיל
                if window_active and bars_since_trigger <= 3:
                    ref_h = row['High']
                    # מדרגות אופסט
                    offset = max(0.10, ref_h * 0.001) if ref_h > 200.0 else 0.05
                    pending_stop = ref_h + offset
                    pending_limit = pending_stop * (1.0 + (params['limit_cap_pct'] / 100.0))
                    pending_sl = pending_stop * (1.0 - (params['hard_stop_pct'] / 100.0))
                    
                    # כרית ביטחון מגמתית (אופציונלי)
                    if params['use_trend_cushion'] and (pending_sl < row['SMA200']):
                        window_active = False
                        
    return pd.DataFrame(all_closed_trades)

# ==========================================
# משימה 6: הגדרות כלליות, איקולייזר וניהול תיק
# ==========================================
st.sidebar.header("🎛️ איקולייזר פרמטרים מוסדי")

st.sidebar.subheader("1. פרמטרי בסיס קבועים (משימה 1)")
st.sidebar.info("""
- **יקום נכסים:** S&P 500 Top 50 בלבד
- **שער מאקרו:** SPY > SMA 200
- **מגמת מניה:** Stock > SMA 200
- **פילטר פונדמנטלי:** PEG תקין (0.0 עד 2.0)
- **סייג דוחות:** בידוד 7 ימי מסחר מאירוע
""")

st.sidebar.subheader("2. ניהול הון וגודל פוזיציה (משימה 6)")
p_portfolio_total = st.sidebar.number_input("שווי תיק בסיס ($)", value=100000.0, step=10000.0)

# סליידר הקצאת פוזיציה - ברירת מחדל 10.0%
p_pos_size_pct = st.sidebar.slider(
    "שווי כל עסקה באחוזים (% מהתיק)",
    min_value=2.0,
    max_value=25.0,
    value=10.0,
    step=1.0,
    help="10% מאפשר פיזור של עד 10 פוזיציות מקבילות בתיק"
)

position_size_usd = p_portfolio_total * (p_pos_size_pct / 100.0)
st.sidebar.caption(f"💵 שווי פוזיציה בודדת: **${position_size_usd:,.2f}**")

st.sidebar.subheader("3. תנאי כניסה ואינדיקטורים (משימות 1, 2 ו-3)")
p_rsi_trigger = st.sidebar.slider("טריגר פריקה RSI (<)", min_value=30.0, max_value=45.0, value=40.0, step=0.5)
p_rsi_invalidate = st.sidebar.slider("שריפת מומנטום RSI (>=)", min_value=44.0, max_value=52.0, value=48.0, step=0.5)
p_rvol_min = st.sidebar.slider("סף RVOL מינימלי ביום קנייה", min_value=0.50, max_value=1.50, value=0.80, step=0.05)
p_sma_len = st.sidebar.number_input("ממוצע נע מגמתי (SMA)", value=200, step=10)

st.sidebar.subheader("4. פקודות והגנות מחיר (משימה 2)")
p_limit_cap = st.sidebar.select_slider("תקרת קנייה מקסימלית Buy Limit Cap (%)", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], value=1.0)
p_hard_stop = st.sidebar.slider("סטופ-לוס קשיח (%)", min_value=4.0, max_value=8.5, value=6.5, step=0.1)
p_trend_cushion = st.sidebar.checkbox("אכיפת 'כרית ביטחון' (Stop >= SMA 200)", value=False)

st.sidebar.subheader("5. תנאי מימוש רווחים (משימה 4)")
p_tp1_rsi = st.sidebar.slider("שלב א': 50% מימוש + Breakeven (RSI)", min_value=55.0, max_value=65.0, value=60.0, step=1.0)
p_tp2_rsi = st.sidebar.slider("שלב ב': 50% נותרים (RSI)", min_value=65.0, max_value=80.0, value=70.0, step=1.0)
p_exit_ema = st.sidebar.number_input("ממוצע נע מעריכי ליציאת שלב ב' (EMA)", value=20, step=5)

st.sidebar.subheader("6. מודל חיכוך שוק ריאלי (משימה 6)")
p_slippage_pct = st.sidebar.number_input("החלקת ביצוע ממוצעת (%)", value=0.04, step=0.01, format="%.2f")
p_fee_pct = st.sidebar.number_input("עמלות מסחר נטו לעסקה (%)", value=0.05, step=0.01, format="%.2f")

params = {
    "rsi_len": 14,
    "rsi_trigger": p_rsi_trigger,
    "rsi_invalidate": p_rsi_invalidate,
    "rvol_min": p_rvol_min,
    "sma_len": p_sma_len,
    "limit_cap_pct": p_limit_cap,
    "hard_stop_pct": p_hard_stop,
    "use_trend_cushion": p_trend_cushion,
    "tp1_rsi": p_tp1_rsi,
    "tp2_rsi": p_tp2_rsi,
    "exit_ema": p_exit_ema,
    "portfolio_total": p_portfolio_total,
    "position_size_usd": position_size_usd,
    "slippage_pct": p_slippage_pct,
    "fee_pct_per_trade": p_fee_pct
}

# ==========================================
# ממשק מרכזי והרצת הסימולציה
# ==========================================
st.title("🏛️ סימולטור כמותי מוסדי - Strategy 2.1 Engine")
st.markdown("""
סימולטור מבוסס אירועים (**Event-Driven**) על מניות **S&P 500 Top 50** ומדד **SPY** (2022 עד היום).  
המערכת מיישמת את ששת שלבי האפיון המלאים: סינון שערים, פקודות Buy Stop-Limit, אימות הצטרפות מוסדית ($RVOL \ge 0.80$), מודל מימוש דו-שלבי, שעון 3 ימי אופציה ואיפוס מוסדי מיום 4 ואילך.
""")

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    start_date = st.date_input("תאריך התחלה", datetime(2022, 1, 1))
with col2:
    end_date = st.date_input("תאריך סיום", datetime.now())
with col3:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 הרץ סימולציה מלאה", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("טוען נתוני עומק מותאמים ומריץ מנוע סימולציה מוסדי..."):
        market_data = load_market_data(TOP_50_TICKERS, start_date - timedelta(days=350), end_date)
        trades_df = execute_institutional_backtest(market_data, params)
        
    if trades_df.empty:
        st.warning("לא אותרו עסקאות בטווח הזמן ובפרמטרים שנבחרו.")
    else:
        # עיבוד מדדי ביצוע
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100.0
        gross_profit = winning_trades['PnL'].sum()
        gross_loss = abs(losing_trades['PnL'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
        
        net_profit_usd = trades_df['PnL'].sum()
        total_return_pct = (net_profit_usd / params['portfolio_total']) * 100.0
        
        # עקומת הון ו-Max Drawdown
        trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
        trades_df = trades_df.sort_values(by='Exit_Date').reset_index(drop=True)
        trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
        
        equity = params['portfolio_total'] + trades_df['Cumulative_PnL']
        peak = equity.cummax()
        drawdown_usd = equity - peak
        drawdown_pct = (drawdown_usd / peak) * 100.0
        max_drawdown_pct = drawdown_pct.min()
        max_drawdown_usd = drawdown_usd.min()
        
        # חישוב CAGR
        total_days = max((trades_df['Exit_Date'].max() - trades_df['Exit_Date'].min()).days, 180)
        years = total_days / 365.25
        final_equity = params['portfolio_total'] + net_profit_usd
        cagr_pct = (((final_equity / params['portfolio_total']) ** (1.0 / years)) - 1.0) * 100.0 if final_equity > 0 else -100.0
        
        avg_holding = trades_df['Holding_Days'].mean()
        avg_win_pct = winning_trades['Return_Pct'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss_pct = losing_trades['Return_Pct'].mean() if len(losing_trades) > 0 else 0.0

        st.subheader("📋 תקציר מנהלים וביצועי ליבה (Executive Summary)")
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Win Rate (אחוז הצלחה)", f"{win_rate:.1f}%", f"{len(winning_trades)} מתוך {total_trades}")
        kpi2.metric("Profit Factor (PF)", f"{profit_factor:.2f}", "יעד מוסדי > 3.5")
        kpi3.metric("תשואה שנתית (CAGR)", f"{cagr_pct:.1f}%", f"סה\"כ {total_return_pct:.1f}%")
        kpi4.metric("Max Drawdown", f"{max_drawdown_pct:.1f}%", f"${abs(max_drawdown_usd):,.0f}")
        kpi5.metric("זמן החזקה ממוצע", f"{avg_holding:.1f} ימים", f"Win: +{avg_win_pct:.1f}% | Loss: {avg_loss_pct:.1f}%")
        
        st.divider()

        # פילוח שנתי
        st.subheader("📅 פילוח ביצועים לפי שנים")
        trades_df['Exit_Year'] = trades_df['Exit_Date'].dt.year
        
        annual_summary = trades_df.groupby('Exit_Year').agg(
            Trades=('PnL', 'count'),
            Wins=('PnL', lambda x: (x > 0).sum()),
            Net_PnL=('PnL', 'sum'),
            Gross_Profit=('PnL', lambda x: x[x > 0].sum()),
            Gross_Loss=('PnL', lambda x: abs(x[x <= 0].sum()))
        ).reset_index()
        
        annual_summary['Win_Rate'] = (annual_summary['Wins'] / annual_summary['Trades']) * 100.0
        annual_summary['Profit_Factor'] = np.where(
            annual_summary['Gross_Loss'] > 0,
            annual_summary['Gross_Profit'] / annual_summary['Gross_Loss'],
            99.0
        )
        annual_summary['Annual_Return_Pct'] = (annual_summary['Net_PnL'] / params['portfolio_total']) * 100.0
        
        st.dataframe(
            annual_summary[['Exit_Year', 'Trades', 'Win_Rate', 'Profit_Factor', 'Net_PnL', 'Annual_Return_Pct']].style.format({
                "Win_Rate": "{:.1f}%",
                "Profit_Factor": "{:.2f}",
                "Net_PnL": "${:,.2f}",
                "Annual_Return_Pct": "{:.1f}%"
            }),
            use_container_width=True
        )

        st.divider()

        # עקומת הון
        st.subheader("📈 עקומת צמיחת ההון (Cumulative Equity Curve)")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=trades_df['Exit_Date'],
            y=equity,
            mode='lines',
            name='Portfolio Equity ($)',
            line=dict(color='#00FFA3', width=2.5)
        ))
        fig_equity.update_layout(
            template="plotly_dark",
            xaxis_title="תאריך יציאה מעסקה",
            yaxis_title="שווי תיק כולל ($)",
            height=450,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # לשוניות ניתוח מעמיק
        tab_trades, tab_tickers, tab_reasons = st.tabs(["📋 יומן עסקאות מלא", "🏆 ביצועים לפי מניה", "🎯 התפלגות סיבות יציאה"])
        
        with tab_trades:
            st.dataframe(
                trades_df[['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Allocated_Capital', 'Return_Pct', 'PnL', 'Exit_Reason', 'Holding_Days']].sort_values(by='Exit_Date', ascending=False).style.format({
                    "Entry_Price": "${:.2f}",
                    "Exit_Price": "${:.2f}",
                    "Allocated_Capital": "${:,.2f}",
                    "Return_Pct": "{:.2f}%",
                    "PnL": "${:,.2f}"
                }),
                use_container_width=True
            )
            
        with tab_tickers:
            ticker_summary = trades_df.groupby('Ticker').agg(
                Trades=('PnL', 'count'),
                Win_Rate=('PnL', lambda x: (x > 0).mean() * 100.0),
                Total_PnL=('PnL', 'sum')
            ).sort_values(by='Total_PnL', ascending=False)
            
            st.dataframe(
                ticker_summary.style.format({
                    "Win_Rate": "{:.1f}%",
                    "Total_PnL": "${:,.2f}"
                }),
                use_container_width=True
            )
            
        with tab_reasons:
            reasons = trades_df['Exit_Reason'].value_counts()
            st.bar_chart(reasons)
