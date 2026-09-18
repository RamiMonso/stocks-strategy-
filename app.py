import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =====================================================================
# 0. הגדרות תצורה וממשק Streamlit
# =====================================================================
st.set_page_config(
    page_title="Institutional Swing Engine 2.1 | Portfolio Simulator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# משימה 1: יקום הנכסים (S&P 500 Top 50)
# =====================================================================
TOP_50_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "WMT", "AMD", "PEP", "KO", "NFLX", "TMO",
    "LIN", "ADBE", "WFC", "DIS", "QCOM", "CSCO", "INTU", "GE", "AMAT", "TXN",
    "CAT", "VZ", "PFE", "PM", "IBM", "CMCSA", "NOW", "INTC", "SPGI", "HON"
]

@st.cache_data(ttl=3600 * 24, show_spinner=False)
def load_all_market_data(tickers, start_date, end_date):
    """טעינת נתונים היסטוריים מתואמים ומלאים עבור המניות ומדד הייחוס SPY"""
    symbols = sorted(list(set(tickers + ["SPY"])))
    raw = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by='ticker',
        progress=False
    )
    
    data_dict = {}
    for sym in symbols:
        try:
            if len(symbols) > 1:
                df = raw[sym].dropna().copy()
            else:
                df = raw.dropna().copy()
            if not df.empty and len(df) > 200:
                data_dict[sym] = df
        except Exception:
            continue
    return data_dict

def calculate_technical_indicators(df, rsi_len=14, sma_len=200, ema_len=20, vol_len=20):
    """חישוב מדדי ניתוח טכני ברמת דיוק מוסדית כולל Wilder's RMA RSI"""
    df = df.copy()
    
    # מגמה ראשית SMA 200 ומגמה מהירה EMA 20
    df['SMA200'] = df['Close'].rolling(window=sma_len).mean()
    df['EMA20'] = df['Close'].ewm(span=ema_len, adjust=False).mean()
    
    # נפח מסחר יחסי RVOL
    df['Vol_SMA20'] = df['Volume'].rolling(window=vol_len).mean()
    df['RVOL'] = np.where(df['Vol_SMA20'] > 0, df['Volume'] / df['Vol_SMA20'], 0.0)
    
    # חישוב RSI(14) לפי Wilder RMA המקורי
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1.0 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1.0 / rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_len, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    
    return df

@st.cache_data(ttl=3600 * 24, show_spinner=False)
def fetch_earnings_calendar(tickers):
    """משיכת לוחות דוחות כספיים לצורך סייג 7 ימי מסחר"""
    earnings_dict = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            ed = tk.get_earnings_dates(limit=40)
            if ed is not None and not ed.empty:
                earnings_dict[t] = ed.index.tz_localize(None).normalize()
            else:
                earnings_dict[t] = pd.to_datetime([])
        except Exception:
            earnings_dict[t] = pd.to_datetime([])
    return earnings_dict

def is_blackout_period(trade_date, earnings_dates):
    """בדיקה האם הנר נמצא בטווח 7 ימי מסחר (כ-10 ימים קלנדריים) לפני דוח"""
    for ed in earnings_dates:
        delta = (ed - trade_date).days
        if 0 <= delta <= 10:
            return True
    return False

# =====================================================================
# מנוע הסימולציה המוסדי ברמת תיק (Portfolio-Level State Machine)
# =====================================================================
def run_portfolio_simulation(data_dict, earnings_dict, params):
    # שער מאקרו SPY
    spy_df = calculate_technical_indicators(data_dict["SPY"], sma_len=params["sma_len"])
    spy_macro = spy_df['Close'] > spy_df['SMA200']
    
    # הכנת נתוני כל המניות
    processed_stocks = {}
    for ticker, raw_df in data_dict.items():
        if ticker == "SPY":
            continue
        df = calculate_technical_indicators(
            raw_df,
            rsi_len=params["rsi_len"],
            sma_len=params["sma_len"],
            ema_len=params["exit_ema"],
            vol_len=20
        )
        df['Macro_OK'] = spy_macro.reindex(df.index).fillna(False)
        e_dates = earnings_dict.get(ticker, pd.to_datetime([]))
        df['Near_Earnings'] = [is_blackout_period(d, e_dates) for d in df.index]
        
        # שער טריגר בסיסי משימה 1
        df['Core_Trigger'] = (
            df['Macro_OK'] &
            (df['Close'] > df['SMA200']) &
            (df['RSI'] < params['rsi_trigger']) &
            (~df['Near_Earnings'])
        )
        processed_stocks[ticker] = df

    # איחוד כל תאריכי המסחר מ-2022
    all_dates = sorted(list(set.union(*[set(df.loc[df.index >= params['start_date']].index) for df in processed_stocks.values()])))
    
    # מצב תיק כולל
    cash = params['initial_capital']
    max_positions = int(100.0 / params['pos_size_pct'])
    allocated_per_position = params['initial_capital'] * (params['pos_size_pct'] / 100.0)
    
    open_positions = {}  # ticker: dict
    pending_orders = {}  # ticker: dict
    closed_trades = []
    daily_equity_history = []
    
    for curr_date in all_dates:
        # -------------------------------------------------------------
        # 1. ניטור פוזיציות פעילות (סטופ לוס, מימוש א', מימוש ב')
        # -------------------------------------------------------------
        active_tickers = list(open_positions.keys())
        for ticker in active_tickers:
            pos = open_positions[ticker]
            df = processed_stocks[ticker]
            if curr_date not in df.index:
                continue
            row = df.loc[curr_date]
            
            entry_p = pos['entry_price']
            stage = pos['stage']
            cur_sl = pos['stop_loss']
            
            # בדיקת פגיעה בסטופ-לוס (תוך-יומי מול Low)
            if row['Low'] <= cur_sl:
                exit_p = cur_sl * (1.0 - (params['slippage_pct'] / 100.0))
                net_ret = (exit_p / entry_p - 1.0) - (params['fee_pct'] / 100.0)
                freed_cap = pos['remaining_capital']
                pnl = freed_cap * net_ret
                cash += (freed_cap + pnl)
                
                reason = "Hard Stop (-6.5%)" if stage == 1 else "Breakeven Stop (Stage 2)"
                closed_trades.append({
                    "Ticker": ticker, "Entry_Date": pos['entry_date'], "Exit_Date": curr_date,
                    "Stage": f"Exit at Stage {stage}", "Entry_P": entry_p, "Exit_P": exit_p,
                    "Return_%": net_ret * 100.0, "PnL_$": pnl, "Reason": reason,
                    "Days": (curr_date - pos['entry_date']).days
                })
                del open_positions[ticker]
                continue
            
            # מימוש שלב א': RSI >= 60 (50% כמות והעלאה ל-Breakeven)
            if stage == 1 and row['RSI'] >= params['tp1_rsi']:
                pos['stage'] = 2
                pos['stop_loss'] = entry_p  # Breakeven
                
                exit_p = row['Close'] * (1.0 - (params['slippage_pct'] / 100.0))
                half_cap = pos['remaining_capital'] * 0.5
                pos['remaining_capital'] -= half_cap
                
                net_ret = (exit_p / entry_p - 1.0) - (params['fee_pct'] / 100.0)
                pnl = half_cap * net_ret
                cash += (half_cap + pnl)
                
                closed_trades.append({
                    "Ticker": ticker, "Entry_Date": pos['entry_date'], "Exit_Date": curr_date,
                    "Stage": "Stage 1 (50% TP)", "Entry_P": entry_p, "Exit_P": exit_p,
                    "Return_%": net_ret * 100.0, "PnL_$": pnl,
                    "Reason": f"RSI 60 TP (RSI={row['RSI']:.1f})", "Days": (curr_date - pos['entry_date']).days
                })
                
            # מימוש שלב ב': RSI >= 70 או Close < EMA 20
            if stage == 2:
                if (row['RSI'] >= params['tp2_rsi']) or (row['Close'] < row['EMA20']):
                    exit_p = row['Close'] * (1.0 - (params['slippage_pct'] / 100.0))
                    net_ret = (exit_p / entry_p - 1.0) - (params['fee_pct'] / 100.0)
                    freed_cap = pos['remaining_capital']
                    pnl = freed_cap * net_ret
                    cash += (freed_cap + pnl)
                    
                    reason = "RSI 70 TP" if row['RSI'] >= params['tp2_rsi'] else "Close < EMA20"
                    closed_trades.append({
                        "Ticker": ticker, "Entry_Date": pos['entry_date'], "Exit_Date": curr_date,
                        "Stage": "Stage 2 (Final)", "Entry_P": entry_p, "Exit_P": exit_p,
                        "Return_%": net_ret * 100.0, "PnL_$": pnl, "Reason": reason,
                        "Days": (curr_date - pos['entry_date']).days
                    })
                    del open_positions[ticker]
                    continue
        
        # -------------------------------------------------------------
        # 2. בדיקת ביצוע פקודות Buy Stop-Limit ממתינות
        # -------------------------------------------------------------
        eligible_fills = []
        pending_tickers = list(pending_orders.keys())
        
        for ticker in pending_tickers:
            if ticker in open_positions:
                del pending_orders[ticker]
                continue
                
            p_order = pending_orders[ticker]
            p_order['bars_active'] += 1
            df = processed_stocks[ticker]
            if curr_date not in df.index:
                continue
            row = df.loc[curr_date]
            
            # בדיקת תנאי פריצה תוך-יומית
            if row['High'] >= p_order['buy_stop']:
                if row['Open'] <= p_order['limit_cap']:
                    fill_p = max(row['Open'], p_order['buy_stop']) * (1.0 + (params['slippage_pct'] / 100.0))
                    if fill_p <= p_order['limit_cap']:
                        # אימות הצטרפות מוסדית: RVOL >= סף מינימום בסיום היום
                        if row['RVOL'] >= params['rvol_min']:
                            eligible_fills.append({
                                "ticker": ticker,
                                "fill_price": fill_p,
                                "initial_sl": p_order['initial_sl'],
                                "rvol": row['RVOL'],
                                "rsi": row['RSI']
                            })
                            del pending_orders[ticker]
                            continue
            
            # פקיעת חלון 3 ימים או ביטול עקב שריפת מומנטום / שבירת מגמה
            if (p_order['bars_active'] >= 3) or (row['RSI'] >= params['rsi_invalidate']) or (row['Close'] <= row['SMA200']) or (not row['Macro_OK']):
                del pending_orders[ticker]

        # הקצאת הון לפקודות שנתפסו לפי קיבולת התיק
        for fill in eligible_fills:
            t = fill['ticker']
            if len(open_positions) < max_positions and cash >= allocated_per_position:
                cash -= allocated_per_position
                open_positions[t] = {
                    "entry_date": curr_date,
                    "entry_price": fill['fill_price'],
                    "stage": 1,
                    "stop_loss": fill['initial_sl'],
                    "remaining_capital": allocated_per_position
                }
        
        # -------------------------------------------------------------
        # 3. סריקת נרות DAY 0 בנעילת יום והגדרת פקודות חדשות / איפוס
        # -------------------------------------------------------------
        for ticker, df in processed_stocks.items():
            if ticker in open_positions or ticker in pending_orders:
                continue
            if curr_date not in df.index:
                continue
            idx = df.index.get_loc(curr_date)
            if idx < 201:
                continue
                
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            
            if row['Core_Trigger']:
                ref_h = row['High']
                offset = max(0.10, ref_h * 0.001) if ref_h > 200.0 else 0.05
                b_stop = ref_h + offset
                l_cap = b_stop * (1.0 + (params['limit_cap_pct'] / 100.0))
                i_sl = b_stop * (1.0 - (params['hard_stop_pct'] / 100.0))
                
                # כרית ביטחון מגמתית (אופציונלי)
                if params['use_trend_cushion'] and (i_sl < row['SMA200']):
                    continue
                    
                pending_orders[ticker] = {
                    "setup_date": curr_date,
                    "buy_stop": b_stop,
                    "limit_cap": l_cap,
                    "initial_sl": i_sl,
                    "bars_active": 0
                }
        
        # חישוב שווי התיק היומי (Mark-to-Market)
        current_positions_value = 0.0
        for t, pos in open_positions.items():
            c_price = processed_stocks[t].loc[curr_date, 'Close']
            unrealized_ret = (c_price / pos['entry_price']) - 1.0
            current_positions_value += pos['remaining_capital'] * (1.0 + unrealized_ret)
            
        total_day_equity = cash + current_positions_value
        daily_equity_history.append({"Date": curr_date, "Equity": total_day_equity})

    return pd.DataFrame(closed_trades), pd.DataFrame(daily_equity_history)

# =====================================================================
# משימה 6: ממשק איקולייזר וניהול פרמטרים ב-Sidebar
# =====================================================================
st.sidebar.header("🎛️ איקולייזר מוסדי | Strategy 2.1")

st.sidebar.subheader("1. מבנה התיק והקצאות (Portfolio Rules)")
p_capital = st.sidebar.number_input("שווי תיק בסיס ($)", value=100000.0, step=10000.0)
p_pos_size = st.sidebar.slider("הקצאה לכל עסקה (% מהתיק)", min_value=2.0, max_value=25.0, value=10.0, step=1.0)
st.sidebar.caption(f"סך פוזיציות מקבילות מקסימלי: **{int(100.0 / p_pos_size)}** | ${p_capital * (p_pos_size / 100.0):,.0f} לעסקה")

st.sidebar.subheader("2. תנאי כניסה ואינדיקטורים")
p_rsi_trig = st.sidebar.slider("טריגר פריקה RSI (<)", min_value=30.0, max_value=45.0, value=40.0, step=0.5)
p_rsi_inval = st.sidebar.slider("שריפת מומנטום RSI (>=)", min_value=44.0, max_value=55.0, value=48.0, step=0.5)
p_rvol_min = st.sidebar.slider("סף RVOL מינימלי ביום כניסה", min_value=0.50, max_value=1.50, value=0.80, step=0.05)
p_sma_len = st.sidebar.number_input("ממוצע נע מגמתי (SMA)", value=200, step=10)

st.sidebar.subheader("3. הגנת פקודות וכרית ביטחון")
p_limit_cap = st.sidebar.select_slider("תקרת Limit Cap מעל ה-Stop (%)", options=[0.5, 1.0, 1.5, 2.0], value=1.0)
p_hard_stop = st.sidebar.slider("סטופ-לוס קשיח התחלתי (%)", min_value=4.0, max_value=8.0, value=6.5, step=0.1)
p_trend_cushion = st.sidebar.checkbox("אכיפת 'כרית ביטחון' (Stop > SMA 200)", value=False)

st.sidebar.subheader("4. תנאי מימוש רווחים")
p_tp1_rsi = st.sidebar.slider("שלב א': 50% מימוש + Breakeven (RSI)", min_value=55.0, max_value=65.0, value=60.0, step=1.0)
p_tp2_rsi = st.sidebar.slider("שלב ב': 50% נותרים (RSI)", min_value=65.0, max_value=80.0, value=70.0, step=1.0)
p_exit_ema = st.sidebar.number_input("ממוצע מעריכי ליציאת שארית (EMA)", value=20, step=5)

st.sidebar.subheader("5. חיכוך שוק ריאלי")
p_slip = st.sidebar.number_input("החלקת ביצוע ממוצעת (%)", value=0.04, step=0.01, format="%.2f")
p_fee = st.sidebar.number_input("עמלות מסחר נטו לעסקה (%)", value=0.05, step=0.01, format="%.2f")

# =====================================================================
# מסך ראשי והרצה
# =====================================================================
st.title("🏛️ סימולטור כמותי מוסדי - Strategy 2.1 Full Portfolio")
st.markdown("""
סימולציית תיק מלאה מבוססת אירועים (**Event-Driven**) על מניות **S&P 500 Top 50** ומדד **SPY**.  
כל ששת שלבי האפיון נאכפים במלואם: שער מאקרו, שער מגמה, סייג דוחות 7 ימים, פקודות Buy Stop-Limit עם מדרגות אופסט, אימות מוסדי ($RVOL \ge 0.80$), שעון 3 ימים, איפוס מוסדי ומודל מימוש דו-שלבי.
""")

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    start_d = st.date_input("תאריך התחלה", datetime(2022, 1, 1))
with col2:
    end_d = st.date_input("תאריך סיום", datetime(2026, 9, 18))
with col3:
    st.write("")
    st.write("")
    run_button = st.button("🚀 הרץ סימולציית תיק מלאה", type="primary", use_container_width=True)

if run_button:
    sim_params = {
        "start_date": pd.to_datetime(start_d),
        "rsi_len": 14,
        "rsi_trigger": p_rsi_trig,
        "rsi_invalidate": p_rsi_inval,
        "rvol_min": p_rvol_min,
        "sma_len": p_sma_len,
        "limit_cap_pct": p_limit_cap,
        "hard_stop_pct": p_hard_stop,
        "use_trend_cushion": p_trend_cushion,
        "tp1_rsi": p_tp1_rsi,
        "tp2_rsi": p_tp2_rsi,
        "exit_ema": p_exit_ema,
        "initial_capital": p_capital,
        "pos_size_pct": p_pos_size,
        "slippage_pct": p_slip,
        "fee_pct": p_fee
    }
    
    with st.spinner("טוען נתוני עומק מתואמים, מסנכרן לוחות דוחות ומריץ סימולציית תיק מלאה..."):
        m_data = load_all_market_data(TOP_50_TICKERS, pd.to_datetime(start_d) - timedelta(days=365), pd.to_datetime(end_d))
        e_data = fetch_earnings_calendar(TOP_50_TICKERS)
        trades_df, equity_df = run_portfolio_simulation(m_data, e_data, sim_params)
        
    if trades_df.empty:
        st.warning("לא אותרו עסקאות בטווח הזמן ובפרמטרים שנבחרו.")
    else:
        total_actions = len(trades_df)
        winning_actions = trades_df[trades_df['PnL_$'] > 0]
        losing_actions = trades_df[trades_df['PnL_$'] <= 0]
        
        win_rate = (len(winning_actions) / total_actions) * 100.0
        gross_profit = winning_actions['PnL_$'].sum()
        gross_loss = abs(losing_actions['PnL_$'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
        
        net_profit_usd = trades_df['PnL_$'].sum()
        total_return_pct = (net_profit_usd / p_capital) * 100.0
        
        # מדדי סיכון מעקומת ההון
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown_USD'] = equity_df['Equity'] - equity_df['Peak']
        equity_df['Drawdown_Pct'] = (equity_df['Drawdown_USD'] / equity_df['Peak']) * 100.0
        max_dd_pct = equity_df['Drawdown_Pct'].min()
        max_dd_usd = equity_df['Drawdown_USD'].min()
        
        # CAGR
        days_total = max((equity_df['Date'].max() - equity_df['Date'].min()).days, 180)
        years = days_total / 365.25
        final_eq = equity_df['Equity'].iloc[-1]
        cagr = (((final_eq / p_capital) ** (1.0 / years)) - 1.0) * 100.0 if final_eq > 0 else -100.0
        
        st.subheader("📋 תקציר מנהלים וביצועי ליבה (Executive Summary)")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Win Rate (אחוז הצלחה)", f"{win_rate:.1f}%", f"{len(winning_actions)} מתוך {total_actions}")
        k2.metric("Profit Factor (PF)", f"{profit_factor:.2f}", "יעד מוסדי > 3.5")
        k3.metric("תשואה שנתית (CAGR)", f"{cagr:.1f}%", f"סה\"כ {total_return_pct:.1f}%")
        k4.metric("Max Drawdown", f"{max_dd_pct:.1f}%", f"${abs(max_dd_usd):,.0f}")
        k5.metric("זמן החזקה ממוצע", f"{trades_df['Days'].mean():.1f} ימים", f"עסקאות שנפתחו: {len(trades_df[trades_df['Stage'].str.contains('Stage 1')])}")
        
        st.divider()

        # עקומת הון
        st.subheader("📈 עקומת צמיחת תיק ההשקעות (Portfolio Equity Curve)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['Date'],
            y=equity_df['Equity'],
            mode='lines',
            name='Portfolio Equity ($)',
            line=dict(color='#00FFA3', width=2.5)
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="תאריך",
            yaxis_title="שווי תיק ($)",
            height=450,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # פילוח לפי שנים
        st.subheader("📅 פילוח ביצועים לפי שנים")
        trades_df['Exit_Year'] = pd.to_datetime(trades_df['Exit_Date']).dt.year
        annual_summary = trades_df.groupby('Exit_Year').agg(
            Actions=('PnL_$', 'count'),
            Wins=('PnL_$', lambda x: (x > 0).sum()),
            Net_PnL=('PnL_$', 'sum'),
            Gross_Profit=('PnL_$', lambda x: x[x > 0].sum()),
            Gross_Loss=('PnL_$', lambda x: abs(x[x <= 0].sum()))
        ).reset_index()
        
        annual_summary['Win_Rate'] = (annual_summary['Wins'] / annual_summary['Actions']) * 100.0
        annual_summary['Profit_Factor'] = np.where(
            annual_summary['Gross_Loss'] > 0,
            annual_summary['Gross_Profit'] / annual_summary['Gross_Loss'],
            99.0
        )
        annual_summary['Return_on_Capital'] = (annual_summary['Net_PnL'] / p_capital) * 100.0
        
        st.dataframe(
            annual_summary[['Exit_Year', 'Actions', 'Win_Rate', 'Profit_Factor', 'Net_PnL', 'Return_on_Capital']].style.format({
                "Win_Rate": "{:.1f}%",
                "Profit_Factor": "{:.2f}",
                "Net_PnL": "${:,.2f}",
                "Return_on_Capital": "{:.1f}%"
            }),
            use_container_width=True
        )

        # לשוניות פירוט
        tab_log, tab_stocks, tab_reasons = st.tabs(["📋 יומן פעולות מלא", "🏆 ביצועים לפי מניה", "🎯 התפלגות סיבות יציאה"])
        
        with tab_log:
            st.dataframe(
                trades_df[['Ticker', 'Entry_Date', 'Exit_Date', 'Stage', 'Entry_P', 'Exit_P', 'Return_%', 'PnL_$', 'Reason', 'Days']].sort_values(by='Exit_Date', ascending=False).style.format({
                    "Entry_P": "${:.2f}",
                    "Exit_P": "${:.2f}",
                    "Return_%": "{:.2f}%",
                    "PnL_$": "${:,.2f}"
                }),
                use_container_width=True
            )
            
        with tab_stocks:
            stock_summary = trades_df.groupby('Ticker').agg(
                Actions=('PnL_$', 'count'),
                Win_Rate=('PnL_$', lambda x: (x > 0).mean() * 100.0),
                Total_PnL=('PnL_$', 'sum')
            ).sort_values(by='Total_PnL', ascending=False)
            
            st.dataframe(
                stock_summary.style.format({
                    "Win_Rate": "{:.1f}%",
                    "Total_PnL": "${:,.2f}"
                }),
                use_container_width=True
            )
            
        with tab_reasons:
            reasons = trades_df['Reason'].value_counts()
            st.bar_chart(reasons)
