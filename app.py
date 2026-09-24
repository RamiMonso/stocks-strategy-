import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# הגדרת תצורה רחבה
st.set_page_config(page_title="AlphaSim | Pro Backtester", page_icon="⚡", layout="wide")

# הזרקת CSS מותאם אישית לעיצוב פרימיום
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AlphaSim Pro — מעבדת סימולציות מסחר והשקעות")
st.caption("פלטפורמת בדיקת אסטרטגיות RSI, ניהול סיכונים ופריצת נרות רב-ממדית")

# --- רשימות S&P מובנות ---
SP500_BASE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'JPM',
    'LLY', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'JNJ', 'WMT',
    'ABBV', 'BAC', 'MRK', 'KO', 'PEP', 'CVX', 'ADBE', 'LIN', 'CRM', 'ACN',
    'NFLX', 'TMO', 'MCD', 'AMD', 'CSCO', 'ABT', 'ORCL', 'INTU', 'QCOM', 'GE',
    'CAT', 'AMAT', 'DIS', 'TXN', 'DHR', 'PFE', 'VZ', 'NOW', 'PM', 'COP'
]
SP500_EXTENDED = SP500_BASE + [
    'IBM', 'UNP', 'SPGI', 'RTX', 'HON', 'AMGN', 'LOW', 'GS', 'BA', 'INTC',
    'ISRG', 'ELV', 'SYK', 'BKNG', 'LMT', 'BLK', 'TJX', 'T', 'MDLZ', 'VRTX',
    'REGN', 'ADI', 'C', 'MMC', 'CI', 'PLD', 'CB', 'PGR', 'ZTS', 'SCHW',
    'BSX', 'ETN', 'FI', 'PANW', 'DE', 'GILD', 'BDX', 'MU', 'SLB', 'EOG',
    'CVS', 'ITW', 'MO', 'WM', 'SHW', 'NOC', 'CDNS', 'CL', 'APD', 'EQIX'
]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ הגדרות אסטרטגיה ותיק")
    
    universe = st.selectbox(
        "נכסי מסחר:",
        ["Top 10 S&P", "Top 20 S&P", "Top 50 S&P", "Top 100 S&P", "הזנה חופשית / מותאמת"]
    )
    if universe == "הזנה חופשית / מותאמת":
        raw_input = st.text_input("הזן סימולים מופרדים בפסיק:", "AAPL, NVDA, MSFT, TSLA, AMZN")
        tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
    elif universe == "Top 10 S&P":
        tickers = SP500_BASE[:10]
    elif universe == "Top 20 S&P":
        tickers = SP500_BASE[:20]
    elif universe == "Top 50 S&P":
        tickers = SP500_BASE
    else:
        tickers = SP500_EXTENDED

    st.markdown("---")
    st.subheader("📅 טווח זמן")
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("תאריך התחלה", datetime.today() - timedelta(days=365*2))
    end_date = col_d2.date_input("תאריך סיום", datetime.today())

    st.markdown("---")
    st.subheader("💰 ניהול הון")
    initial_capital = st.number_input("הון התחלתי ($):", min_value=1000, value=100000, step=5000)
    pos_type = st.radio("אופן הקצאת הון:", ["אחוז משווי התיק הדינמי", "סכום קבוע לפוזיציה"])
    if pos_type == "אחוז משווי התיק הדינמי":
        pos_size = st.slider("אחוז לכל פוזיציה (%):", 2, 100, 15, 1) / 100.0
    else:
        pos_fixed_val = st.number_input("סכום ב-$ לעסקה:", min_value=500, value=10000, step=500)
    commission = st.number_input("עמלה לפעולה ($):", min_value=0.0, value=1.0, step=0.5)

    st.markdown("---")
    st.subheader("🎯 כללי איתות RSI")
    rsi_len = st.number_input("תקופת RSI (נרות):", min_value=2, max_value=50, value=14)
    rsi_buy = st.slider("רף כניסה (Oversold):", 5, 50, 30, 1)
    rsi_sell = st.slider("רף יציאה (Overbought):", 50, 95, 70, 1)

    st.markdown("---")
    st.subheader("🛡️ ניהול עסקאות מתקדם")
    use_confirmation = st.toggle("מנגנון פריצת High (חלון 3 ימים)", value=True, help="אם כבוי: כניסה מיידית בסגירת נר האיתות")
    use_sl = st.toggle("הפעלת Stop-Loss", value=True)
    sl_pct = st.slider("Stop-Loss ממחיר הכניסה (%):", 1.0, 25.0, 7.0, 0.5) / 100.0 if use_sl else None

    run_sim = st.button("🚀 הפעל סימולציה מלאה", use_container_width=True, type="primary")

# פונקציית Wilder RSI
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

if run_sim:
    if not tickers:
        st.error("אנא בחר או הזן לפחות מניה אחת.")
        st.stop()
    if start_date >= end_date:
        st.error("תאריך התחלה חייב להיות מוקדם מתאריך סיום.")
        st.stop()

    with st.spinner(f"טוען נתונים עבור {len(tickers)} מניות ומבצע חישובי סימולציה..."):
        dl_start = pd.to_datetime(start_date) - timedelta(days=90)
        # נוריד גם SPY לצורך השוואת בנצ'מרק
        all_download_tickers = list(set(tickers + ['SPY']))
        raw_data = yf.download(all_download_tickers, start=dl_start, end=end_date, group_by='ticker', progress=False)

        stock_dfs = {}
        for t in tickers:
            if len(all_download_tickers) == 1:
                df = raw_data.copy()
            else:
                if t in raw_data.columns.levels[0]:
                    df = raw_data[t].dropna(how='all').copy()
                else:
                    continue
            if not df.empty and 'Close' in df and len(df) > rsi_len:
                df['RSI'] = calc_rsi(df['Close'], period=rsi_len)
                df = df[df.index >= pd.to_datetime(start_date)]
                stock_dfs[t] = df

        # נתוני SPY להשוואה
        spy_df = pd.DataFrame()
        if 'SPY' in raw_data.columns.levels[0]:
            spy_df = raw_data['SPY'].dropna(how='all').copy()
            spy_df = spy_df[spy_df.index >= pd.to_datetime(start_date)]

        all_dates = sorted(list(set.union(*[set(df.index) for df in stock_dfs.values()])))

        cash = float(initial_capital)
        open_positions = {}
        pending_setups = {}
        trade_log = []
        equity_curve = []

        # לולאת הסימולציה הראשית
        for curr_dt in all_dates:
            # 1. בדיקת יציאות
            to_close = []
            for t, pos in open_positions.items():
                df = stock_dfs.get(t)
                if df is None or curr_dt not in df.index:
                    continue
                row = df.loc[curr_dt]
                c_close, c_low, c_rsi = row['Close'], row['Low'], row['RSI']

                hit_sl = False
                if sl_pct:
                    stop_lvl = pos['entry_price'] * (1 - sl_pct)
                    if c_low <= stop_lvl:
                        hit_sl = True
                        exit_price = stop_lvl
                        reason = f"Stop Loss (-{sl_pct*100:.1f}%)"

                if not hit_sl and c_rsi >= rsi_sell:
                    exit_price = c_close
                    reason = f"RSI Exit ({c_rsi:.1f})"
                elif not hit_sl:
                    continue

                gross_return = (pos['shares'] * exit_price) - (pos['shares'] * pos['entry_price'])
                net_pnl = gross_return - (commission * 2)
                pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                cash += (pos['shares'] * exit_price) - commission

                trade_log.append({
                    'Date': curr_dt.strftime('%Y-%m-%d'),
                    'Ticker': t,
                    'Action': 'SELL',
                    'Price': round(exit_price, 2),
                    'Shares': pos['shares'],
                    'P&L ($)': round(net_pnl, 2),
                    'Return (%)': round(pnl_pct, 2),
                    'Hold Days': (curr_dt - pos['entry_date']).days,
                    'Status': 'Executed',
                    'Reason': reason
                })
                to_close.append(t)

            for t in to_close:
                del open_positions[t]

            # חישוב שווי תיק רגעי
            pos_equity = sum(
                p['shares'] * stock_dfs[t].loc[curr_dt]['Close']
                for t, p in open_positions.items()
                if curr_dt in stock_dfs[t].index
            )
            curr_portfolio_val = cash + pos_equity

            # 2. בדיקת טריגרי כניסה
            if use_confirmation:
                expired = []
                for t, s in pending_setups.items():
                    if t in open_positions:
                        expired.append(t)
                        continue
                    df = stock_dfs.get(t)
                    if df is None or curr_dt not in df.index:
                        continue
                    row = df.loc[curr_dt]

                    if row['High'] > s['trigger_high']:
                        exec_price = max(row['Open'], s['trigger_high'])
                        alloc = (curr_portfolio_val * pos_size) if pos_type == "אחוז משווי התיק הדינמי" else pos_fixed_val
                        shares = int((alloc - commission) // exec_price)
                        cost = (shares * exec_price) + commission

                        if shares > 0 and cash >= cost:
                            cash -= cost
                            open_positions[t] = {'shares': shares, 'entry_price': exec_price, 'entry_date': curr_dt}
                            trade_log.append({
                                'Date': curr_dt.strftime('%Y-%m-%d'),
                                'Ticker': t,
                                'Action': 'BUY',
                                'Price': round(exec_price, 2),
                                'Shares': shares,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Hold Days': 0,
                                'Status': 'Executed',
                                'Reason': f"Breakout day {4 - s['days_left']}"
                            })
                        else:
                            trade_log.append({
                                'Date': curr_dt.strftime('%Y-%m-%d'),
                                'Ticker': t,
                                'Action': 'BUY',
                                'Price': round(exec_price, 2),
                                'Shares': 0,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Hold Days': 0,
                                'Status': 'Skipped (No Cash)',
                                'Reason': f"נדרש: ${cost:.0f}, יתרה: ${cash:.0f}"
                            })
                        expired.append(t)
                    else:
                        s['trigger_high'] = row['High']
                        s['days_left'] -= 1
                        if s['days_left'] <= 0:
                            trade_log.append({
                                'Date': curr_dt.strftime('%Y-%m-%d'),
                                'Ticker': t,
                                'Action': 'SETUP',
                                'Price': round(row['Close'], 2),
                                'Shares': 0,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Hold Days': 0,
                                'Status': 'Cancelled',
                                'Reason': 'לא פרץ High תוך 3 ימים'
                            })
                            expired.append(t)

                for t in expired:
                    del pending_setups[t]

            # 3. זיהוי איתותי יום 0
            for t, df in stock_dfs.items():
                if t in open_positions or (use_confirmation and t in pending_setups):
                    continue
                if curr_dt not in df.index:
                    continue
                row = df.loc[curr_dt]

                if row['RSI'] <= rsi_buy:
                    if use_confirmation:
                        pending_setups[t] = {'days_left': 3, 'trigger_high': row['High'], 'signal_date': curr_dt}
                    else:
                        # כניסה מיידית בסגירה
                        exec_price = row['Close']
                        alloc = (curr_portfolio_val * pos_size) if pos_type == "אחוז משווי התיק הדינמי" else pos_fixed_val
                        shares = int((alloc - commission) // exec_price)
                        cost = (shares * exec_price) + commission

                        if shares > 0 and cash >= cost:
                            cash -= cost
                            open_positions[t] = {'shares': shares, 'entry_price': exec_price, 'entry_date': curr_dt}
                            trade_log.append({
                                'Date': curr_dt.strftime('%Y-%m-%d'),
                                'Ticker': t,
                                'Action': 'BUY',
                                'Price': round(exec_price, 2),
                                'Shares': shares,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Hold Days': 0,
                                'Status': 'Executed',
                                'Reason': f"RSI Entry ({row['RSI']:.1f})"
                            })
                        else:
                            trade_log.append({
                                'Date': curr_dt.strftime('%Y-%m-%d'),
                                'Ticker': t,
                                'Action': 'BUY',
                                'Price': round(exec_price, 2),
                                'Shares': 0,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Hold Days': 0,
                                'Status': 'Skipped (No Cash)',
                                'Reason': f"נדרש: ${cost:.0f}, יתרה: ${cash:.0f}"
                            })

            # שערוך סוף יום
            end_pos_equity = sum(
                p['shares'] * stock_dfs[t].loc[curr_dt]['Close']
                for t, p in open_positions.items()
                if curr_dt in stock_dfs[t].index
            )
            equity_curve.append({
                'Date': curr_dt,
                'Portfolio': cash + end_pos_equity,
                'Cash': cash,
                'Positions Count': len(open_positions)
            })

        # --- יצירת דוחות ועיבוד תוצאות ---
        eq_df = pd.DataFrame(equity_curve).set_index('Date')
        trades_df = pd.DataFrame(trade_log)

        final_val = eq_df['Portfolio'].iloc[-1]
        tot_ret = ((final_val - initial_capital) / initial_capital) * 100

        # חישוב Drawdown
        eq_df['Peak'] = eq_df['Portfolio'].cummax()
        eq_df['DD'] = (eq_df['Portfolio'] - eq_df['Peak']) / eq_df['Peak'] * 100
        max_dd = eq_df['DD'].min()

        # תשואת SPY
        spy_ret = 0.0
        if not spy_df.empty:
            spy_ret = ((spy_df['Close'].iloc[-1] - spy_df['Close'].iloc[0]) / spy_df['Close'].iloc[0]) * 100

        # מדדי עסקאות
        sells = trades_df[(trades_df['Status'] == 'Executed') & (trades_df['Action'] == 'SELL')] if not trades_df.empty else pd.DataFrame()
        n_trades = len(sells)
        wins = sells[sells['P&L ($)'] > 0] if n_trades > 0 else pd.DataFrame()
        losses = sells[sells['P&L ($)'] <= 0] if n_trades > 0 else pd.DataFrame()
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0

        avg_win = wins['P&L ($)'].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses['P&L ($)'].mean()) if len(losses) > 0 else 0.0
        profit_factor = (wins['P&L ($)'].sum() / abs(losses['P&L ($)'].sum())) if (len(losses) > 0 and losses['P&L ($)'].sum() != 0) else np.nan
        avg_hold = sells['Hold Days'].mean() if n_trades > 0 else 0

        # מדד שארפ
        daily_ret = eq_df['Portfolio'].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0.0

        st.session_state['results'] = {
            'eq_df': eq_df,
            'trades_df': trades_df,
            'stock_dfs': stock_dfs,
            'spy_df': spy_df,
            'tickers': tickers,
            'metrics': {
                'final_val': final_val,
                'tot_ret': tot_ret,
                'max_dd': max_dd,
                'spy_ret': spy_ret,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_hold': avg_hold,
                'sharpe': sharpe
            }
        }

# --- הצגת לוח התוצאות ---
if 'results' in st.session_state:
    res = st.session_state['results']
    m = res['metrics']
    trades_df = res['trades_df']
    eq_df = res['eq_df']
    stock_dfs = res['stock_dfs']

    st.markdown("### 🏆 סקירת ביצועים מרכזית")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    ret_color = "positive" if m['tot_ret'] >= 0 else "negative"
    spy_diff = m['tot_ret'] - m['spy_ret']
    diff_color = "positive" if spy_diff >= 0 else "negative"

    c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">שווי תיק סופי</div>
            <div class="metric-value">${m['final_val']:,.0f}</div>
            <div class="metric-sub {ret_color}">{m['tot_ret']:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">S&P 500 (Buy & Hold)</div>
            <div class="metric-value">{m['spy_ret']:+.2f}%</div>
            <div class="metric-sub {diff_color}">אלפא: {spy_diff:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" style="color:#ef4444;">{m['max_dd']:.2f}%</div>
            <div class="metric-sub" style="color:#9ca3af;">שארפ: {m['sharpe']:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">עסקאות שנסגרו</div>
            <div class="metric-value">{m['n_trades']}</div>
            <div class="metric-sub" style="color:#9ca3af;">משך ממוצע: {m['avg_hold']:.1f} ימים</div>
        </div>
    """, unsafe_allow_html=True)

    c5.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">אחוז הצלחה (Win Rate)</div>
            <div class="metric-value">{m['win_rate']:.1f}%</div>
            <div class="metric-sub positive">רווח ממוצע: ${m['avg_win']:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

    c6.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">יחס רווח/הפסד (PF)</div>
            <div class="metric-value">{m['profit_factor']:.2f}</div>
            <div class="metric-sub negative">הפסד ממוצע: ${m['avg_loss']:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- טאבים לתצוגה מתקדמת ---
    tab_chart, tab_stocks, tab_log = st.tabs([
        "📊 עקומת שווי תיק ו-Drawdown",
        "🕯️ ניתוח מניה בודדת (כניסות/יציאות על הגרף)",
        "📋 יומן עסקאות מלא וסטטוסים"
    ])

    with tab_chart:
        fig_equity = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08, row_heights=[0.7, 0.3],
            subplot_titles=("שווי תיק מול S&P 500 ומזומן", "אחוז Drawdown מתמשך")
        )
        # נרמול S&P 500 לשווי התיק ההתחלתי
        if not res['spy_df'].empty:
            spy_norm = (res['spy_df']['Close'] / res['spy_df']['Close'].iloc[0]) * eq_df['Portfolio'].iloc[0]
            fig_equity.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, name="S&P 500 Benchmark", line=dict(color='#64748b', dash='dot')), row=1, col=1)

        fig_equity.add_trace(go.Scatter(x=eq_df.index, y=eq_df['Portfolio'], name="שווי תיק (Portfolio)", line=dict(color='#10b981', width=2.5)), row=1, col=1)
        fig_equity.add_trace(go.Scatter(x=eq_df.index, y=eq_df['Cash'], name="מזומן פנוי (Cash)", line=dict(color='#8b5cf6', dash='dash')), row=1, col=1)

        fig_equity.add_trace(go.Scatter(x=eq_df.index, y=eq_df['DD'], name="Drawdown %", fill='tozeroy', line=dict(color='#ef4444', width=1)), row=2, col=1)
        fig_equity.update_layout(template="plotly_dark", height=580, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig_equity, use_container_width=True)

    with tab_stocks:
        st.subheader("ויזואליזציית פוזיציות על גבי גרף נרות יפניים")
        available_tickers = list(stock_dfs.keys())
        inspected_ticker = st.selectbox("בחר מניה להצגה:", available_tickers)

        if inspected_ticker:
            stk_df = stock_dfs[inspected_ticker]
            stk_trades = trades_df[trades_df['Ticker'] == inspected_ticker]

            buys = stk_trades[(stk_trades['Action'] == 'BUY') & (stk_trades['Status'] == 'Executed')]
            sells = stk_trades[(stk_trades['Action'] == 'SELL') & (stk_trades['Status'] == 'Executed')]
            skipped = stk_trades[stk_trades['Status'].str.contains('Skipped|Cancelled', na=False)]

            fig_stock = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.06, row_heights=[0.75, 0.25],
                subplot_titles=(f"{inspected_ticker} - מחיר ועסקאות", "RSI (14)")
            )

            # נרות יפניים
            fig_stock.add_trace(go.Candlestick(
                x=stk_df.index, open=stk_df['Open'], high=stk_df['High'],
                low=stk_df['Low'], close=stk_df['Close'], name="מחיר"
            ), row=1, col=1)

            # סימון קניות (חץ ירוק כלפי מעלה)
            if not buys.empty:
                buy_dates = pd.to_datetime(buys['Date'])
                fig_stock.add_trace(go.Scatter(
                    x=buy_dates, y=buys['Price'] * 0.985, mode='markers+text',
                    marker=dict(symbol='triangle-up', size=14, color='#10b981'),
                    text=[f"קנייה: ${p}" for p in buys['Price']], textposition="bottom center",
                    name="קנייה (Buy)"
                ), row=1, col=1)

            # סימון מכירות (חץ אדום כלפי מטה)
            if not sells.empty:
                sell_dates = pd.to_datetime(sells['Date'])
                fig_stock.add_trace(go.Scatter(
                    x=sell_dates, y=sells['Price'] * 1.015, mode='markers+text',
                    marker=dict(symbol='triangle-down', size=14, color='#ef4444'),
                    text=[f"מכירה ({r.split()[0]}): ${p}" for p, r in zip(sells['Price'], sells['Reason'])],
                    textposition="top center", name="מכירה (Sell)"
                ), row=1, col=1)

            # סימון עסקאות שבוטלו / חוסר מזומן (עיגול צהוב/כתום)
            if not skipped.empty:
                skip_dates = pd.to_datetime(skipped['Date'])
                fig_stock.add_trace(go.Scatter(
                    x=skip_dates, y=skipped['Price'], mode='markers',
                    marker=dict(symbol='x', size=10, color='#f59e0b'),
                    name="דולג/בוטל (Skipped)", hovertext=skipped['Reason']
                ), row=1, col=1)

            # גרף RSI
            fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df['RSI'], name="RSI", line=dict(color='#38bdf8', width=1.5)), row=2, col=1)
            fig_stock.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
            fig_stock.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)

            fig_stock.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_stock, use_container_width=True)

    with tab_log:
        st.subheader("ריכוז פעולות, איתותים וביצועים")
        if not trades_df.empty:
            # אפשרויות סינון מתקדמות
            f_col1, f_col2 = st.columns([1, 3])
            status_filter = f_col1.multiselect("סנן לפי סטטוס:", trades_df['Status'].unique(), default=trades_df['Status'].unique())
            filtered_trades = trades_df[trades_df['Status'].isin(status_filter)]

            # הורדה ל-CSV
            csv_data = filtered_trades.to_csv(index=False).encode('utf-8')
            f_col2.download_button("📥 הורד יומן עסקאות ל-CSV", data=csv_data, file_name="trades_log.csv", mime="text/csv")

            def color_trade_row(val):
                if val == 'Executed':
                    return 'color: #10b981; font-weight: bold;'
                elif 'Skipped' in str(val):
                    return 'color: #f59e0b;'
                elif 'Cancelled' in str(val):
                    return 'color: #ef4444;'
                return ''

            style_obj = filtered_trades.style
            if hasattr(style_obj, "map"):
                rendered_log = style_obj.map(color_trade_row, subset=['Status'])
            else:
                rendered_log = style_obj.applymap(color_trade_row, subset=['Status'])

            st.dataframe(rendered_log, use_container_width=True, height=450)
        else:
            st.info("אין עסקאות להצגה בטווח זה.")
