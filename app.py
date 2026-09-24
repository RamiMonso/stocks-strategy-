import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Strategy Backtester", layout="wide")
st.title("📈 מערכת סימולציית השקעות ואסטרטגיות מסחר")

# --- רשימות מניות מובנות ---
SP500_TOP10 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'AVGO', 'JPM']
SP500_TOP20 = SP500_TOP10 + ['LLY', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'JNJ', 'WMT']

# --- Sidebar: פרמטרי קלט ---
st.sidebar.header("⚙️ הגדרות סימולציה")

ticker_mode = st.sidebar.radio("אופן בחירת מניות:", ["ידני", "Top 10 S&P 500", "Top 20 S&P 500"])
if ticker_mode == "ידני":
    tickers_input = st.sidebar.text_input("הזן סימולים (מופרדים בפסיק):", "AAPL, MSFT, NVDA")
    selected_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
elif ticker_mode == "Top 10 S&P 500":
    selected_tickers = SP500_TOP10
else:
    selected_tickers = SP500_TOP20

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("תאריך התחלה", datetime.today() - timedelta(days=365*3))
end_date = col2.date_input("תאריך סיום", datetime.today())

st.sidebar.subheader("💰 ניהול תיק והון")
initial_capital = st.sidebar.number_input("שווי תיק התחלתי ($):", min_value=1000, value=100000, step=1000)
position_mode = st.sidebar.selectbox("שיטת הקצאת סכום לפוזיציה:", ["אחוז משווי התיק הדינמי", "סכום קבוע בדולר"])

if position_mode == "אחוז משווי התיק הדינמי":
    position_pct = st.sidebar.slider("אחוז לכל פוזיציה (%):", min_value=5, max_value=100, value=20, step=5) / 100.0
else:
    position_fixed_amount = st.sidebar.number_input("סכום קבוע לפוזיציה ($):", min_value=500, value=20000, step=500)

commission = st.sidebar.number_input("עמלת מסחר לפעולה ($):", min_value=0.0, value=1.0, step=0.5)

st.sidebar.subheader("📊 פרמטרי אסטרטגיה: RSI וטריגר פריצה")
rsi_entry = st.sidebar.slider("רף כניסה (RSI Oversold):", min_value=10, max_value=40, value=30, step=1)
rsi_exit = st.sidebar.slider("רף יציאה (RSI Overbought):", min_value=60, max_value=90, value=70, step=1)

use_stop_loss = st.sidebar.checkbox("הפעל Stop-Loss (%)", value=True)
stop_loss_pct = st.sidebar.slider("אחוז Stop-Loss ממחיר הכניסה (%):", min_value=1.0, max_value=20.0, value=7.0, step=0.5) / 100.0 if use_stop_loss else None

run_button = st.sidebar.button("🚀 הרץ סימולציה", use_container_width=True)

# --- פונקציית חישוב RSI תקני של Wilder ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- מנוע הסימולציה ---
if run_button:
    if not selected_tickers:
        st.error("אנא בחר לפחות מניה אחת.")
    elif start_date >= end_date:
        st.error("תאריך התחלה חייב להיות מוקדם מתאריך סיום.")
    else:
        with st.spinner("מוריד נתוני שוק ומבצע חישובי סימולציה..."):
            # תוספת ימים אחורה כדי לחשב RSI כראוי מיום ההתחלה
            dl_start = pd.to_datetime(start_date) - timedelta(days=60)
            data = yf.download(selected_tickers, start=dl_start, end=end_date, group_by='ticker')
            
            # עיבוד מבנה הנתונים ל-dictionary של DataFrames לכל מניה
            stock_dfs = {}
            for t in selected_tickers:
                if len(selected_tickers) == 1:
                    df = data.copy()
                else:
                    if t in data.columns.levels[0]:
                        df = data[t].dropna(how='all').copy()
                    else:
                        continue
                if not df.empty and 'Close' in df:
                    df['RSI'] = calculate_rsi(df['Close'])
                    df = df[df.index >= pd.to_datetime(start_date)]
                    stock_dfs[t] = df

            # איחוד כל תאריכי המסחר הזמינים
            all_dates = sorted(list(set.union(*[set(df.index) for df in stock_dfs.values()])))

            cash = float(initial_capital)
            open_positions = {}  # {ticker: {'shares', 'entry_price', 'entry_date'}}
            pending_setups = {}  # {ticker: {'days_left', 'trigger_high', 'signal_date'}}
            
            trade_log = []
            portfolio_history = []

            # לולאת זמן יום-יום
            for current_date in all_dates:
                # 1. בדיקת יציאות מפוזיציות קיימות (RSI Exit או Stop-Loss)
                closed_today = []
                for ticker, pos in open_positions.items():
                    df = stock_dfs.get(ticker)
                    if df is None or current_date not in df.index:
                        continue
                    row = df.loc[current_date]
                    price_close = row['Close']
                    price_low = row['Low']
                    curr_rsi = row['RSI']

                    # בדיקת Stop-Loss לפי ה-Low של היום
                    hit_stop = False
                    if stop_loss_pct:
                        stop_price = pos['entry_price'] * (1 - stop_loss_pct)
                        if price_low <= stop_price:
                            hit_stop = True
                            exit_price = stop_price
                            reason = f"Stop-Loss (-{stop_loss_pct*100:.1f}%)"

                    # בדיקת יציאת RSI
                    if not hit_stop and curr_rsi >= rsi_exit:
                        exit_price = price_close
                        reason = f"RSI Exit ({curr_rsi:.1f})"
                    elif not hit_stop:
                        continue

                    # סגירת הפוזיציה בפועל
                    revenue = (pos['shares'] * exit_price) - commission
                    cash += revenue
                    pnl = revenue - (pos['shares'] * pos['entry_price'])
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100

                    trade_log.append({
                        'Date': current_date.strftime('%Y-%m-%d'),
                        'Ticker': ticker,
                        'Action': 'SELL',
                        'Price': round(exit_price, 2),
                        'Shares': pos['shares'],
                        'P&L ($)': round(pnl, 2),
                        'Return (%)': round(pnl_pct, 2),
                        'Status': 'Executed',
                        'Note': reason
                    })
                    closed_today.append(ticker)

                for ticker in closed_today:
                    del open_positions[ticker]

                # 2. בדיקת טריגרי כניסה קיימים (ממתינים לפריצת High עד 3 ימים)
                expired_setups = []
                for ticker, setup in pending_setups.items():
                    if ticker in open_positions:
                        expired_setups.append(ticker)
                        continue

                    df = stock_dfs.get(ticker)
                    if df is None or current_date not in df.index:
                        continue
                    row = df.loc[current_date]

                    # פריצת High
                    if row['High'] > setup['trigger_high']:
                        # מחיר כניסה: פתיחה (אם פתח בגאפ מעל) או מחיר ה-High שנפרץ
                        entry_price = max(row['Open'], setup['trigger_high'])
                        
                        # חישוב שווי התיק לצורך הקצאה
                        current_equity = cash + sum(
                            p['shares'] * stock_dfs[t].loc[current_date]['Close']
                            for t, p in open_positions.items()
                            if current_date in stock_dfs[t].index
                        )
                        
                        target_allocation = (current_equity * position_pct) if position_mode == "אחוז משווי התיק הדינמי" else position_fixed_amount
                        shares = int((target_allocation - commission) // entry_price)

                        cost = (shares * entry_price) + commission
                        if shares > 0 and cash >= cost:
                            cash -= cost
                            open_positions[ticker] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': current_date
                            }
                            trade_log.append({
                                'Date': current_date.strftime('%Y-%m-%d'),
                                'Ticker': ticker,
                                'Action': 'BUY',
                                'Price': round(entry_price, 2),
                                'Shares': shares,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Status': 'Executed',
                                'Note': f"Breakout day {4 - setup['days_left']}"
                            })
                        else:
                            trade_log.append({
                                'Date': current_date.strftime('%Y-%m-%d'),
                                'Ticker': ticker,
                                'Action': 'BUY',
                                'Price': round(entry_price, 2),
                                'Shares': 0,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Status': 'Skipped (No Cash)',
                                'Note': f"Needed ${cost:.0f}, Available ${cash:.0f}"
                            })
                        expired_setups.append(ticker)
                    else:
                        # לא נפרץ: עדכון High נמוך יותר וקידום ספירת ימים
                        setup['trigger_high'] = row['High']
                        setup['days_left'] -= 1
                        if setup['days_left'] <= 0:
                            trade_log.append({
                                'Date': current_date.strftime('%Y-%m-%d'),
                                'Ticker': ticker,
                                'Action': 'SETUP',
                                'Price': round(row['Close'], 2),
                                'Shares': 0,
                                'P&L ($)': 0.0,
                                'Return (%)': 0.0,
                                'Status': 'Cancelled (Expired)',
                                'Note': 'High not breached within 3 days'
                            })
                            expired_setups.append(ticker)

                for ticker in expired_setups:
                    del pending_setups[ticker]

                # 3. זיהוי יום 0 חדש (RSI נמוך מהרף)
                for ticker, df in stock_dfs.items():
                    if ticker in open_positions or ticker in pending_setups:
                        continue
                    if current_date not in df.index:
                        continue

                    row = df.loc[current_date]
                    if row['RSI'] <= rsi_entry:
                        pending_setups[ticker] = {
                            'days_left': 3,
                            'trigger_high': row['High'],
                            'signal_date': current_date
                        }

                # 4. שערוך שווי תיק יומי (Equity)
                pos_value = sum(
                    p['shares'] * stock_dfs[t].loc[current_date]['Close']
                    for t, p in open_positions.items()
                    if current_date in stock_dfs[t].index
                )
                portfolio_history.append({
                    'Date': current_date,
                    'Portfolio Value': cash + pos_value,
                    'Cash': cash
                })

            # --- עיבוד נתונים ותוצאות ---
            history_df = pd.DataFrame(portfolio_history).set_index('Date')
            final_equity = history_df['Portfolio Value'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            
            # חישוב Max Drawdown
            history_df['Peak'] = history_df['Portfolio Value'].cummax()
            history_df['Drawdown'] = (history_df['Portfolio Value'] - history_df['Peak']) / history_df['Peak'] * 100
            max_drawdown = history_df['Drawdown'].min()

            trades_df = pd.DataFrame(trade_log)

            # --- הצגת לוח מדדים (KPIs) ---
            st.subheader("📊 סיכום ביצועים")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("שווי תיק סופי", f"${final_equity:,.0f}", f"{total_return:.2f}%")
            m2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
            executed_trades = trades_df[trades_df['Status'] == 'Executed'] if not trades_df.empty else pd.DataFrame()
            sells = executed_trades[executed_trades['Action'] == 'SELL'] if not executed_trades.empty else pd.DataFrame()
            
            win_rate = (len(sells[sells['P&L ($)'] > 0]) / len(sells) * 100) if len(sells) > 0 else 0
            m3.metric("עסקאות שבוצעו", len(sells))
            m4.metric("אחוז הצלחה (Win Rate)", f"{win_rate:.1f}%")

            # --- גרף שווי תיק ---
            st.subheader("📈 עקומת שווי התיק (Equity Curve)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Portfolio Value'], mode='lines', name='שווי תיק כולל', line=dict(color='#00CC96', width=2)))
            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Cash'], mode='lines', name='מזומן פנוי', line=dict(color='#AB63FA', dash='dash')))
            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- יומן פעולות מפורט ---
            st.subheader("📜 יומן פעולות ואיתותים")
            if not trades_df.empty:
                # הדגשה צבעונית לפי סטטוס
                def color_status(val):
                    if val == 'Executed':
                        return 'color: #00ff7f'
                    elif 'Skipped' in val:
                        return 'color: #ffaa00'
                    elif 'Cancelled' in val:
                        return 'color: #ff5555'
                    return ''

                # תמיכה ב-map החדש של pandas תוך תאימות לאחור (מונע את שגיאת applymap)
                style_styler = trades_df.style
                if hasattr(style_styler, "map"):
                    styled_df = style_styler.map(color_status, subset=['Status'])
                else:
                    styled_df = style_styler.applymap(color_status, subset=['Status'])

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=350
                )
            else:
                st.info("לא נוצרו איתותים בטווח התאריכים והפרמטרים שנבחרו.")
