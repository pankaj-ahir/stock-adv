# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import time
# from utils import (
#     get_stock_data, add_technical_indicators, train_model,
#     predict_next_day, batch_predict, calculate_stop_loss_target,
#     detect_volume_spike, get_swing_signal, calculate_profit,
#     get_morning_suggestions, get_best_pick
# )

# # Set Streamlit page configuration
# st.set_page_config(page_title="Stock Advisor", layout="wide")

# def main():
#     st.title("📈 Advanced Stock Market Advisor")

#     st.subheader("🌅 Morning Dashboard - Live Suggestions")
#     if st.button("🔄 Refresh Suggestions Now"):
#         st.session_state.last_refresh = time.time()

#     if "last_refresh" not in st.session_state:
#         st.session_state.last_refresh = time.time()

#     if time.time() - st.session_state.last_refresh >= 300:
#         st.session_state.last_refresh = time.time()
#         st.experimental_rerun()

#     morning_results = get_morning_suggestions()
#     if not morning_results:
#         st.info("👭 No strong buy suggestion found yet. Market may be flat or no strong signals.")
#     else:
#         for result in morning_results:
#             st.success(f"✅ Suggested Buy: **{result['ticker']}**")
#             st.markdown(f"- RSI: `{result['rsi']:.2f}`")
#             st.markdown(f"- SMA20: `{result['sma']:.2f}`")
#             st.markdown(f"- Close: `{result['close']:.2f}`")
#             st.markdown(f"- Volume: `{result['volume']}`")
#             st.markdown(f"- 📌 Action: **Buy** near `{result['close']:.2f}`, Target: `{result['target']:.2f}`, Stop Loss: `{result['stop_loss']:.2f}`")

#     st.subheader("🔍 Analyze Individual Stock")
#     ticker = st.text_input("Enter Stock Ticker", "RELIANCE.NS").upper()

#     if ticker:
#         data = get_stock_data(ticker)
#         if data.empty:
#             st.error("No data found.")
#         else:
#             st.subheader("📈 Recent Stock Data")
#             st.dataframe(data.tail())

#             data = add_technical_indicators(data)
#             st.subheader("📈 Technical Indicators")
#             st.dataframe(data[['Close', 'rsi', 'sma_20', 'Volume']].tail())

#             model = train_model(data)
#             prediction = predict_next_day(model, data)
#             st.success(f"📌 ML Prediction: {'Price Up 📈' if prediction == 1 else 'Price Down 📉'}")

#             stop_loss, target_price = calculate_stop_loss_target(data)
#             st.info(f"🎯 Target: {target_price:.2f}, 🛌 Stop Loss: {stop_loss:.2f}")

#             profit = calculate_profit(data)
#             st.markdown(f"💰 **Estimated Profit (if bought 10 shares)**: ₹{profit:.2f}")

#             spike = detect_volume_spike(data)
#             if spike:
#                 st.warning("⚠️ Volume Spike Detected!")

#             signal = get_swing_signal(data)
#             st.info(f"🥊 Swing Trade Signal: {signal}")

#             # Drop NaNs for candlestick chart rendering
#             data_for_chart = data[['Open', 'High', 'Low', 'Close']].dropna()

#             if data_for_chart.empty:
#                 st.warning("⚠️ Not enough data to render chart.")
#             else:
#                 fig = go.Figure(data=[go.Candlestick(
#                     x=data_for_chart.index,
#                     open=data_for_chart['Open'],
#                     high=data_for_chart['High'],
#                     low=data_for_chart['Low'],
#                     close=data_for_chart['Close']
#                 )])
#                 st.plotly_chart(fig, use_container_width=True)

#             csv = data.to_csv(index=True).encode("utf-8")
#             st.download_button("⬇️ Download CSV", csv, "stock_data.csv", "text/csv")

#     st.subheader("📦 Batch Prediction")
#     raw_input = st.text_area("Enter comma-separated tickers", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, LT.NS, "
#     "SBIN.NS, AXISBANK.NS, KOTAKBANK.NS, BHARTIARTL.NS, ASIANPAINT.NS, BAJFINANCE.NS, DMART.NS, "
#     "MARUTI.NS, SUNPHARMA.NS, WIPRO.NS, TECHM.NS, HCLTECH.NS, POWERGRID.NS, NTPC.NS, ONGC.NS, "
#     "COALINDIA.NS, ULTRACEMCO.NS, TITAN.NS, NESTLEIND.NS, BAJAJFINSV.NS, INDUSINDBK.NS, JSWSTEEL.NS, "
#     "HDFCLIFE.NS, ADANIENT.NS, ADANIPORTS.NS, DIVISLAB.NS, GRASIM.NS, BRITANNIA.NS, HEROMOTOCO.NS, "
#     "EICHERMOT.NS, TATASTEEL.NS, HINDALCO.NS, SBILIFE.NS, UPL.NS, CIPLA.NS, BAJAJ-AUTO.NS, DRREDDY.NS, "
#     "TATAMOTORS.NS, BPCL.NS, SHREECEM.NS, M&M.NS, APOLLOHOSP.NS")
#     tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

#     if st.button("Run Batch Prediction"):
#         results = batch_predict(tickers)
#         ups = [k for k, v in results.items() if v == 1]
#         downs = [k for k, v in results.items() if v == 0]
#         print("📊 Prediction Results:", results)
#         st.markdown("### 📈 Likely to Go Up")
#         for s in ups:
#             st.success(f"✅ {s}")

#         st.markdown("### 📉 Likely to Go Down")
#         for s in downs:
#             st.error(f"❌ {s}")

#         best_picks = get_best_pick(results)
#         if best_picks:
#             st.markdown("🏆 **Best Picks for Next Day**")
#             for pick in best_picks:
#                 st.markdown(
#                     f"✅ **{pick['ticker']}** — Score: {pick['score']}/4  \n"
#                     f"💹 RSI: {pick['rsi']:.1f}, SMA20: {pick['sma']:.1f}, Close: {pick['close']:.1f}, Volume: {pick['volume']}"
#                 )
#         else:
#             st.info("No best picks found based on current criteria.")

# if __name__ == "__main__":
#     main()




# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import time

# from utils import (
#     get_stock_data, add_technical_indicators, train_model,
#     predict_next_day, batch_predict, calculate_stop_loss_target,
#     detect_volume_spike, get_swing_signal, calculate_profit,
#     get_morning_suggestions, get_best_pick ,save_swing_pick, load_swing_picks, clear_swing_picks
# )

# st.set_page_config(page_title="Stock Advisor", layout="wide")

# def main():
#     st.title("📈 Advanced Stock Market Advisor")

#     # Morning Live Dashboard
#     st.subheader("🌅 Morning Dashboard - Live Suggestions")
#     if st.button("🔄 Refresh Suggestions Now"):
#         st.session_state.last_refresh = time.time()

#     if "last_refresh" not in st.session_state:
#         st.session_state.last_refresh = time.time()

#     if time.time() - st.session_state.last_refresh >= 300:
#         st.session_state.last_refresh = time.time()
#         st.experimental_rerun()

#     morning_results = get_morning_suggestions()
#     if not morning_results:
#         st.info("👭 No strong buy suggestion found yet.")
#     else:
#         for result in morning_results:
#             st.success(f"✅ Suggested Buy: **{result['ticker']}**")
#             st.markdown(f"- RSI: `{result['rsi']:.2f}`")
#             st.markdown(f"- SMA20: `{result['sma']:.2f}`")
#             st.markdown(f"- Close: `{result['close']:.2f}`")
#             st.markdown(f"- Volume: `{result['volume']}`")
#             st.markdown(f"- 📌 Action: **Buy near** `{result['close']:.2f}`, 🎯 Target: `{result['target']:.2f}`, 🛑 Stop Loss: `{result['stop_loss']:.2f}`")

# # NIFTY 50 टिकर्स की लिस्ट
#     NIFTY50_TICKERS = [
#                 "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
#                 "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", 
#                 "ASIANPAINT.NS", "BAJFINANCE.NS", "DMART.NS", "MARUTI.NS", "SUNPHARMA.NS", "WIPRO.NS",
#                 "TECHM.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS",
#                 "ULTRACEMCO.NS", "TITAN.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS",
#                 "JSWSTEEL.NS", "HDFCLIFE.NS", "ADANIENT.NS", "ADANIPORTS.NS", "DIVISLAB.NS", 
#                 "GRASIM.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TATASTEEL.NS", 
#                 "HINDALCO.NS", "SBILIFE.NS", "UPL.NS", "CIPLA.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS",
#                 "TATAMOTORS.NS", "BPCL.NS", "SHREECEM.NS", "M&M.NS", "APOLLOHOSP.NS"
#             ]

#     # Individual Stock Analysis
#     st.subheader("🔍 Analyze Individual Stock")
#     use_dropdown = st.radio("Select Mode", ["Dropdown", "Manual Entry"])

#     if use_dropdown == "Dropdown":
#         ticker = st.selectbox("🔍 NIFTY 50 स्टॉक चुनें", options=NIFTY50_TICKERS)
#     else:
#         ticker = st.text_input("✍️ टाइप करें स्टॉक टिकर (जैसे RELIANCE.NS)", "RELIANCE.NS").upper()
        
#     st.write(f"आपने चुना है: {ticker}")

#     if ticker:
#         try:
#             data = get_stock_data(ticker)
#         except Exception as e:
#             st.error(f"❌ Error fetching data: {e}")
#             return

#         if data.empty:
#             st.error("No data found.")
#         else:
#             st.subheader("📈 Recent Stock Data")
#             st.dataframe(data.tail())

#             data = add_technical_indicators(data)
#             st.subheader("📊 Technical Indicators")
#             st.dataframe(data[['Close', 'rsi', 'sma_20', 'Volume']].tail())

#             model = train_model(data)
#             prediction = predict_next_day(model, data)
#             st.success(f"📌 ML Prediction: {'📈 Price Up' if prediction == 1 else '📉 Price Down'}")

#             stop_loss, target_price = calculate_stop_loss_target(data)
#             st.info(f"🎯 Target: {target_price:.2f} | 🛑 Stop Loss: {stop_loss:.2f}")

#             profit = calculate_profit(data)
#             st.markdown(f"💰 **Estimated Profit (10 shares): ₹{profit:.2f}**")

#             if detect_volume_spike(data):
#                 st.warning("⚠️ Volume Spike Detected!")

#             signal = get_swing_signal(data)
#             st.info(f"🥊 Swing Signal: {signal}")

#             # RSI + SMA20 Touch alert
#             try:
#                 latest = data.dropna().iloc[-1]
#                 if float(latest['rsi']) < 35 and abs(float(latest['Close']) - float(latest['sma_20'])) < 2:
#                     st.info("📉 RSI is low and price is near SMA20 — Possible Reversal Opportunity!")
#             except Exception as e:
#                 st.warning(f"Error checking RSI/SMA alert: {e}")

#             # Candlestick chart
#             data_for_chart = data[['Open', 'High', 'Low', 'Close']].dropna()
#             if not data_for_chart.empty:
#                 fig = go.Figure(data=[go.Candlestick(
#                     x=data_for_chart.index,
#                     open=data_for_chart['Open'],
#                     high=data_for_chart['High'],
#                     low=data_for_chart['Low'],
#                     close=data_for_chart['Close']
#                 )])
#                 fig.update_layout(title=f"{ticker} - Candlestick Chart", xaxis_rangeslider_visible=False)
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("⚠️ Not enough data to show candlestick chart.")

#             csv = data.to_csv(index=True).encode("utf-8")
#             st.download_button("⬇️ Download CSV", csv, "stock_data.csv", "text/csv")

#     # Batch Prediction Section
#     st.subheader("📦 Batch Prediction")
#     raw_input = st.text_area("Enter comma-separated tickers", 
#         "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, LT.NS, "
#     "SBIN.NS, AXISBANK.NS, KOTAKBANK.NS, BHARTIARTL.NS, ASIANPAINT.NS, BAJFINANCE.NS, DMART.NS, "
#     "MARUTI.NS, SUNPHARMA.NS, WIPRO.NS, TECHM.NS, HCLTECH.NS, POWERGRID.NS, NTPC.NS, ONGC.NS, "
#     "COALINDIA.NS, ULTRACEMCO.NS, TITAN.NS, NESTLEIND.NS, BAJAJFINSV.NS, INDUSINDBK.NS, JSWSTEEL.NS, "
#     "HDFCLIFE.NS, ADANIENT.NS, ADANIPORTS.NS, DIVISLAB.NS, GRASIM.NS, BRITANNIA.NS, HEROMOTOCO.NS, "
#     "EICHERMOT.NS, TATASTEEL.NS, HINDALCO.NS, SBILIFE.NS, UPL.NS, CIPLA.NS, BAJAJ-AUTO.NS, DRREDDY.NS, "
#     "TATAMOTORS.NS, BPCL.NS, SHREECEM.NS, M&M.NS, APOLLOHOSP.NS"
#     )
#     tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

#     if st.button("Run Batch Prediction"):
#         results = batch_predict(tickers)
#         ups = [k for k, v in results.items() if v == 1]
#         downs = [k for k, v in results.items() if v == 0]

#         st.markdown("### 📈 Likely to Go Up")
#         for s in ups:
#             st.success(f"✅ {s}")

#         st.markdown("### 📉 Likely to Go Down")
#         for s in downs:
#             st.error(f"❌ {s}")

#         best_picks = get_best_pick(results)
#         if best_picks:
#             st.markdown("🏆 **Best Picks for Next Day**")
#             for pick in best_picks:
#                 st.markdown(
#                     f"✅ **{pick['ticker']}** — Score: {pick['score']}/4  \n"
#                     f"💹 RSI: {pick['rsi']:.1f}, SMA20: {pick['sma']:.1f}, Close: {pick['close']:.1f}, Volume: {pick['volume']}"
#                 )
#         else:
#             st.info("No strong picks found today.")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

from utils import (
    get_stock_data, add_technical_indicators, train_model,
    predict_next_day, batch_predict, calculate_stop_loss_target,
    detect_volume_spike, get_swing_signal, calculate_profit,
    get_morning_suggestions, get_best_pick, save_swing_pick, load_swing_picks, clear_swing_picks
)

st.set_page_config(page_title="Stock Advisor", layout="wide")

def main():
    st.subheader("📈 Advanced Stock Market Advisor")

    # Morning Live Dashboard
    st.subheader("🌅 Morning Dashboard - Live Suggestions")
    if st.button("🔄 Refresh Suggestions Now"):
        st.session_state.last_refresh = time.time()

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    if time.time() - st.session_state.last_refresh >= 300:
        st.session_state.last_refresh = time.time()
        st.experimental_rerun()

    morning_results = get_morning_suggestions()
    if not morning_results:
        st.info("👭 No strong buy suggestion found yet.")
    else:
        for result in morning_results:
            st.success(f"✅ Suggested Buy: **{result['ticker']}**")
            st.markdown(f"- RSI: `{result['rsi']:.2f}`")
            st.markdown(f"- SMA20: `{result['sma']:.2f}`")
            st.markdown(f"- Close: `{result['close']:.2f}`")
            st.markdown(f"- Volume: `{result['volume']}`")
            st.markdown(f"- 📌 Action: **Buy near** `{result['close']:.2f}`, 🎯 Target: `{result['target']:.2f}`, 🛑 Stop Loss: `{result['stop_loss']:.2f}`")

    # NIFTY 50 टिकर्स की लिस्ट
    NIFTY50_TICKERS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
        "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", 
        "ASIANPAINT.NS", "BAJFINANCE.NS", "DMART.NS", "MARUTI.NS", "SUNPHARMA.NS", "WIPRO.NS",
        "TECHM.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS",
        "ULTRACEMCO.NS", "TITAN.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS",
        "JSWSTEEL.NS", "HDFCLIFE.NS", "ADANIENT.NS", "ADANIPORTS.NS", "DIVISLAB.NS", 
        "GRASIM.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TATASTEEL.NS", 
        "HINDALCO.NS", "SBILIFE.NS", "UPL.NS", "CIPLA.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS",
        "TATAMOTORS.NS", "BPCL.NS", "SHREECEM.NS", "M&M.NS", "APOLLOHOSP.NS"
    ]

    # Individual Stock Analysis
    st.subheader("🔍 Analyze Individual Stock")
    use_dropdown = st.radio("Select Mode", ["Dropdown", "Manual Entry"])

    if use_dropdown == "Dropdown":
        ticker = st.selectbox("🔍 NIFTY 50 स्टॉक चुनें", options=NIFTY50_TICKERS)
    else:
        ticker = st.text_input("✍️ टाइप करें स्टॉक टिकर (जैसे RELIANCE.NS)", "RELIANCE.NS").upper()
        
    st.write(f"आपने चुना है: {ticker}")

    if ticker:
        try:
            data = get_stock_data(ticker)
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return

        if data.empty:
            st.error("No data found.")
        else:
            st.subheader("📈 Recent Stock Data")
            st.dataframe(data.tail())

            data = add_technical_indicators(data)
            st.subheader("📊 Technical Indicators")
            st.dataframe(data[['Close', 'rsi', 'sma_20', 'Volume']].tail())

            model = train_model(data)
            prediction = predict_next_day(model, data)
            st.success(f"📌 ML Prediction: {'📈 Price Up' if prediction == 1 else '📉 Price Down'}")

            stop_loss, target_price = calculate_stop_loss_target(data)
            st.info(f"🎯 Target: {target_price:.2f} | 🛑 Stop Loss: {stop_loss:.2f}")

            profit = calculate_profit(data)
            st.markdown(f"💰 **Estimated Profit (10 shares): ₹{profit:.2f}**")

            if detect_volume_spike(data):
                st.warning("⚠️ Volume Spike Detected!")

            signal = get_swing_signal(data)
            st.info(f"🥊 Swing Signal: {signal}")

            # RSI + SMA20 Touch alert
            try:
                latest = data.dropna().iloc[-1]
                if float(latest['rsi']) < 35 and abs(float(latest['Close']) - float(latest['sma_20'])) < 2:
                    st.info("📉 RSI is low and price is near SMA20 — Possible Reversal Opportunity!")
            except Exception as e:
                st.warning(f"Error checking RSI/SMA alert: {e}")

            # Candlestick chart
            data_for_chart = data[['Open', 'High', 'Low', 'Close']].dropna()
            if not data_for_chart.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=data_for_chart.index,
                    open=data_for_chart['Open'],
                    high=data_for_chart['High'],
                    low=data_for_chart['Low'],
                    close=data_for_chart['Close']
                )])
                fig.update_layout(title=f"{ticker} - Candlestick Chart", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Not enough data to show candlestick chart.")

            csv = data.to_csv(index=True).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "stock_data.csv", "text/csv")

    # Batch Prediction Section
    st.subheader("📦 Batch Prediction")
    raw_input = st.text_area("Enter comma-separated tickers", 
        "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, LT.NS, "
        "SBIN.NS, AXISBANK.NS, KOTAKBANK.NS, BHARTIARTL.NS, ASIANPAINT.NS, BAJFINANCE.NS, DMART.NS, "
        "MARUTI.NS, SUNPHARMA.NS, WIPRO.NS, TECHM.NS, HCLTECH.NS, POWERGRID.NS, NTPC.NS, ONGC.NS, "
        "COALINDIA.NS, ULTRACEMCO.NS, TITAN.NS, NESTLEIND.NS, BAJAJFINSV.NS, INDUSINDBK.NS, JSWSTEEL.NS, "
        "HDFCLIFE.NS, ADANIENT.NS, ADANIPORTS.NS, DIVISLAB.NS, GRASIM.NS, BRITANNIA.NS, HEROMOTOCO.NS, "
        "EICHERMOT.NS, TATASTEEL.NS, HINDALCO.NS, SBILIFE.NS, UPL.NS, CIPLA.NS, BAJAJ-AUTO.NS, DRREDDY.NS, "
        "TATAMOTORS.NS, BPCL.NS, SHREECEM.NS, M&M.NS, APOLLOHOSP.NS"
    )
    tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

    if st.button("Run Batch Prediction"):
        results = batch_predict(tickers)
        ups = [k for k, v in results.items() if v == 1]
        downs = [k for k, v in results.items() if v == 0]

        st.markdown("### 📈 Likely to Go Up")
        for s in ups:
            st.success(f"✅ {s}")

        st.markdown("### 📉 Likely to Go Down")
        for s in downs:
            st.error(f"❌ {s}")

        best_picks = get_best_pick(results)
        if best_picks:
            st.markdown("🏆 **Best Picks for Next Day**")
            for pick in best_picks:
                st.markdown(
                    f"✅ **{pick['ticker']}** — Score: {pick['score']}/4  \n"
                    f"💹 RSI: {pick['rsi']:.1f}, SMA20: {pick['sma']:.1f}, Close: {pick['close']:.1f}, Volume: {pick['volume']}"
                )
        else:
            st.info("No strong picks found today.")

if __name__ == "__main__":
    main()
