import yfinance as yf
import pandas as pd
import os
import json
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
import xgboost as xgb

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

SWING_PICK_FILE = "swing_trades.json"

# def get_stock_data(ticker):
#     try:
#         df = yf.download(ticker, period="6mo", interval="1d")
#         return df if not df.empty else pd.DataFrame()
#     except:
#         return pd.DataFrame()

def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty or 'Close' not in df.columns:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


# def add_technical_indicators(df):
#     close_series = df['Close']
#     if isinstance(close_series, pd.DataFrame):
#         close_series = close_series.squeeze()
#     df['rsi'] = RSIIndicator(close_series).rsi()
#     df['sma_20'] = close_series.rolling(window=20).mean()
#     df['target'] = (close_series.shift(-1) > close_series).astype(int)
#     return df.dropna()

from ta.momentum import RSIIndicator

def add_technical_indicators(df):
    # Step 1: Null या empty dataframe को छोड़ दें
    if df is None or df.empty:
        return pd.DataFrame()

    # Step 2: 'Close' column होना चाहिए
    if 'Close' not in df.columns:
        return pd.DataFrame()

    # Step 3: Close column को safe तरीके से Series में बदलें
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    # Step 4: NaN check करें - अगर close Series है तभी चेक करें
    if not isinstance(close, (pd.Series, pd.DataFrame)):
        # अगर close Series नहीं है, तो df लौटाओ (या खाली df)
        return pd.DataFrame()

    # NaN चेक करते वक्त double all() से बचने के लिए पहले type चेक करें
    if (hasattr(close, 'isnull') and close.isnull().all()):
        return pd.DataFrame()

    # Step 5: Indicators लगाएं
    try:
        df['rsi'] = RSIIndicator(close).rsi()
        df['sma_20'] = close.rolling(window=20).mean()
        df['target'] = (close.shift(-1) > close).astype(int)
    except Exception as e:
        print(f"Indicator error: {e}")
        return pd.DataFrame()

    return df.dropna()



def train_model(df):
    X = df[['rsi', 'sma_20']]
    y = df['target']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def predict_next_day(model, df):
    X_last = df[['rsi', 'sma_20']].iloc[-1:].values
    return model.predict(X_last)[0]

def batch_predict(tickers):
    results = {}
    for t in tickers:
        df = get_stock_data(t)
        if df.empty: 
            continue
        df = add_technical_indicators(df)
        model = train_model(df)
        pred = predict_next_day(model, df)
        results[t] = pred
    return results

def calculate_stop_loss_target(df):
    last_close = df['Close'].iloc[-1]
    stop_loss = float(last_close * 0.97)
    target = float(last_close * 1.05)
    return stop_loss, target

def detect_volume_spike(df):
    avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
    current_volume = df['Volume'].iloc[-1]

    if isinstance(avg_volume, (pd.Series, pd.DataFrame)):
        avg_volume = avg_volume.values[0] if len(avg_volume) > 0 else float('nan')
    if isinstance(current_volume, (pd.Series, pd.DataFrame)):
        current_volume = current_volume.values[0] if len(current_volume) > 0 else float('nan')

    if pd.isna(avg_volume) or pd.isna(current_volume):
        return False

    return current_volume > 1.5 * avg_volume

def get_swing_signal(df):
    rsi_last = df['rsi'].iloc[-1]
    if rsi_last < 30:
        return "Buy (Oversold)"
    elif rsi_last > 70:
        return "Sell (Overbought)"
    else:
        return "Hold"

def calculate_profit(df):
    buy_price = df['Close'].iloc[-2]
    sell_price = df['Close'].iloc[-1]
    profit = (sell_price - buy_price) * 10
    return float(profit)

def get_morning_suggestions():
    suggestions = []
    for ticker in NIFTY50_TICKERS:
        try:
            df = yf.download(ticker, period="15d", interval="5m", progress=False)
            if df.empty or len(df) < 20:
                continue
            df['rsi'] = RSIIndicator(df['Close']).rsi()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['avg_vol'] = df['Volume'].rolling(window=20).mean()
            latest = df.iloc[-1]
            close = float(latest['Close'])
            rsi = float(latest['rsi'])
            sma = float(latest['sma_20'])
            vol = float(latest['Volume'])
            avg_vol = float(latest['avg_vol'])

            if rsi < 35 and abs(close - sma) < 2 and vol > 1.2 * avg_vol:
                stop_loss = close * 0.98
                target = close * 1.03
                suggestions.append({
                    "ticker": ticker,
                    "rsi": rsi,
                    "sma": sma,
                    "close": close,
                    "volume": int(vol),
                    "target": target,
                    "stop_loss": stop_loss
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    return sorted(suggestions, key=lambda x: x['rsi'])[:5]

def get_best_pick(results):
    best_picks = []
    for ticker, prediction in results.items():
        if prediction != 1:
            continue
        try:
            df = get_stock_data(ticker)
            df = add_technical_indicators(df)
            if df.empty:
                continue
            latest = df.iloc[-1]
            rsi = float(latest['rsi'])
            sma = float(latest['sma_20'])
            close = float(latest['Close'])
            volume = int(latest['Volume'])
            avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1])

            score = 0
            if 35 < rsi < 65:
                score += 1
            if close > sma:
                score += 1
            if volume > avg_vol:
                score += 1
            if rsi > 40:
                score += 1

            if score >= 2:
                best_picks.append({
                    "ticker": ticker,
                    "score": score,
                    "rsi": rsi,
                    "sma": sma,
                    "close": close,
                    "volume": volume
                })
        except Exception as e:
            print(f"⚠️ Error in best pick for {ticker}: {e}")
            continue
    return sorted(best_picks, key=lambda x: x['score'], reverse=True)[:5]

def save_swing_pick(pick):
    if os.path.exists(SWING_PICK_FILE):
        with open(SWING_PICK_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    if not any(p['ticker'] == pick['ticker'] for p in data):
        data.append(pick)
        with open(SWING_PICK_FILE, "w") as f:
            json.dump(data, f, indent=2)

def load_swing_picks():
    if os.path.exists(SWING_PICK_FILE):
        with open(SWING_PICK_FILE, "r") as f:
            return json.load(f)
    return []

def clear_swing_picks():
    if os.path.exists(SWING_PICK_FILE):
        os.remove(SWING_PICK_FILE)
