# import pandas as pd
# import numpy as np

# def calculate_rsi(data, period=14, column='Close'):
#     """
#     Calculate Relative Strength Index (RSI) for given data.
#     data: pandas DataFrame with price data
#     period: RSI calculation period (default 14)
#     column: price column to use (default 'Close')
#     """
#     delta = data[column].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)

#     avg_gain = gain.rolling(window=period, min_periods=period).mean()
#     avg_loss = loss.rolling(window=period, min_periods=period).mean()

#     rs = avg_gain / avg_loss
#     rsi = 100 - (100 / (1 + rs))
#     data['rsi'] = rsi.fillna(0)  # या आप fillna(method='bfill') भी कर सकते हैं

#     return data


# def calculate_sma(data, period=20, column='Close'):
#     """
#     Calculate Simple Moving Average (SMA) for given data.
#     data: pandas DataFrame with price data
#     period: SMA period (default 20)
#     column: price column to use (default 'Close')
#     """
#     data[f'sma_{period}'] = data[column].rolling(window=period, min_periods=1).mean()
#     return data


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def calculate_sma(series, period=20):
    return series.rolling(window=period, min_periods=1).mean()
