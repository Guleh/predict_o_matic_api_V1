from binance.client import Client
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
import datetime
import json
import pandas as pd
import numpy as np
import os


BASE_URL = 'https://www.bitmex.com'

timeframes = {'1m':1, '5m':5, '15m':15, '1h':60, '2h':120, '4h':240, '1d':1440}

def make_request(symbol, timeframe):
    try:
        response = requests.get(f'https://www.bitmex.com/api/v1/trade/bucketed?binSize={timeframe}&partial=true&symbol={symbol}&count=1000&reverse=true')
        print(response)
    except Exception as e:
        print(f'connection error while making {method} request to {endpoint}: {e}')
    if response.status_code == 200:
        return response.json()
    else:
        print(f'error while making request to {endpoint}: {response.status_code}')
        return None

def get_historical_candles(symbol, timeframe):
    print(symbol)
    print(timeframe)
    raw_candles = make_request(symbol, timeframe)   
    if raw_candles is not None:
        raw_candles.reverse()
        df = pd.DataFrame.from_dict(raw_candles)    
    df['timestamp'] = pd.to_datetime(df.iloc[:,0])        
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', "vwap", 
                      "turnover"]]
    raw_candles = df[['timestamp', 'open', 'high', 'low', 'close']].rename(columns = {'timestamp':'time'})
    raw_candles['time'] = raw_candles.time.values.astype(np.int64) // 10 ** 9
    df.set_index("timestamp", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = 'coerce')
    df['returns'] = np.log(df.close / df.close.shift(1))
    df.dropna(inplace = True)    
    return df, raw_candles


def get_data(symbol, tf):
    data, raw_candles = get_historical_candles(symbol, tf)
    return data, raw_candles.iloc[-100:,:]
    
