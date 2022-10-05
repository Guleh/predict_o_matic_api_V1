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

def make_request(symbol, timeframe):
    try:
        response = requests.get(f'https://www.bitmex.com/api/v1/trade/bucketed?binSize={timeframe}&partial=true&symbol={symbol}&count=1000&reverse=true')
        print(response)
    except Exception as e:
        print(f'connection error while making request: {e}')
    if response.status_code == 200:
        return response.json()
    else:
        print(f'error while making request')
        return None

def get_historical_candles(symbol, timeframe):
    print(symbol)
    print(timeframe)
    temptimeframe = None
    if timeframe == '4h':
        timeframe = '1h'
        temptimeframe = '4h'
    raw_candles = make_request(symbol, timeframe)   
    if raw_candles is not None:
        raw_candles.reverse()
        df = pd.DataFrame.from_dict(raw_candles)    
    df['timestamp'] = pd.to_datetime(df.iloc[:,0])        
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', "vwap", "turnover"]]
    if temptimeframe:
        rows = []
        for ind in df.index:
            div = str(df['timestamp'][ind])[10:13]           
            if int(div)%4 == 0:
                if ind+3 < len(df):
                    highs = [int(df['high'][ind]), 
                             int(df['high'][ind+1]), 
                             int(df['high'][ind+2]), 
                             int(df['high'][ind+3])]
                    lows = [int(df['low'][ind]), 
                            int(df['low'][ind+1]), 
                            int(df['low'][ind+2]), 
                            int(df['low'][ind+3])]
                    vol = [int(df['volume'][ind]), 
                            int(df['volume'][ind+1]), 
                            int(df['volume'][ind+2]), 
                            int(df['volume'][ind+3])]
                    vwap = mean([(df['vwap'][ind]), 
                            (df['vwap'][ind+1]), 
                            (df['vwap'][ind+2]), 
                            (df['vwap'][ind+3])])
                    turnover = sum([(df['turnover'][ind]), 
                            (df['turnover'][ind+1]), 
                            (df['turnover'][ind+2]), 
                            (df['turnover'][ind+3])])
                    close = df['close'][ind+3]
                    close = df['close'][ind+3]
                    high = max(highs)
                    low = min(lows)
                    volume = sum(vol)
                    row = {
                            'timestamp':(df['timestamp'][ind]), 
                            'open':df['open'][ind],
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume,
                            'vwap': vwap,
                            'turnover': turnover
                          }
                    rows.append(row)
        df = pd.DataFrame(rows)
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
    
