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
    if timeframe != '1h' or timeframe != '1d':
        temptimeframe = timeframe
        tf = int(timeframe[0:1])
        timeframe = '1h'
    raw_candles = make_request(symbol, timeframe)   
    if raw_candles is not None:
        raw_candles.reverse()
        df = pd.DataFrame.from_dict(raw_candles)    
    df['timestamp'] = pd.to_datetime(df.iloc[:,0])        
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', "vwap", 
                      "turnover"]]
    if temptimeframe:
        rows = []
        for ind in df.index:
            div = str(df['timestamp'][ind])[10:13]           
            if int(div)%tf == 0:
                if ind+tf-1 < len(df):
                    highs = []
                    for i in range(0,tf):
                        highs.append(float(df['high'][ind+i]))
                    lows = []
                    for i in range(0,tf):
                        lows.append(float(df['low'][ind+i]))    
                    vol = []
                    for i in range(0,tf):
                        vol.append(float(df['volume'][ind+i])) 
                    vw = []
                    for i in range(0,tf):
                        vw.append(df['vwap'][ind+i])
                    to = []
                    for i in range(0,tf):
                        to.append(df['turnover'][ind+i])
                    
                    close = df['close'][ind+tf-1]
                    high = max(highs)
                    low = min(lows)
                    volume = sum(vol)  
                    vwap = mean(vw)  
                    turnover = sum(to)
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
    
