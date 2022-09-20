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


PUBLIC_ENV = os.environ.get('B_PK') 
if PUBLIC_ENV:
    print('pk from env------------------------')
    PUBLIC = PUBLIC_ENV
else:
    PUBLIC = 'WZWyDV3jEFP6c-xnoNtYbkhK'

SECRET_ENV = os.environ.get('B_SK') 
if SECRET_ENV:
    print('sk from env------------------------')
    SECRET = SECRET_ENV
else:
    SECRET = 'E-wC7jcD65Bfy-4qnvbD4YWWTNET6cN_nARAAyP3io0re4ab'

BASE_URL = 'https://www.bitmex.com'

timeframes = {'1m':1, '5m':5, '15m':15, '1h':60, '2h':120, '4h':240, '1d':1440}

def generate_signature(endpoint, expires, data, method):
    message = 'GET' + endpoint + '?' + urlencode(data) + expires if len(data) > 0 else method + endpoint + expires
    return hmac.new(SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()

def make_request(endpoint, data):
    headers = dict()
    expires = str(int(time.time()) + 5)
    headers['api-expires'] = expires
    headers['api-key'] = PUBLIC
    headers['api-signature'] = generate_signature(endpoint, expires, data, 'GET')
    try:
        response = requests.get(BASE_URL + endpoint, params=data, headers=headers)
    except Exception as e:
        print(f'connection error while making {method} request to {endpoint}: {e}')
    if response.status_code == 200:
        return response.json()
    else:
        print(f'error while making {method} request to {endpoint}: {response.status_code}')
        return None

def get_historical_candles(symbol, timeframe):
    data = dict()
    data['symbol'] = symbol
    data['partial'] = True
    data['binSize'] = timeframe
    data['count'] = 1000
    data['reverse'] = True
    raw_candles = make_request("/api/v1/trade/bucketed", data)   
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
    