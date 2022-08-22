from binance.client import Client
import pandas as pd
import numpy as np
import os

PUBLIC_ENV = os.environ.get('B_PK') 
if PUBLIC_ENV:
    print('pk from env------------------------')
    PUBLIC = PUBLIC_ENV
else:
    PUBLIC = 'GGnXPyNDlN6iNkSzPXCdri7O3YEZT76d9KSdwYbp8dm1r2EiIaa5uyBvoV6vN8cI'

SECRET_ENV = os.environ.get('B_SK') 
if SECRET_ENV:
    print('sk from env------------------------')
    SECRET = SECRET_ENV
else:
    SECRET = 'hKr3M9QqU2NDPjeFY2V3ULsDztevYfwG1orn9pSFePhARXZBtcgx2rXQPKBnoUwJ'

client = Client(api_key = PUBLIC, api_secret = SECRET, tld = 'com')


def get_history(symbol, interval, start = None, end = None):
    if start is None:
         start = client._get_earliest_valid_timestamp(symbol='BTCUSDT', interval = interval)
    bars = client.get_historical_klines(symbol = symbol, interval = interval, 
                                        start_str = start, end_str = end, limit = 1000)
    df = pd.DataFrame(bars)
    df['Date'] = pd.to_datetime(df.iloc[:,0],unit = 'ms')
    df.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close Time", 
                  "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume", 
                  "Taker Buy Quote Asset Volume", "Ignore", "Date"]
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns = {'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
    df.set_index("Date", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = 'coerce')
    df['returns'] = np.log(df.close / df.close.shift(1))
    df.dropna(inplace = True)   
    return df