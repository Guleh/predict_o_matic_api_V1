from .models import Asset, Strategy
import numpy as np
import pandas as pd


def prepare_features(df, asset):
    strategy = Strategy.objects.get(asset=asset)
    window_s = strategy.window_s
    window_m = strategy.window_m
    window_l = strategy.window_l
    lag = strategy.lag
    df['dir'] = np.where(df['returns'] > 0, 1, 0)
    df['boll'] = (df['open'] - df['open'].rolling(window_m).mean()) / df['open'].rolling(window_m).std()
    df['min'] = df['open'].rolling(window_m).min() / df['open'] - 1
    df['max'] = df['open'].rolling(window_m).max() / df['open'] - 1
    df['volume'] = df['volume'].rolling(window_m).std()  
    df['volatility'] = df['returns'].rolling(window_m).std()  
    df['ema_s'] = EMA(df, window_s)
    df['ema_m'] = EMA(df, window_m)
    df['ema_l'] = EMA(df, window_l)
    df['ma_s'] = MA(df, window_s)
    df['ma_m'] = MA(df, window_m)
    df['ma_l'] = MA(df, window_l)
    df['roc10'] = ROC(df['open'], 10)
    df['roc30'] = ROC(df['open'], 30)
    df['mom10'] = MOM(df['open'], 10)
    df['mom30'] = MOM(df['open'], 30)
    df["rsi10"] = RSI(df['open'], 10)
    df["rsi30"] = RSI(df['open'], 30)
    df["rsi200"] = RSI(df['open'], 200)
    df["k10"] = STOK(df['open'], df['low'], df['high'], 10)
    df["k30"] = STOK(df['open'], df['low'], df['high'], 30)
    df["k200"] = STOK(df['open'], df['low'], df['high'], 200)
    df["d10"] = STOD(df['open'], df['low'], df['high'], 10)
    df["d30"] = STOD(df['open'], df['low'], df['high'], 30)
    df["d200"] = STOD(df['open'], df['low'], df['high'], 200)
    features = ['dir', 'ema_s', 'ema_m', 'ema_l','ma_s', 'ma_m', 'ma_l', 'boll', 'min', 'max', 'volume', 'volatility',
               'roc10', 'roc30', 'mom10','mom30', 'rsi10','rsi30','rsi200', 'k10', 'k30', 'k200',
               'd10', 'd30', 'd200']
    df, cols = LAG(df, lag, features)
    df['dir'] = df['dir'].shift(periods=-1)
    df = df.replace(np.nan,0)
    df['dir'] = df['dir'].apply(np.int64) 
    df.drop(df.tail(1).index,inplace = True)
    return df, cols



def LAG(df, lag, features):
    cols = []
    for f in features:
        for l in range(1, lag +1):
            col = f'{f}_lag{l}'
            column = df[f].shift(l)
            cols.append(col)            
            df = pd.concat([df, column.rename(f'{f}_lag{l}')], axis=1)
    df.dropna(inplace = True) 
    return df, cols

def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name = 'ROC_' + str(n))
    return ROC

def MOM(df, n):
    MOM = pd.Series(df.diff(n), name = 'Momentum_' + str(n))
    return MOM

def EMA(df, n):
    EMA = pd.Series(df['open'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA

def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    u[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean(u[:period]) #sum of average gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period]) #sum of average losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean()
    d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100/(1+rs)

def STOK(close, low, high, n): 
    STOK = ((close - low.rolling(n).min())/(high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def STOD(close, low, high, n):
    return STOK(close, low, high, n).rolling(3).mean()
    
def MA(df, n):
    MA = pd.Series(df["open"].rolling(n, min_periods = n).mean(), name='MA_'+ str(n))
    return MA