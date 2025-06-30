import yfinance as yf
import pandas as pd
import pandas_datareader.data as web

def get_etf_returns(ticker, start='2000-01-01'):
    data = yf.download(ticker, start=start, progress=False, auto_adjust=False)['Adj Close']
    returns = data.pct_change().dropna()
    returns.name = ticker
    return returns

def get_fama_french_factors(start='2000-01-01'):
    ff = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start)[0]
    ff = ff / 100
    return ff

def align_data(etf, ff):
    if isinstance(ff.index, pd.PeriodIndex):
        ff.index = ff.index.to_timestamp(how='end').normalize()
    merged = etf.join(ff, how='inner')
    merged['Excess'] = merged.iloc[:, 0] - merged['RF']
    return merged

def get_etf_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName", ticker.upper())
    except Exception:
        return ticker.upper()
