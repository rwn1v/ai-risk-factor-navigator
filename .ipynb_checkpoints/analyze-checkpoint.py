from data_loader import get_etf_returns, get_fama_french_factors, align data

returns = get_etf_returns('SPY')
ff = get_fama_french_factos()
aligned = align_data(returns, ff)

print(aligned.head())