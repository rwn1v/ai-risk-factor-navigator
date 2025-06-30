from data_loader import get_etf_returns, get_fama_french_factors, align_data
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from openai import OpenAI

print("[analyze.py] Module loaded")

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_factor_regression(data):
    print("[run_factor_regression] Starting regression")
    X = data[['Mkt-RF', 'SMB', 'HML']]
    y = data['Excess']
    model = LinearRegression().fit(X, y)

    results = {
        'Intercept (alpha)': model.intercept_,
        'Beta Mkt-RF': model.coef_[0],
        'Beta SMB': model.coef_[1],
        'Beta HML': model.coef_[2],
        'R-squared': model.score(X, y)
    }
    print("[run_factor_regression] Completed", results)
    return results

# Debug example
try:
    returns = get_etf_returns('ARKK')
    ff = get_fama_french_factors()
    merged = align_data(returns, ff)
    results = run_factor_regression(merged)
    print("[DEBUG] Example Results:", results)
except Exception as e:
    print("[ERROR] Example debug run failed:", e)

def summarize_factor_regression(results: dict, ticker: str) -> str:
    print("[summarize_factor_regression] Generating summary for", ticker)
    prompt = f"""
    You are a financial analyst. Interpret the following factor regression results for ETF {ticker}.
    Keep it concise and use plain language suitable for a portfolio manager.

    Results:
        - Intercept (alpha):  {results['Intercept (alpha)']:.4f}
        - Beta Mkt-RF: {results['Beta Mkt-RF']:.4f}
        - Beta SMB: {results['Beta SMB']:.4f}
        - Beta HML:{results['Beta HML']:.4f}
        - R-squared: {results['R-squared']:.4f}

    Return a short paragraph explaining what these numbers suggest about the ETF's behavior and factor exposures.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": 'user', "content": prompt}],
        temperature=0.6,
    )
    summary = response.choices[0].message.content.strip()
    print("[summarize_factor_regression] Summary:", summary)
    return summary

def summarize_active_exposures(etf_results: dict, bm_results: dict, etf_ticker: str, benchmark_ticker: str) -> str:
    print("[summarize_active_exposures] Comparing", etf_ticker, "vs", benchmark_ticker)
    active = {
        "Beta Mkt-RF": etf_results["Beta Mkt-RF"] - bm_results["Beta Mkt-RF"],
        "Beta SMB": etf_results["Beta SMB"] - bm_results["Beta SMB"],
        "Beta HML": etf_results["Beta HML"] - bm_results["Beta HML"],
        "Alpha Diff": etf_results["Intercept (alpha)"] - bm_results["Intercept (alpha)"]
    }

    prompt = f"""
    You are a financial analyst. Compare the factor exposures of ETF {etf_ticker} against its benchmark {benchmark_ticker}.

    Factor exposures (Beta):
    {etf_ticker}:
        - Beta Mkt-RF: {etf_results['Beta Mkt-RF']:.4f}
        - Beta SMB: {etf_results['Beta SMB']:.4f}
        - Beta HML: {etf_results['Beta HML']:.4f}
        - Alpha: {etf_results['Intercept (alpha)']:.4f}

    {benchmark_ticker}:
        - Beta Mkt-RF: {bm_results['Beta Mkt-RF']:.4f}
        - Beta SMB: {bm_results['Beta SMB']:.4f}
        - Beta HML: {bm_results['Beta HML']:.4f}
        - Alpha: {bm_results['Intercept (alpha)']:.4f}

    Active Exposures (ETF - Benchmark):
        - Beta Mkt-RF: {active['Beta Mkt-RF']:.4f}
        - Beta SMB: {active['Beta SMB']:.4f}
        - Beta HML: {active['Beta HML']:.4f}
        - Alpha Diff: {active['Alpha Diff']:.4f}

    Write a short paragraph that summarizes these differences and what they imply about how {etf_ticker} is positioned relative to {benchmark_ticker}. Use plain language.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    summary = response.choices[0].message.content.strip()
    print("[summarize_active_exposures] Summary:", summary)
    return summary

def summarize_cumulative_returns(cum_returns, etf_ticker, benchmark_ticker):
    print("[summarize_cumulative_returns] Generating cumulative return summary")
    etf_final = cum_returns['ETF'].iloc[-1]
    benchmark_final = cum_returns['Benchmark'].iloc[-1]
    outperformance = etf_final - benchmark_final

    direction = (
        "outperformed" if outperformance > 0 else
        "underperformed" if outperformance < 0 else
        "performed similarly to"
    )

    summary = (
        f"From the selected start date to the present, {etf_ticker.upper()} has {direction} {benchmark_ticker.upper()} "
        f"by a cumulative margin of {outperformance:.2%}. This suggests that {etf_ticker.upper()} "
        f"has delivered {'stronger' if outperformance > 0 else 'weaker' if outperformance < 0 else 'comparable'} performance "
        f"relative to the benchmark over the observed period."
    )
    print("[summarize_cumulative_returns] Summary:", summary)
    return summary
