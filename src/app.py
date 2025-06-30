import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from data_loader import get_etf_returns, get_fama_french_factors, align_data, get_etf_name
from analyze import run_factor_regression, summarize_active_exposures, summarize_cumulative_returns

app = dash.Dash(__name__)
app.title = "AI Risk Factor Navigator"

app.layout = html.Div([
    html.H2("AI Risk Factor Navigator"),

    html.Div([
        dcc.Input(id='etf-ticker', type='text', placeholder='Enter ETF Ticker (e.g. ARKK)', debounce=True),
        dcc.Input(id='benchmark-ticker', type='text', placeholder='Enter Benchmark Ticker (e.g. SPY)', debounce=True),
        html.Button('Run Analysis', id='analyze-btn', n_clicks=0),
    ], style={'margin-bottom': '20px'}),

    html.Div(id='summary-output', style={'whiteSpace': 'pre-line', 'margin-top': '20px'}),

    html.H4(id='etf-name', style={'margin-top': '10px'}),

    dcc.Graph(id='factor-exposure-graph'),
    dcc.Graph(id='cumulative-return-graph'),

    html.Div(id='gpt-summary', style={'whiteSpace': 'pre-line', 'margin-top': '40px', 'font-style': 'italic'}),
    html.Div(id='gpt-cumulative-summary', style={'whiteSpace': 'pre-line', 'margin-top': '20px', 'font-style': 'italic'})
])

@app.callback(
    [Output('factor-exposure-graph', 'figure'),
     Output('cumulative-return-graph', 'figure'),
     Output('summary-output', 'children'),
     Output('gpt-summary', 'children'),
     Output('gpt-cumulative-summary', 'children'),
     Output('etf-name', 'children')],
    [Input('analyze-btn', 'n_clicks')],
    [State('etf-ticker', 'value'),
     State('benchmark-ticker', 'value')]
)
def update_output(n_clicks, etf_ticker, benchmark_ticker):
    if n_clicks == 0 or not etf_ticker:
        return go.Figure(), go.Figure(), "", "", "", ""

    try:
        print(f"Fetching ETF returns for: {etf_ticker}")
        returns = get_etf_returns(etf_ticker)
        print(f"Returns shape: {returns.shape}")

        print("Fetching Fama-French factors")
        ff = get_fama_french_factors()
        print(f"FF shape: {ff.shape}")

        merged = align_data(returns, ff)
        print(f"Merged data shape: {merged.shape}")

        etf_results = run_factor_regression(merged)
        etf_name = get_etf_name(etf_ticker)

        summary_text = (
            f"Intercept (alpha): {etf_results['Intercept (alpha)']:.4f}\n"
            f"Beta Mkt-RF: {etf_results['Beta Mkt-RF']:.4f}\n"
            f"Beta SMB: {etf_results['Beta SMB']:.4f}\n"
            f"Beta HML: {etf_results['Beta HML']:.4f}\n"
            f"R-squared: {etf_results['R-squared']:.4f}"
        )

        exposure_fig = go.Figure()
        cumulative_fig = go.Figure()
        gpt_text = ""
        gpt_cum_text = ""
        benchmark_name = ""

        if benchmark_ticker:
            print(f"Fetching Benchmark returns for: {benchmark_ticker}")
            bm_returns = get_etf_returns(benchmark_ticker)
            bm_merged = align_data(bm_returns, ff)
            bm_results = run_factor_regression(bm_merged)
            benchmark_name = get_etf_name(benchmark_ticker)

            active_exposures = {
                f: etf_results[f] - bm_results[f]
                for f in ['Beta Mkt-RF', 'Beta SMB', 'Beta HML']
            }

            exposure_fig = go.Figure([go.Bar(
                x=list(active_exposures.keys()),
                y=list(active_exposures.values()),
                name='Active Exposures'
            )])
            exposure_fig.update_layout(
                title="Active Factor Exposures",
                yaxis_title="Active Beta",
                xaxis_title="Factor"
            )

            gpt_text = summarize_active_exposures(etf_results, bm_results, etf_ticker, benchmark_ticker)

            combined = pd.concat([returns, bm_returns], axis=1).dropna()
            combined.columns = ['ETF', 'Benchmark']
            cumulative_returns = (1 + combined).cumprod() - 1

            cumulative_fig = go.Figure()
            cumulative_fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['ETF'], name=etf_ticker.upper()))
            cumulative_fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Benchmark'], name=benchmark_ticker.upper()))
            cumulative_fig.update_layout(title="Cumulative Returns", yaxis_title="Return", xaxis_title="Date")

            gpt_cum_text = summarize_cumulative_returns(cumulative_returns, etf_ticker, benchmark_ticker)
        else:
            exposure_fig = go.Figure([go.Bar(
                x=['Mkt-RF', 'SMB', 'HML'],
                y=[etf_results['Beta Mkt-RF'], etf_results['Beta SMB'], etf_results['Beta HML']],
                name=etf_ticker.upper()
            )])
            exposure_fig.update_layout(title=f"Factor Exposures for {etf_ticker.upper()}",
                                       yaxis_title="Beta", xaxis_title="Factor")

        full_name = f"Analysis Portfolio: {etf_name}"
        if benchmark_name:
            full_name += f"\nBenchmark Portfolio: {benchmark_name}"

        return exposure_fig, cumulative_fig, summary_text, gpt_text, gpt_cum_text, full_name

    except Exception as e:
        import traceback
        traceback.print_exc()
        return go.Figure(), go.Figure(), f"Error: {str(e)}", "", "", ""

if __name__ == '__main__':
    app.run(debug=True)