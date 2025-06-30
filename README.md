# AI Risk Factor Navigator

A lightweight Dash app that performs Fama-French factor analysis on any U.S.-listed ETF and generates human-readable summaries using OpenAI's GPT API.

## Features
- Upload any ETF ticker (e.g., ARKK)
- Compare it to a benchmark ETF (e.g., SPY)
- Regression on Fama-French 3 Factors (Mkt-RF, SMB, HML)
- GPT-generated summary of:
  - Factor exposures
  - Active exposures vs benchmark
  - Cumulative performance difference

## How to Run Locally

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key**  
In your terminal or `.env`:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. **Run the app**
```bash
cd src
python app.py
```

Then open your browser to `http://127.0.0.1:8050`

## Example Output
![Example Screenshot](./screenshots/factor_exposures_example.png)

## Notes
- Data pulled from Yahoo Finance and FRED (via `pandas_datareader`)
- GPT used for interpreting model results
- No data stored — all analysis is on-the-fly
