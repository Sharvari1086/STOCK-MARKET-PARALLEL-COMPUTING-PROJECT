
import yfinance as yf
import pandas as pd
from pathlib import Path
from urllib.request import Request, urlopen
from io import StringIO

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "stock_prices.csv"

URL = (
    "https://raw.githubusercontent.com/"
    "chinobing/historical_sp500_constituents/"
    "master/sp500_constituents.csv"
)

print("Fetching current S&P 500 ticker list...")

request = Request(
    URL,
    headers={"User-Agent": "StockMarketProject/1.0"}
)

with urlopen(request, timeout=30) as response:
    content = response.read().decode("utf-8")

constituents = pd.read_csv(StringIO(content))
ticker_column = next(
    (col for col in constituents.columns
     if col.lower() in ["symbol", "ticker", "tickers"]),
    None
)

if ticker_column is None:
    raise RuntimeError(
        "Could not identify ticker column. "
        f"Available columns: {list(constituents.columns)}"
    )

tickers = (
    constituents[ticker_column]
    .dropna()
    .astype(str)
    .str.replace(".", "-", regex=False)
    .unique()
    .tolist()
)

print("Tickers found:", len(tickers))
print("Downloading daily prices from Yahoo Finance...")

all_prices = []

# Download in batches to reduce request failures.
for start in range(0, len(tickers), 50):
    batch = tickers[start:start + 50]
    print(f"Batch {start + 1} to {start + len(batch)}")

    data = yf.download(
        batch,
        start="2021-01-01",
        end="2026-10-27",
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False
    )

    if data.empty:
        print("No data in this batch; skipping.")
        continue

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            print("No Close prices in this batch; skipping.")
            continue
        prices = data["Close"]
    else:
        if "Close" not in data.columns:
            print("No Close prices in this batch; skipping.")
            continue
        prices = data[["Close"]]
        prices.columns = [batch[0]]

    all_prices.append(prices)

if not all_prices:
    raise RuntimeError(
        "No stock prices downloaded. Check your internet connection."
    )

close_prices = pd.concat(all_prices, axis=1)
close_prices = close_prices.loc[
    :, ~close_prices.columns.duplicated()
]
close_prices = close_prices.dropna(axis=1, how="all")
close_prices = close_prices.sort_index()
close_prices.index.name = "Date"

close_prices.to_csv(OUTPUT_FILE)

print("\nDownload complete.")
print("Stocks with data:", len(close_prices.columns))
print("Trading dates:", len(close_prices))
print("CSV saved to:", OUTPUT_FILE)
print(close_prices.head())
