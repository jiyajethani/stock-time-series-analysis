import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Download stock data (example: Apple)
stock = yf.download("AAPL", start="2018-01-01", end="2024-01-01")

# Save to CSV
stock.to_csv("../data/stock_data.csv")

# Display first few rows
print(stock.head())

# Plot stock closing price
plt.figure(figsize=(12,6))
plt.plot(stock['Close'])
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Moving Average
stock['MA30'] = stock['Close'].rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(stock['Close'], label='Original')
plt.plot(stock['MA30'], label='30 Day Moving Average')
plt.legend()
plt.title("Stock Price with Moving Average")
plt.show()

# ADF Test Function
def adf_test(series):
    result = adfuller(series.dropna())

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    if result[1] <= 0.05:
        print("Series is Stationary")
    else:
        print("Series is NOT Stationary")

# Run ADF Test
adf_test(stock['Close'])

# Differencing
stock['Differenced'] = stock['Close'].diff()

plt.figure(figsize=(12,6))
plt.plot(stock['Differenced'])
plt.title("Differenced Time Series")
plt.show()

# ADF test after differencing
adf_test(stock['Differenced'])

# Autocorrelation plots
plot_acf(stock['Close'].dropna())
plt.title("Autocorrelation Plot")
plt.show()

plot_pacf(stock['Close'].dropna())
plt.title("Partial Autocorrelation Plot")
plt.show()

