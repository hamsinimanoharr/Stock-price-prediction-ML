import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Fetch Stock Data
ticker = 'AAPL'  # Change this to any stock symbol (e.g., TSLA, JPM, MSFT)
stock_data = yf.download(ticker, start="2022-01-01", end="2024-03-01")

# Step 2: Prepare Data for Machine Learning
stock_data = stock_data[['Close']].dropna()  # Keep only 'Close' price & drop NaN values
stock_data['Day'] = np.arange(len(stock_data))  # Convert dates into numerical format

# Selecting Features (X) and Target (Y)
X = stock_data[['Day']]
y = stock_data['Close']

# Step 3: Train the Machine Learning Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Step 6: Visualize Actual vs Predicted Prices
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', label="Actual Prices")
plt.scatter(X_test, y_pred, color='red', label="Predicted Prices")
plt.xlabel("Days Since Start")
plt.ylabel("Stock Closing Price (USD)")
plt.legend()
plt.title(f"Stock Price Prediction for {ticker}")
plt.show()

