import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import ta
import requests
import matplotlib.pyplot as plt

from zipfile import ZipFile
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path, coin

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
training_price_data_path = os.path.join(data_base_path, f"{coin}_price_data.csv")

def download_actual_data():
    intervals = ["1m"]
    years = ["2024","2023","2022"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    symbols = [f"{coin}USDT"]

    download_data(symbols, intervals, years, months, binance_data_path)

    current_year = datetime.now().year
    current_month = datetime.now().month

    download_binance_daily_data('um', symbols, intervals, current_year, current_month, binance_data_path)

def download_data(symbols, intervals, years, months, download_path):
    cm_or_um = "um"

    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"Downloaded monthly data to {download_path}.")

def format_actual_data():    
    format_data(binance_data_path, training_price_data_path)

def format_data(path, output_path):
    files = sorted([x for x in os.listdir(path)])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()

    for file in files:
        zip_file_path = os.path.join(path, file)

        if not zip_file_path.endswith(".zip"):
            continue

        myzip = ZipFile(zip_file_path)
        with myzip.open(myzip.filelist[0]) as f:
            line = f.readline()
            header = 0 if line.decode("utf-8").startswith("open_time") else None
        df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
        df.columns = [
            "start_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "end_time",
            "volume_usd",
            "n_trades",
            "taker_volume",
            "taker_volume_usd",
        ]
        df.index = [pd.Timestamp(x + 1, unit="ms") for x in df["end_time"]]
        df.index.name = "date"
        price_df = pd.concat([price_df, df])

    price_df.sort_index().to_csv(output_path)

def train_model():
    # Load the processed price data
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp)
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    # Calculate the average price and add technical indicators
    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)
    
    # Adding RSI (14)
    df["RSI_14"] = ta.momentum.RSIIndicator(price_data["close"], window=14).rsi()
    df["EMA_14"] = ta.trend.EMAIndicator(price_data["close"], window=14).ema_indicator()
    df["MACD"] = ta.trend.MACD(price_data["close"]).macd()
    df["StochasticOscillator"] = ta.momentum.StochasticOscillator(price_data["high"], price_data["low"], price_data["close"]).stoch()  # Stochastic Oscillator

    # Shift the price to create a 10-minute ahead prediction target
    df["price_10min_ahead"] = df["price"].shift(-10)  # Assuming data is at 1-minute intervals

    # Drop rows where the target is NaN due to shifting
    df.dropna(inplace=True)

    # Prepare the features and target variable
    x = df[["date", "RSI_14", "EMA_14", "MACD", "StochasticOscillator"]].values  # Features
    y = df["price_10min_ahead"].values  # Target: 10-minute ahead price

    # Split the data into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0)
    model.fit(x_train, y_train)

    # Create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

    # Test the model with the test set
    y_pred = model.predict(x_test)
    
    print("Test predictions:", y_pred)

def get_current_indicator_values(coin):
    # Symbol and exchange (example for Bitcoin on Binance)
    symbol = f'{coin}USDT'
    interval = '5m'  # 5-minute timeframe
    limit = '1000'  # Fetch the last 100 data points

    # API endpoint for batch request
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'

    # Make the API request
    response = requests.get(url)
    data = response.json()

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    df = pd.DataFrame(data, columns=columns)

    df['close'] = df['close'].astype(float)
    rsi_value = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]

    # Calculate EMA (14)
    ema_value = ta.trend.EMAIndicator(df["close"], window=14).ema_indicator().iloc[-1]

    # Calculate MACD (12, 26, 9)
    macd = ta.trend.MACD(df["close"])
    macd_value = macd.macd().iloc[-1]

    # Calculate Stochastic Oscillator (14, 3, 3)
    stochastic = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    stochastic_value = stochastic.stoch().iloc[-1]

    return rsi_value, ema_value, macd_value, stochastic_value

def get_model_file_path(token):
    lower_token = str.lower(token)

    return os.path.join(data_base_path, f"{lower_token}_price_data.csv")

def get_price_prediction(token):

    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    rsi_value, ema_value, macd_value, stochastic_value = get_current_indicator_values(token)

    future_timestamp = pd.Timestamp(datetime.now() + timedelta(minutes=10)).timestamp()
    future_features = np.array([[future_timestamp, rsi_value, ema_value, macd_value, stochastic_value]])    
    future_price_pred = loaded_model.predict(future_features)

    return future_price_pred[0]
    # Prepare the data
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)
    df["RSI_14"] = ta.momentum.RSIIndicator(price_data["close"], window=14).rsi()
    df["EMA_14"] = ta.trend.EMAIndicator(price_data["close"], window=14).ema_indicator()
    df["MACD"] = ta.trend.MACD(price_data["close"]).macd()
    df["StochasticOscillator"] = ta.momentum.StochasticOscillator(price_data["high"], price_data["low"], price_data["close"]).stoch()

    # Shift the price to create a 20-minute ahead prediction target
    df["price_20min_ahead"] = df["price"].shift(-20)
    df.dropna(inplace=True)

    # Prepare the features and target variable
    x = df[["date", "RSI_14", "EMA_14", "MACD", "StochasticOscillator"]].values
    y_true = df["price_20min_ahead"].values

    # Generate predictions
    y_pred = model.predict(x)

    # Calculate accuracy metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy (percentage of times the predicted direction matches the actual direction)
    directional_accuracy = ((y_pred > df["price"]) == (y_true > df["price"])).mean()

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, y_true, label="Actual Price", color="blue")
    plt.plot(df.index, y_pred, label="Predicted Price", color="red", linestyle="--")
    plt.title("Backtest: Actual vs Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Print the accuracy metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")