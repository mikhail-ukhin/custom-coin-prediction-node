import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import ta
import requests
import requests_cache

from zipfile import ZipFile
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path, coin_pair

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
training_price_data_path = os.path.join(data_base_path, "sol_price_data.csv")

def download_data():
    cm_or_um = "um"
    symbols = [coin_pair]

    intervals = ["10m", "20m", "1h", "1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    
    download_path = binance_data_path
    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"Downloaded monthly data to {download_path}.")
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )
    print(f"Downloaded daily data to {download_path}.")

def download_data_frequent():
    cm_or_um = "um"
    symbols = [coin_pair]

    intervals = ["1m"]
    
    download_path = binance_data_path

    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )

def format_data():
    files = sorted([x for x in os.listdir(binance_data_path)])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()

    for file in files:
        zip_file_path = os.path.join(binance_data_path, file)

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

    price_df.sort_index().to_csv(training_price_data_path)

def train_model():
    # Load the eth price data
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    # Reshape the data to the shape expected by sklearn
    x = df["date"].values.reshape(-1, 1)
    y = df["price"].values.reshape(-1, 1)

    # Split the data into training set and test set
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def train_model_xgb():
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    # Calculate the average price and RSI
    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)
    df["RSI_14"] = ta.momentum.RSIIndicator(price_data["close"], window=14).rsi()

    # Prepare the features and target variable
    x = df[["date", "RSI_14"]].values  # Include RSI as a feature
    y = df["price"].values

    # Split the data into training set and test set
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=0)
    model.fit(x_train, y_train)

    # Create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def get_current_indicator_values():
    # Symbol and exchange (example for Bitcoin on Binance)
    symbol = coin_pair
    interval = '5m'  # 5-minute timeframe
    limit = '100'  # Fetch the last 100 data points

    # API endpoint for batch request
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'

    # Make the API request
    response = requests.get(url)
    data = response.json()

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    df = pd.DataFrame(data, columns=columns)
    df['close'] = df['close'].astype(float)
    df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    rsi_value = df['RSI_14'].iloc[-1]

    return rsi_value

def get_price_prediction():
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    rsi_value = get_current_indicator_values()

    future_timestamp = pd.Timestamp(datetime.now() + timedelta(minutes=30)).timestamp()
    # Make the prediction
    future_features = np.array([[future_timestamp, rsi_value]])
    
    future_price_pred = loaded_model.predict(future_features)

    return future_price_pred[0]

if __name__ == "__main__":
    # download_data()
    format_data()
    train_model_xgb()

    price = get_price_prediction()

    print(price)
