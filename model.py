import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import ta

from zipfile import ZipFile
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from updater import download_binance_monthly_data, download_binance_daily_data, get_latest_data
from config import data_base_path, model_file_path, TIMEFRAME, TRAINING_DAYS, TOKEN, MINUTES_PRICE_PREDICTION

binance_data_path = os.path.join(data_base_path, "binance")
training_price_data_path = os.path.join(data_base_path, f"{TOKEN}_price_data.csv")

def download_actual_data():
    symbols = f"{TOKEN}USDT"

    download_data(symbols, TRAINING_DAYS, TIMEFRAME, binance_data_path)

def download_data(symbols, training_days, interval, download_path):
    download_binance_daily_data(symbols, training_days, interval, 'EU', download_path)

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
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp)
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    # Calculate the average price and add technical indicators
    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    # Exponential Moving Average (EMA)
    df["EMA_14"] = ta.trend.EMAIndicator(price_data["close"], window=14).ema_indicator()

    # Simple Moving Average (SMA)
    df["SMA_14"] = ta.trend.SMAIndicator(price_data["close"], window=14).sma_indicator()

    # Average True Range (ATR) - Volatility Indicator
    df["ATR_14"] = ta.volatility.AverageTrueRange(price_data["high"], price_data["low"], price_data["close"], window=14).average_true_range()

    # On-Balance Volume (OBV)
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(price_data["close"], price_data["volume"]).on_balance_volume()

    # Shift the price to create a N-minute ahead prediction target
    df["price_ahead"] = df["price"].shift(-int(MINUTES_PRICE_PREDICTION))
    
    # Drop rows where the target is NaN due to shifting
    df.dropna(inplace=True)

    # Updated feature list without StochasticOscillator (f4), RSI (f1), and MACD (f3)
    features = ["date", "EMA_14", "SMA_14", "ATR_14", "OBV"]

    # Prepare the new features and target variable
    x = df[features].values  # Updated Features without the low importance ones
    y = df["price_ahead"].values  # Target: N-minute ahead price

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

    # # Test the model with the test set
    y_pred = model.predict(x_test)
    
    print("Test predictions:", y_pred)

    # Now we pass the trained model into the backtest function for evaluation
    y_test, y_pred, accuracy = backtest_directional_accuracy(model, df, features, "price_ahead")

    print(f"Backtest Directional Accuracy: {accuracy:.2f}%")

def get_current_indicator_values(coin):
    symbol = f'{TOKEN}USDT'
    limit = '1000'  # Fetch the last 1000 data points

    # API endpoint for batch request
    data = get_latest_data(symbol, TIMEFRAME, limit)

    # Define column names for the Binance response
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(data, columns=columns)

    # Ensure the close and volume are of float type for calculation
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Calculate EMA (14)
    ema_value = ta.trend.EMAIndicator(df["close"], window=14).ema_indicator().iloc[-1]

    # Calculate SMA (14)
    sma_value = ta.trend.SMAIndicator(df["close"], window=14).sma_indicator().iloc[-1]

    # Calculate ATR (14) for volatility
    atr_value = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]

    # Calculate OBV (On-Balance Volume)
    obv_value = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume().iloc[-1]

    return ema_value, sma_value, atr_value, obv_value

def get_model_file_path(token):
    lower_token = str.lower(token)

    return os.path.join(data_base_path, f"{lower_token}_price_data.csv")

def get_price_prediction(token):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current indicator values (EMA, SMA, ATR, OBV)
    ema_value, sma_value, atr_value, obv_value = get_current_indicator_values(token)

    future_timestamp = pd.Timestamp(datetime.now() + timedelta(minutes=int(MINUTES_PRICE_PREDICTION))).timestamp()
    future_features = np.array([[future_timestamp, ema_value, sma_value, atr_value, obv_value]])    
    future_price_pred = loaded_model.predict(future_features)

    return future_price_pred[0]

def backtest_directional_accuracy(model, df, features, target):
    # Split the data into training and test sets
    x = df[features].values  # Features
    y = df[target].values  # Actual future prices (target)
    
    # We assume the model is already trained
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Generate predictions for the test set
    y_pred = model.predict(x_test)

    # Calculate directional accuracy
    actual_direction = np.sign(np.diff(y_test))  # +1 for up, -1 for down
    predicted_direction = np.sign(np.diff(y_pred))  # +1 for predicted up, -1 for predicted down

    # Compare actual and predicted directions
    correct_predictions = (actual_direction == predicted_direction).sum()
    total_predictions = len(actual_direction)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Directional Accuracy: {accuracy:.2f}%")
    
    return y_test, y_pred, accuracy