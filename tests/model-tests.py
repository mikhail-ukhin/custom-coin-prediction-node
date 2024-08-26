from model import download_actual_data, train_model_xgb, format_data, get_price_prediction, validate_model_perfomance
from config import data_base_path, model_file_path, coin_pair

import os

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
binance_data_backtest_path = os.path.join(data_base_path, "binance/futures-klines/backtest")

training_price_data_path = os.path.join(data_base_path, "sol_price_data.csv")
backtrack_price_data_path = os.path.join(data_base_path, "sol_price__backtrack_data.csv")

if __name__ == '__main__':
    download_actual_data() # original data for training
    format_data(binance_data_path, training_price_data_path)
    train_model_xgb()

    validate_model_perfomance()

    print(get_price_prediction())