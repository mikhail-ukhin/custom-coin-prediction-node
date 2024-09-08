import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, Response
from model import get_price_prediction, train_model, download_actual_data, format_actual_data, training_price_data_path, binance_data_path
from config import model_file_path
from tools import remove_file, remove_files_in_dir

app = Flask(__name__)

def get_coin_inference(token):
    return get_price_prediction(token)

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_coin_inference(token)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    
@app.route('/healthcheck')
def healthcheck():
    return Response('OK', status=200)

@app.route("/update")
def update():
    print('Endpoint update data triggered')
    try:
        update_data_with_cleanup()
        return "0"
    except Exception:
        return "1"
    
def cleanup_files():
    remove_file(model_file_path)
    remove_file(training_price_data_path)
    remove_files_in_dir(binance_data_path)

def update_data_with_cleanup():
    cleanup_files()
    update_data()

def update_data():
    # cleanup_files()

    print('Starting to download the data')
    download_actual_data()

    print('Starting to format the data')

    try:
        format_actual_data()
    except Exception as e:
        print(f"Error formatting data: {e}")

    print('Starting to train the model')

    try:
        train_model()
    except Exception as e:
        print(f"Error training model: {e}")

if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
