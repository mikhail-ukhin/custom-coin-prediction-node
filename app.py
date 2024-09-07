import json
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
from flask import Flask, Response
from model import get_price_prediction, train_model, download_actual_data, format_actual_data, training_price_data_path
from config import model_file_path

app = Flask(__name__)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    else:
        print("The file does not exist.")

def update_data():
    """Download price data, format data and train model."""

    remove_file(model_file_path)
    remove_file(training_price_data_path)

    download_actual_data()
    format_actual_data()
    train_model()

def get_coin_inference():
    """Load model and predict current price."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    now_timestamp = pd.Timestamp(datetime.now()).timestamp()
    X_new = np.array([now_timestamp]).reshape(-1, 1)
    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0][0]

def get_coin_inference_alternative(token):
    return get_price_prediction(token)

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_coin_inference_alternative(token)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    
@app.route('/healthcheck')
def healthcheck():
    return Response('OK', status=200)

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
