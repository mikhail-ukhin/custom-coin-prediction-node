import json
from flask import Flask, Response
from model import get_price_prediction, train_model, download_actual_data, format_actual_data

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
        update_data()
        return "0"
    except Exception:
        return "1"

def update_data():
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