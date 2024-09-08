import os
import json

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
model_file_path = os.path.join(data_base_path, "model.pkl")
# coin = 'ETH'
# minutes_price_prediction = 20

_config = None  # Global variable to store configuration

def load_config(config_file='settings.json'):
    """Loads the configuration from a JSON file and stores it globally."""
    global _config
    if _config is None:
        with open(config_file, 'r') as f:
            _config = json.load(f)
            print(_config)
            print('config loaded')
    return _config

def get_config():
    """Returns the global configuration object."""
    if _config is None:
        load_config()
    return _config