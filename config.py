import os
from dotenv import load_dotenv

load_dotenv()

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
model_file_path = os.path.join(data_base_path, "model.pkl")

TOKEN = os.getenv("TOKEN").upper()
TRAINING_DAYS = os.getenv("TRAINING_DAYS")
TIMEFRAME = os.getenv("TIMEFRAME")
REGION = os.getenv("REGION").lower()
MINUTES_PRICE_PREDICTION = os.getenv("MINUTES_PRICE_PREDICTION")

if REGION in ["us", "com", "usa"]:
    REGION = "us"
else:
    REGION = "com"

CG_API_KEY = os.getenv("CG_API_KEY", default=None)