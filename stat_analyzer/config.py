from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "vgsales.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "vgsales_clean.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
print("DEBUG OPENROUTER_API_KEY:", repr(OPENAI_API_KEY))