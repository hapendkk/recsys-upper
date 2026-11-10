from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
NOTEBOOK_DIR = ROOT / "notebooks"
APP_DIR = ROOT / "app"

DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"
RATINGS_REPO_PATH = "benchmark/0core/rating_only/All_Beauty.csv"
META_REPO_PATH = "raw/meta_categories/meta_All_Beauty.jsonl"
RATINGS_FILENAME = "All_Beauty_ratings.csv"
META_FILENAME = "meta_All_Beauty.jsonl"
EXPECTED_SIZES = {
    RATINGS_FILENAME: 40_249_617,
    META_FILENAME: 212_990_142,
}

MIN_POSITIVE_RATING = 3.0
DEFAULT_MIN_USER_INTERACTIONS = 3
DEFAULT_MIN_ITEM_INTERACTIONS = 5
TOP_K_SIMILAR_ITEMS = 100
MAX_HISTORY_ITEMS = 50
