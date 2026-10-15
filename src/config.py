from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"
CSV_PATH = BASE_DIR / "GroundTruth.csv"

CLASSES = [
    "MEL", "NV", "BCC", "AKIEC",
    "BKL", "DF", "VASC"
]