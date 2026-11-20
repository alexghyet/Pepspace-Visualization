"""Download the raw Veltri dataset to the project's data directory."""

from pathlib import Path
import sys

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import VELTRI_PATH

DATASET_URL = "https://raw.githubusercontent.com/GIST-CSBL/AMP-BERT/main/all_veltri.csv"


def main():
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    output_path = PROJECT_ROOT / VELTRI_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"Veltri dataset saved to {output_path}")


if __name__ == "__main__":
    main()
