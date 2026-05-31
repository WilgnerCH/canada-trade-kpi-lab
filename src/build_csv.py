import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_DIR = os.path.join(BASE_DIR, "data_csv")

os.makedirs(CSV_DIR, exist_ok=True)

def json_to_csv(json_file, csv_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return

    keys = data[0].keys()

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

# Arquivos
files = [
    "countries",
    "monthly",
    "products",
    "products_top100"
]

for file in files:
    json_path = os.path.join(DATA_DIR, f"{file}.json")
    csv_path = os.path.join(CSV_DIR, f"{file}.csv")
    
    if os.path.exists(json_path):
        json_to_csv(json_path, csv_path)
        print(f"✅ Converted {file}.json → CSV")
    else:
        print(f"⚠️ File not found: {json_path}")
