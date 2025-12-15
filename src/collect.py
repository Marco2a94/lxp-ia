# src/collect.py
import requests
import csv
import os
from datetime import datetime

URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records"
PARAMS = {
    "limit": 100,
    "refine": 'nom_arrondissement_communes:"Villejuif"'
}

OUTPUT_FILE = "data/velib_raw.csv"

FIELDS = [
    "timestamp",
    "stationcode",
    "name",
    "nom_arrondissement_communes",
    "numbikesavailable",
    "numdocksavailable",
    "mechanical",
    "ebike",
    "is_installed",
    "is_renting",
    "is_returning",
]

def main():
    response = requests.get(URL, params=PARAMS)
    response.raise_for_status()
    data = response.json()["results"]

    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.utcnow().isoformat()

        for row in data:
            record = {
                "timestamp": timestamp,
                "stationcode": row.get("stationcode"),
                "name": row.get("name"),
                "nom_arrondissement_communes": row.get("nom_arrondissement_communes"),
                "numbikesavailable": row.get("numbikesavailable"),
                "numdocksavailable": row.get("numdocksavailable"),
                "mechanical": row.get("mechanical"),
                "ebike": row.get("ebike"),
                "is_installed": row.get("is_installed"),
                "is_renting": row.get("is_renting"),
                "is_returning": row.get("is_returning"),
            }
            writer.writerow(record)

if __name__ == "__main__":
    main()
