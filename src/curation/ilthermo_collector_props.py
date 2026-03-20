"""
ILThermo Property Collector (Viscosity, Density)
=================================================
Collects pure-component IL property data from NIST ILThermo.
Focuses on viscosity and density — the two most data-rich properties.

Usage:
    python -m src.curation.ilthermo_collector_props
"""

import json
import time
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://ilthermo.boulder.nist.gov/ILT2"
RAW_DIR = Path("data/raw")

TARGET_PROPERTIES = ["Viscosity", "Density"]


def search_pure_il_datasets(property_name: str) -> list[dict]:
    """Search ILThermo for pure IL datasets of a specific property."""
    logger.info(f"Searching ILThermo for {property_name} data...")
    r = requests.get(
        f"{BASE_URL}/ilsearch",
        params={"cmp1": "", "ncmp": 1, "prp": ""},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    header = data["header"]
    entries = []
    for row in data["res"]:
        entry = dict(zip(header, row))
        if entry["prp"] == property_name:
            entry["il_name"] = entry.get("nm1") or ""
            entries.append(entry)

    logger.info(f"Found {len(entries)} {property_name} datasets")
    return entries


def fetch_dataset(set_id: str, retries: int = 3) -> dict | None:
    """Fetch a single dataset from ILThermo."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}/ilset", params={"set": set_id}, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                logger.warning(f"Failed to fetch {set_id}: {e}")
                return None


def parse_dataset(raw: dict, meta: dict) -> list[dict]:
    """Parse a raw ILThermo dataset into flat records."""
    records = []
    headers = raw.get("dhead", [])
    data_rows = raw.get("data", [])
    ref_info = raw.get("ref", {})
    reference = ref_info.get("full", "")
    property_name = meta["prp"]
    il_name = meta["il_name"]
    set_id = meta["setid"]

    # Map header names to column indices
    col_map = {}
    for i, h in enumerate(headers):
        name = h[0].lower() if h[0] else ""
        if "temperature" in name:
            col_map["temperature"] = i
        elif "pressure" in name:
            col_map["pressure"] = i
        elif "viscosity" in name:
            col_map["value"] = i
        elif "density" in name:
            col_map["value"] = i

    for row in data_rows:
        record = {
            "set_id": set_id,
            "il_name": il_name,
            "property": property_name,
            "reference": reference,
            "temperature_K": None,
            "pressure_kPa": None,
            "value": None,
            "uncertainty": None,
        }

        for col_name, col_idx in col_map.items():
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if not cell:
                continue

            try:
                val = float(cell[0]) if cell[0] else None
            except (ValueError, TypeError):
                val = None

            try:
                unc = float(cell[1]) if len(cell) > 1 else None
            except (ValueError, TypeError):
                unc = None

            if col_name == "temperature":
                record["temperature_K"] = val
            elif col_name == "pressure":
                record["pressure_kPa"] = val
            elif col_name == "value":
                record["value"] = val
                record["uncertainty"] = unc

        if record["value"] is not None and record["temperature_K"] is not None:
            records.append(record)

    return records


def collect_property(property_name: str, delay: float = 0.15, max_datasets: int = 2000) -> pd.DataFrame:
    """Collect all data for a single property."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    entries = search_pure_il_datasets(property_name)
    entries = entries[:max_datasets]

    all_records = []
    failed = 0

    for meta in tqdm(entries, desc=f"Fetching {property_name}"):
        raw = fetch_dataset(meta["setid"])
        if raw is None:
            failed += 1
            continue
        records = parse_dataset(raw, meta)
        all_records.extend(records)
        time.sleep(delay)

    logger.info(f"Collected {len(all_records)} {property_name} points ({failed} failed)")

    df = pd.DataFrame(all_records)
    safe_name = property_name.lower().replace(" ", "_")
    df.to_parquet(RAW_DIR / f"ilthermo_{safe_name}_raw.parquet", index=False)
    df.to_csv(RAW_DIR / f"ilthermo_{safe_name}_raw.csv", index=False)

    logger.info(f"Unique ILs: {df['il_name'].nunique()}")
    logger.info(f"T range: {df['temperature_K'].min():.1f} - {df['temperature_K'].max():.1f} K")

    return df


def main():
    for prop in TARGET_PROPERTIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting {prop}")
        logger.info(f"{'='*60}")
        df = collect_property(prop)
        logger.info(f"Done: {len(df)} records for {prop}")


if __name__ == "__main__":
    main()
