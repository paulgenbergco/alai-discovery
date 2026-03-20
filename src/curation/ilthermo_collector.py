"""
ILThermo Data Collector
=======================
Collects ionic liquid + CO2 binary mixture data from the NIST ILThermo database.
Focuses on CO2 solubility-related properties:
  - Composition at phase equilibrium (mole fraction)
  - Henry's Law constant
  - Equilibrium pressure

Usage:
    python -m src.curation.ilthermo_collector
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

# Properties relevant to CO2 solubility
SOLUBILITY_PROPERTIES = {
    "Composition at phase equilibrium",
    "Henry's Law constant",
    "Equilibrium pressure",
}


def search_co2_binary_mixtures() -> list[dict]:
    """Search ILThermo for all binary CO2 + IL datasets."""
    logger.info("Searching ILThermo for CO2 binary mixtures...")
    r = requests.get(
        f"{BASE_URL}/ilsearch",
        params={"cmp1": "carbon dioxide", "ncmp": 2},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    header = data["header"]  # ['setid','ref','prp','phases','cmp1','cmp2','cmp3','np','nm1','nm2','nm3']

    entries = []
    for row in data["res"]:
        entry = dict(zip(header, row))
        # Only keep entries where CO2 is actually a named component
        nm1 = (entry.get("nm1") or "").lower()
        nm2 = (entry.get("nm2") or "").lower()
        if "carbon dioxide" not in nm1 and "carbon dioxide" not in nm2:
            continue
        # Only keep solubility-related properties
        if entry["prp"] not in SOLUBILITY_PROPERTIES:
            continue
        # Identify which component is the IL
        if "carbon dioxide" in nm1:
            entry["il_name"] = entry.get("nm2") or ""
        else:
            entry["il_name"] = entry.get("nm1") or ""
        entries.append(entry)

    logger.info(f"Found {len(entries)} CO2 solubility datasets")
    return entries


def fetch_dataset(set_id: str, retries: int = 3) -> dict | None:
    """Fetch a single dataset from ILThermo by set ID."""
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{BASE_URL}/ilset",
                params={"set": set_id},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                logger.warning(f"Failed to fetch dataset {set_id}: {e}")
                return None


def parse_dataset(raw: dict, meta: dict) -> list[dict]:
    """Parse a raw ILThermo dataset into flat records.

    Returns a list of dicts, one per data point, with columns:
      il_name, property, temperature_K, pressure_kPa, value, uncertainty,
      mole_fraction_co2, phase, reference, set_id
    """
    records = []
    headers = raw.get("dhead", [])
    data_rows = raw.get("data", [])
    components = raw.get("components", [])
    ref_info = raw.get("ref", {})
    reference = ref_info.get("full", "")
    property_name = meta["prp"]
    il_name = meta["il_name"]
    set_id = meta["setid"]

    # Map header names to column indices (integers only)
    col_map = {}
    for i, h in enumerate(headers):
        name = h[0].lower() if h[0] else ""
        if "temperature" in name:
            col_map["temperature"] = i
        elif "mole fraction" in name:
            col_map["mole_fraction"] = i
        elif "henry" in name:
            col_map["henry"] = i
        elif "equilibrium pressure" in name or "pressure" in name:
            col_map["pressure"] = i

    for row in data_rows:
        record = {
            "set_id": set_id,
            "il_name": il_name,
            "property": property_name,
            "reference": reference,
            "temperature_K": None,
            "pressure_kPa": None,
            "mole_fraction_co2": None,
            "value": None,
            "uncertainty": None,
        }

        for col_name, col_idx in col_map.items():
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if not cell:
                continue

            val_str = cell[0] if cell else None
            unc_str = cell[1] if len(cell) > 1 else None

            try:
                val = float(val_str) if val_str else None
            except (ValueError, TypeError):
                val = None

            try:
                unc = float(unc_str) if unc_str else None
            except (ValueError, TypeError):
                unc = None

            if col_name == "temperature":
                record["temperature_K"] = val
            elif col_name == "pressure":
                record["pressure_kPa"] = val
            elif col_name == "mole_fraction":
                record["mole_fraction_co2"] = val
                record["value"] = val
                record["uncertainty"] = unc
            elif col_name == "henry":
                record["value"] = val
                record["uncertainty"] = unc

        # Only keep records with at least a value and temperature
        if record["value"] is not None and record["temperature_K"] is not None:
            records.append(record)

    return records


def collect_all(delay: float = 0.2) -> pd.DataFrame:
    """Collect all CO2 solubility data from ILThermo.

    Args:
        delay: Seconds to wait between API calls (be polite to NIST).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Search for all relevant datasets
    entries = search_co2_binary_mixtures()

    # Step 2: Fetch each dataset and parse
    all_records = []
    failed = 0

    for meta in tqdm(entries, desc="Fetching datasets"):
        raw = fetch_dataset(meta["setid"])
        if raw is None:
            failed += 1
            continue

        records = parse_dataset(raw, meta)
        all_records.extend(records)
        time.sleep(delay)

    logger.info(f"Collected {len(all_records)} data points from {len(entries) - failed} datasets ({failed} failed)")

    # Step 3: Convert to DataFrame and save
    df = pd.DataFrame(all_records)

    # Save raw data
    raw_path = RAW_DIR / "ilthermo_co2_solubility_raw.parquet"
    df.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw data to {raw_path}")

    # Also save as CSV for inspection
    csv_path = RAW_DIR / "ilthermo_co2_solubility_raw.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")

    # Summary statistics
    logger.info(f"\n--- Collection Summary ---")
    logger.info(f"Total data points: {len(df)}")
    logger.info(f"Unique ionic liquids: {df['il_name'].nunique()}")
    logger.info(f"Property breakdown:\n{df['property'].value_counts().to_string()}")
    logger.info(f"Temperature range: {df['temperature_K'].min():.1f} - {df['temperature_K'].max():.1f} K")

    return df


if __name__ == "__main__":
    df = collect_all()
    print(f"\nDone! Collected {len(df)} records for {df['il_name'].nunique()} unique ILs.")
