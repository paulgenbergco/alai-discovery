"""
Process Explorer — Pre-compute T/P Sensitivity Data
=====================================================
For the top N candidates, predict CO2 solubility at a grid of
temperature and pressure conditions. Saves as JSON for the
Streamlit demo to render interactively.

Usage:
    python -m src.screening.process_explorer
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
GENERATED_DIR = Path("data/generated")

TEMPERATURES = [283.15, 288.15, 293.15, 298.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15]
PRESSURES = [1.0, 5.0, 10.0, 20.0, 50.0]


def precompute_process_data(top_n: int = 20):
    """Pre-compute CO2 solubility predictions at a T/P grid for top candidates."""
    from src.screening.pareto_ranker import load_ensemble, predict_batch

    # Load ranked candidates
    ranked = pd.read_parquet(GENERATED_DIR / "candidates_ranked.parquet")
    top = ranked.head(top_n)

    # Load CO2 solubility models
    models = load_ensemble(MODEL_DIR)

    # Convert SMILES to mols
    candidates = []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            candidates.append({
                "smiles": row["smiles"],
                "mol": mol,
                "rank": int(row["rank"]),
                "cation_name": row["cation_name"],
                "anion_name": row["anion_name"],
            })

    logger.info(f"Pre-computing for {len(candidates)} candidates at {len(TEMPERATURES)} temps x {len(PRESSURES)} pressures")

    results = []

    for cand in candidates:
        mol = cand["mol"]
        cand_data = {
            "smiles": cand["smiles"],
            "rank": cand["rank"],
            "cation_name": cand["cation_name"],
            "anion_name": cand["anion_name"],
            "predictions": [],
        }

        for pressure in PRESSURES:
            mols_batch = [mol] * len(TEMPERATURES)
            features = np.array([[t, pressure] for t in TEMPERATURES], dtype=np.float32)
            means, stds = predict_batch(models, mols_batch, features)

            for i, temp in enumerate(TEMPERATURES):
                cand_data["predictions"].append({
                    "temperature_K": temp,
                    "temperature_C": round(temp - 273.15, 1),
                    "pressure_bar": pressure,
                    "log_solubility": round(float(means[i]), 4),
                    "solubility": round(float(10 ** means[i]), 6),
                    "uncertainty": round(float(stds[i]), 4),
                })

        results.append(cand_data)

    # Save as JSON
    output_path = GENERATED_DIR / "process_explorer.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved process explorer data to {output_path}")
    logger.info(f"Total predictions: {len(candidates) * len(TEMPERATURES) * len(PRESSURES)}")

    return results


if __name__ == "__main__":
    precompute_process_data(top_n=20)
