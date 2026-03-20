"""
Candidate Screening and Ranking
================================
Scores all generated IL candidates using the trained ensemble and ranks them
by predicted CO2 solubility with uncertainty filtering.

Usage:
    python -m src.screening.pareto_ranker
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
GENERATED_DIR = Path("data/generated")
CURATED_DIR = Path("data/curated")


def load_ensemble():
    """Load all trained ensemble models."""
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, RegressionFFN, MeanAggregation

    models = []
    model_files = sorted(MODEL_DIR.glob("model_*.pt"))

    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}/")

    for model_path in model_files:
        mp = BondMessagePassing(d_h=300)
        agg = MeanAggregation()
        ffn = RegressionFFN(input_dim=300 + 2, hidden_dim=300, n_layers=2)
        model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        models.append(model)

    logger.info(f"Loaded ensemble of {len(models)} models")
    return models


def predict_batch(models, mols, features):
    """Run ensemble prediction on a batch of molecules."""
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    data = [
        MoleculeDatapoint(mol=mol, x_d=np.array(feat, dtype=np.float32))
        for mol, feat in zip(mols, features)
    ]
    dataset = MoleculeDataset(data, featurizer=featurizer)
    loader = build_dataloader(dataset, batch_size=256, shuffle=False)

    all_preds = []
    for model in models:
        preds = []
        with torch.no_grad():
            for batch in loader:
                output = model.predict_step(batch, 0)
                preds.extend(output.squeeze(-1).cpu().numpy().tolist())
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    return all_preds.mean(axis=0), all_preds.std(axis=0)


def screen_candidates(
    models,
    candidates_df: pd.DataFrame,
    temperature_K: float = 298.15,
    pressure_bar: float = 10.0,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """Screen all candidates at specified conditions.

    Args:
        models: List of trained ensemble models
        candidates_df: DataFrame with 'smiles' column
        temperature_K: Operating temperature in Kelvin
        pressure_bar: Operating pressure in bar
        batch_size: Number of molecules to process at once
    """
    logger.info(f"Screening {len(candidates_df)} candidates at T={temperature_K}K, P={pressure_bar} bar")

    smiles_list = candidates_df["smiles"].tolist()

    # Convert SMILES to Mol objects
    mols = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    logger.info(f"Valid molecules: {len(mols)}/{len(smiles_list)}")

    # Create features (same T, P for all)
    features = np.array([[temperature_K, pressure_bar]] * len(mols), dtype=np.float32)

    # Predict in batches
    all_means = []
    all_stds = []

    for i in tqdm(range(0, len(mols), batch_size), desc="Scoring candidates"):
        batch_mols = mols[i:i + batch_size]
        batch_features = features[i:i + batch_size]
        means, stds = predict_batch(models, batch_mols, batch_features)
        all_means.extend(means)
        all_stds.extend(stds)

    # Build results DataFrame
    results = candidates_df.iloc[valid_idx].copy()
    results["log_co2_solubility_pred"] = all_means
    results["co2_solubility_pred"] = 10 ** np.array(all_means)  # Convert from log10
    results["uncertainty"] = all_stds
    results["temperature_K"] = temperature_K
    results["pressure_bar"] = pressure_bar

    return results


def rank_candidates(results_df: pd.DataFrame, max_uncertainty_frac: float = 0.3) -> pd.DataFrame:
    """Rank candidates by predicted CO2 solubility with uncertainty filtering.

    Args:
        results_df: DataFrame with predictions
        max_uncertainty_frac: Maximum allowed uncertainty as fraction of prediction
    """
    df = results_df.copy()

    # Filter out high-uncertainty predictions
    before = len(df)
    df["rel_uncertainty"] = df["uncertainty"] / np.abs(df["log_co2_solubility_pred"]).clip(lower=0.01)
    df = df[df["rel_uncertainty"] < max_uncertainty_frac]
    logger.info(f"Uncertainty filter: {before} → {len(df)} candidates")

    # Filter physically reasonable predictions (mole fraction 0-1)
    df = df[(df["co2_solubility_pred"] > 0) & (df["co2_solubility_pred"] <= 1)]

    # Rank by CO2 solubility (higher is better for CO2 capture)
    df = df.sort_values("co2_solubility_pred", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # MEA benchmark: typical CO2 loading ~0.5 mol CO2/mol amine at 40°C, 1 bar
    # In mole fraction terms, this is roughly 0.15-0.25
    mea_baseline = 0.20
    df["vs_mea"] = (df["co2_solubility_pred"] / mea_baseline - 1) * 100  # % better than MEA

    logger.info(f"Ranked {len(df)} candidates")
    logger.info(f"Top candidate: x_CO2 = {df.iloc[0]['co2_solubility_pred']:.4f}")
    logger.info(f"Candidates better than MEA ({mea_baseline}): {(df['co2_solubility_pred'] > mea_baseline).sum()}")

    return df


def main():
    # Load models
    models = load_ensemble()

    # Load candidates
    candidates = pd.read_parquet(GENERATED_DIR / "il_candidates.parquet")
    logger.info(f"Loaded {len(candidates)} candidates")

    # Screen at standard conditions
    results = screen_candidates(
        models, candidates,
        temperature_K=298.15,
        pressure_bar=10.0,
    )

    # Rank
    ranked = rank_candidates(results)

    # Save results
    output_path = GENERATED_DIR / "candidates_ranked.parquet"
    ranked.to_parquet(output_path, index=False)
    logger.info(f"Saved ranked candidates to {output_path}")

    # Save top 100 as CSV for easy inspection
    top100 = ranked.head(100)
    top100.to_csv(GENERATED_DIR / "top_100_candidates.csv", index=False)
    logger.info(f"Saved top 100 to {GENERATED_DIR}/top_100_candidates.csv")

    # Summary
    logger.info(f"\n--- Screening Summary ---")
    logger.info(f"Total candidates screened: {len(results)}")
    logger.info(f"After filtering: {len(ranked)}")
    logger.info(f"Top 10 candidates:")
    for _, row in ranked.head(10).iterrows():
        logger.info(f"  #{int(row['rank'])}: {row['cation_name']}-{row['anion_name']} "
                    f"x_CO2={row['co2_solubility_pred']:.4f} ± {row['uncertainty']:.3f} "
                    f"({row['vs_mea']:+.1f}% vs MEA)")


if __name__ == "__main__":
    main()
