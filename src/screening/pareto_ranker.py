"""
Multi-Property Candidate Screening and Pareto Ranking
======================================================
Scores all generated IL candidates on CO2 solubility, viscosity, and density
using trained D-MPNN ensembles, then computes Pareto-optimal rankings.

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

# Auto-detect GPU: uses CUDA (NVIDIA GPU) if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

MODEL_DIR = Path("models")
GENERATED_DIR = Path("data/generated")
CURATED_DIR = Path("data/curated")


def load_ensemble(model_dir: Path = None):
    """Load trained ensemble models from a directory."""
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, RegressionFFN, MeanAggregation

    if model_dir is None:
        model_dir = MODEL_DIR

    models = []
    model_files = sorted(model_dir.glob("model_*.pt"))

    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}/")

    for model_path in model_files:
        mp = BondMessagePassing(d_h=300)
        agg = MeanAggregation()
        ffn = RegressionFFN(input_dim=300 + 2, hidden_dim=300, n_layers=2)
        model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
        model.to(DEVICE)  # Move model to GPU if available
        model.eval()
        models.append(model)

    logger.info(f"Loaded {len(models)} models from {model_dir}")
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
                # Move batch components to GPU (Chemprop TrainingBatch has no .to())
                if DEVICE.type == "cuda":
                    batch.bmg = batch.bmg.to(DEVICE)
                    if batch.X_d is not None:
                        batch.X_d = batch.X_d.to(DEVICE)
                output = model.predict_step(batch, 0)
                preds.extend(output.squeeze(-1).cpu().numpy().tolist())
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    return all_preds.mean(axis=0), all_preds.std(axis=0)


def _score_property(models, mols, features, batch_size=1000, desc="Scoring"):
    """Score all molecules with an ensemble, batched."""
    all_means, all_stds = [], []
    for i in tqdm(range(0, len(mols), batch_size), desc=desc):
        means, stds = predict_batch(models, mols[i:i+batch_size], features[i:i+batch_size])
        all_means.extend(means)
        all_stds.extend(stds)
    return np.array(all_means), np.array(all_stds)


def screen_candidates(
    candidates_df: pd.DataFrame,
    temperature_K: float = 298.15,
    pressure_bar: float = 10.0,
) -> pd.DataFrame:
    """Screen all candidates on CO2 solubility, viscosity, and density."""

    smiles_list = candidates_df["smiles"].tolist()
    mols, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    logger.info(f"Valid molecules: {len(mols)}/{len(smiles_list)}")
    features = np.array([[temperature_K, pressure_bar]] * len(mols), dtype=np.float32)

    results = candidates_df.iloc[valid_idx].copy()
    results["temperature_K"] = temperature_K
    results["pressure_bar"] = pressure_bar

    # --- CO2 Solubility ---
    logger.info("Scoring CO2 solubility...")
    sol_models = load_ensemble(MODEL_DIR)
    sol_means, sol_stds = _score_property(sol_models, mols, features, desc="CO2 solubility")
    results["log_co2_solubility_pred"] = sol_means
    results["co2_solubility_pred"] = 10 ** sol_means
    results["co2_uncertainty"] = sol_stds
    # Keep backward-compatible column name
    results["uncertainty"] = sol_stds

    # --- Viscosity ---
    visc_dir = MODEL_DIR / "viscosity"
    if visc_dir.exists() and list(visc_dir.glob("model_*.pt")):
        logger.info("Scoring viscosity...")
        visc_models = load_ensemble(visc_dir)
        visc_means, visc_stds = _score_property(visc_models, mols, features, desc="Viscosity")
        results["log_viscosity_pred"] = visc_means
        results["viscosity_pred"] = 10 ** visc_means  # Convert from log10 to mPa.s
        results["viscosity_uncertainty"] = visc_stds
    else:
        logger.warning("Viscosity models not found, skipping")

    # --- Density ---
    dens_dir = MODEL_DIR / "density"
    if dens_dir.exists() and list(dens_dir.glob("model_*.pt")):
        logger.info("Scoring density...")
        dens_models = load_ensemble(dens_dir)
        dens_means, dens_stds = _score_property(dens_models, mols, features, desc="Density")
        results["density_pred"] = dens_means  # Density was NOT log-transformed
        results["density_uncertainty"] = dens_stds
    else:
        logger.warning("Density models not found, skipping")

    return results, mols


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto front: maximize solubility, minimize viscosity."""
    df = df.copy()

    if "viscosity_pred" not in df.columns:
        df["pareto_front"] = False
        df["pareto_rank"] = np.nan
        return df

    # For Pareto: we want HIGH solubility and LOW viscosity
    sol = df["co2_solubility_pred"].values
    visc = df["viscosity_pred"].values

    # Non-dominated sorting (simplified — mark Pareto-optimal points)
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j has higher solubility AND lower viscosity
            if sol[j] >= sol[i] and visc[j] <= visc[i] and (sol[j] > sol[i] or visc[j] < visc[i]):
                is_pareto[i] = False
                break

    df["pareto_front"] = is_pareto

    # Pareto rank: distance to ideal point (max sol, min visc), normalized
    sol_norm = (sol - sol.min()) / (sol.max() - sol.min() + 1e-10)
    visc_norm = (visc - visc.min()) / (visc.max() - visc.min() + 1e-10)
    # Ideal = high sol (1.0), low visc (0.0)
    df["pareto_distance"] = np.sqrt((1 - sol_norm)**2 + visc_norm**2)
    df["pareto_rank"] = df["pareto_distance"].rank(method="min").astype(int)

    n_pareto = is_pareto.sum()
    logger.info(f"Pareto front: {n_pareto} candidates on the front")

    return df


def rank_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by CO2 solubility with Pareto optimization."""
    df = results_df.copy()

    # Filter uncertainty
    before = len(df)
    df["rel_uncertainty"] = df["uncertainty"] / np.abs(df["log_co2_solubility_pred"]).clip(lower=0.01)
    df = df[df["rel_uncertainty"] < 0.3]
    logger.info(f"Uncertainty filter: {before} -> {len(df)}")

    # Filter physically reasonable
    df = df[(df["co2_solubility_pred"] > 0) & (df["co2_solubility_pred"] <= 1)]

    # Primary rank by solubility
    df = df.sort_values("co2_solubility_pred", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # MEA comparison
    mea_baseline = 0.20
    df["vs_mea"] = (df["co2_solubility_pred"] / mea_baseline - 1) * 100

    # Pareto ranking on top 5000 (full dataset is too slow for non-dominated sort)
    top_for_pareto = min(5000, len(df))
    top_df = compute_pareto_front(df.head(top_for_pareto))
    for col in ["pareto_front", "pareto_rank", "pareto_distance"]:
        if col in top_df.columns:
            df.loc[df.index[:top_for_pareto], col] = top_df[col].values

    logger.info(f"Ranked {len(df)} candidates")
    logger.info(f"Top candidate: x_CO2 = {df.iloc[0]['co2_solubility_pred']:.4f}")

    return df


def main():
    # Load candidates
    all_path = GENERATED_DIR / "il_candidates_all.parquet"
    combo_path = GENERATED_DIR / "il_candidates.parquet"
    if all_path.exists():
        candidates = pd.read_parquet(all_path)
        logger.info(f"Loaded {len(candidates)} candidates")
    elif combo_path.exists():
        candidates = pd.read_parquet(combo_path)
        logger.info(f"Loaded {len(candidates)} candidates")
    else:
        raise FileNotFoundError("No candidate files found.")

    # Multi-property screening
    results, mols = screen_candidates(candidates, temperature_K=298.15, pressure_bar=10.0)

    # Rank with Pareto
    ranked = rank_candidates(results)

    # Van't Hoff physics validation on top 500
    try:
        from src.physics.vant_hoff import validate_candidates_physics
        sol_models = load_ensemble(MODEL_DIR)
        logger.info("Running Van't Hoff validation on top 500...")
        top500 = ranked.head(500)
        validated = validate_candidates_physics(top500, sol_models, pressure_bar=10.0)
        n_consistent = validated["physics_consistent"].sum()
        logger.info(f"Physics consistent: {n_consistent}/500")
        physics_cols = ["physics_consistent", "delta_H_kJ_mol", "delta_S_kJ_mol_K", "vant_hoff_r2", "physics_flags"]
        for col in physics_cols:
            if col in validated.columns:
                ranked.loc[ranked.index[:500], col] = validated[col].values
    except Exception as e:
        logger.warning(f"Van't Hoff skipped: {e}")

    # Save
    ranked.to_parquet(GENERATED_DIR / "candidates_ranked.parquet", index=False)
    ranked.head(100).to_csv(GENERATED_DIR / "top_100_candidates.csv", index=False)

    # Summary
    logger.info(f"\n--- Screening Summary ---")
    logger.info(f"Total screened: {len(results)}")
    logger.info(f"After filtering: {len(ranked)}")
    has_visc = "viscosity_pred" in ranked.columns
    has_dens = "density_pred" in ranked.columns
    logger.info(f"Properties: CO2 solubility{' + viscosity' if has_visc else ''}{' + density' if has_dens else ''}")
    if has_visc:
        logger.info(f"Viscosity range: {ranked['viscosity_pred'].min():.1f} - {ranked['viscosity_pred'].max():.1f} mPa.s")
    if has_dens:
        logger.info(f"Density range: {ranked['density_pred'].min():.0f} - {ranked['density_pred'].max():.0f} kg/m3")
    if "pareto_front" in ranked.columns:
        logger.info(f"Pareto front candidates: {ranked['pareto_front'].sum()}")

    logger.info(f"\nTop 10:")
    for _, row in ranked.head(10).iterrows():
        visc_str = f" visc={row['viscosity_pred']:.0f}" if has_visc else ""
        dens_str = f" dens={row['density_pred']:.0f}" if has_dens else ""
        logger.info(f"  #{int(row['rank'])}: {row['cation_name']}-{row['anion_name']} "
                    f"x_CO2={row['co2_solubility_pred']:.4f}{visc_str}{dens_str}")


if __name__ == "__main__":
    main()
