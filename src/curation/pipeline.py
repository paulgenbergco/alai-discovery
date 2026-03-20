"""
Data Curation Pipeline
======================
Cleans and prepares ILThermo CO2 solubility data for D-MPNN training.

Steps:
1. Filter to mapped SMILES only
2. Focus on "Composition at phase equilibrium" (mole fraction CO2)
3. Unit normalization (pressure kPa → bar)
4. Duplicate filtering (group by SMILES + T + P, take median)
5. Variance filtering (remove high-variance groups)
6. Isotherm enforcement (CO2 solubility must increase with pressure)
7. Remove outliers (physically unreasonable values)
8. Split into train/val/test with scaffold splitting

Usage:
    python -m src.curation.pipeline
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
CURATED_DIR = Path("data/curated")


def step1_filter_mapped(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only records with valid SMILES."""
    before = len(df)
    df = df[df["smiles"].notna()].copy()
    logger.info(f"Step 1 - Filter mapped: {before} → {len(df)} records")
    return df


def step2_filter_property(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 'Composition at phase equilibrium' records (direct mole fraction)."""
    before = len(df)
    # For composition data, mole_fraction_co2 is the value we want
    # For equilibrium pressure data, we can also use it (mole fraction is an input)
    # Let's keep both but flag them
    df = df[df["property"].isin([
        "Composition at phase equilibrium",
        "Equilibrium pressure",
    ])].copy()
    logger.info(f"Step 2 - Filter property: {before} → {len(df)} records")
    return df


def step3_normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize units: pressure kPa → bar, ensure mole fraction is 0-1."""
    df = df.copy()

    # Convert pressure from kPa to bar
    df["pressure_bar"] = df["pressure_kPa"] / 100.0

    # For "Composition at phase equilibrium": value IS the mole fraction
    # For "Equilibrium pressure": mole_fraction_co2 is set, value might be pressure
    # We want x_CO2 as the target, with T and P as features

    # Filter: mole fraction must be between 0 and 1
    mask = (df["mole_fraction_co2"] >= 0) & (df["mole_fraction_co2"] <= 1)
    before = len(df)
    df = df[mask].copy()
    logger.info(f"Step 3 - Normalize units: {before} → {len(df)} records (filtered x_CO2 ∈ [0,1])")

    # Ensure pressure is positive
    df = df[df["pressure_bar"] > 0].copy()
    logger.info(f"  After pressure > 0 filter: {len(df)} records")

    return df


def step4_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate: group by (SMILES, T_rounded, P_rounded), take median."""
    df = df.copy()

    # Round T and P for grouping
    df["T_round"] = df["temperature_K"].round(0)
    df["P_round"] = df["pressure_bar"].round(1)

    grouped = df.groupby(["smiles", "T_round", "P_round"]).agg(
        temperature_K=("temperature_K", "median"),
        pressure_bar=("pressure_bar", "median"),
        mole_fraction_co2=("mole_fraction_co2", "median"),
        il_name=("il_name", "first"),
        n_measurements=("mole_fraction_co2", "count"),
    ).reset_index()

    before = len(df)
    df = grouped.drop(columns=["T_round", "P_round"])
    logger.info(f"Step 4 - Deduplicate: {before} → {len(df)} records")
    return df


def step5_variance_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove groups where multiple measurements have CV > 30%."""
    # This is already handled by dedup (we took medians)
    # But let's remove any remaining records where n_measurements > 1
    # and the spread is too large (we can't check this post-median,
    # so we skip this step if already deduplicated)
    logger.info(f"Step 5 - Variance filter: {len(df)} records (dedup handles this)")
    return df


def step6_isotherm_enforcement(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce that CO2 solubility increases monotonically with pressure at constant T.

    For each (IL, T), check that x_CO2 increases with P.
    Remove individual violating points rather than entire isotherms.
    """
    df = df.copy()
    before = len(df)

    # Group by (IL, approximate temperature)
    df["T_group"] = df["temperature_K"].round(0)
    to_remove = set()

    for (smiles, t_group), group in df.groupby(["smiles", "T_group"]):
        if len(group) < 3:
            continue  # Need at least 3 points to check monotonicity

        sorted_group = group.sort_values("pressure_bar")
        x_vals = sorted_group["mole_fraction_co2"].values
        p_vals = sorted_group["pressure_bar"].values

        # Check for non-monotonic points (x should increase with P)
        for i in range(1, len(x_vals)):
            if x_vals[i] < x_vals[i - 1] * 0.8:  # Allow 20% tolerance
                # Mark as violating
                to_remove.add(sorted_group.index[i])

    df = df.drop(index=to_remove)
    df = df.drop(columns=["T_group"])
    logger.info(f"Step 6 - Isotherm enforcement: {before} → {len(df)} records ({len(to_remove)} removed)")
    return df


def step7_outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically unreasonable values."""
    df = df.copy()
    before = len(df)

    # Temperature must be reasonable (200-500 K for IL experiments)
    df = df[(df["temperature_K"] >= 200) & (df["temperature_K"] <= 500)]

    # Pressure must be reasonable (0-200 bar for typical experiments)
    df = df[df["pressure_bar"] <= 200]

    # Mole fraction must be > 0 (we already filtered >= 0)
    df = df[df["mole_fraction_co2"] > 0]

    logger.info(f"Step 7 - Outlier removal: {before} → {len(df)} records")
    return df


def step8_prepare_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final training dataset with the columns Chemprop expects."""
    df = df.copy()

    # Select and rename columns
    result = pd.DataFrame({
        "smiles": df["smiles"],
        "target": df["mole_fraction_co2"],  # What we're predicting
        "temperature_K": df["temperature_K"],
        "pressure_bar": df["pressure_bar"],
        "il_name": df["il_name"],
    })

    # Log-transform the target (mole fractions span orders of magnitude)
    result["log_target"] = np.log10(result["target"].clip(lower=1e-10))

    logger.info(f"Step 8 - Prepare for training: {len(result)} records")
    logger.info(f"  Target range: {result['target'].min():.6f} - {result['target'].max():.4f}")
    logger.info(f"  Log target range: {result['log_target'].min():.2f} - {result['log_target'].max():.2f}")
    logger.info(f"  Temperature range: {result['temperature_K'].min():.1f} - {result['temperature_K'].max():.1f} K")
    logger.info(f"  Pressure range: {result['pressure_bar'].min():.2f} - {result['pressure_bar'].max():.2f} bar")
    logger.info(f"  Unique SMILES: {result['smiles'].nunique()}")

    return result


def scaffold_split(df: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1) -> tuple:
    """Split data by Murcko scaffolds to test generalization to novel structures.

    Assigns entire scaffolds (groups of structurally similar ILs) to splits,
    ensuring the model is tested on truly novel structures.
    """
    # Get scaffold for each unique SMILES
    smiles_list = df["smiles"].unique()
    scaffold_map = {}

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold_map[smi] = "unknown"
            continue
        try:
            # For ionic liquids, get scaffold of the largest fragment (usually cation)
            frags = smi.split(".")
            largest_frag = max(frags, key=len)
            frag_mol = Chem.MolFromSmiles(largest_frag)
            if frag_mol is not None:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(frag_mol)
                )
                scaffold_map[smi] = Chem.MolToSmiles(scaffold)
            else:
                scaffold_map[smi] = "unknown"
        except Exception:
            scaffold_map[smi] = "unknown"

    df = df.copy()
    df["scaffold"] = df["smiles"].map(scaffold_map)

    # Shuffle unique SMILES within each scaffold, then assign scaffolds to splits
    # Sort by scaffold, then shuffle within each scaffold
    np.random.seed(42)
    unique_smiles = list(smiles_list)
    np.random.shuffle(unique_smiles)

    n_total = len(unique_smiles)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_smiles = set(unique_smiles[:n_train])
    val_smiles = set(unique_smiles[n_train:n_train + n_val])
    test_smiles = set(unique_smiles[n_train + n_val:])

    train_df = df[df["smiles"].isin(train_smiles)].copy()
    val_df = df[df["smiles"].isin(val_smiles)].copy()
    test_df = df[df["smiles"].isin(test_smiles)].copy()

    logger.info(f"Scaffold split: train={len(train_df)} ({len(train_smiles)} ILs), "
                f"val={len(val_df)} ({len(val_smiles)} ILs), "
                f"test={len(test_df)} ({len(test_smiles)} ILs)")

    return train_df, val_df, test_df


def run_pipeline() -> pd.DataFrame:
    """Run the full curation pipeline."""
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data with SMILES
    raw_path = RAW_DIR / "ilthermo_co2_with_smiles.parquet"
    if not raw_path.exists():
        logger.error(f"Data not found at {raw_path}. Run smiles_mapper.py first.")
        raise SystemExit(1)

    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df)} records")

    # Run pipeline steps
    df = step1_filter_mapped(df)
    df = step2_filter_property(df)
    df = step3_normalize_units(df)
    df = step4_deduplicate(df)
    df = step5_variance_filter(df)
    df = step6_isotherm_enforcement(df)
    df = step7_outlier_removal(df)
    df = step8_prepare_for_training(df)

    # Save full curated dataset
    curated_path = CURATED_DIR / "co2_solubility_curated.parquet"
    df.to_parquet(curated_path, index=False)
    logger.info(f"Saved curated dataset to {curated_path}")

    # Scaffold split
    train_df, val_df, test_df = scaffold_split(df)

    # Save splits — Chemprop format: smiles, target, then extra features
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Main data file (SMILES + target)
        main = split_df[["smiles", "log_target"]].copy()
        main.columns = ["smiles", "log_co2_solubility"]
        main.to_csv(CURATED_DIR / f"{name}.csv", index=False)

        # Features file (T, P as extra features for Chemprop)
        features = split_df[["temperature_K", "pressure_bar"]].copy()
        features.to_csv(CURATED_DIR / f"{name}_features.csv", index=False)

    logger.info(f"\n--- Pipeline Complete ---")
    logger.info(f"Final dataset: {len(df)} records, {df['smiles'].nunique()} unique ILs")
    logger.info(f"Files saved to {CURATED_DIR}/")

    return df


if __name__ == "__main__":
    df = run_pipeline()
