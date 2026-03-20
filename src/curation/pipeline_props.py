"""
Curation + Training Pipeline for Viscosity and Density
=======================================================
Curates viscosity and density data from ILThermo, maps to SMILES,
and trains D-MPNN models.

Usage:
    python -m src.curation.pipeline_props
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
CURATED_DIR = Path("data/curated")


def curate_property(property_name: str, log_transform: bool = True) -> pd.DataFrame | None:
    """Curate a single property dataset and prepare for training."""
    safe_name = property_name.lower().replace(" ", "_")
    raw_path = RAW_DIR / f"ilthermo_{safe_name}_raw.parquet"

    if not raw_path.exists():
        logger.warning(f"Raw data not found: {raw_path}")
        return None

    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df)} {property_name} records")

    # Load SMILES mapping from CO2 work
    mapping_path = RAW_DIR / "il_smiles_mapping.csv"
    if not mapping_path.exists():
        logger.error("SMILES mapping not found. Run smiles_mapper.py first.")
        return None

    mapping = pd.read_csv(mapping_path)
    smiles_dict = dict(zip(mapping["il_name"], mapping["smiles"]))

    # Map names to SMILES
    df["smiles"] = df["il_name"].map(smiles_dict)
    before = len(df)
    df = df[df["smiles"].notna()].copy()
    logger.info(f"SMILES mapped: {before} -> {len(df)} records ({df['il_name'].nunique()} ILs)")

    # Filter valid values
    df = df[df["value"] > 0].copy()
    df = df[df["temperature_K"].between(200, 500)].copy()

    # Set default pressure for pure IL measurements
    df["pressure_kPa"] = df["pressure_kPa"].fillna(101.325)
    df["pressure_bar"] = df["pressure_kPa"] / 100.0

    # Deduplicate
    df["T_round"] = df["temperature_K"].round(0)
    df = df.groupby(["smiles", "T_round"]).agg(
        temperature_K=("temperature_K", "median"),
        pressure_bar=("pressure_bar", "median"),
        value=("value", "median"),
        il_name=("il_name", "first"),
    ).reset_index().drop(columns=["T_round"])

    logger.info(f"After dedup: {len(df)} records")

    # Log transform (viscosity spans orders of magnitude)
    if log_transform:
        df["log_value"] = np.log10(df["value"].clip(lower=1e-10))
    else:
        df["log_value"] = df["value"]

    # Split: 80/10/10 by SMILES
    np.random.seed(42)
    unique_smiles = list(df["smiles"].unique())
    np.random.shuffle(unique_smiles)
    n = len(unique_smiles)
    train_smi = set(unique_smiles[:int(n * 0.8)])
    val_smi = set(unique_smiles[int(n * 0.8):int(n * 0.9)])
    test_smi = set(unique_smiles[int(n * 0.9):])

    train_df = df[df["smiles"].isin(train_smi)]
    val_df = df[df["smiles"].isin(val_smi)]
    test_df = df[df["smiles"].isin(test_smi)]

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save
    prop_dir = CURATED_DIR / safe_name
    prop_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"log_{safe_name}"
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        main = pd.DataFrame({"smiles": split_df["smiles"], target_col: split_df["log_value"]})
        main.to_csv(prop_dir / f"{name}.csv", index=False)

        features = split_df[["temperature_K", "pressure_bar"]].copy()
        features.to_csv(prop_dir / f"{name}_features.csv", index=False)

    logger.info(f"Saved to {prop_dir}/")
    return df


def train_property_model(property_name: str, epochs: int = 30, ensemble_size: int = 4):
    """Train D-MPNN ensemble for a property."""
    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, RegressionFFN, MeanAggregation
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    import lightning as L
    from sklearn.metrics import r2_score, mean_squared_error

    safe_name = property_name.lower().replace(" ", "_")
    prop_dir = CURATED_DIR / safe_name
    model_dir = Path("models") / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"log_{safe_name}"

    # Load data
    splits = {}
    for split_name in ["train", "val", "test"]:
        main = pd.read_csv(prop_dir / f"{split_name}.csv")
        features = pd.read_csv(prop_dir / f"{split_name}_features.csv")

        mols, valid_idx = [], []
        for i, smi in enumerate(main["smiles"]):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_idx.append(i)

        splits[split_name] = {
            "smiles": [main["smiles"].iloc[i] for i in valid_idx],
            "mols": mols,
            "targets": main[target_col].values[valid_idx],
            "features": features.values[valid_idx],
        }

    logger.info(f"Training {property_name}: train={len(splits['train']['mols'])}, "
                f"val={len(splits['val']['mols'])}, test={len(splits['test']['mols'])}")

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    models = []

    for model_idx in range(ensemble_size):
        seed = 42 + model_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info(f"  Model {model_idx + 1}/{ensemble_size} (seed={seed})")

        train_data = [
            MoleculeDatapoint(mol=mol, y=np.array([t]), x_d=np.array(f, dtype=np.float32))
            for mol, t, f in zip(splits["train"]["mols"], splits["train"]["targets"], splits["train"]["features"])
        ]
        val_data = [
            MoleculeDatapoint(mol=mol, y=np.array([t]), x_d=np.array(f, dtype=np.float32))
            for mol, t, f in zip(splits["val"]["mols"], splits["val"]["targets"], splits["val"]["features"])
        ]

        train_dataset = MoleculeDataset(train_data, featurizer=featurizer)
        val_dataset = MoleculeDataset(val_data, featurizer=featurizer)

        train_loader = build_dataloader(train_dataset, batch_size=64, shuffle=True, seed=seed)
        val_loader = build_dataloader(val_dataset, batch_size=64, shuffle=False)

        mp = BondMessagePassing(d_h=300)
        agg = MeanAggregation()
        ffn = RegressionFFN(input_dim=300 + 2, hidden_dim=300, n_layers=2)
        model = MPNN(message_passing=mp, agg=agg, predictor=ffn)

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, train_loader, val_loader)

        save_path = model_dir / f"model_{model_idx}.pt"
        torch.save(model.state_dict(), save_path)
        models.append(model)

    # Evaluate on test set
    all_preds = []
    for model in models:
        model.eval()
        test_data = [
            MoleculeDatapoint(mol=mol, x_d=np.array(f, dtype=np.float32))
            for mol, f in zip(splits["test"]["mols"], splits["test"]["features"])
        ]
        test_dataset = MoleculeDataset(test_data, featurizer=featurizer)
        test_loader = build_dataloader(test_dataset, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                output = model.predict_step(batch, 0)
                preds.extend(output.squeeze(-1).cpu().numpy().tolist())
        all_preds.append(preds)

    mean_preds = np.array(all_preds).mean(axis=0)
    std_preds = np.array(all_preds).std(axis=0)
    targets = splits["test"]["targets"]

    r2 = r2_score(targets, mean_preds)
    rmse = np.sqrt(mean_squared_error(targets, mean_preds))

    logger.info(f"\n{property_name} TEST Results:")
    logger.info(f"  R² = {r2:.4f}")
    logger.info(f"  RMSE = {rmse:.4f}")

    # Save predictions
    results = pd.DataFrame({
        "smiles": splits["test"]["smiles"],
        "actual": targets,
        "predicted": mean_preds,
        "uncertainty": std_preds,
    })
    results.to_csv(prop_dir / "test_predictions.csv", index=False)

    return r2


def main():
    results = {}

    for prop, log_xform in [("viscosity", True), ("density", False)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {prop}")
        logger.info(f"{'='*60}")

        df = curate_property(prop, log_transform=log_xform)
        if df is not None:
            r2 = train_property_model(prop, epochs=30, ensemble_size=4)
            results[prop] = r2

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    for prop, r2 in results.items():
        logger.info(f"  {prop}: Test R² = {r2:.4f}")


if __name__ == "__main__":
    main()
