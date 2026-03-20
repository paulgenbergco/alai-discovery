"""
D-MPNN Ensemble Training
=========================
Trains an ensemble of D-MPNN models for CO2 solubility prediction using Chemprop v2.

Architecture:
- 8 D-MPNN models with different random seeds
- Process-conditioned: T(K) and P(bar) as extra features
- Target: log10(mole fraction CO2)
- Scaffold-split validation

Usage:
    python -m src.models.train_ensemble
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CURATED_DIR = Path("data/curated")
MODEL_DIR = Path("models")
ENSEMBLE_SIZE = 8


def _smiles_to_mol(smiles_list):
    """Convert SMILES to RDKit Mol objects, filtering invalids."""
    mols = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)
    return mols, valid_idx


def load_data():
    """Load train/val/test splits."""
    splits = {}
    for name in ["train", "val", "test"]:
        main = pd.read_csv(CURATED_DIR / f"{name}.csv")
        features = pd.read_csv(CURATED_DIR / f"{name}_features.csv")

        smiles = main["smiles"].tolist()
        mols, valid_idx = _smiles_to_mol(smiles)

        splits[name] = {
            "smiles": [smiles[i] for i in valid_idx],
            "mols": mols,
            "targets": main["log_co2_solubility"].values[valid_idx],
            "features": features.values[valid_idx],
        }
    return splits


def _make_datapoints(mols, targets, features):
    """Create Chemprop MoleculeDatapoint list."""
    from chemprop.data import MoleculeDatapoint

    return [
        MoleculeDatapoint(
            mol=mol,
            y=np.array([target]),
            x_d=np.array(feat, dtype=np.float32),
        )
        for mol, target, feat in zip(mols, targets, features)
    ]


def train_single_model(splits, model_idx: int, save_dir: Path):
    """Train a single D-MPNN model using Chemprop v2 API."""
    from chemprop.data import MoleculeDataset, build_dataloader
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, RegressionFFN, MeanAggregation
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    import lightning as L

    seed = 42 + model_idx
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"Training model {model_idx + 1}/{ENSEMBLE_SIZE} (seed={seed})")

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    train_data = _make_datapoints(
        splits["train"]["mols"],
        splits["train"]["targets"],
        splits["train"]["features"],
    )
    val_data = _make_datapoints(
        splits["val"]["mols"],
        splits["val"]["targets"],
        splits["val"]["features"],
    )

    train_dataset = MoleculeDataset(train_data, featurizer=featurizer)
    val_dataset = MoleculeDataset(val_data, featurizer=featurizer)

    train_loader = build_dataloader(train_dataset, batch_size=64, shuffle=True, seed=seed)
    val_loader = build_dataloader(val_dataset, batch_size=64, shuffle=False)

    # Build D-MPNN model
    n_extra = splits["train"]["features"].shape[1]  # 2 (T, P)

    mp = BondMessagePassing(d_h=300)
    agg = MeanAggregation()
    ffn = RegressionFFN(input_dim=300 + n_extra, hidden_dim=300, n_layers=2)

    model = MPNN(message_passing=mp, agg=agg, predictor=ffn)

    trainer = L.Trainer(
        max_epochs=30,
        accelerator="cpu",  # MPS can be unstable for small models; CPU is fast enough
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save model
    save_path = save_dir / f"model_{model_idx}.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")

    return model


def predict_with_model(model, mols, features):
    """Get predictions from a single model."""
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    data = [
        MoleculeDatapoint(mol=mol, x_d=np.array(feat, dtype=np.float32))
        for mol, feat in zip(mols, features)
    ]
    dataset = MoleculeDataset(data, featurizer=featurizer)
    loader = build_dataloader(dataset, batch_size=64, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            output = model.predict_step(batch, 0)
            preds.extend(output.squeeze(-1).cpu().numpy().tolist())

    return np.array(preds)


def predict_ensemble(models, mols, features):
    """Run ensemble prediction and return mean + std."""
    all_preds = []
    for model in models:
        preds = predict_with_model(model, mols, features)
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # (n_models, n_samples)
    mean_preds = all_preds.mean(axis=0)
    std_preds = all_preds.std(axis=0)

    return mean_preds, std_preds


def evaluate(splits, models):
    """Evaluate ensemble on val and test sets."""
    for split_name in ["val", "test"]:
        split = splits[split_name]
        if len(split["smiles"]) == 0:
            logger.warning(f"No data in {split_name} set, skipping evaluation")
            continue

        mean_preds, std_preds = predict_ensemble(
            models, split["mols"], split["features"]
        )

        targets = split["targets"]
        rmse = np.sqrt(mean_squared_error(targets, mean_preds))
        r2 = r2_score(targets, mean_preds)
        mae = np.mean(np.abs(targets - mean_preds))

        logger.info(f"\n{split_name.upper()} Results:")
        logger.info(f"  R² = {r2:.4f}")
        logger.info(f"  RMSE = {rmse:.4f}")
        logger.info(f"  MAE = {mae:.4f}")
        logger.info(f"  Mean uncertainty (ensemble std): {std_preds.mean():.4f}")

        # Save predictions for demo
        results = pd.DataFrame({
            "smiles": split["smiles"],
            "actual": targets,
            "predicted": mean_preds,
            "uncertainty": std_preds,
        })
        results.to_csv(CURATED_DIR / f"{split_name}_predictions.csv", index=False)
        logger.info(f"Saved predictions to {CURATED_DIR}/{split_name}_predictions.csv")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    splits = load_data()
    logger.info(f"Train: {len(splits['train']['smiles'])} records")
    logger.info(f"Val: {len(splits['val']['smiles'])} records")
    logger.info(f"Test: {len(splits['test']['smiles'])} records")

    # Train ensemble
    models = []
    for i in range(ENSEMBLE_SIZE):
        model = train_single_model(splits, i, MODEL_DIR)
        models.append(model)

    # Evaluate
    evaluate(splits, models)

    logger.info(f"\nEnsemble of {ENSEMBLE_SIZE} models trained and saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
