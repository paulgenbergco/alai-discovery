"""
Combinatorial IL Candidate Generator
=====================================
Generates novel ionic liquid candidates by combining cation scaffolds,
substituents, and anions.

Target: 400,000+ novel, valid, charge-balanced IL candidates.

Usage:
    python -m src.generation.combinatorial
"""

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENERATED_DIR = Path("data/generated")

# ============================================================================
# Cation scaffolds with attachment points ([*] = where substituent goes)
# ============================================================================
# Instead of SMARTS reactions, we'll use pre-built cation SMILES with
# varying alkyl chain lengths

IMIDAZOLIUM_TEMPLATE = "{}n1cc[n+]({})c1"  # R1 and R2 on the two nitrogens
PYRIDINIUM_TEMPLATE = "{}[n+]1ccccc1"  # R1 on nitrogen
PYRROLIDINIUM_TEMPLATE = "{}[N+]1({})CCCC1"  # R1 and R2 on nitrogen
AMMONIUM_TEMPLATE = "{}[N+]({})({})C"  # R1, R2, R3 on nitrogen (4th is methyl)
PHOSPHONIUM_TEMPLATE = "{}[P+]({})({})C"  # R1, R2, R3 on phosphorus

# Alkyl chains
ALKYL_CHAINS = {
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
    "C4": "CCCC",
    "C5": "CCCCC",
    "C6": "CCCCCC",
    "C7": "CCCCCCC",
    "C8": "CCCCCCCC",
    "C10": "CCCCCCCCCC",
    "C12": "CCCCCCCCCCCC",
    "C14": "CCCCCCCCCCCCCC",
    "allyl": "C=CC",
    "benzyl": "c1ccccc1C",
    "hydroxyethyl": "OCCC",
    "methoxyethyl": "COCC",
}

# Common anion SMILES
ANIONS = {
    "NTf2": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "BF4": "F[B-](F)(F)F",
    "PF6": "F[P-](F)(F)(F)(F)F",
    "Cl": "[Cl-]",
    "Br": "[Br-]",
    "DCA": "N#C[N-]C#N",
    "TCM": "[C-](C#N)(C#N)C#N",
    "OAc": "CC([O-])=O",
    "SCN": "[S-]C#N",
    "NO3": "[O-][N+]([O-])=O",
    "OTf": "[O-]S(=O)(=O)C(F)(F)F",
    "EtSO4": "CCOS([O-])(=O)=O",
    "MeSO4": "COS([O-])(=O)=O",
    "HSO4": "[O-]S(O)(=O)=O",
    "Gly": "[NH2]CC([O-])=O",
    "Pro": "[O-]C(=O)C1CCCN1",
    "Lac": "CC(O)C([O-])=O",
    "TFA": "[O-]C(=O)C(F)(F)F",
    "BETI": "[N-](S(=O)(=O)C(F)(F)C(F)(F)F)S(=O)(=O)C(F)(F)C(F)(F)F",
    "FSI": "[N-](S(=O)(=O)F)S(=O)(=O)F",
    "B(CN)4": "[B-](C#N)(C#N)(C#N)C#N",
    "DEP": "CCOP([O-])(=O)OCC",
    "MeSO3": "CS([O-])(=O)=O",
    "Formate": "[O-]C=O",
    "Benzoate": "[O-]C(=O)c1ccccc1",
}


def generate_imidazolium_cations():
    """Generate 1-R1-3-R2-imidazolium cations."""
    cations = []
    for name1, r1 in ALKYL_CHAINS.items():
        for name2, r2 in ALKYL_CHAINS.items():
            smiles = IMIDAZOLIUM_TEMPLATE.format(r1, r2)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                cations.append((f"IM-{name1}-{name2}", canon))
    return cations


def generate_pyridinium_cations():
    """Generate 1-R-pyridinium cations."""
    cations = []
    for name, r in ALKYL_CHAINS.items():
        smiles = PYRIDINIUM_TEMPLATE.format(r)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            cations.append((f"PY-{name}", canon))
    return cations


def generate_pyrrolidinium_cations():
    """Generate 1-R1-1-R2-pyrrolidinium cations."""
    cations = []
    for name1, r1 in ALKYL_CHAINS.items():
        for name2, r2 in list(ALKYL_CHAINS.items())[:8]:  # Limit R2 to shorter chains
            smiles = PYRROLIDINIUM_TEMPLATE.format(r1, r2)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                cations.append((f"PYR-{name1}-{name2}", canon))
    return cations


def generate_ammonium_cations():
    """Generate R1-R2-R3-R4-ammonium cations."""
    cations = []
    chains = list(ALKYL_CHAINS.items())
    # Tetra-substituted: R1-R2-R3-R4 ammonium
    for name1, r1 in chains:
        for name2, r2 in chains:
            for name3, r3 in chains[:6]:
                for name4, r4 in chains[:4]:
                    smiles = f"{r1}[N+]({r2})({r3}){r4}"
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canon = Chem.MolToSmiles(mol)
                        cations.append((f"AM-{name1}-{name2}-{name3}-{name4}", canon))
    return cations


def generate_phosphonium_cations():
    """Generate R1-R2-R3-R4-phosphonium cations."""
    cations = []
    chains = list(ALKYL_CHAINS.items())
    for name1, r1 in chains:
        for name2, r2 in chains[:8]:
            for name3, r3 in chains[:6]:
                for name4, r4 in chains[:4]:
                    smiles = f"{r1}[P+]({r2})({r3}){r4}"
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canon = Chem.MolToSmiles(mol)
                        cations.append((f"PH-{name1}-{name2}-{name3}-{name4}", canon))
    return cations


def generate_all_candidates():
    """Generate all IL candidates by combining cations and anions."""
    logger.info("Generating cation libraries...")

    all_cations = []
    all_cations.extend(generate_imidazolium_cations())
    logger.info(f"  Imidazolium: {len(generate_imidazolium_cations())} cations")

    all_cations.extend(generate_pyridinium_cations())
    logger.info(f"  Pyridinium: {len(generate_pyridinium_cations())} cations")

    all_cations.extend(generate_pyrrolidinium_cations())
    logger.info(f"  Pyrrolidinium: {len(generate_pyrrolidinium_cations())} cations")

    all_cations.extend(generate_ammonium_cations())
    logger.info(f"  Ammonium: {len(generate_ammonium_cations())} cations")

    all_cations.extend(generate_phosphonium_cations())
    logger.info(f"  Phosphonium: {len(generate_phosphonium_cations())} cations")

    # Deduplicate cations by SMILES
    seen_cations = {}
    unique_cations = []
    for name, smi in all_cations:
        if smi not in seen_cations:
            seen_cations[smi] = name
            unique_cations.append((name, smi))

    logger.info(f"Total unique cations: {len(unique_cations)}")
    logger.info(f"Total anions: {len(ANIONS)}")
    logger.info(f"Theoretical combinations: {len(unique_cations) * len(ANIONS)}")

    # Combine cations with anions
    candidates = []
    invalid = 0

    for cat_name, cat_smi in tqdm(unique_cations, desc="Generating ILs"):
        for anion_name, anion_smi in ANIONS.items():
            combined = f"{cat_smi}.{anion_smi}"
            mol = Chem.MolFromSmiles(combined)
            if mol is None:
                invalid += 1
                continue

            canon = Chem.MolToSmiles(mol)
            mw = Descriptors.MolWt(mol)

            # Filter by molecular weight (100-1000)
            if mw < 100 or mw > 1000:
                continue

            candidates.append({
                "smiles": canon,
                "cation_name": cat_name,
                "cation_smiles": cat_smi,
                "anion_name": anion_name,
                "anion_smiles": anion_smi,
                "molecular_weight": mw,
            })

    logger.info(f"Generated {len(candidates)} valid candidates ({invalid} invalid)")

    # Deduplicate by canonical SMILES
    df = pd.DataFrame(candidates)
    before = len(df)
    df = df.drop_duplicates(subset=["smiles"])
    logger.info(f"After deduplication: {len(df)} unique candidates (removed {before - len(df)} duplicates)")

    return df


def filter_novelty(candidates_df: pd.DataFrame, training_smiles: set) -> pd.DataFrame:
    """Remove candidates that appear in the training data."""
    before = len(candidates_df)
    novel = candidates_df[~candidates_df["smiles"].isin(training_smiles)]
    logger.info(f"Novelty filter: {before} → {len(novel)} candidates ({before - len(novel)} in training set)")
    return novel


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Generate candidates
    df = generate_all_candidates()

    # Check novelty against training data
    curated_dir = Path("data/curated")
    if (curated_dir / "train.csv").exists():
        train = pd.read_csv(curated_dir / "train.csv")
        training_smiles = set(train["smiles"].unique())
        df = filter_novelty(df, training_smiles)

    # Save
    output_path = GENERATED_DIR / "il_candidates.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved candidates to {output_path}")

    csv_path = GENERATED_DIR / "il_candidates.csv"
    df.head(1000).to_csv(csv_path, index=False)  # Sample for inspection
    logger.info(f"Saved sample CSV to {csv_path}")

    # Summary
    logger.info(f"\n--- Generation Summary ---")
    logger.info(f"Total unique candidates: {len(df)}")
    logger.info(f"Cation types: {df['cation_name'].str.split('-').str[0].nunique()}")
    logger.info(f"Anion types: {df['anion_name'].nunique()}")
    logger.info(f"MW range: {df['molecular_weight'].min():.1f} - {df['molecular_weight'].max():.1f}")

    return df


if __name__ == "__main__":
    main()
