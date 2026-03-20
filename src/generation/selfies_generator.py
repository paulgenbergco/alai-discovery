"""
SELFIES-Based Generative Design
=================================
Uses STONED-SELFIES mutations to generate novel ionic liquids
by mutating known high-performing ILs.

SELFIES (Self-Referencing Embedded Strings) guarantee 100% chemical
validity — every mutation produces a valid molecule.

Usage:
    python -m src.generation.selfies_generator
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENERATED_DIR = Path("data/generated")
CURATED_DIR = Path("data/curated")

# Seed ILs: known high-performers for CO2 capture (from literature + our screening)
SEED_CATIONS = {
    "EMIM": "CCn1cc[n+](C)c1",
    "BMIM": "CCCCn1cc[n+](C)c1",
    "HMIM": "CCCCCCn1cc[n+](C)c1",
    "BMPyr": "CCCC[N+]1(C)CCCC1",
    "P4444": "CCCC[P+](CCCC)(CCCC)CCCC",
    "P66614": "CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC",
    "Choline": "OCC[N+](C)(C)C",
    "N4444": "CCCC[N+](CCCC)(CCCC)CCCC",
}

SEED_ANIONS = {
    "NTf2": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "BF4": "F[B-](F)(F)F",
    "DCA": "N#C[N-]C#N",
    "OAc": "CC([O-])=O",
    "Gly": "[NH2]CC([O-])=O",
    "Cl": "[Cl-]",
    "SCN": "[S-]C#N",
}


def smiles_to_selfies(smiles: str) -> str | None:
    """Convert SMILES to SELFIES safely."""
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_smiles(selfies_str: str) -> str | None:
    """Convert SELFIES to SMILES safely."""
    try:
        smi = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return None
    except Exception:
        return None


def mutate_selfies(selfies_str: str, n_mutations: int = 1) -> str:
    """Apply random mutations to a SELFIES string.

    Mutations: insert, delete, or replace a random SELFIES token.
    """
    tokens = list(sf.split_selfies(selfies_str))
    alphabet = list(sf.get_semantic_robust_alphabet())

    for _ in range(n_mutations):
        if len(tokens) == 0:
            break

        mutation_type = random.choice(["insert", "delete", "replace"])

        if mutation_type == "insert" and len(tokens) < 50:
            pos = random.randint(0, len(tokens))
            new_token = random.choice(alphabet)
            tokens.insert(pos, new_token)
        elif mutation_type == "delete" and len(tokens) > 3:
            pos = random.randint(0, len(tokens) - 1)
            tokens.pop(pos)
        elif mutation_type == "replace":
            pos = random.randint(0, len(tokens) - 1)
            tokens[pos] = random.choice(alphabet)

    return "".join(tokens)


def tanimoto_distance(smi1: str, smi2: str) -> float:
    """Compute Tanimoto distance between two molecules."""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 1.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


def generate_mutants(
    seed_smiles: str,
    n_candidates: int = 100,
    n_mutations: int = 2,
    max_attempts: int = 500,
) -> list[str]:
    """Generate novel molecules by mutating a seed SMILES."""
    seed_selfies = smiles_to_selfies(seed_smiles)
    if seed_selfies is None:
        return []

    candidates = set()
    attempts = 0

    while len(candidates) < n_candidates and attempts < max_attempts:
        mutated = mutate_selfies(seed_selfies, n_mutations=n_mutations)
        smi = selfies_to_smiles(mutated)
        if smi is not None and smi != seed_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                if 50 < mw < 600:  # Reasonable MW for ion fragments
                    candidates.add(smi)
        attempts += 1

    return list(candidates)


def generate_selfies_candidates(
    n_per_seed: int = 200,
    n_mutations: int = 2,
) -> pd.DataFrame:
    """Generate novel IL candidates by mutating seed cations and anions."""
    logger.info("Generating SELFIES-based IL candidates...")

    # Generate mutant cations
    all_cation_mutants = {}
    for name, smi in SEED_CATIONS.items():
        mutants = generate_mutants(smi, n_candidates=n_per_seed, n_mutations=n_mutations)
        all_cation_mutants[name] = mutants
        logger.info(f"  {name}: {len(mutants)} cation mutants")

    # Generate mutant anions
    all_anion_mutants = {}
    for name, smi in SEED_ANIONS.items():
        mutants = generate_mutants(smi, n_candidates=n_per_seed // 2, n_mutations=n_mutations)
        all_anion_mutants[name] = mutants
        logger.info(f"  {name}: {len(mutants)} anion mutants")

    # Combine: each mutant cation with original anions + each mutant anion with original cations
    candidates = []

    # Mutant cations x original anions
    for cat_parent, mutants in all_cation_mutants.items():
        for cation_smi in mutants:
            for anion_name, anion_smi in SEED_ANIONS.items():
                combined = f"{cation_smi}.{anion_smi}"
                mol = Chem.MolFromSmiles(combined)
                if mol is not None:
                    canon = Chem.MolToSmiles(mol)
                    mw = Descriptors.MolWt(mol)
                    if 100 < mw < 1000:
                        candidates.append({
                            "smiles": canon,
                            "cation_name": f"SELFIES-{cat_parent}-mut",
                            "cation_smiles": cation_smi,
                            "anion_name": anion_name,
                            "anion_smiles": anion_smi,
                            "molecular_weight": mw,
                            "source": "selfies",
                        })

    # Original cations x mutant anions
    for anion_parent, mutants in all_anion_mutants.items():
        for anion_smi in mutants:
            for cation_name, cation_smi in SEED_CATIONS.items():
                combined = f"{cation_smi}.{anion_smi}"
                mol = Chem.MolFromSmiles(combined)
                if mol is not None:
                    canon = Chem.MolToSmiles(mol)
                    mw = Descriptors.MolWt(mol)
                    if 100 < mw < 1000:
                        candidates.append({
                            "smiles": canon,
                            "cation_name": cation_name,
                            "cation_smiles": cation_smi,
                            "anion_name": f"SELFIES-{anion_parent}-mut",
                            "anion_smiles": anion_smi,
                            "molecular_weight": mw,
                            "source": "selfies",
                        })

    df = pd.DataFrame(candidates)
    df = df.drop_duplicates(subset=["smiles"])
    logger.info(f"Total SELFIES-generated candidates: {len(df)}")

    return df


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # Generate SELFIES candidates
    selfies_df = generate_selfies_candidates(n_per_seed=200, n_mutations=2)

    # Load existing combinatorial candidates
    combo_path = GENERATED_DIR / "il_candidates.parquet"
    if combo_path.exists():
        combo_df = pd.read_parquet(combo_path)
        combo_df["source"] = "combinatorial"
        logger.info(f"Existing combinatorial candidates: {len(combo_df)}")

        # Merge and deduplicate
        merged = pd.concat([combo_df, selfies_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["smiles"])
        logger.info(f"Merged total: {len(merged)} ({len(merged) - len(combo_df)} new from SELFIES)")
    else:
        merged = selfies_df

    # Save merged candidates
    merged.to_parquet(GENERATED_DIR / "il_candidates_all.parquet", index=False)
    logger.info(f"Saved all candidates to {GENERATED_DIR}/il_candidates_all.parquet")

    # Summary
    logger.info(f"\n--- SELFIES Generation Summary ---")
    logger.info(f"SELFIES candidates: {len(selfies_df)}")
    logger.info(f"Total candidates (merged): {len(merged)}")
    if "source" in merged.columns:
        logger.info(f"By source:\n{merged['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
