"""
Expanded Combinatorial IL Candidate Generator (v2)
====================================================
Generates 400,000+ novel ionic liquid candidates with broader
chemical diversity: more chain lengths, branching, functional groups,
and anion variety.

Usage:
    python -m src.generation.combinatorial_v2
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENERATED_DIR = Path("data/generated")

# ============================================================================
# Expanded substituent library
# ============================================================================
ALKYL = {
    "Me": "C", "Et": "CC", "Pr": "CCC", "Bu": "CCCC",
    "Pe": "CCCCC", "Hex": "CCCCCC", "Hep": "CCCCCCC", "Oct": "CCCCCCCC",
    "Non": "CCCCCCCCC", "Dec": "CCCCCCCCCC", "Dod": "CCCCCCCCCCCC",
    "Tet": "CCCCCCCCCCCCCC", "Hex16": "CCCCCCCCCCCCCCCC",
    # Branched
    "iPr": "CC(C)C", "iBu": "CC(C)CC", "sBu": "CCC(C)C", "neoP": "CC(C)(C)C",
    "2EtHex": "CCCCC(CC)C",
    # Functional
    "allyl": "C=CC", "vinyl": "C=C",
    "benzyl": "c1ccccc1C", "phenyl": "c1ccccc1",
    "hydroxyEt": "OCCC", "methoxyEt": "COCC", "ethoxyEt": "CCOCC",
    "cyanoMe": "N#CC", "cyanoPr": "N#CCCC",
    # Fluorinated
    "CF3Et": "CC(F)(F)F", "perfluoroBu": "C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    # Silyl
    "TMS": "[Si](C)(C)C",
}

# Expanded anion library (50+ anions)
ANIONS = {
    # Fluorinated sulfonylimides
    "NTf2": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "BETI": "[N-](S(=O)(=O)C(F)(F)C(F)(F)F)S(=O)(=O)C(F)(F)C(F)(F)F",
    "FSI": "[N-](S(=O)(=O)F)S(=O)(=O)F",
    # Fluorinated anions
    "BF4": "F[B-](F)(F)F",
    "PF6": "F[P-](F)(F)(F)(F)F",
    "OTf": "[O-]S(=O)(=O)C(F)(F)F",
    "TFA": "[O-]C(=O)C(F)(F)F",
    "TFES": "[O-]S(=O)(=O)C(F)(F)C(F)(F)F",
    "NfO": "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "FAP": "F[P-](F)(OC(F)(F)C(F)(F)F)(OC(F)(F)C(F)(F)F)(OC(F)(F)C(F)(F)F)F",
    # Cyano anions
    "DCA": "N#C[N-]C#N",
    "TCM": "[C-](C#N)(C#N)C#N",
    "B(CN)4": "[B-](C#N)(C#N)(C#N)C#N",
    "SCN": "[S-]C#N",
    # Halides
    "Cl": "[Cl-]",
    "Br": "[Br-]",
    "I": "[I-]",
    # Carboxylates
    "OAc": "CC([O-])=O",
    "Formate": "[O-]C=O",
    "Propanoate": "CCC([O-])=O",
    "Butanoate": "CCCC([O-])=O",
    "Hexanoate": "CCCCCC([O-])=O",
    "Octanoate": "CCCCCCCC([O-])=O",
    "Benzoate": "[O-]C(=O)c1ccccc1",
    "Salicylate": "[O-]C(=O)c1ccccc1O",
    "Lactate": "CC(O)C([O-])=O",
    "Glycinate": "[NH2]CC([O-])=O",
    "Prolinate": "[O-]C(=O)C1CCCN1",
    "Alaninate": "[NH2]C(C)C([O-])=O",
    "Taurate": "[NH-]CCS(=O)(=O)O",
    # Sulfates/sulfonates
    "EtSO4": "CCOS([O-])(=O)=O",
    "MeSO4": "COS([O-])(=O)=O",
    "HSO4": "[O-]S(O)(=O)=O",
    "MeSO3": "CS([O-])(=O)=O",
    "Tosylate": "Cc1ccc(S([O-])(=O)=O)cc1",
    "DBS": "CCCCCCCCCCCCc1ccccc1S([O-])(=O)=O",
    # Phosphates
    "DEP": "CCOP([O-])(=O)OCC",
    "DMP": "COP([O-])(=O)OC",
    "DBP": "CCCCOP([O-])(=O)OCCCC",
    "H2PO4": "[O-]P(O)(O)=O",
    "MeHPO3": "CP([O-])(=O)O",
    # Other
    "NO3": "[O-][N+]([O-])=O",
    "MeCO3": "COC([O-])=O",
    "Sac": "O=C1NS([O-])(=O)c2ccccc21",
    "Ace": "CC(=O)[N-]S(C)(=O)=O",
}

# Template definitions
IM_TEMPLATE = "{}n1cc[n+]({})c1"        # 1,3-disubstituted imidazolium
IM2_TEMPLATE = "{}n1cc[n+]({})c1{}"     # 1,2,3-trisubstituted imidazolium
PY_TEMPLATE = "{}[n+]1ccccc1"           # N-substituted pyridinium
PY3_TEMPLATE = "{}[n+]1cccc(C)c1"       # N-substituted 3-methylpyridinium
PY4_TEMPLATE = "{}[n+]1ccc(C)cc1"       # N-substituted 4-methylpyridinium
PYR_TEMPLATE = "{}[N+]1({})CCCC1"       # 1,1-disubstituted pyrrolidinium
PIP_TEMPLATE = "{}[N+]1({})CCCCC1"      # 1,1-disubstituted piperidinium
MOR_TEMPLATE = "{}[N+]1({})CCOCC1"      # 4,4-disubstituted morpholinium


def generate_cations():
    """Generate all cation SMILES from templates + substituents."""
    cations = {}
    chains = list(ALKYL.items())

    # Imidazolium: R1-N...N-R2 (all pairs)
    for n1, r1 in chains:
        for n2, r2 in chains:
            smi = IM_TEMPLATE.format(r1, r2)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"IM-{n1}-{n2}"] = Chem.MolToSmiles(mol)

    # 2-methyl imidazolium: R1-N...N-R2 with C2-methyl
    for n1, r1 in chains[:15]:  # Limit for size
        for n2, r2 in chains[:10]:
            smi = IM2_TEMPLATE.format(r1, r2, "C")
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"IM2-{n1}-{n2}"] = Chem.MolToSmiles(mol)

    # Pyridinium variants
    for n1, r1 in chains:
        for template, prefix in [(PY_TEMPLATE, "PY"), (PY3_TEMPLATE, "PY3"), (PY4_TEMPLATE, "PY4")]:
            smi = template.format(r1)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"{prefix}-{n1}"] = Chem.MolToSmiles(mol)

    # Pyrrolidinium
    for n1, r1 in chains:
        for n2, r2 in chains[:12]:
            smi = PYR_TEMPLATE.format(r1, r2)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"PYR-{n1}-{n2}"] = Chem.MolToSmiles(mol)

    # Piperidinium
    for n1, r1 in chains[:15]:
        for n2, r2 in chains[:10]:
            smi = PIP_TEMPLATE.format(r1, r2)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"PIP-{n1}-{n2}"] = Chem.MolToSmiles(mol)

    # Morpholinium
    for n1, r1 in chains[:15]:
        for n2, r2 in chains[:10]:
            smi = MOR_TEMPLATE.format(r1, r2)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"MOR-{n1}-{n2}"] = Chem.MolToSmiles(mol)

    # Ammonium: R1-N+(R2)(R3)-R4
    for n1, r1 in chains[:18]:
        for n2, r2 in chains[:15]:
            for n3, r3 in chains[:10]:
                for n4, r4 in chains[:8]:
                    smi = f"{r1}[N+]({r2})({r3}){r4}"
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        cations[f"AM-{n1}-{n2}-{n3}-{n4}"] = Chem.MolToSmiles(mol)

    # Phosphonium: R1-P+(R2)(R3)-R4
    for n1, r1 in chains[:18]:
        for n2, r2 in chains[:12]:
            for n3, r3 in chains[:10]:
                for n4, r4 in chains[:6]:
                    smi = f"{r1}[P+]({r2})({r3}){r4}"
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        cations[f"PH-{n1}-{n2}-{n3}-{n4}"] = Chem.MolToSmiles(mol)

    # Sulfonium: R1-S+(R2)-R3
    for n1, r1 in chains[:12]:
        for n2, r2 in chains[:10]:
            for n3, r3 in chains[:8]:
                smi = f"{r1}[S+]({r2}){r3}"
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    cations[f"SU-{n1}-{n2}-{n3}"] = Chem.MolToSmiles(mol)

    # Guanidinium
    for n1, r1 in chains[:8]:
        for n2, r2 in chains[:6]:
            smi = f"{r1}N(C)C(=[NH2+])N({r2})C"
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cations[f"GU-{n1}-{n2}"] = Chem.MolToSmiles(mol)

    # Deduplicate by SMILES
    unique = {}
    for name, smi in cations.items():
        if smi not in unique.values():
            unique[name] = smi

    return unique


def generate_all_candidates():
    """Combine all cations x all anions."""
    logger.info("Generating expanded cation library...")
    cations = generate_cations()
    logger.info(f"Unique cations: {len(cations)}")
    logger.info(f"Anions: {len(ANIONS)}")
    logger.info(f"Theoretical max: {len(cations) * len(ANIONS):,}")

    candidates = []
    invalid = 0

    for cat_name, cat_smi in tqdm(cations.items(), desc="Generating ILs"):
        for anion_name, anion_smi in ANIONS.items():
            combined = f"{cat_smi}.{anion_smi}"
            mol = Chem.MolFromSmiles(combined)
            if mol is None:
                invalid += 1
                continue

            mw = Descriptors.MolWt(mol)
            if mw < 100 or mw > 1200:
                continue

            candidates.append({
                "smiles": Chem.MolToSmiles(mol),
                "cation_name": cat_name,
                "cation_smiles": cat_smi,
                "anion_name": anion_name,
                "anion_smiles": anion_smi,
                "molecular_weight": mw,
                "source": "combinatorial",
            })

    logger.info(f"Raw candidates: {len(candidates):,} ({invalid} invalid)")

    df = pd.DataFrame(candidates)
    before = len(df)
    df = df.drop_duplicates(subset=["smiles"])
    logger.info(f"After dedup: {len(df):,} (removed {before - len(df):,})")

    return df


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Generate expanded combinatorial set
    combo_df = generate_all_candidates()

    # Also run SELFIES with more mutations
    logger.info("\nGenerating SELFIES candidates (expanded)...")
    from src.generation.selfies_generator import generate_selfies_candidates
    import random
    random.seed(42)
    np.random.seed(42)
    selfies_df = generate_selfies_candidates(n_per_seed=500, n_mutations=3)

    # Merge
    merged = pd.concat([combo_df, selfies_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["smiles"])

    # Filter novelty vs training data
    from pathlib import Path
    curated_dir = Path("data/curated")
    if (curated_dir / "train.csv").exists():
        train = pd.read_csv(curated_dir / "train.csv")
        training_smiles = set(train["smiles"].unique())
        before = len(merged)
        merged = merged[~merged["smiles"].isin(training_smiles)]
        logger.info(f"Novelty filter: {before:,} -> {len(merged):,}")

    # Save
    merged.to_parquet(GENERATED_DIR / "il_candidates_all.parquet", index=False)
    logger.info(f"\nSaved {len(merged):,} candidates to {GENERATED_DIR}/il_candidates_all.parquet")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total unique candidates: {len(merged):,}")
    if "source" in merged.columns:
        logger.info(f"By source:\n{merged['source'].value_counts().to_string()}")
    logger.info(f"Cation families: {merged['cation_name'].str.split('-').str[0].nunique()}")
    logger.info(f"Anion types: {merged['anion_name'].nunique()}")
    logger.info(f"MW range: {merged['molecular_weight'].min():.0f} - {merged['molecular_weight'].max():.0f}")


if __name__ == "__main__":
    main()
