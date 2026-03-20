"""
IL Name-to-SMILES Mapper
========================
Maps ionic liquid chemical names from ILThermo to SMILES representations.

Strategy:
1. Manual lookup table for the most common IL cation/anion fragments
2. PubChem PUG REST API fallback for unmapped names
3. RDKit validation of all generated SMILES

Usage:
    python -m src.curation.smiles_mapper
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# ============================================================================
# Manual SMILES lookup for common IL cations
# ============================================================================
CATION_SMILES = {
    # Imidazolium-based (1-R-3-methylimidazolium series)
    "1-ethyl-3-methylimidazolium": "CCn1cc[n+](C)c1",
    "1-ethyl-3-methyl-1h-imidazolium": "CCn1cc[n+](C)c1",
    "1-propyl-3-methylimidazolium": "CCCn1cc[n+](C)c1",
    "1-methyl-3-propylimidazolium": "CCCn1cc[n+](C)c1",
    "1-butyl-3-methylimidazolium": "CCCCn1cc[n+](C)c1",
    "1-methyl-3-pentylimidazolium": "CCCCCn1cc[n+](C)c1",
    "1-pentyl-3-methylimidazolium": "CCCCCn1cc[n+](C)c1",
    "1-hexyl-3-methylimidazolium": "CCCCCCn1cc[n+](C)c1",
    "1-heptyl-3-methylimidazolium": "CCCCCCCn1cc[n+](C)c1",
    "1-methyl-3-octylimidazolium": "CCCCCCCCn1cc[n+](C)c1",
    "1-octyl-3-methylimidazolium": "CCCCCCCCn1cc[n+](C)c1",
    "1-methyl-3-nonylimidazolium": "CCCCCCCCCn1cc[n+](C)c1",
    "1-decyl-3-methylimidazolium": "CCCCCCCCCCn1cc[n+](C)c1",
    "1-dodecyl-3-methylimidazolium": "CCCCCCCCCCCCn1cc[n+](C)c1",
    # 2,3-dimethyl variants
    "1-butyl-2,3-dimethylimidazolium": "CCCCn1cc[n+](C)c1C",
    "1-ethyl-2,3-dimethylimidazolium": "CCn1cc[n+](C)c1C",
    "2,3-dimethyl-1-octylimidazolium": "CCCCCCCCn1cc[n+](C)c1C",
    "2,3-dimethyl-1-octyl-1h-imidazolium": "CCCCCCCCn1cc[n+](C)c1C",
    "1,2-dimethyl-3-propylimidazolium": "CCCn1cc[n+](C)c1C",
    # Other imidazolium
    "1,3-dimethylimidazolium": "Cn1cc[n+](C)c1",
    "1-allyl-3-methylimidazolium": "C=CCn1cc[n+](C)c1",
    "1-(2-hydroxyethyl)-3-methylimidazolium": "OCCn1cc[n+](C)c1",
    "1-benzyl-3-methylimidazolium": "c1ccc(Cn2cc[n+](C)c2)cc1",
    "1-methylimidazolium": "Cn1cc[nH+]c1",
    "1-butyl-3-ethylimidazolium": "CCCCn1cc[n+](CC)c1",
    "1,3-diethylimidazolium": "CCn1cc[n+](CC)c1",
    "1,3-dibutylimidazolium": "CCCCn1cc[n+](CCCC)c1",
    "1,3-dibutyl-1h-imidazolium": "CCCCn1cc[n+](CCCC)c1",
    "1-(2-methoxyethyl)-3-methylimidazolium": "COCCn1cc[n+](C)c1",

    # Pyridinium-based
    "1-butylpyridinium": "CCCC[n+]1ccccc1",
    "1-butyl-3-methylpyridinium": "CCCC[n+]1cccc(C)c1",
    "1-butyl-4-methylpyridinium": "CCCC[n+]1ccc(C)cc1",
    "1-ethylpyridinium": "CC[n+]1ccccc1",
    "1-hexyl-3-methylpyridinium": "CCCCCC[n+]1cccc(C)c1",
    "n-butylpyridinium": "CCCC[n+]1ccccc1",
    "n-benzylpyridinium": "c1ccc(C[n+]2ccccc2)cc1",

    # Pyrrolidinium-based
    "1-butyl-1-methylpyrrolidinium": "CCCC[N+]1(C)CCCC1",
    "1-methyl-1-propylpyrrolidinium": "CCC[N+]1(C)CCCC1",
    "1-ethyl-1-methylpyrrolidinium": "CC[N+]1(C)CCCC1",
    "1-methyl-1-butylpyrrolidinium": "CCCC[N+]1(C)CCCC1",
    "1-hexyl-1-methylpyrrolidinium": "CCCCCC[N+]1(C)CCCC1",
    "1-methyl-1-pentylpyrrolidinium": "CCCCC[N+]1(C)CCCC1",
    "1-pentyl-1-methylpyrrolidinium": "CCCCC[N+]1(C)CCCC1",
    "1-heptyl-1-methylpyrrolidinium": "CCCCCCC[N+]1(C)CCCC1",
    "1-heptyl-1-methylpyrrolidin-1-ium": "CCCCCCC[N+]1(C)CCCC1",
    "1-methyl-1-octylpyrrolidinium": "CCCCCCCC[N+]1(C)CCCC1",
    "1-methyl-1-nonylpyrrolidinium": "CCCCCCCCC[N+]1(C)CCCC1",

    # Ammonium-based
    "tetrabutylammonium": "CCCC[N+](CCCC)(CCCC)CCCC",
    "tetraethylammonium": "CC[N+](CC)(CC)CC",
    "tetramethylammonium": "C[N+](C)(C)C",
    "trimethylbutylammonium": "CCCC[N+](C)(C)C",
    "butyltrimethylammonium": "CCCC[N+](C)(C)C",
    "tributylmethylammonium": "CCCC[N+](C)(CCCC)CCCC",
    "methyltrioctylammonium": "CCCCCCCC[N+](C)(CCCCCCCC)CCCCCCCC",
    "triethylbutylammonium": "CCCC[N+](CC)(CC)CC",
    "butyltriethylammonium": "CCCC[N+](CC)(CC)CC",
    "choline": "OCC[N+](C)(C)C",
    "cholinium": "OCC[N+](C)(C)C",
    "2-hydroxy-n,n,n-trimethylethanaminium": "OCC[N+](C)(C)C",
    "n,n,n-trimethyl-n-propylammonium": "CCC[N+](C)(C)C",

    # Phosphonium-based
    "trihexyl(tetradecyl)phosphonium": "CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC",
    "trihexyltetradecylphosphonium": "CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC",
    "tetradecyl(trihexyl)phosphonium": "CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC",
    "tetradecyltrihexylphosphonium": "CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC",
    "tetrabutylphosphonium": "CCCC[P+](CCCC)(CCCC)CCCC",
    "tributyl(tetradecyl)phosphonium": "CCCCCCCCCCCCCC[P+](CCCC)(CCCC)CCCC",
    "tributylmethylphosphonium": "CCCC[P+](C)(CCCC)CCCC",
    "triethyl(octyl)phosphonium": "CCCCCCCC[P+](CC)(CC)CC",
    "triethyloctylphosphonium": "CCCCCCCC[P+](CC)(CC)CC",
    "triethyl(pentyl)phosphonium": "CCCCC[P+](CC)(CC)CC",
    "triethyl(2-methoxyethyl)phosphonium": "COCC[P+](CC)(CC)CC",

    # Sulfonium-based
    "triethylsulfonium": "CC[S+](CC)CC",

    # Guanidinium-based
    "1,1,3,3-tetramethylguanidinium": "CN(C)C(=[NH2+])N(C)C",
}

# ============================================================================
# Manual SMILES lookup for common IL anions
# ============================================================================
ANION_SMILES = {
    # NTf2 variants (most common IL anion, many naming conventions)
    "bis[(trifluoromethyl)sulfonyl]imide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "bis(trifluoromethylsulfonyl)imide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "bis((trifluoromethyl)sulfonyl)imide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "bis(trifluoromethylsulfonyl)amide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "bis((trifluoromethyl)sulfonyl)amide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "bis(trifluoromethanesulfonyl)imide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "bis(trifluoromethanesulfonyl)amide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "1,1,1-trifluoro-n-[(trifluoromethyl)sulfonyl]methanesulfonamide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "1,1,1-trifluoro-n-(trifluoromethylsulfonyl)methanesulfonamide": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",

    # BETI (bis(perfluoroethylsulfonyl)imide)
    "bis(perfluoroethylsulfonyl)imide": "[N-](S(=O)(=O)C(F)(F)C(F)(F)F)S(=O)(=O)C(F)(F)C(F)(F)F",
    "bis((perfluoroethyl)sulfonyl)amide": "[N-](S(=O)(=O)C(F)(F)C(F)(F)F)S(=O)(=O)C(F)(F)C(F)(F)F",
    "bis((perfluoroethyl)sulfonyl)imide": "[N-](S(=O)(=O)C(F)(F)C(F)(F)F)S(=O)(=O)C(F)(F)C(F)(F)F",

    # Other fluorinated
    "bis(fluorosulfonyl)imide": "[N-](S(=O)(=O)F)S(=O)(=O)F",
    "trifluoromethanesulfonate": "[O-]S(=O)(=O)C(F)(F)F",
    "triflate": "[O-]S(=O)(=O)C(F)(F)F",
    "hexafluorophosphate": "F[P-](F)(F)(F)(F)F",
    "tetrafluoroborate": "F[B-](F)(F)F",
    "trifluoroacetate": "[O-]C(=O)C(F)(F)F",
    "1,1,2,2-tetrafluoroethanesulfonate": "[O-]S(=O)(=O)C(F)(F)C(F)(F)F",
    "1,1,2,2-tetrafluoroethane-1-sulfonate": "[O-]S(=O)(=O)C(F)(F)C(F)(F)F",
    "perfluorohexylsulfonate": "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "nonafluorobutanesulfonate": "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "tris(pentafluoroethyl)trifluorophosphate": "FC(F)(F)C(F)(F)[P-](F)(F)(F)C(F)(F)C(F)(F)F",

    # Cyano-based
    "dicyanamide": "N#C[N-]C#N",
    "tricyanomethane": "[C-](C#N)(C#N)C#N",
    "tricyanomethanide": "[C-](C#N)(C#N)C#N",
    "tetracyanoborate": "[B-](C#N)(C#N)(C#N)C#N",
    "thiocyanate": "[S-]C#N",

    # Halides
    "chloride": "[Cl-]",
    "bromide": "[Br-]",
    "iodide": "[I-]",
    "fluoride": "[F-]",

    # Carboxylates
    "acetate": "CC([O-])=O",
    "formate": "[O-]C=O",
    "propanoate": "CCC([O-])=O",
    "butanoate": "CCCC([O-])=O",
    "benzoate": "[O-]C(=O)c1ccccc1",
    "lactate": "CC(O)C([O-])=O",
    "glycinate": "[NH2]CC([O-])=O",
    "prolinate": "[O-]C(=O)[C@@H]1CCCN1",
    "l-prolinate": "[O-]C(=O)[C@@H]1CCCN1",
    "l-tyrosinate": "[O-]C(=O)[C@@H](N)Cc1ccc(O)cc1",
    "l-glutamate": "[O-]C(=O)[C@@H](N)CCC([O-])=O",
    "l-glutaminate": "[O-]C(=O)[C@@H](N)CCC(N)=O",
    "dodecyl-benzenesulfonate": "CCCCCCCCCCCCS(=O)(=O)c1ccccc1",

    # Phosphorus
    "diethyl phosphate": "CCOP([O-])(=O)OCC",
    "dimethyl phosphate": "COP([O-])(=O)OC",
    "dihydrogen phosphate": "[O-]P(O)(O)=O",
    "methyl phosphonate": "CP([O-])(=O)O",
    "bis(2,4,4-trimethylpentyl)phosphinate": "CC(CC(C)(C)C)[P-](=O)CC(C)CC(C)(C)C",

    # Sulfates/sulfonates
    "ethyl sulfate": "CCOS([O-])(=O)=O",
    "methyl sulfate": "COS([O-])(=O)=O",
    "hydrogen sulfate": "[O-]S(O)(=O)=O",
    "methylsulfonate": "CS([O-])(=O)=O",
    "methanesulfonate": "CS([O-])(=O)=O",
    "p-toluenesulfonate": "Cc1ccc(S([O-])(=O)=O)cc1",
    "tosylate": "Cc1ccc(S([O-])(=O)=O)cc1",

    # Nitrate
    "nitrate": "[O-][N+]([O-])=O",

    # Other
    "saccharinate": "[O-]c1nsc2ccccc12",
    "methylcarbonate": "COC([O-])=O",
}


def _normalize_name(name: str) -> str:
    """Normalize IL name for matching."""
    name = name.strip().lower()
    # Remove "-1H-" form (e.g., "1-ethyl-3-methyl-1H-imidazolium" → "1-ethyl-3-methylimidazolium")
    name = re.sub(r"-1h-imidazolium", "imidazolium", name)
    # Remove "pyrrolidin-1-ium" → "pyrrolidinium"
    name = name.replace("pyrrolidin-1-ium", "pyrrolidinium")
    # Common abbreviation substitutions
    name = name.replace("ntf2", "bis(trifluoromethylsulfonyl)imide")
    name = name.replace("[tf2n]", "bis(trifluoromethylsulfonyl)imide")
    name = name.replace("[bmim]", "1-butyl-3-methylimidazolium")
    name = name.replace("[emim]", "1-ethyl-3-methylimidazolium")
    name = name.replace("[hmim]", "1-hexyl-3-methylimidazolium")
    name = name.replace("[omim]", "1-methyl-3-octylimidazolium")
    return name


def _try_manual_lookup(il_name: str) -> str | None:
    """Try to construct SMILES from manual cation+anion lookup."""
    name = _normalize_name(il_name)

    # Try to match cation and anion
    best_cation = None
    best_cation_len = 0
    for cation_name, cation_smi in CATION_SMILES.items():
        cn = cation_name.lower()
        if cn in name and len(cn) > best_cation_len:
            best_cation = cation_smi
            best_cation_len = len(cn)

    best_anion = None
    best_anion_len = 0
    for anion_name, anion_smi in ANION_SMILES.items():
        an = anion_name.lower()
        if an in name and len(an) > best_anion_len:
            best_anion = anion_smi
            best_anion_len = len(an)

    if best_cation and best_anion:
        combined = f"{best_cation}.{best_anion}"
        mol = Chem.MolFromSmiles(combined)
        if mol is not None:
            return Chem.MolToSmiles(mol)

    return None


def _try_pubchem(il_name: str, retries: int = 2) -> str | None:
    """Try to get SMILES from PubChem by chemical name."""
    for attempt in range(retries):
        try:
            r = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(il_name)}/property/CanonicalSMILES/JSON",
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return smiles
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def map_names_to_smiles(df: pd.DataFrame, pubchem_delay: float = 0.3) -> pd.DataFrame:
    """Add SMILES column to dataframe of IL records.

    Args:
        df: DataFrame with 'il_name' column
        pubchem_delay: Delay between PubChem API calls

    Returns:
        DataFrame with added 'smiles' column
    """
    unique_names = df["il_name"].unique()
    logger.info(f"Mapping {len(unique_names)} unique IL names to SMILES...")

    smiles_map = {}
    manual_hits = 0
    pubchem_hits = 0
    misses = 0

    for name in tqdm(unique_names, desc="Mapping SMILES"):
        # Try manual lookup first
        smi = _try_manual_lookup(name)
        if smi:
            smiles_map[name] = smi
            manual_hits += 1
            continue

        # Try PubChem
        smi = _try_pubchem(name)
        if smi:
            smiles_map[name] = smi
            pubchem_hits += 1
            time.sleep(pubchem_delay)
            continue

        smiles_map[name] = None
        misses += 1
        time.sleep(pubchem_delay)

    logger.info(f"Mapping results: {manual_hits} manual, {pubchem_hits} PubChem, {misses} unmapped")
    logger.info(f"Coverage: {(manual_hits + pubchem_hits) / len(unique_names) * 100:.1f}%")

    # Save the mapping for reuse
    mapping_df = pd.DataFrame(
        [(name, smi) for name, smi in smiles_map.items()],
        columns=["il_name", "smiles"],
    )
    mapping_path = RAW_DIR / "il_smiles_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"Saved SMILES mapping to {mapping_path}")

    # Log unmapped names
    unmapped = [name for name, smi in smiles_map.items() if smi is None]
    if unmapped:
        logger.warning(f"Unmapped IL names ({len(unmapped)}):")
        for name in unmapped[:20]:
            logger.warning(f"  - {name}")
        if len(unmapped) > 20:
            logger.warning(f"  ... and {len(unmapped) - 20} more")

    # Apply mapping to dataframe
    df = df.copy()
    df["smiles"] = df["il_name"].map(smiles_map)
    return df


if __name__ == "__main__":
    # Load raw data
    raw_path = RAW_DIR / "ilthermo_co2_solubility_raw.parquet"
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run ilthermo_collector.py first.")
        raise SystemExit(1)

    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df)} records for {df['il_name'].nunique()} unique ILs")

    df = map_names_to_smiles(df)

    # Save with SMILES
    output_path = RAW_DIR / "ilthermo_co2_with_smiles.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved data with SMILES to {output_path}")

    # Stats
    mapped = df["smiles"].notna().sum()
    total = len(df)
    logger.info(f"Records with SMILES: {mapped}/{total} ({mapped/total*100:.1f}%)")
