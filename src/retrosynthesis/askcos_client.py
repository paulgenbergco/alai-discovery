"""
ASKCOS Retrosynthesis Client
==============================
Queries MIT's ASKCOS platform for retrosynthetic routes to synthesize
target ionic liquids. Supports both live API and pre-computed routes.

Usage:
    python -m src.retrosynthesis.askcos_client

Requires:
    - ASKCOS account at https://askcos.mit.edu
    - Set ASKCOS_TOKEN environment variable
"""

import json
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ASKCOS_BASE_URL = os.environ.get("ASKCOS_URL", "https://askcos.mit.edu")
ASKCOS_TOKEN = os.environ.get("ASKCOS_TOKEN", "")
GENERATED_DIR = Path("data/generated")


def get_retro_routes(smiles: str, max_depth: int = 3, expansion_time: int = 60) -> dict:
    """Query ASKCOS tree builder for retrosynthetic routes.

    Args:
        smiles: Target molecule SMILES
        max_depth: Maximum retrosynthetic depth
        expansion_time: Time limit in seconds

    Returns:
        dict with routes, or error info
    """
    if not ASKCOS_TOKEN:
        logger.warning("ASKCOS_TOKEN not set. Use pre-computed routes or set token.")
        return {"error": "No ASKCOS token configured", "smiles": smiles}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ASKCOS_TOKEN}",
    }

    # Submit tree builder job
    payload = {
        "smiles": smiles,
        "max_depth": max_depth,
        "max_branching": 25,
        "expansion_time": expansion_time,
        "max_cum_prob": 0.999,
        "num_templates": 100,
    }

    try:
        resp = requests.post(
            f"{ASKCOS_BASE_URL}/api/tree-builder/",
            json=payload, headers=headers, timeout=10
        )
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        logger.info(f"Submitted ASKCOS job: {task_id}")

        # Poll for results
        for _ in range(30):  # Max 5 min wait
            time.sleep(10)
            result = requests.get(
                f"{ASKCOS_BASE_URL}/api/tree-builder/result/{task_id}/",
                headers=headers, timeout=10
            )
            data = result.json()
            if data.get("state") == "completed":
                return {
                    "smiles": smiles,
                    "routes": data.get("result", {}).get("trees", []),
                    "num_routes": len(data.get("result", {}).get("trees", [])),
                    "source": "askcos_live",
                }
            elif data.get("state") == "failed":
                return {"error": "ASKCOS job failed", "smiles": smiles}

        return {"error": "ASKCOS job timed out", "smiles": smiles}

    except requests.RequestException as e:
        logger.error(f"ASKCOS API error: {e}")
        return {"error": str(e), "smiles": smiles}


def generate_example_routes():
    """Generate pre-computed retrosynthesis data for the demo.

    For ionic liquids, retrosynthesis is relatively straightforward:
    most ILs are made via quaternization (cation) + anion exchange.
    We encode this domain knowledge as structured routes.
    """
    import pandas as pd
    from rdkit import Chem

    ranked = pd.read_parquet(GENERATED_DIR / "candidates_ranked.parquet")
    top = ranked.head(20)

    routes = []
    for _, row in top.iterrows():
        cation = row["cation_name"]
        anion = row["anion_name"]
        smiles = row["smiles"]

        # Parse cation type for synthesis route
        route = build_il_synthesis_route(cation, anion, smiles, row)
        routes.append(route)

    # Save
    output = GENERATED_DIR / "retrosynthesis_routes.json"
    with open(output, "w") as f:
        json.dump(routes, f, indent=2)

    logger.info(f"Generated {len(routes)} retrosynthesis routes → {output}")
    return routes


def build_il_synthesis_route(cation_name: str, anion_name: str, smiles: str, row: dict) -> dict:
    """Build a retrosynthetic route for an ionic liquid.

    Most ILs follow a standard 2-3 step synthesis:
    1. Quaternization: amine/phosphine + alkyl halide → cation halide salt
    2. Anion exchange: cation halide + metal/acid salt → target IL

    Some amino acid ILs require additional steps.
    """
    # Determine cation family
    cation_family = cation_name.split("-")[0] if "-" in cation_name else cation_name

    # Common synthesis patterns by cation family
    synthesis_patterns = {
        "IM": {  # Imidazolium
            "step1": "N-alkylation of 1-methylimidazole with alkyl halide",
            "reagents1": ["1-Methylimidazole", "Alkyl halide (R-X)"],
            "conditions1": "Reflux in acetonitrile, 24h, N₂ atmosphere",
            "intermediate": f"[{cation_name}][Halide]",
            "step2": f"Anion metathesis with {anion_name} salt",
            "reagents2": [f"Metal {anion_name} salt", "Dichloromethane/water"],
            "conditions2": "Room temperature, stirring 24h, wash with water",
            "yield_est": "70-85%",
            "difficulty": "Easy",
            "cost_indicator": "$",
        },
        "PH": {  # Phosphonium
            "step1": "Quaternization of trialkylphosphine with alkyl halide",
            "reagents1": ["Trialkylphosphine (R₃P)", "Alkyl halide (R'-X)"],
            "conditions1": "Heating at 100-150°C, 48h, sealed tube",
            "intermediate": f"[{cation_name}][Halide]",
            "step2": f"Anion exchange with {anion_name}",
            "reagents2": [f"Li/Na/K {anion_name}", "Methanol/water"],
            "conditions2": "Room temperature, stirring 12h",
            "yield_est": "60-75%",
            "difficulty": "Moderate",
            "cost_indicator": "$$",
        },
        "PYR": {  # Pyrrolidinium
            "step1": "N-alkylation of N-methylpyrrolidine",
            "reagents1": ["N-Methylpyrrolidine", "Alkyl halide (R-X)"],
            "conditions1": "Reflux in ethyl acetate, 48h",
            "intermediate": f"[{cation_name}][Halide]",
            "step2": f"Anion metathesis to {anion_name}",
            "reagents2": [f"{anion_name} acid or salt", "Water/DCM biphasic"],
            "conditions2": "Room temperature, 24h",
            "yield_est": "65-80%",
            "difficulty": "Easy",
            "cost_indicator": "$",
        },
        "AM": {  # Ammonium
            "step1": "Quaternization of trialkylamine with alkyl halide",
            "reagents1": ["Trialkylamine (R₃N)", "Alkyl halide (R'-X)"],
            "conditions1": "Reflux in acetonitrile, 24-48h",
            "intermediate": f"[{cation_name}][Halide]",
            "step2": f"Anion exchange with {anion_name}",
            "reagents2": [f"{anion_name} salt", "Water"],
            "conditions2": "Room temperature, stirring 24h",
            "yield_est": "70-85%",
            "difficulty": "Easy",
            "cost_indicator": "$",
        },
        "P4444": {  # Specific phosphonium
            "step1": "Quaternization of tributylphosphine with butyl bromide",
            "reagents1": ["Tributylphosphine", "1-Bromobutane"],
            "conditions1": "120°C, 48h, N₂",
            "intermediate": f"[P₄₄₄₄][Br]",
            "step2": f"Anion exchange with {anion_name}",
            "reagents2": [f"Li{anion_name} or H{anion_name}", "Methanol"],
            "conditions2": "Room temperature, 24h",
            "yield_est": "65-78%",
            "difficulty": "Moderate",
            "cost_indicator": "$$",
        },
        "N4444": {  # Specific ammonium
            "step1": "Quaternization of tributylamine with butyl bromide",
            "reagents1": ["Tributylamine", "1-Bromobutane"],
            "conditions1": "Reflux in acetonitrile, 48h",
            "intermediate": f"[N₄₄₄₄][Br]",
            "step2": f"Anion exchange with {anion_name}",
            "reagents2": [f"Na/K {anion_name}", "Water/DCM"],
            "conditions2": "Room temperature, 24h",
            "yield_est": "70-82%",
            "difficulty": "Easy",
            "cost_indicator": "$",
        },
    }

    # Match pattern (try exact, then family prefix)
    pattern = None
    for key in [cation_name.split("-")[0], cation_family]:
        if key in synthesis_patterns:
            pattern = synthesis_patterns[key]
            break

    # Default pattern for unknown cation families (including SELFIES mutations)
    if pattern is None:
        pattern = {
            "step1": "Quaternization reaction (alkylation of base heterocycle)",
            "reagents1": ["Base heterocycle", "Alkyl halide"],
            "conditions1": "Reflux, 24-48h, inert atmosphere",
            "intermediate": f"[{cation_name}][Halide]",
            "step2": f"Anion metathesis with {anion_name}",
            "reagents2": [f"{anion_name} salt/acid", "Biphasic extraction"],
            "conditions2": "Room temperature, 24h",
            "yield_est": "50-70%",
            "difficulty": "Moderate-Hard (novel structure)",
            "cost_indicator": "$$$",
        }

    # Anion-specific notes
    anion_notes = {
        "OAc": "Acetate ILs: use silver acetate or ion exchange resin. Hygroscopic — handle under N₂.",
        "NTf2": "Bis(trifluoromethylsulfonyl)imide: use LiNTf₂. Hydrophobic IL — easy purification.",
        "BF4": "Tetrafluoroborate: use NaBF₄. May hydrolyze — avoid prolonged water contact.",
        "PF6": "Hexafluorophosphate: use KPF₆. Hydrophobic — simple extraction.",
        "DCA": "Dicyanamide: use NaDCA. Relatively low viscosity ILs.",
        "SCN": "Thiocyanate: use KSCN. Simple anion exchange.",
        "B(CN)4": "Tetracyanoborate: use K[B(CN)₄]. Specialty reagent — higher cost.",
        "TCM": "Tricyanomethanide: use Na[C(CN)₃]. Good thermal stability.",
        "Lactate": "Lactate: neutralization of cation hydroxide with lactic acid.",
        "Prolinate": "Prolinate: neutralization with L-proline. Chiral IL.",
        "Glycinate": "Glycinate: neutralization with glycine. Amino acid IL — enhanced CO₂ capture via carbamate.",
        "Alaninate": "Alaninate: neutralization with alanine. Amino acid IL.",
    }

    return {
        "smiles": smiles,
        "cation_name": cation_name,
        "anion_name": anion_name,
        "rank": int(row.get("rank", 0)),
        "co2_solubility": round(float(row.get("co2_solubility_pred", 0)), 4),
        "synthesis": {
            "n_steps": 2,
            "overall_yield": pattern["yield_est"],
            "difficulty": pattern["difficulty"],
            "cost_indicator": pattern["cost_indicator"],
            "steps": [
                {
                    "step": 1,
                    "name": "Cation Formation",
                    "description": pattern["step1"],
                    "reagents": pattern["reagents1"],
                    "conditions": pattern["conditions1"],
                    "product": pattern["intermediate"],
                },
                {
                    "step": 2,
                    "name": "Anion Exchange",
                    "description": pattern["step2"],
                    "reagents": pattern["reagents2"],
                    "conditions": pattern["conditions2"],
                    "product": f"[{cation_name}][{anion_name}]",
                },
            ],
            "notes": anion_notes.get(anion_name, f"Standard anion exchange with {anion_name} salt."),
            "purification": "Wash with water (3×), dry under vacuum at 60°C for 24h. Verify purity by ¹H NMR and Karl Fischer titration (water < 100 ppm).",
        },
        "source": "domain_knowledge",
    }


if __name__ == "__main__":
    routes = generate_example_routes()
    print(f"\nGenerated routes for {len(routes)} candidates")
    for r in routes[:5]:
        syn = r["synthesis"]
        print(f"  #{r['rank']}: {r['cation_name']}-{r['anion_name']} | "
              f"{syn['n_steps']} steps | yield: {syn['overall_yield']} | "
              f"difficulty: {syn['difficulty']}")
