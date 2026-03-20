"""
Van't Hoff Physics Validation
==============================
Validates CO2 solubility predictions using thermodynamic consistency checks.

For CO2 absorption:
  ln(K) = -ΔH_abs / (R * T) + ΔS / R

Where K is the solubility equilibrium constant (proportional to mole fraction).

Physics constraints:
1. CO2 absorption is exothermic → ΔH_abs should be negative
2. Solubility should decrease with increasing temperature (at constant P)
3. ΔH_abs magnitude should be in a reasonable range (-10 to -80 kJ/mol)

Usage:
    python -m src.physics.vant_hoff
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

R = 8.314e-3  # Gas constant in kJ/(mol·K)


def vant_hoff_fit(temperatures_K: np.ndarray, solubilities: np.ndarray) -> dict:
    """Fit Van't Hoff equation to temperature-dependent solubility data.

    ln(x_CO2) = -ΔH_abs / (R * T) + ΔS / R

    Returns dict with deltaH, deltaS, r_squared, and physics flags.
    """
    if len(temperatures_K) < 3:
        return {"valid": False, "reason": "insufficient_data"}

    inv_T = 1.0 / temperatures_K
    ln_x = np.log(solubilities)

    try:
        # Linear fit: ln(x) = slope * (1/T) + intercept
        # slope = -ΔH / R, intercept = ΔS / R
        coeffs = np.polyfit(inv_T, ln_x, 1)
        slope, intercept = coeffs

        delta_H = -slope * R  # kJ/mol
        delta_S = intercept * R  # kJ/(mol·K)

        # R-squared
        ln_x_pred = np.polyval(coeffs, inv_T)
        ss_res = np.sum((ln_x - ln_x_pred) ** 2)
        ss_tot = np.sum((ln_x - np.mean(ln_x)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Physics checks
        flags = []

        # CO2 absorption should be exothermic (ΔH < 0)
        if delta_H > 0:
            flags.append("endothermic")

        # Magnitude should be reasonable (-10 to -80 kJ/mol for physical absorption)
        if delta_H < -100:
            flags.append("deltaH_too_large")
        elif -10 < delta_H < 0:
            flags.append("deltaH_weak")  # Very weak absorption

        # Fit quality
        if r_squared < 0.7:
            flags.append("poor_fit")

        return {
            "valid": True,
            "delta_H_kJ_mol": delta_H,
            "delta_S_kJ_mol_K": delta_S,
            "r_squared": r_squared,
            "n_points": len(temperatures_K),
            "physics_flags": flags,
            "physics_consistent": len(flags) == 0,
        }

    except Exception as e:
        return {"valid": False, "reason": str(e)}


def validate_candidates_physics(
    candidates_df: pd.DataFrame,
    models,
    temperatures: list[float] = None,
    pressure_bar: float = 10.0,
) -> pd.DataFrame:
    """Run Van't Hoff validation on candidate predictions at multiple temperatures.

    Predicts solubility at 3+ temperatures, fits Van't Hoff equation,
    and flags thermodynamically inconsistent candidates.
    """
    from src.models.train_ensemble import predict_ensemble, _smiles_to_mol

    if temperatures is None:
        temperatures = [283.15, 298.15, 313.15, 333.15, 353.15]  # 10°C to 80°C

    logger.info(f"Running Van't Hoff validation at {len(temperatures)} temperatures...")

    smiles_list = candidates_df["smiles"].tolist()
    mols, valid_idx = _smiles_to_mol(smiles_list)

    # Predict at each temperature
    predictions_by_temp = {}
    for temp in temperatures:
        features = np.array([[temp, pressure_bar]] * len(mols), dtype=np.float32)
        means, stds = predict_ensemble(models, mols, features)
        # Convert from log10 to linear
        predictions_by_temp[temp] = 10 ** means

    # For each candidate, fit Van't Hoff
    results = []
    for i, idx in enumerate(valid_idx):
        temps_arr = np.array(temperatures)
        sols_arr = np.array([predictions_by_temp[t][i] for t in temperatures])

        # Remove any non-positive predictions
        mask = sols_arr > 0
        if mask.sum() < 3:
            results.append({
                "idx": idx,
                "physics_consistent": False,
                "delta_H_kJ_mol": np.nan,
                "vant_hoff_r2": np.nan,
                "physics_flags": "insufficient_valid_predictions",
            })
            continue

        fit = vant_hoff_fit(temps_arr[mask], sols_arr[mask])

        if fit["valid"]:
            results.append({
                "idx": idx,
                "physics_consistent": fit["physics_consistent"],
                "delta_H_kJ_mol": fit["delta_H_kJ_mol"],
                "delta_S_kJ_mol_K": fit["delta_S_kJ_mol_K"],
                "vant_hoff_r2": fit["r_squared"],
                "physics_flags": ",".join(fit["physics_flags"]) if fit["physics_flags"] else "",
            })
        else:
            results.append({
                "idx": idx,
                "physics_consistent": False,
                "delta_H_kJ_mol": np.nan,
                "vant_hoff_r2": np.nan,
                "physics_flags": fit.get("reason", "fit_failed"),
            })

    results_df = pd.DataFrame(results)

    # Merge with candidates
    out = candidates_df.iloc[valid_idx].copy().reset_index(drop=True)
    results_df = results_df.reset_index(drop=True)
    for col in ["physics_consistent", "delta_H_kJ_mol", "delta_S_kJ_mol_K", "vant_hoff_r2", "physics_flags"]:
        if col in results_df.columns:
            out[col] = results_df[col].values

    n_consistent = out["physics_consistent"].sum()
    logger.info(f"Physics validation: {n_consistent}/{len(out)} candidates are thermodynamically consistent")
    logger.info(f"Mean ΔH_abs: {out['delta_H_kJ_mol'].mean():.1f} kJ/mol")

    return out


if __name__ == "__main__":
    # Standalone test with existing predictions
    logger.info("Van't Hoff module ready. Use validate_candidates_physics() with trained models.")
