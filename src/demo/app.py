"""
ALAi Discovery — Investor Demo
================================
Interactive Streamlit dashboard demonstrating the IL discovery pipeline.

Usage:
    streamlit run src/demo/app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

CURATED_DIR = Path("data/curated")
GENERATED_DIR = Path("data/generated")

st.set_page_config(
    page_title="ALAi Discovery Engine",
    page_icon="🧪",
    layout="wide",
)


@st.cache_data
def load_curated_data():
    if (CURATED_DIR / "co2_solubility_curated.parquet").exists():
        return pd.read_parquet(CURATED_DIR / "co2_solubility_curated.parquet")
    return None


@st.cache_data
def load_predictions(split_name):
    path = CURATED_DIR / f"{split_name}_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_ranked_candidates():
    path = GENERATED_DIR / "candidates_ranked.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def mol_to_svg(smiles, size=(300, 200)):
    """Convert SMILES to PNG image bytes."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# Sidebar
# ============================================================================
st.sidebar.title("ALAi Discovery Engine")
st.sidebar.markdown("*AI-native materials discovery*")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Performance", "Discovery Engine", "Top Candidates", "Physics Validation"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("Chemprop D-MPNN | RDKit | PyTorch")
st.sidebar.markdown("Data: NIST ILThermo")

# ============================================================================
# Page: Overview
# ============================================================================
if page == "Overview":
    st.title("ALAi — AI-Driven Ionic Liquid Discovery")
    st.markdown("### Accelerating CO₂ capture materials discovery by 100x")

    col1, col2, col3, col4, col5 = st.columns(5)

    curated = load_curated_data()
    ranked = load_ranked_candidates()

    with col1:
        n_data = len(curated) if curated is not None else 0
        st.metric("Curated Data Points", f"{n_data:,}")
    with col2:
        n_ils = curated["smiles"].nunique() if curated is not None else 0
        st.metric("Training ILs", n_ils)
    with col3:
        n_candidates = len(ranked) if ranked is not None else 0
        st.metric("Candidates Screened", f"{n_candidates:,}")
    with col4:
        st.metric("Ensemble Models", "8")
    with col5:
        # Load test R2 if available
        test_preds = load_predictions("test")
        if test_preds is not None:
            from sklearn.metrics import r2_score
            r2 = r2_score(test_preds["actual"], test_preds["predicted"])
            st.metric("Test R²", f"{r2:.3f}")
        else:
            st.metric("Test R²", "—")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        **End-to-End Pipeline:**
        1. **Data Curation** — 10,000+ data points from NIST ILThermo with physics-informed quality checks
        2. **D-MPNN Training** — 8-model ensemble of Directed Message Passing Neural Networks
        3. **Molecular Generation** — 98,000+ novel candidates via combinatorial enumeration + SELFIES mutations
        4. **Property Prediction** — CO₂ solubility with ensemble uncertainty quantification
        5. **Physics Validation** — Van't Hoff thermodynamic consistency checks
        6. **Screening** — Multi-objective ranking to identify top candidates
        """)

    with col_right:
        if ranked is not None and "source" in ranked.columns:
            source_counts = ranked["source"].value_counts()
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Candidate Sources",
                color_discrete_sequence=["#0d9488", "#6366f1"],
            )
            fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

    if curated is not None:
        st.markdown("### Training Data Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                curated, x="target", nbins=50,
                title="CO₂ Solubility Distribution (mole fraction)",
                labels={"target": "CO₂ Mole Fraction", "count": "Count"},
                color_discrete_sequence=["#0d9488"],
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                curated, x="temperature_K", y="pressure_bar",
                color="target", color_continuous_scale="Viridis",
                title="Data Coverage: Temperature vs Pressure",
                labels={"temperature_K": "Temperature (K)", "pressure_bar": "Pressure (bar)", "target": "x_CO₂"},
                opacity=0.5,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Model Performance
# ============================================================================
elif page == "Model Performance":
    st.title("D-MPNN Ensemble — Model Performance")
    st.markdown("8-model ensemble with scaffold-split validation (structurally novel test molecules)")

    for split in ["val", "test"]:
        preds = load_predictions(split)
        if preds is None:
            continue

        st.markdown(f"### {split.upper()} Set")

        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(preds["actual"], preds["predicted"])
        rmse = np.sqrt(mean_squared_error(preds["actual"], preds["predicted"]))
        mae = np.mean(np.abs(preds["actual"] - preds["predicted"]))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("MAE", f"{mae:.4f}")
        col4.metric("Data Points", len(preds))

        # Parity plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preds["actual"], y=preds["predicted"],
            mode="markers",
            marker=dict(
                size=6,
                color=preds["uncertainty"],
                colorscale="Viridis",
                colorbar=dict(title="Uncertainty"),
                opacity=0.7,
            ),
            text=[f"SMILES: {s}<br>Actual: {a:.3f}<br>Pred: {p:.3f}<br>Unc: {u:.3f}"
                  for s, a, p, u in zip(preds["smiles"], preds["actual"],
                                         preds["predicted"], preds["uncertainty"])],
            hoverinfo="text",
        ))

        min_val = min(preds["actual"].min(), preds["predicted"].min())
        max_val = max(preds["actual"].max(), preds["predicted"].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(dash="dash", color="red"),
            showlegend=False,
        ))

        fig.update_layout(
            title=f"Predicted vs Actual — log₁₀(CO₂ Solubility) [{split.upper()}]",
            xaxis_title="Actual log₁₀(x_CO₂)",
            yaxis_title="Predicted log₁₀(x_CO₂)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        errors = preds["predicted"] - preds["actual"]
        fig2 = px.histogram(
            x=errors, nbins=40,
            title=f"Prediction Error Distribution [{split.upper()}]",
            labels={"x": "Prediction Error (log₁₀)", "count": "Count"},
            color_discrete_sequence=["#6366f1"],
        )
        fig2.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# Page: Discovery Engine
# ============================================================================
elif page == "Discovery Engine":
    st.title("Discovery Engine — Screen Novel IL Candidates")

    ranked = load_ranked_candidates()
    if ranked is None:
        st.warning("No screening results found. Run `python -m src.screening.pareto_ranker` first.")
    else:
        st.markdown(f"**{len(ranked):,} candidates** screened and ranked by CO₂ capture performance")

        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            min_solubility = st.slider("Min CO₂ Solubility", 0.0, 0.5, 0.05, 0.01)
        with col2:
            max_uncertainty = st.slider("Max Uncertainty", 0.0, 1.0, 0.3, 0.05)
        with col3:
            source_filter = st.multiselect(
                "Source",
                options=ranked["source"].unique().tolist() if "source" in ranked.columns else ["all"],
                default=ranked["source"].unique().tolist() if "source" in ranked.columns else ["all"],
            )

        # Filter
        filtered = ranked[
            (ranked["co2_solubility_pred"] >= min_solubility) &
            (ranked["uncertainty"] <= max_uncertainty)
        ]
        if "source" in filtered.columns and source_filter:
            filtered = filtered[filtered["source"].isin(source_filter)]

        st.markdown(f"Showing **{len(filtered):,}** candidates matching filters")

        if len(filtered) > 0:
            filtered = filtered.copy()
            filtered["cation_type"] = filtered["cation_name"].str.split("-").str[0]

            fig = px.scatter(
                filtered.head(5000),
                x="co2_solubility_pred",
                y="uncertainty",
                color="cation_type",
                symbol="source" if "source" in filtered.columns else None,
                hover_data=["cation_name", "anion_name", "molecular_weight", "vs_mea"],
                title="Candidate Landscape: CO₂ Solubility vs Prediction Uncertainty",
                labels={
                    "co2_solubility_pred": "Predicted CO₂ Solubility (mole fraction)",
                    "uncertainty": "Ensemble Uncertainty (log₁₀)",
                    "cation_type": "Cation Family",
                },
                opacity=0.5,
            )
            fig.add_vline(x=0.20, line_dash="dash", line_color="red",
                         annotation_text="MEA Baseline", annotation_position="top")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Distribution by cation family
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.box(
                    filtered.head(5000), x="cation_type", y="co2_solubility_pred",
                    title="CO₂ Solubility by Cation Family",
                    labels={"cation_type": "Cation Family", "co2_solubility_pred": "x_CO₂"},
                    color="cation_type",
                )
                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                anion_means = filtered.groupby("anion_name")["co2_solubility_pred"].mean().sort_values(ascending=True).tail(15)
                fig3 = px.bar(
                    x=anion_means.values, y=anion_means.index,
                    orientation="h",
                    title="Mean CO₂ Solubility by Anion (Top 15)",
                    labels={"x": "Mean x_CO₂", "y": "Anion"},
                    color_discrete_sequence=["#0d9488"],
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)


# ============================================================================
# Page: Top Candidates
# ============================================================================
elif page == "Top Candidates":
    st.title("Top IL Candidates for CO₂ Capture")

    ranked = load_ranked_candidates()
    if ranked is None:
        st.warning("No screening results found.")
    else:
        top_n = st.selectbox("Show top N candidates", [10, 25, 50, 100], index=0)
        top = ranked.head(top_n)

        # Summary table
        display_cols = ["rank", "cation_name", "anion_name", "co2_solubility_pred",
                       "uncertainty", "vs_mea", "molecular_weight"]
        if "source" in top.columns:
            display_cols.append("source")
        if "physics_consistent" in top.columns:
            display_cols.append("physics_consistent")

        display_df = top[display_cols].copy()
        col_names = {"rank": "Rank", "cation_name": "Cation", "anion_name": "Anion",
                    "co2_solubility_pred": "x_CO₂", "uncertainty": "Unc.",
                    "vs_mea": "% vs MEA", "molecular_weight": "MW",
                    "source": "Source", "physics_consistent": "Physics OK"}
        display_df = display_df.rename(columns=col_names)
        if "x_CO₂" in display_df.columns:
            display_df["x_CO₂"] = display_df["x_CO₂"].round(4)
        if "Unc." in display_df.columns:
            display_df["Unc."] = display_df["Unc."].round(3)
        if "% vs MEA" in display_df.columns:
            display_df["% vs MEA"] = display_df["% vs MEA"].round(1)
        if "MW" in display_df.columns:
            display_df["MW"] = display_df["MW"].round(1)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Detailed view
        st.markdown("---")
        st.markdown("### Candidate Detail")

        selected_rank = st.selectbox(
            "Select a candidate to view",
            options=top["rank"].tolist(),
            format_func=lambda x: f"#{x}: {top[top['rank']==x].iloc[0]['cation_name']}-{top[top['rank']==x].iloc[0]['anion_name']}",
        )

        candidate = top[top["rank"] == selected_rank].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            img_bytes = mol_to_svg(candidate["smiles"], size=(400, 300))
            if img_bytes:
                st.image(img_bytes, caption="Molecular Structure")

        with col2:
            st.markdown(f"**SMILES:** `{candidate['smiles']}`")
            st.markdown(f"**Cation:** {candidate['cation_name']}")
            st.markdown(f"**Anion:** {candidate['anion_name']}")
            if "source" in candidate.index:
                st.markdown(f"**Source:** {candidate.get('source', 'unknown')}")

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("CO₂ Solubility", f"{candidate['co2_solubility_pred']:.4f}")
            mcol2.metric("Uncertainty", f"±{candidate['uncertainty']:.3f}")
            mcol3.metric("vs MEA", f"{candidate['vs_mea']:+.1f}%")

            mcol4, mcol5 = st.columns(2)
            mcol4.metric("Molecular Weight", f"{candidate['molecular_weight']:.1f} g/mol")
            mcol5.metric("Conditions", f"T={candidate['temperature_K']:.0f}K, P={candidate['pressure_bar']:.0f} bar")

            if "physics_consistent" in candidate.index and pd.notna(candidate.get("physics_consistent")):
                if candidate["physics_consistent"]:
                    st.success("Thermodynamically consistent (Van't Hoff validated)")
                else:
                    flags = candidate.get("physics_flags", "")
                    st.warning(f"Physics flags: {flags}")

            if "delta_H_kJ_mol" in candidate.index and pd.notna(candidate.get("delta_H_kJ_mol")):
                st.markdown(f"**Enthalpy of absorption:** {candidate['delta_H_kJ_mol']:.1f} kJ/mol")


# ============================================================================
# Page: Physics Validation
# ============================================================================
elif page == "Physics Validation":
    st.title("Van't Hoff Thermodynamic Validation")
    st.markdown("""
    For each candidate, we predict CO₂ solubility at 5 temperatures (10-80°C) and fit the Van't Hoff equation:

    **ln(x_CO₂) = -ΔH_abs / (R·T) + ΔS / R**

    A thermodynamically consistent candidate should have:
    - **Negative ΔH** (exothermic CO₂ absorption)
    - **Good fit quality** (R² > 0.7)
    - **Reasonable magnitude** (-10 to -80 kJ/mol)
    """)

    ranked = load_ranked_candidates()
    if ranked is None or "physics_consistent" not in ranked.columns:
        st.info("Physics validation data not yet available. Run screening with Van't Hoff validation enabled.")
    else:
        validated = ranked[ranked["physics_consistent"].notna()]
        if len(validated) == 0:
            st.info("No physics validation results found.")
        else:
            n_consistent = validated["physics_consistent"].sum()
            n_total = len(validated)

            col1, col2, col3 = st.columns(3)
            col1.metric("Validated Candidates", n_total)
            col2.metric("Thermodynamically Consistent", int(n_consistent))
            col3.metric("Consistency Rate", f"{n_consistent/n_total*100:.0f}%")

            if "delta_H_kJ_mol" in validated.columns:
                fig = px.histogram(
                    validated, x="delta_H_kJ_mol", nbins=40,
                    color="physics_consistent",
                    title="Distribution of Absorption Enthalpy (ΔH_abs)",
                    labels={"delta_H_kJ_mol": "ΔH_abs (kJ/mol)", "physics_consistent": "Physics OK"},
                    color_discrete_map={True: "#16a34a", False: "#dc2626"},
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            if "vant_hoff_r2" in validated.columns:
                fig2 = px.scatter(
                    validated, x="co2_solubility_pred", y="delta_H_kJ_mol",
                    color="physics_consistent",
                    hover_data=["cation_name", "anion_name", "vant_hoff_r2"],
                    title="CO₂ Solubility vs Absorption Enthalpy",
                    labels={
                        "co2_solubility_pred": "Predicted x_CO₂",
                        "delta_H_kJ_mol": "ΔH_abs (kJ/mol)",
                        "physics_consistent": "Physics OK",
                    },
                    color_discrete_map={True: "#16a34a", False: "#dc2626"},
                )
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
