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
    """Convert SMILES to SVG image."""
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
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Performance", "Discovery Engine", "Top Candidates"],
)

# ============================================================================
# Page: Overview
# ============================================================================
if page == "Overview":
    st.title("ALAi — AI-Driven Ionic Liquid Discovery")
    st.markdown("### Accelerating CO₂ capture materials discovery by 100x")

    col1, col2, col3, col4 = st.columns(4)

    curated = load_curated_data()
    ranked = load_ranked_candidates()

    with col1:
        n_data = len(curated) if curated is not None else 0
        st.metric("Curated Data Points", f"{n_data:,}")
    with col2:
        n_ils = curated["smiles"].nunique() if curated is not None else 0
        st.metric("Unique ILs in Training", n_ils)
    with col3:
        n_candidates = len(ranked) if ranked is not None else 0
        st.metric("Candidates Screened", f"{n_candidates:,}")
    with col4:
        st.metric("Ensemble Models", "8")

    st.markdown("---")
    st.markdown("""
    **Pipeline:**
    1. **Data Curation** — 10,000+ data points from NIST ILThermo, with physics-informed quality checks
    2. **D-MPNN Training** — 8-model ensemble of Directed Message Passing Neural Networks
    3. **Molecular Generation** — 80,000+ novel ionic liquid candidates via combinatorial enumeration
    4. **Property Prediction** — CO₂ solubility prediction with uncertainty quantification
    5. **Screening** — Multi-objective ranking to identify top candidates
    """)

    if curated is not None:
        st.markdown("### Training Data Distribution")
        fig = px.histogram(
            curated, x="target", nbins=50,
            title="CO₂ Solubility Distribution (mole fraction)",
            labels={"target": "CO₂ Mole Fraction", "count": "Count"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Model Performance
# ============================================================================
elif page == "Model Performance":
    st.title("D-MPNN Ensemble — Model Performance")

    for split in ["val", "test"]:
        preds = load_predictions(split)
        if preds is None:
            continue

        st.markdown(f"### {split.upper()} Set")

        # Metrics
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

        # Perfect prediction line
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
        col1, col2 = st.columns(2)
        with col1:
            min_solubility = st.slider(
                "Min CO₂ Solubility (mole fraction)",
                0.0, 0.5, 0.05, 0.01,
            )
        with col2:
            max_uncertainty = st.slider(
                "Max Prediction Uncertainty",
                0.0, 1.0, 0.3, 0.05,
            )

        # Filter
        filtered = ranked[
            (ranked["co2_solubility_pred"] >= min_solubility) &
            (ranked["uncertainty"] <= max_uncertainty)
        ]

        st.markdown(f"Showing **{len(filtered):,}** candidates matching filters")

        # Scatter plot: solubility vs uncertainty, colored by cation type
        if len(filtered) > 0:
            filtered = filtered.copy()
            filtered["cation_type"] = filtered["cation_name"].str.split("-").str[0]

            fig = px.scatter(
                filtered.head(5000),  # Limit for performance
                x="co2_solubility_pred",
                y="uncertainty",
                color="cation_type",
                hover_data=["cation_name", "anion_name", "molecular_weight", "vs_mea"],
                title="Candidate Landscape: CO₂ Solubility vs Prediction Uncertainty",
                labels={
                    "co2_solubility_pred": "Predicted CO₂ Solubility (mole fraction)",
                    "uncertainty": "Ensemble Uncertainty (log₁₀)",
                    "cation_type": "Cation Family",
                },
                opacity=0.6,
            )

            # Add MEA baseline
            fig.add_vline(x=0.20, line_dash="dash", line_color="red",
                         annotation_text="MEA Baseline", annotation_position="top")

            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)


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
        display_df = top[display_cols].copy()
        display_df.columns = ["Rank", "Cation", "Anion", "x_CO₂", "Uncertainty",
                             "% vs MEA", "MW"]
        display_df["x_CO₂"] = display_df["x_CO₂"].round(4)
        display_df["Uncertainty"] = display_df["Uncertainty"].round(3)
        display_df["% vs MEA"] = display_df["% vs MEA"].round(1)
        display_df["MW"] = display_df["MW"].round(1)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Detailed view for selected candidate
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
            # Molecular structure
            img_bytes = mol_to_svg(candidate["smiles"], size=(400, 300))
            if img_bytes:
                st.image(img_bytes, caption="Molecular Structure")

        with col2:
            st.markdown(f"**SMILES:** `{candidate['smiles']}`")
            st.markdown(f"**Cation:** {candidate['cation_name']}")
            st.markdown(f"**Anion:** {candidate['anion_name']}")

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("CO₂ Solubility", f"{candidate['co2_solubility_pred']:.4f}")
            mcol2.metric("Uncertainty", f"±{candidate['uncertainty']:.3f}")
            mcol3.metric("vs MEA", f"{candidate['vs_mea']:+.1f}%")

            st.markdown(f"**Molecular Weight:** {candidate['molecular_weight']:.1f} g/mol")
            st.markdown(f"**Conditions:** T={candidate['temperature_K']:.1f} K, P={candidate['pressure_bar']:.1f} bar")
