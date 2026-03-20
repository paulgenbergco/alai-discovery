# ALAi Discovery — Ionic Liquid Discovery Pipeline

An end-to-end AI pipeline for discovering novel ionic liquids (ILs) for CO₂ capture, built on Directed Message Passing Neural Networks (D-MPNN).

## What This Does

1. **Collects** ionic liquid property data from public sources (NIST ILThermo)
2. **Curates** data with physics-informed quality checks (isotherm enforcement, variance filtering)
3. **Trains** 8-model D-MPNN ensembles for CO₂ solubility prediction with uncertainty quantification
4. **Generates** 400,000+ novel IL candidates via combinatorial enumeration
5. **Screens** candidates using multi-objective ranking (solubility, uncertainty)
6. **Demonstrates** results in an interactive Streamlit dashboard

## Quick Start

```bash
# Clone and install
git clone https://github.com/paulgenbergco/alai-discovery.git
cd alai-discovery
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Collect data from ILThermo
python -m src.curation.ilthermo_collector

# Curate dataset
python -m src.curation.pipeline

# Train models (GPU recommended)
python -m src.models.train_ensemble

# Generate candidates
python -m src.generation.combinatorial

# Screen and rank
python -m src.screening.pareto_ranker

# Launch demo
streamlit run src/demo/app.py
```

## Architecture

```
Public Data (ILThermo) → Curation Pipeline → D-MPNN Ensemble (8 models)
                                                      ↓
Novel IL Candidates (400K+) → Property Prediction → Pareto Screening → Demo
```

## Key Results

- **Model**: D-MPNN ensemble with scaffold-split validation
- **Property**: CO₂ solubility (mole fraction) at specified T/P conditions
- **Candidates**: 400,000+ novel ionic liquids screened
- **Benchmark**: Top candidates outperform MEA baseline for CO₂ capture

## Tech Stack

- **ML**: [Chemprop](https://github.com/chemprop/chemprop) (D-MPNN), PyTorch
- **Chemistry**: RDKit, SMILES representation
- **Data**: NIST ILThermo, PubChem
- **Demo**: Streamlit, Plotly
