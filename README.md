# NASA Astro Detector — Noble-Bohr

An interactive near-Earth asteroid analysis dashboard combining NASA/JPL catalog data, machine-learning hazard classification, asteroid orbital calculations, and a 3D orbit visualization.

> **Scientific note:** This project is an educational analysis and visualization tool. Its ML classification is not an official impact prediction and does not replace NASA/JPL Sentry or professional orbital-risk calculations.

## Features

- **Asteroid catalog explorer:** Search and inspect near-Earth asteroid records from cached NASA/JPL datasets.
- **NASA/JPL data integration:** Data pipeline uses SBDB, Sentry, and Close Approach Data (CAD) sources.
- **Feature engineering:** Calculates/imputes selected orbital and physical features, including estimated diameter, relative encounter velocity, and approximate kinetic energy.
- **ML classification and benchmarking:** Compares Logistic Regression, Random Forest, Gradient Boosting, and MLP classifiers using a stratified train/test workflow and evaluation metrics.
- **Risk explanations:** Displays feature importance and asteroid-level explanatory signals.
- **3D orbital visualization:** Interactive Sun/Earth/asteroid orbit display using a Keplerian two-body propagation approximation.
- **Custom risk input:** Enter orbital parameters or use included example presets to explore the classifier.
- **REST API:** FastAPI endpoints provide dashboard metrics, catalog/search data, model predictions, and orbit-propagation results.

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Data & ML:** pandas, NumPy, SciPy, scikit-learn, joblib
- **Visualization:** Plotly, Matplotlib, Three.js (browser-side)
- **Frontend:** HTML, CSS, JavaScript, Jinja2 templates
- **Testing:** pytest, httpx

## Data sources

The project fetches or uses cached data from NASA/JPL Small-Body Database (SBDB), Sentry Earth Impact Monitoring, and Close Approach Data (CAD). API URLs are defined in `config.py`; raw data snapshots and processed data are stored under `data/`.

## Run locally

### 1. Clone and enter the repository

```bash
git clone https://github.com/<your-username>/NASA-Astro-Detector.git
cd NASA-Astro-Detector
```

### 2. Create a virtual environment (recommended)

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the web application

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

> The repository includes cached datasets and model artifacts for local startup. Refreshing NASA data or retraining the models is optional and may take longer.

## Optional: Rebuild the dataset and train models

```bash
python data/data_pipeline.py
python models/train.py
```

## Optional: Run tests

```bash
pytest -v
```

## Project structure

```text
NASA-Astro-Detector/
├── app.py                     # FastAPI app, dashboard and API routes
├── config.py                  # Paths, API URLs and training configuration
├── requirements.txt
├── data/
│   ├── fetch_raw.py            # NASA/JPL data fetch and cache
│   ├── data_pipeline.py        # Cleaning, preprocessing, feature engineering
│   ├── raw/                   # Cached raw API snapshots
│   └── processed/              # Processed asteroid dataset and analysis metadata
├── models/
│   ├── train.py               # Model training and comparison
│   ├── predictor.py            # Prediction and explanation logic
│   └── saved/                  # Saved model and evaluation artifacts
├── orbit/
│   └── propagator.py           # Keplerian orbit propagation utilities
├── templates/                   # Dashboard HTML
├── static/                      # CSS, JavaScript, charts and 3D view
└── tests/                       # API and project tests
```

## Interpretation and limitations

- The classification target is derived from catalog hazard criteria/data, not from a model learning observed Earth impacts.
- Machine-learning scores should be interpreted as estimates for the project's defined classification task; they are not an actual collision probability.
- NASA/JPL Sentry's authoritative impact-risk solutions use orbital uncertainty analysis and are not replaced by this project.
- The 3D orbital display uses a simplified Keplerian two-body model; it is intended for visualization and intuition, not high-precision long-term prediction.
- Dataset snapshots become stale. Use the data-refresh functionality to update records when network access is available.

## Acknowledgements

Data is sourced from NASA/JPL Small-Body Dynamics resources. Please review the relevant NASA/JPL source documentation when reusing the data.
