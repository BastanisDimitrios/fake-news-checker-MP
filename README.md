# Fake News Detection Capstone (Starter)

This project is a starter template for a **Fake News Detection** capstone using **Machine Learning + NLP**.

## Folder structure
- `data/raw/` : put `Fake.csv` and `True.csv` here (not committed to git)
- `data/processed/` : cleaned/merged datasets
- `notebooks/` : exploration notebooks
- `src/` : reusable python modules (preprocessing, training, evaluation)
- `models/` : saved models (e.g., `model.pkl`, `vectorizer.pkl`)
- `reports/figures/` : plots for your report

## Quick start (VS Code)
1. Create and activate a virtual environment:
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Put your Kaggle files into:
   - `data/raw/Fake.csv`
   - `data/raw/True.csv`

4. Open `notebooks/01_exploration.ipynb` and run the cells.

## Notes
- This template avoids databases / complex UI to keep scope controlled.
- You can add a simple Streamlit UI later if needed.
