"""Note un nouveau dossier de crédit :  python src/predire.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from src.features import preparer  # noqa: E402

paquet = joblib.load(RACINE / "models" / "xgb.joblib")
dossier = pd.DataFrame([{
    "person_age": 29, "person_income": 42000, "person_home_ownership": "RENT", "person_emp_length": 3.0,
    "loan_intent": "EDUCATION", "loan_grade": "C", "loan_amnt": 12000, "loan_int_rate": 13.5,
    "loan_percent_income": 0.29, "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 4,
}])
X, _ = preparer(dossier)
X = X.reindex(columns=paquet["colonnes"], fill_value=0.0)
print(f"Probabilité de défaut : {paquet['modele'].predict_proba(X)[0, 1]:.1%}")
