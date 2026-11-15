import pandas as pd

NUMERIQUES = ["person_age", "person_income", "person_emp_length", "loan_amnt",
              "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
CATEGORIELLES = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
CIBLE = "loan_status"


def preparer(df):
    """Les variables du modèle : les nombres tels quels, les catégories en colonnes 0/1."""
    X = pd.get_dummies(df[NUMERIQUES + CATEGORIELLES], columns=CATEGORIELLES, drop_first=True, dtype=float)
    y = df[CIBLE] if CIBLE in df else None
    return X, y
