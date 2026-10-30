"""API de prédiction de la durée de séjour (patients diabétiques).

Lancer :  uvicorn main:app --host 0.0.0.0 --port 8000
Documentation interactive : http://localhost:8000/docs
"""
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# --- chargement du modèle (une seule fois, au démarrage) ---------------------
bundle = joblib.load(Path(__file__).parent / "modele_final.joblib")
FEATURES = bundle["features"]
DEFAUTS = bundle["valeurs_par_defaut"]
BORNES = bundle["bornes"]
PREPRO = bundle["preprocesseur"]
MODELE = bundle["modele"]

# valeurs autorisées pour chaque variable catégorielle (lues dans l'encodeur)
CATEGORIES = {}
for nom, transfo, colonnes in PREPRO.transformers_:
    if nom == "cat":
        for col, cats in zip(colonnes, transfo.categories_):
            CATEGORIES[col] = [str(c) for c in cats]

app = FastAPI(title="Prédiction de la durée d'hospitalisation",
              description="Modèle HistGradientBoosting entraîné sur Diabetes 130-US Hospitals. "
                          "Prévision à l'admission, à utiliser comme aide à la planification des lits.",
              version="1.0")

# le site de ton collègue peut appeler l'API depuis le navigateur
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Patient(BaseModel):
    """Toutes les variables sont facultatives : celles qu'on n'envoie pas prennent la valeur la plus fréquente."""
    model_config = ConfigDict(extra="forbid")
    age_num: Optional[float] = Field(None, ge=0, le=120, description="Âge en années (ex. 65)")
    diag_groupe: Optional[str] = Field(None, description="Famille du diagnostic principal")
    insulin: Optional[str] = Field(None, description="Down, No, Steady ou Up")
    medical_specialty: Optional[str] = None
    admission_type_id: Optional[str] = None
    admission_source_id: Optional[str] = None
    payer_code: Optional[str] = None
    diabetesMed: Optional[str] = None
    change: Optional[str] = None
    A1Cresult: Optional[str] = None
    max_glu_serum: Optional[str] = None
    gender: Optional[str] = None
    glyburide: Optional[str] = None
    glipizide: Optional[str] = None
    glimepiride: Optional[str] = None
    repaglinide: Optional[str] = None
    rosiglitazone: Optional[str] = None
    pioglitazone: Optional[str] = None
    total_visites: Optional[float] = Field(None, ge=0, description="Visites passées (ambulatoire + urgences + hospitalisations)")
    number_outpatient: Optional[float] = Field(None, ge=0, description="Visites ambulatoires passées")


def predire(patient: dict) -> float:
    ligne = pd.DataFrame([DEFAUTS])[FEATURES]
    for variable, valeur in patient.items():
        if variable in CATEGORIES and str(valeur) not in CATEGORIES[variable]:
            raise ValueError(f"Valeur '{valeur}' inconnue pour {variable}. Valeurs possibles : {CATEGORIES[variable]}")
        ligne[variable] = valeur
    for c in BORNES.columns:                       # même plafonnement que pendant l'entraînement
        if c in ligne.columns:
            ligne[c] = ligne[c].clip(BORNES.loc[0.01, c], BORNES.loc[0.99, c])
    x = pd.DataFrame(PREPRO.transform(ligne), columns=PREPRO.get_feature_names_out())
    return round(float(MODELE.predict(x)[0]), 1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/variables")
def variables():
    """Liste des variables acceptées, valeurs autorisées et valeur par défaut (pour construire le formulaire du site)."""
    return {v: {"type": "categorie" if v in CATEGORIES else "nombre",
                "valeurs_possibles": CATEGORIES.get(v),
                "valeur_par_defaut": DEFAUTS[v]} for v in FEATURES}


@app.post("/predict")
def predict(patient: Patient):
    donnees = patient.model_dump(exclude_none=True)
    try:
        duree = predire(donnees)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur))
    return {"duree_prevue_jours": duree,
            "erreur_moyenne_jours": 2.1,
            "avertissement": "Aide à la planification uniquement. Les séjours longs (8 jours et plus) sont sous-estimés."}
