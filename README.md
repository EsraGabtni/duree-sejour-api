# API de prédiction de la durée d'hospitalisation

Modèle : HistGradientBoosting (erreur moyenne ≈ 2,1 jours). Prévision à l'admission d'un patient diabétique.

## Lancer l'API
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Documentation interactive (on peut tester directement) : http://localhost:8000/docs

> Les versions de `requirements.txt` doivent être respectées (le modèle a été sauvegardé avec scikit-learn 1.9.1).

## Routes
| Route | Rôle |
|---|---|
| `GET /health` | vérifier que l'API tourne |
| `GET /variables` | liste des variables, valeurs autorisées et valeur par défaut (pour construire le formulaire) |
| `POST /predict` | renvoie la durée de séjour prévue |

## Exemple
Requête :
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age_num": 65, "diag_groupe": "Tumeurs", "insulin": "Up"}'
```
Réponse :
```json
{"duree_prevue_jours": 6.8, "erreur_moyenne_jours": 2.1, "avertissement": "..."}
```
Toutes les variables sont facultatives : celles qu'on n'envoie pas prennent la valeur la plus fréquente.
Une valeur non autorisée renvoie une erreur 422 avec la liste des valeurs possibles.

## Appel depuis le site (JavaScript)
```javascript
const reponse = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({age_num: 65, diag_groupe: "Tumeurs", insulin: "Up"})
});
const resultat = await reponse.json();
console.log(resultat.duree_prevue_jours);
```
(CORS est activé : le navigateur peut appeler l'API depuis un autre domaine.)

## Variables principales
`age_num` (nombre), `diag_groupe` (Circulatoire, Diabete, Digestif, Respiratoire, Tumeurs, Traumatisme, Genito-urinaire, Musculo-squelettique, Autre), `insulin` (No, Steady, Up, Down), `medical_specialty`, `admission_type_id`, `admission_source_id`, `payer_code`, `diabetesMed`, `change`, `A1Cresult`, `max_glu_serum`, `gender`, médicaments (`glyburide`, `glipizide`, `glimepiride`, `repaglinide`, `rosiglitazone`, `pioglitazone`), `total_visites`, `number_outpatient`. La liste complète et les valeurs autorisées sont dans `GET /variables`.

## Limites
Aide à la planification des lits, pas un outil médical. Les séjours longs (8 jours et plus) sont sous-estimés d'environ 5 jours.
