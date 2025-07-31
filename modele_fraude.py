import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Chargement des données
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Prétraitement
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Modèles
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Entraînement et évaluation
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    auc = roc_auc_score(y_test, preds)
    print("AUC:", auc)
    if auc > best_score:
        best_score = auc
        best_model = model

# Sauvegarde du meilleur modèle
joblib.dump(best_model, "modele_fraude.pkl")
joblib.dump(scaler, "scaler.pkl")
