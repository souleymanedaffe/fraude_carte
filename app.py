import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

st.set_page_config(page_title="Débogage", layout="centered")
st.title("🧪 Débogage de l'application")

st.write("✅ Début de l'exécution")

# --------------------------
# Charger les données
# --------------------------
try:
    df = pd.read_csv("fake_transactions_balanced.csv")
    st.write("✅ CSV chargé")
except Exception as e:
    st.error(f"❌ Erreur de lecture CSV : {e}")
    st.stop()

try:
    encoders = {}
    for col in ["Pays", "PaysResidence", "Carte", "DeviceType", "EnLigne"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    st.write("✅ Encodage terminé")
except Exception as e:
    st.error(f"❌ Erreur encodage : {e}")
    st.stop()

# --------------------------
# Entraîner le modèle
# --------------------------
try:
    X = df.drop(columns=["Fraude"])
    y = df["Fraude"]
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)
    st.write("✅ Modèle entraîné")
except Exception as e:
    st.error(f"❌ Erreur entraînement : {e}")
    st.stop()

# --------------------------
# Affichage test
# --------------------------
st.subheader("✅ Toutes les étapes ont réussi 🎉")
st.write(df.head())
