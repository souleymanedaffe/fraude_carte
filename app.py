# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1) Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("fake_transactions_balanced.csv")
    except FileNotFoundError:
        st.error("❌ 'fake_transactions_balanced.csv' introuvable. Place-le à la racine du repo.")
        st.stop()
    encoders = {}
    for col in ["Pays", "PaysResidence", "Carte", "DeviceType", "EnLigne"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

# 2) Entraînement du modèle
@st.cache_resource
def train_model(df):
    X = df.drop("Fraude", axis=1)
    y = df["Fraude"]
    model = RandomForestClassifier(
        n_estimators=50, n_jobs=-1, class_weight="balanced", random_state=42
    )
    model.fit(X, y)
    return model

# 3) Interface
st.set_page_config(page_title="Détection de Fraude", layout="centered")
st.title("💳 Détection de Fraude Bancaire")

df, encoders = load_data()
model = train_model(df)

# Importance des variables
st.header("📊 Importance des variables")
fig, ax = plt.subplots()
ax.barh(df.drop("Fraude", axis=1).columns, model.feature_importances_)
ax.set_xlabel("Importance")
st.pyplot(fig)

# Formulaire
st.header("📝 Saisir une transaction")
with st.form("tx_form"):
    client_id    = st.number_input("ID Client", 1000, 1100, 1005)
    amount       = st.number_input("Montant (€)", 0.01, 10000.0, 100.0)
    heure        = st.slider("Heure de la transaction", 0, 23, 12)
    heure_pref   = st.slider("Heure habituelle d'achat", 0, 23, 14)
    delta_heure  = abs(heure - heure_pref)
    nb_tx24h     = st.slider("Nb transactions (24h)", 0, 50, 2)
    pays_tx      = st.selectbox("Pays de transaction", sorted(encoders["Pays"].classes_))
    pays_res     = st.selectbox("Pays de résidence", sorted(encoders["PaysResidence"].classes_))
    carte        = st.selectbox("Type de carte", sorted(encoders["Carte"].classes_))
    device       = st.selectbox("Type d'appareil", sorted(encoders["DeviceType"].classes_))
    en_ligne     = st.selectbox("En ligne ?", ["Oui", "Non"])
    submit       = st.form_submit_button("🔍 Vérifier")

# 4) Prédiction & résultat
if submit:
    inp = {
        "ClientID": client_id,
        "Amount": amount,
        "Heure": heure,
        "HeurePreferee": heure_pref,
        "DeltaHeure": delta_heure,
        "NbTransactions24h": nb_tx24h,
        "Pays": encoders["Pays"].transform([pays_tx])[0],
        "PaysResidence": encoders["PaysResidence"].transform([pays_res])[0],
        "Carte": encoders["Carte"].transform([carte])[0],
        "DeviceType": encoders["DeviceType"].transform([device])[0],
        "EnLigne": encoders["EnLigne"].transform([en_ligne])[0],
    }
    df_in = pd.DataFrame([inp])
    prob = model.predict_proba(df_in)[0][1]
    seuil = 0.5

    if prob > seuil:
        # choix d’action
        if amount <= 500:
            st.warning("⚠️ Transaction suspecte. Vérification manuelle requise.")
        elif amount <= 1000:
            st.warning("🔔 Suspecte : demande de confirmation par SMS.")
        else:
            st.error("🚫 Bloquée : montant élevé.")
        st.error(f"🚨 Probabilité de fraude : {prob:.2%}")
    else:
        st.success(f"✅ Transaction normale (Probabilité de fraude : {prob:.2%})")

    # Histogramme du score
    fig2, ax2 = plt.subplots()
    ax2.bar(["Normale","Fraude"], model.predict_proba(df_in)[0])
    ax2.set_ylabel("Probabilité")
    st.pyplot(fig2)
