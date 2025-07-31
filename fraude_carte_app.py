import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
import matplotlib.pyplot as plt

# --------------------------
# Charger les données (avec cache)
# --------------------------
@st.cache_data
def charger_donnees():
    df = pd.read_csv("fake_transactions_balanced.csv")
    encoders = {}
    for col in ["Pays", "PaysResidence", "Carte", "DeviceType", "EnLigne"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

# --------------------------
# Entraîner le modèle
# --------------------------
@st.cache_resource
def charger_modele(df):
    X = df.drop(columns=["Fraude"])
    y = df["Fraude"]
    model = RandomForestClassifier(
        n_estimators=50,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model

# --------------------------
# Enregistrement dans un CSV
# --------------------------
def enregistrer_historique(client_id, amount, proba, is_fraude, action):
    ligne = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ClientID": client_id,
        "Montant": amount,
        "Probabilité": round(proba, 4),
        "Fraude": "Oui" if is_fraude else "Non",
        "Action": action
    }
    chemin = "historique_fraude.csv"
    df_ligne = pd.DataFrame([ligne])
    df_ligne.to_csv(chemin, mode="a" if os.path.exists(chemin) else "w", header=not os.path.exists(chemin), index=False)

# --------------------------
# Interface Streamlit
# --------------------------
st.set_page_config(page_title="Détection de Fraude", layout="centered")
st.title("💳 Détection de Fraude Bancaire")

# Chargement des données et du modèle
df, encoders = charger_donnees()
model = charger_modele(df)

# Afficher l'importance des variables
st.subheader("📊 Importance des variables")
fig, ax = plt.subplots()
ax.barh(df.drop("Fraude", axis=1).columns, model.feature_importances_)
ax.set_xlabel("Importance")
st.pyplot(fig)

# Formulaire utilisateur
with st.form("tx_form"):
    client_id = st.number_input("ID Client", 1000, 1100, 1005)
    amount = st.number_input("Montant (€)", 0.01, 10000.0, 100.0)
    heure = st.slider("Heure transaction", 0, 23, 12)
    heure_pref = st.slider("Heure habituelle", 0, 23, 14)
    delta_heure = abs(heure - heure_pref)
    nb_tx = st.slider("Nb tx 24h", 0, 50, 2)
    pays = st.selectbox("Pays", sorted(encoders["Pays"].classes_))
    pays_res = st.selectbox("Pays résidence", sorted(encoders["PaysResidence"].classes_))
    carte = st.selectbox("Carte", sorted(encoders["Carte"].classes_))
    device = st.selectbox("Appareil", sorted(encoders["DeviceType"].classes_))
    en_ligne = st.selectbox("En ligne ?", ["Oui", "Non"])
    submit = st.form_submit_button("🔍 Vérifier")

# --------------------------
# Prédiction
# --------------------------
if submit:
    input_data = {
        "ClientID": client_id,
        "Amount": amount,
        "Heure": heure,
        "HeurePreferee": heure_pref,
        "DeltaHeure": delta_heure,
        "NbTransactions24h": nb_tx,
        "Pays": encoders["Pays"].transform([pays])[0],
        "PaysResidence": encoders["PaysResidence"].transform([pays_res])[0],
        "Carte": encoders["Carte"].transform([carte])[0],
        "DeviceType": encoders["DeviceType"].transform([device])[0],
        "EnLigne": encoders["EnLigne"].transform([en_ligne])[0]
    }

    df_input = pd.DataFrame([input_data])
    proba = model.predict_proba(df_input)[0][1]
    seuil = 0.5

    if proba > seuil:
        if amount <= 500:
            action = "Confirmation manuelle"
            st.warning("⚠️ Transaction suspecte. Veuillez confirmer si vous l'avez autorisée.")
        elif 100 < amount <= 1000:
            action = "Demande SMS"
            st.warning("🔔 Transaction moyenne détectée comme suspecte.")
        else:
            action = "Blocage et contact conseiller"
            st.error("🚫 Transaction à montant élevé bloquée temporairement.")

        st.markdown(f"🚨 **Probabilité de fraude : {proba:.2%}**", unsafe_allow_html=True)
        enregistrer_historique(client_id, amount, proba, True, action)

    else:
        action = "Aucune"
        st.success(f"✅ Transaction normale. Probabilité de fraude : {proba:.2%}")
        enregistrer_historique(client_id, amount, proba, False, action)

    fig2, ax2 = plt.subplots()
    ax2.bar(["Normale", "Fraude"], model.predict_proba(df_input)[0])
    ax2.set_ylabel("Probabilité")
    st.pyplot(fig2)
