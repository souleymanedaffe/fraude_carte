import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt

# --------------------------
# Chargement optimisé des données (CSV -> Parquet)
# --------------------------
@st.cache_data
def charger_donnees_parquet():
    parquet_file = "fake_transactions.parquet"
    encoders_file = "encoders.pkl"

    # Si Parquet et encoders non existants, créer à partir du CSV
    if not os.path.exists(parquet_file) or not os.path.exists(encoders_file):
        df_raw = pd.read_csv("fake_transactions_balanced.csv")
        encoders = {}
        for col in ["Pays", "PaysResidence", "Carte", "DeviceType", "EnLigne"]:
            le = LabelEncoder()
            df_raw[col] = le.fit_transform(df_raw[col])
            encoders[col] = le
        df_raw.to_parquet(parquet_file, index=False)
        with open(encoders_file, "wb") as f:
            pickle.dump(encoders, f)
        return df_raw, encoders

    # Sinon charger directement Parquet et encoders
    df = pd.read_parquet(parquet_file)
    with open(encoders_file, "rb") as f:
        encoders = pickle.load(f)
    return df, encoders

# --------------------------
# Chargement ou entraînement du modèle (persisté avec joblib)
# --------------------------
@st.cache_resource
def obtenir_modele():
    model_file = "rf_model.joblib"
    df, _ = charger_donnees_parquet()
    X = df.drop(columns=["Fraude"])
    y = df["Fraude"]

    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        model.fit(X, y)
        joblib.dump(model, model_file)
        return model

# --------------------------
# Enregistrement de l'historique (inchangé)
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
    if not os.path.exists(chemin):
        pd.DataFrame([ligne]).to_csv(chemin, index=False)
    else:
        pd.DataFrame([ligne]).to_csv(chemin, mode="a", header=False, index=False)

# --------------------------
# Interface Streamlit
# --------------------------
st.set_page_config(page_title="Détection de Fraude", layout="centered")
st.title("💳 Détection de Fraude Bancaire (Optimisé)")

# Charger données et modèle
chemin_histo = "historique_fraude.csv"
df, encoders = charger_donnees_parquet()
model = obtenir_modele()

# Affichage de l'importance
st.subheader("📊 Importance des variables")
importances = model.feature_importances_
features = df.drop("Fraude", axis=1).columns
fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importance")
ax.set_title("Poids des variables")
st.pyplot(fig)

# Formulaire de transaction
with st.form("form_tx"):
    st.subheader("📝 Saisir une transaction")
    client_id = st.number_input("🆔 ID Client", min_value=1000, max_value=1100, value=1005)
    amount = st.number_input("💰 Montant (€)", min_value=0.01, value=100.0)
    heure = st.slider("🕒 Heure de la transaction", 0, 23, 12)
    heure_pref = st.slider("🕕 Heure habituelle d'achat", 0, 23, 14)
    delta_heure = abs(heure - heure_pref)
    nb_tx_24h = st.slider("🔁 Nb transactions (24h)", 0, 20, 2)
    pays = st.selectbox("🌍 Pays de transaction", sorted(encoders["Pays"].classes_))
    pays_res = st.selectbox("🏠 Pays de résidence", sorted(encoders["PaysResidence"].classes_))
    carte = st.selectbox("💳 Type de carte", sorted(encoders["Carte"].classes_))
    device = st.selectbox("📱 Type d'appareil", sorted(encoders["DeviceType"].classes_))
    en_ligne = st.selectbox("🛒 En ligne ?", ["Oui", "Non"])
    submit = st.form_submit_button("🔍 Vérifier la transaction")

seuil = 0.5
if submit:
    data = {
        "ClientID": client_id,
        "Amount": amount,
        "Heure": heure,
        "HeurePreferee": heure_pref,
        "DeltaHeure": delta_heure,
        "NbTransactions24h": nb_tx_24h,
        "Pays": encoders["Pays"].transform([pays])[0],
        "PaysResidence": encoders["PaysResidence"].transform([pays_res])[0],
        "Carte": encoders["Carte"].transform([carte])[0],
        "DeviceType": encoders["DeviceType"].transform([device])[0],
        "EnLigne": encoders["EnLigne"].transform([en_ligne])[0]
    }
    df_input = pd.DataFrame([data])
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.subheader("🔍 Résultat")
    if proba > seuil:
        if amount <= 500:
            action = "Confirmation manuelle"
            st.info("Transaction suspecte. Veuillez confirmer si vous l'avez autorisée.")
            st.button("✅ Je confirme cette transaction")
            st.button("❌ Ce n'était pas moi")
        elif amount <= 1000:
            action = "Demande SMS"
            st.warning("Transaction moyenne suspecte. Confirmez par SMS.")
            st.button("📩 Demander SMS")
            st.button("✅ Je confirme manuellement")
        else:
            action = "Blocage & contact conseiller"
            st.error("🚫 Montant élevé bloqué temporairement.")
            st.button("📞 Contacter conseiller")
            st.button("🔁 Demande vérification agent")
        st.error(f"🚨 FRAUDE détectée ! Probabilité : {proba:.2%}")
        enregistrer_historique(client_id, amount, proba, True, action)
    else:
        action = "Aucune"
        st.success(f"✅ Transaction normale. Probabilité de fraude : {proba:.2%}")
        enregistrer_historique(client_id, amount, proba, False, action)

    # Affichage des probabilités
    fig2, ax2 = plt.subplots()
    ax2.bar(["Normale", "Fraude"], model.predict_proba(df_input)[0])
    ax2.set_ylabel("Probabilité")
    st.pyplot(fig2)

# Historique
st.subheader("🧾 Historique des détections")
if os.path.exists(chemin_histo):
    hist = pd.read_csv(chemin_histo)
    st.dataframe(hist)
    if st.button("🗑️ Réinitialiser l'historique"):
        os.remove(chemin_histo)
        st.success("Historique supprimé !")
else:
    st.info("Aucune transaction enregistrée.")
