import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
import matplotlib.pyplot as plt
from fpdf import FPDF

# --------------------------
# Charger les données équilibrées
# --------------------------
@st.cache_data
def charger_donnees():
    try:
        df = pd.read_csv("fake_transactions_balanced.csv", encoding="utf-8")
    except FileNotFoundError:
        st.error("❌ Le fichier 'fake_transactions_balanced.csv' est introuvable. Vérifiez qu'il est bien dans le dépôt.")
        st.stop()

    encoders = {}
    for col in ["Pays", "PaysResidence", "Carte", "DeviceType", "EnLigne"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

# --------------------------
# Entraîner le modèle
# --------------------------
@st.cache_data
def entrainer_modele(df):
    X = df.drop(columns=["Fraude"])
    y = df["Fraude"]
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model

# --------------------------
# Enregistrer dans un historique
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
# Génération PDF
# --------------------------
def generer_pdf(chemin_csv, chemin_pdf="rapport_fraude.pdf"):
    df = pd.read_csv(chemin_csv)
    nb_total = len(df)
    nb_fraudes = df[df["Fraude"] == "Oui"].shape[0]
    nb_normales = df[df["Fraude"] == "Non"].shape[0]
    montant_fraude = df[df["Fraude"] == "Oui"]["Montant"].mean()
    montant_normal = df[df["Fraude"] == "Non"]["Montant"].mean()
    derniere_date = df["Date"].max()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rapport de Détection de Fraude", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Total transactions : {nb_total}", ln=True)
    pdf.cell(0, 10, f"Transactions normales : {nb_normales}", ln=True)
    pdf.cell(0, 10, f"Fraudes détectées : {nb_fraudes}", ln=True)
    pdf.cell(0, 10, f"Montant moyen fraude : {montant_fraude:.2f} €", ln=True)
    pdf.cell(0, 10, f"Montant moyen normal : {montant_normal:.2f} €", ln=True)
    pdf.cell(0, 10, f"Dernière détection : {derniere_date}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Dernières transactions :", ln=True)
    pdf.set_font("Arial", "", 10)

    for _, row in df.tail(5).iterrows():
        ligne = f"{row['Date']} | Client {row['ClientID']} | {row['Montant']} € | {row['Fraude']} | {row['Action']}"
        pdf.cell(0, 8, ligne, ln=True)

    pdf.output(chemin_pdf)
    return chemin_pdf

# --------------------------
# Interface principale
# --------------------------
st.set_page_config(page_title="Détection de Fraude", layout="centered")
st.markdown("<h1 style='font-size: 40px;'>💳 Détection de Fraude Bancaire</h1>", unsafe_allow_html=True)

chemin = "historique_fraude.csv"
df, encoders = charger_donnees()

if df.empty:
    st.error("⚠️ Le fichier CSV est vide.")
    st.stop()

model = entrainer_modele(df)

st.markdown("<h2 style='font-size: 26px;'>📊 Importance des variables</h2>", unsafe_allow_html=True)
importances = model.feature_importances_
features = df.drop("Fraude", axis=1).columns
fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importance")
ax.set_title("Poids des variables")
st.pyplot(fig)

with st.form("formulaire_transaction"):
    st.markdown("<h2 style='font-size: 26px;'>📝 Saisir une transaction</h2>", unsafe_allow_html=True)
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
    input_data = {
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

    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.markdown("<h2 style='font-size: 26px;'>🔍 Résultat</h2>", unsafe_allow_html=True)
    if proba > seuil:
        if amount <= 500:
            action = "Confirmation manuelle"
            st.info("Transaction suspecte. Veuillez confirmer si vous l'avez autorisée.")
        elif 100 < amount <= 1000:
            action = "Demande SMS"
            st.warning("Transaction moyenne détectée comme suspecte.")
        else:
            action = "Blocage et contact conseiller"
            st.error("🚫 Transaction à montant élevé bloquée temporairement.")

        st.markdown(f"<div style='color: red; font-size: 22px;'>🚨 <b>FRAUDE détectée !</b><br>Probabilité : {proba:.2%}</div>", unsafe_allow_html=True)
        enregistrer_historique(client_id, amount, proba, True, action)
    else:
        action = "Aucune"
        st.markdown(f"<div style='color: green; font-size: 22px;'>✅ <b>Transaction normale</b><br>Probabilité : {proba:.2%}</div>", unsafe_allow_html=True)
        enregistrer_historique(client_id, amount, proba, False, action)

    fig2, ax2 = plt.subplots()
    ax2.bar(["Normale", "Fraude"], model.predict_proba(df_input)[0])
    ax2.set_ylabel("Probabilité")
    st.pyplot(fig2)

st.markdown("<h2 style='font-size: 24px;'>🧾 Historique des détections</h2>", unsafe_allow_html=True)
if os.path.exists(chemin):
    historique = pd.read_csv(chemin)
    st.dataframe(historique)
    if st.button("🗑️ Réinitialiser l'historique"):
        os.remove(chemin)
        st.success("Historique supprimé avec succès.")
else:
    st.info("Aucune transaction enregistrée pour le moment.")
