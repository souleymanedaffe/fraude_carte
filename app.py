# app.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================================
# CONFIG GLOBALE + STYLES
# =========================================
st.set_page_config(
    page_title="Détection de Fraude",
    page_icon="💳",
    layout="wide",
)

# ---- CSS (animations + styles discrets) ----
st.markdown("""
<style>
.main .block-container {max-width: 1100px;}

.hero {
  background: linear-gradient(120deg, #0ea5e9, #8b5cf6, #22c55e);
  background-size: 300% 300%;
  animation: gradientShift 10s ease infinite;
  border-radius: 20px;
  padding: 24px 28px;
  color: white;
  box-shadow: 0 10px 30px rgba(0,0,0,.12);
  margin-bottom: 10px;
}
@keyframes gradientShift {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.stButton>button {
  border-radius: 12px !important;
  padding: 10px 16px !important;
  font-weight: 600 !important;
  transition: transform .08s ease;
}
.stButton>button:active {transform: translateY(1px)}
.footer { text-align:center; opacity:.8; margin-top:32px; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

def plotly_template():
    return "plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"

# =========================================
# CHARGEMENT DONNÉES + MODÈLE
# =========================================
@st.cache_data(show_spinner=False)
def charger_donnees(path: str):
    df = pd.read_csv(path)
    encoders = {}
    for col in ["Pays", "PaysResidence", "Carte", "DeviceType", "EnLigne"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

@st.cache_data(show_spinner=False)
def entrainer_modele(df: pd.DataFrame):
    X = df.drop(columns=["Fraude"])
    y = df["Fraude"]
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

DATA_PATH = "fake_transactions_balanced.csv"
if not os.path.exists(DATA_PATH):
    st.error("Le fichier 'fake_transactions_balanced.csv' est introuvable à la racine de l'app.")
    st.stop()

with st.spinner("Chargement des données et du modèle..."):
    df, encoders = charger_donnees(DATA_PATH)
    model = entrainer_modele(df)

# Seuil **fixe** (pas d’affichage à l’utilisateur)
SEUIL = 0.50

# =========================================
# EN-TÊTE
# =========================================
st.markdown("""
<div class="hero">
  <div style="font-size:2rem;">💳 Détection de Fraude Bancaire</div>
  <div style="opacity:.9; margin-top:.4rem;">
    Analysez une transaction, obtenez une probabilité de fraude et des actions recommandées.
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# FORMULAIRE + RÉSULTATS
# =========================================
st.subheader("📝 Saisir une transaction")

col1, col2, col3 = st.columns([1,1,1])
with st.form("formulaire_transaction", clear_on_submit=False):
    with col1:
        client_id = st.number_input("🆔 ID Client", min_value=1000, max_value=1100, value=1005, step=1)
        amount = st.number_input("💰 Montant (€)", min_value=0.01, value=100.0, step=1.0)
        nb_tx_24h = st.slider("🔁 Nb transactions (24h)", 0, 30, 2)
    with col2:
        heure = st.slider("🕒 Heure de la transaction", 0, 23, 12)
        heure_pref = st.slider("🕕 Heure habituelle d'achat", 0, 23, 14)
        delta_heure = abs(heure - heure_pref)
    with col3:
        pays = st.selectbox("🌍 Pays de transaction", sorted(encoders["Pays"].classes_))
        pays_res = st.selectbox("🏠 Pays de résidence", sorted(encoders["PaysResidence"].classes_))
        carte = st.selectbox("💳 Type de carte", sorted(encoders["Carte"].classes_))
        device = st.selectbox("📱 Type d'appareil", sorted(encoders["DeviceType"].classes_))
        en_ligne = st.selectbox("🛒 En ligne ?", ["Oui", "Non"])

    submit = st.form_submit_button("🔍 Vérifier la transaction")

def enregistrer_historique(client_id, amount, proba, is_fraude, action, chemin):
    ligne = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ClientID": client_id,
        "Montant": amount,
        "Probabilité": round(float(proba), 4),
        "Fraude": "Oui" if is_fraude else "Non",
        "Action": action
    }
    if not os.path.exists(chemin):
        pd.DataFrame([ligne]).to_csv(chemin, index=False)
    else:
        pd.DataFrame([ligne]).to_csv(chemin, mode="a", header=False, index=False)

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
        "EnLigne": encoders["EnLigne"].transform([en_ligne])[0],
    }
    df_input = pd.DataFrame([input_data])
    proba = float(model.predict_proba(df_input)[0][1])
    pred = int(proba > SEUIL)

    colg, cold = st.columns([1,1])

    with colg:
        if pred == 1:
            st.error(f"🚨 Probabilité de FRAUDE : **{proba:.2%}** (seuil {SEUIL:.0%})")
        else:
            st.success(f"✅ Transaction NORMALE : **{(1-proba):.2%}** de normalité (proba fraude {proba:.2%})")
            st.balloons()

    with cold:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, SEUIL*100], 'color': 'rgba(34,197,94,0.5)'},
                    {'range': [SEUIL*100, 100], 'color': 'rgba(239,68,68,0.5)'}
                ],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': SEUIL*100}
            },
            title={'text': "Proba fraude"}
        ))
        fig_gauge.update_layout(template=plotly_template(), height=250, margin=dict(l=10, r=10, t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Graphe des probabilités
    proba_df = pd.DataFrame({"Classe": ["Normale", "Fraude"], "Probabilité": model.predict_proba(df_input)[0]})
    fig2 = px.bar(
        proba_df, x="Classe", y="Probabilité", title="Probabilité de prédiction",
        text="Probabilité", template=plotly_template()
    )
    fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig2.update_layout(yaxis_range=[0, 1], height=380, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # Historisation
    enregistrer_historique(client_id, amount, proba, proba > SEUIL, "Auto", "historique_fraude.csv")

# =========================================
# HISTORIQUE
# =========================================
st.subheader("🧾 Historique des détections")
hist_path = "historique_fraude.csv"
if os.path.exists(hist_path):
    historique = pd.read_csv(hist_path)
    st.dataframe(historique, use_container_width=True, height=360)
    colh1, colh2 = st.columns([1,1])
    with colh1:
        if st.button("🗑️ Réinitialiser l'historique"):
            os.remove(hist_path)
            st.success("Historique supprimé avec succès.")
    with colh2:
        st.download_button(
            "⬇️ Télécharger l'historique (CSV)",
            data=historique.to_csv(index=False).encode("utf-8"),
            file_name=f"historique_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
else:
    st.info("Aucune transaction enregistrée pour le moment.")

# =========================================
# FOOTER
# =========================================
st.markdown("""
<div class="footer">
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST</strong>
</div>
""", unsafe_allow_html=True)
