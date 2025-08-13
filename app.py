# app.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
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
    initial_sidebar_state="expanded",
)

# ---- CSS (animations + styles de cartes/boutons) ----
st.markdown("""
<style>
/* largeur contenu */
.main .block-container {max-width: 1200px;}

/* header animé */
.hero {
  background: linear-gradient(120deg, #0ea5e9, #8b5cf6, #22c55e);
  background-size: 300% 300%;
  animation: gradientShift 10s ease infinite;
  border-radius: 20px;
  padding: 24px 28px;
  color: white;
  box-shadow: 0 10px 30px rgba(0,0,0,.12);
}
@keyframes gradientShift {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

/* cartes KPI */
.kpi {
  border-radius: 18px;
  padding: 18px;
  background: rgba(255,255,255,.55);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(0,0,0,.06);
  transition: transform .15s ease, box-shadow .15s ease;
}
.kpi:hover {transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,.08);}

/* boutons plus visibles */
.stButton>button {
  border-radius: 12px !important;
  padding: 10px 16px !important;
  font-weight: 600 !important;
  transition: transform .08s ease;
}
.stButton>button:active {transform: translateY(1px)}

/* badges */
.badge {
  display:inline-flex; align-items:center; gap:.5rem;
  background: #0ea5e91a; color:#0369a1; border:1px solid #0ea5e94d;
  padding:.4rem .65rem; border-radius:999px; font-size:.85rem; font-weight:600;
}

/* footer */
.footer {
  text-align:center; opacity:.8; margin-top:32px; font-size:.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HELPERS
# =========================================
def plotly_template():
    # suit le thème Streamlit (clair/sombre) automatiquement
    return "plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"

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
        n_estimators=300,  # un peu plus d’arbres pour la stabilité
        class_weight="balanced",
        random_state=42,
        max_depth=None,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

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

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("⚙️ Paramètres")
data_path_default = "fake_transactions_balanced.csv"
uploaded = st.sidebar.file_uploader("📥 Charger un dataset CSV (optionnel)", type=["csv"])
if uploaded:
    df, encoders = charger_donnees(uploaded)
else:
    if not os.path.exists(data_path_default):
        st.sidebar.error("Le fichier fake_transactions_balanced.csv est introuvable.")
        st.stop()
    df, encoders = charger_donnees(data_path_default)

seuil = st.sidebar.slider("🎚️ Seuil de décision (Fraude si proba > seuil)", 0.05, 0.95, 0.50, 0.01)
show_details = st.sidebar.toggle("Afficher les détails avancés", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Astuce : ajustez le seuil pour moduler la sensibilité.")

# =========================================
# HERO
# =========================================
st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
    <div style="font-size:2rem;">💳 Détection de Fraude Bancaire</div>
    <span class="badge">Temps réel</span>
    <span class="badge">Random Forest</span>
    <span class="badge">Explainability light</span>
  </div>
  <div style="opacity:.9; margin-top:.4rem;">Analysez vos transactions, obtenez une probabilité de fraude et des actions recommandées.</div>
</div>
""", unsafe_allow_html=True)
st.write("")

# =========================================
# MODELE + IMPORTANCES
# =========================================
with st.spinner("Entraînement / chargement du modèle..."):
    model = entrainer_modele(df)

features = df.drop("Fraude", axis=1).columns
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}) \
    .sort_values("Importance", ascending=True)

# KPIs rapides
col_a, col_b, col_c = st.columns(3)
with col_a:
    with st.container():
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("📦 Observations", f"{len(df):,}".replace(",", " "))
        st.markdown('</div>', unsafe_allow_html=True)
with col_b:
    with st.container():
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("🧮 Variables", f"{len(features)}")
        st.markdown('</div>', unsafe_allow_html=True)
with col_c:
    with st.container():
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("🌳 Arbres", "300")
        st.markdown('</div>', unsafe_allow_html=True)

# Tabs: Importance / Formulaire / Historique
tab_imp, tab_form, tab_hist = st.tabs(["📊 Importance", "📝 Transaction", "🧾 Historique"])

with tab_imp:
    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Poids des variables (Random Forest)",
        template=plotly_template(),
    )
    fig_imp.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

with tab_form:
    st.subheader("Saisir une transaction")
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

    # Résultats
    chemin_hist = "historique_fraude.csv"
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
        pred = int(proba > seuil)

        # Affichage résumé + jauge
        colg, cold = st.columns([1,1])
        with colg:
            if pred == 1:
                st.error(f"🚨 Probabilité de FRAUDE : **{proba:.2%}** (seuil {seuil:.0%})")
            else:
                st.success(f"✅ Transaction NORMALE : **{(1-proba):.2%}** de normalité (proba fraude {proba:.2%})")

        with cold:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'thickness': 0.25},
                    'steps': [
                        {'range': [0, seuil*100], 'color': 'rgba(34,197,94,0.5)'},
                        {'range': [seuil*100, 100], 'color': 'rgba(239,68,68,0.5)'}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': seuil*100}
                },
                title={'text': "Proba fraude"}
            ))
            fig_gauge.update_layout(template=plotly_template(), height=250, margin=dict(l=10, r=10, t=40, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Actions recommandées (avec jolies cartes)
        st.markdown("#### Actions recommandées")
        if proba > seuil:
            if amount <= 500:
                action = "Confirmation manuelle"
                c1, c2 = st.columns(2)
                with c1:
                    st.info("🔒 Transaction suspecte. Veuillez confirmer si vous l'avez autorisée.")
                    st.button("✅ Je confirme cette transaction")
                with c2:
                    st.warning("Si vous n'êtes pas à l'origine de ce paiement :")
                    st.button("❌ Ce n'était pas moi")
            elif 100 < amount <= 1000:
                action = "Demande SMS"
                c1, c2 = st.columns(2)
                with c1:
                    st.warning("⚠️ Transaction moyenne détectée comme suspecte.")
                    st.button("📩 Demander un code SMS")
                with c2:
                    st.info("Ou confirmer manuellement si c'est bien vous :")
                    st.button("✅ Je confirme manuellement")
            else:
                action = "Blocage et contact conseiller"
                c1, c2 = st.columns(2)
                with c1:
                    st.error("🚫 Montant élevé : transaction temporairement bloquée.")
                    st.button("🔁 Demander une vérification par un agent")
                with c2:
                    st.info("Besoin d'aide immédiate ?")
                    st.button("📞 Contacter mon conseiller")
        else:
            action = "Aucune"
            st.balloons()  # petite animation 🎈

        # Graphique probas
        proba_df = pd.DataFrame({"Classe": ["Normale", "Fraude"], "Probabilité": model.predict_proba(df_input)[0]})
        fig2 = px.bar(
            proba_df, x="Classe", y="Probabilité", title="Probabilité de prédiction",
            text="Probabilité", template=plotly_template()
        )
        fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig2.update_layout(yaxis_range=[0, 1], height=380, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # Historisation
        enregistrer_historique(client_id, amount, proba, proba > seuil, action, chemin_hist)

with tab_hist:
    st.subheader("Historique des détections")
    chemin = "historique_fraude.csv"
    if os.path.exists(chemin):
        historique = pd.read_csv(chemin)
        st.dataframe(historique, use_container_width=True, height=360)
        colh1, colh2 = st.columns([1,1])
        with colh1:
            if st.button("🗑️ Réinitialiser l'historique"):
                os.remove(chemin)
                st.success("Historique supprimé avec succès.")
        with colh2:
            # téléchargement CSV
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
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST</strong> · UI modernisée ✨
</div>
""", unsafe_allow_html=True)
