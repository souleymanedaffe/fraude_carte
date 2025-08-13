# app.py
# -*- coding: utf-8 -*-
import os, csv
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.errors import ParserError
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG + STYLES
# =========================
st.set_page_config(page_title="Détection de Fraude", page_icon="💳", layout="wide")
st.markdown("""
<style>
.main .block-container {max-width: 1200px;}
.hero {background: linear-gradient(120deg,#0ea5e9,#8b5cf6,#22c55e);background-size:300% 300%;
animation: gradientShift 10s ease infinite;border-radius:20px;padding:24px 28px;color:white;margin-bottom:10px;}
@keyframes gradientShift {0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
.stButton>button {border-radius:12px;padding:10px 16px;font-weight:600;}
.footer {text-align:center;opacity:.8;margin-top:32px;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

def plotly_template():
    return "plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"

# =========================
# HELPERS
# =========================
DATA_PATH = "fake_transactions_balanced.csv"
HISTO_PATH = "historique_fraude.csv"
SEUIL_USER = 0.50

# Colonnes de l'historique (avec toutes les valeurs saisies)
COLUMNS_HISTO = [
    "ID","Date","ClientID","Montant","Probabilité","Fraude",
    "ActionRecommandée","Statut","Décision","Décideur",
    # --- champs de saisie utilisateur ---
    "Heure","HeurePreferee","DeltaHeure","NbTransactions24h",
    "Pays","PaysResidence","Carte","DeviceType","EnLigne"
]

@st.cache_data(show_spinner=False)
def charger_donnees(path: str = DATA_PATH):
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
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def ensure_histo():
    """Crée un fichier d'historique vide et propre s'il n'existe pas ou s'il est vide."""
    if (not os.path.exists(HISTO_PATH)) or os.path.getsize(HISTO_PATH) == 0:
        pd.DataFrame(columns=COLUMNS_HISTO).to_csv(HISTO_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

def charger_historique():
    """Lecture robuste de l'historique. Si corrompu -> sauvegarde .bak + réinitialisation propre."""
    ensure_histo()
    try:
        df = pd.read_csv(HISTO_PATH, encoding="utf-8", engine="python", on_bad_lines="skip", dtype=str)
        for col in COLUMNS_HISTO:
            if col not in df.columns: df[col] = ""
        return df[COLUMNS_HISTO]
    except (ParserError, UnicodeDecodeError, OSError):
        try:
            bak = HISTO_PATH + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
            if os.path.exists(HISTO_PATH): os.replace(HISTO_PATH, bak)
        finally:
            pd.DataFrame(columns=COLUMNS_HISTO).to_csv(HISTO_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        return pd.DataFrame(columns=COLUMNS_HISTO)

def enregistrer_historique(client_id, amount, proba, is_fraude, action_reco, payload: dict):
    """Ajoute une ligne à l'historique en incluant toutes les valeurs saisies."""
    ensure_histo()
    rec_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    base = {
        "ID": rec_id,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ClientID": client_id,
        "Montant": float(amount),
        "Probabilité": round(float(proba), 4),
        "Fraude": "Oui" if is_fraude else "Non",
        "ActionRecommandée": action_reco,
        "Statut": "En attente",
        "Décision": "",
        "Décideur": "",
    }
    # normalise & ajoute champs de saisie
    for k in ["Heure","HeurePreferee","DeltaHeure","NbTransactions24h"]:
        base[k] = int(payload.get(k, 0))
    for k in ["Pays","PaysResidence","Carte","DeviceType","EnLigne"]:
        base[k] = str(payload.get(k, ""))

    write_header = (not os.path.exists(HISTO_PATH)) or os.path.getsize(HISTO_PATH) == 0
    pd.DataFrame([base]).to_csv(HISTO_PATH, mode="a", header=write_header, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return rec_id

def maj_statut(rec_id: str, statut: str, decision: str, decideur: str = "Conseiller"):
    df = charger_historique()
    mask = df["ID"].astype(str) == str(rec_id)
    if mask.any():
        df.loc[mask, ["Statut","Décision","Décideur"]] = [statut, decision, decideur]
        df.to_csv(HISTO_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        return True
    return False

def action_recommandee(proba: float, montant: float, seuil: float):
    if proba <= seuil: return "Aucune"
    if montant <= 500: return "Confirmation manuelle"
    if 100 < montant <= 1000: return "Demande SMS"
    return "Blocage et contact conseiller"

# =========================
# CHARGEMENT MODÈLE
# =========================
if not os.path.exists(DATA_PATH):
    st.error("Fichier 'fake_transactions_balanced.csv' introuvable."); st.stop()
with st.spinner("Chargement des données et entraînement du modèle..."):
    df, encoders = charger_donnees(DATA_PATH)
    model = entrainer_modele(df)

features = df.drop("Fraude", axis=1).columns
importance_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)

# =========================
# UI
# =========================
st.markdown("""
<div class="hero">
  <div style="font-size:2rem;">💳 Détection de Fraude</div>
  <div style="opacity:.9;margin-top:.4rem;">Choisissez un espace pour continuer.</div>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Navigation", ["Espace Utilisateur", "Espace Conseiller"], horizontal=True, label_visibility="collapsed")

# ========= PARTIE UTILISATEUR (SANS GRAPHIQUES) =========
if mode == "Espace Utilisateur":
    st.subheader("📝 Saisir une transaction")
    col1, col2, col3 = st.columns([1,1,1])
    with st.form("form_user", clear_on_submit=False):
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

    if submit:
        x = {
            "ClientID": client_id, "Amount": amount,
            "Heure": heure, "HeurePreferee": heure_pref, "DeltaHeure": delta_heure,
            "NbTransactions24h": nb_tx_24h,
            # pour l'historique on garde les labels lisibles (pas les codes encodés)
            "Pays": pays, "PaysResidence": pays_res, "Carte": carte, "DeviceType": device, "EnLigne": en_ligne,
            # mais pour le modèle on passe les codes
            "_PaysCode": encoders["Pays"].transform([pays])[0],
            "_PaysResidenceCode": encoders["PaysResidence"].transform([pays_res])[0],
            "_CarteCode": encoders["Carte"].transform([carte])[0],
            "_DeviceTypeCode": encoders["DeviceType"].transform([device])[0],
            "_EnLigneCode": encoders["EnLigne"].transform([en_ligne])[0],
        }
        model_input = pd.DataFrame([{
            "ClientID": x["ClientID"], "Amount": x["Amount"],
            "Heure": x["Heure"], "HeurePreferee": x["HeurePreferee"], "DeltaHeure": x["DeltaHeure"],
            "NbTransactions24h": x["NbTransactions24h"],
            "Pays": x["_PaysCode"], "PaysResidence": x["_PaysResidenceCode"],
            "Carte": x["_CarteCode"], "DeviceType": x["_DeviceTypeCode"], "EnLigne": x["_EnLigneCode"],
        }])
        proba = float(model.predict_proba(model_input)[0][1])
        pred_fraude = proba > SEUIL_USER
        reco = action_recommandee(proba, amount, SEUIL_USER)

        if pred_fraude:
            st.error(f"🚨 Probabilité de FRAUDE : **{proba:.2%}** (seuil {SEUIL_USER:.0%})")
        else:
            st.success(f"✅ Transaction NORMALE : **{(1-proba):.2%}** de normalité (proba fraude {proba:.2%})")
            st.balloons()

        st.markdown("#### Actions recommandées")
        if reco == "Aucune":
            st.success("Aucune action requise."); action_txt = "Aucune"
        elif reco == "Confirmation manuelle":
            st.info("Transaction suspecte. Veuillez confirmer si vous l'avez autorisée.")
            st.button("✅ Je confirme cette transaction"); st.button("❌ Ce n'était pas moi")
            action_txt = "Confirmation manuelle"
        elif reco == "Demande SMS":
            st.warning("Transaction moyenne détectée comme suspecte.")
            st.button("📩 Demander un code SMS"); st.button("✅ Je confirme manuellement")
            action_txt = "Demande SMS"
        else:
            st.error("🚫 Montant élevé : transaction temporairement bloquée.")
            st.button("📞 Contacter mon conseiller"); st.button("🔁 Demander vérification par un agent")
            action_txt = "Blocage et contact conseiller"

        # Enregistre toutes les valeurs saisies (lisibles) + proba
        payload_for_history = {
            "Heure": x["Heure"], "HeurePreferee": x["HeurePreferee"], "DeltaHeure": x["DeltaHeure"],
            "NbTransactions24h": x["NbTransactions24h"], "Pays": x["Pays"], "PaysResidence": x["PaysResidence"],
            "Carte": x["Carte"], "DeviceType": x["DeviceType"], "EnLigne": x["EnLigne"],
        }
        rec_id = enregistrer_historique(client_id, amount, proba, pred_fraude, action_txt, payload_for_history)
        st.info(f"🧾 Demande transmise au conseiller (ID : {rec_id}).")

    st.markdown('<div class="footer">Espace Utilisateur</div>', unsafe_allow_html=True)

# ========= PARTIE CONSEILLER (reçoit les mêmes valeurs + formulaire pré-rempli) =========
else:
    st.title("🛡️ Espace Conseiller")
    seuil_conseiller = st.slider("Seuil de décision (interne conseiller)", 0.05, 0.95, 0.50, 0.01)

    tab1, tab2, tab3 = st.tabs(["📥 Dernière saisie (formulaire)", "📊 Analyses", "🧾 Historique & Validation"])

    # --- Tab 1 : Dernière saisie utilisateur dans un formulaire pré-rempli ---
    with tab1:
        histo = charger_historique()
        if len(histo) == 0:
            st.info("Aucune transaction disponible. Demandez à l'utilisateur d'en soumettre une.")
        else:
            # on prend la plus récente (peu importe le statut)
            last = histo.sort_values("Date").iloc[-1]

            st.markdown("#### Détails (formulaire pré-rempli à partir de la dernière saisie utilisateur)")
            edit_mode = st.toggle("Permettre modification avant décision", value=False)

            # valeurs par défaut (textuelles lisibles) — mêmes champs que l'utilisateur
            def_val = {
                "ClientID": int(last.get("ClientID", 1005) or 1005),
                "Montant": float(last.get("Montant", 100.0) or 100.0),
                "NbTransactions24h": int(last.get("NbTransactions24h", 2) or 2),
                "Heure": int(last.get("Heure", 12) or 12),
                "HeurePreferee": int(last.get("HeurePreferee", 14) or 14),
                "DeltaHeure": int(last.get("DeltaHeure", abs(int(last.get("Heure",12))-int(last.get("HeurePreferee",14)))) or 0),
                "Pays": str(last.get("Pays", "")),
                "PaysResidence": str(last.get("PaysResidence", "")),
                "Carte": str(last.get("Carte", "")),
                "DeviceType": str(last.get("DeviceType", "")),
                "EnLigne": str(last.get("EnLigne", "")),
            }

            col1, col2, col3 = st.columns([1,1,1])
            with st.form("form_conseiller_prefilled"):
                with col1:
                    client_id = st.number_input("🆔 ID Client", 1000, 1100, def_val["ClientID"], step=1, disabled=not edit_mode)
                    amount = st.number_input("💰 Montant (€)", min_value=0.01, value=float(def_val["Montant"]), step=1.0, disabled=not edit_mode)
                    nb_tx_24h = st.slider("🔁 Nb transactions (24h)", 0, 30, def_val["NbTransactions24h"], disabled=not edit_mode)
                with col2:
                    heure = st.slider("🕒 Heure de la transaction", 0, 23, def_val["Heure"], disabled=not edit_mode)
                    heure_pref = st.slider("🕕 Heure habituelle d'achat", 0, 23, def_val["HeurePreferee"], disabled=not edit_mode)
                    delta_heure = abs(heure - heure_pref)  # recalcul live si modifié
                with col3:
                    # pour les selectbox, on essaie de remettre la valeur si elle existe sinon le premier élément
                    pays = st.selectbox("🌍 Pays de transaction", sorted(encoders["Pays"].classes_),
                                        index=max(0, sorted(encoders["Pays"].classes_).index(def_val["Pays"])
                                                  if def_val["Pays"] in encoders["Pays"].classes_ else 0),
                                        disabled=not edit_mode)
                    pays_res = st.selectbox("🏠 Pays de résidence", sorted(encoders["PaysResidence"].classes_),
                                            index=max(0, sorted(encoders["PaysResidence"].classes_).index(def_val["PaysResidence"])
                                                      if def_val["PaysResidence"] in encoders["PaysResidence"].classes_ else 0),
                                            disabled=not edit_mode)
                    carte = st.selectbox("💳 Type de carte", sorted(encoders["Carte"].classes_),
                                         index=max(0, sorted(encoders["Carte"].classes_).index(def_val["Carte"])
                                                   if def_val["Carte"] in encoders["Carte"].classes_ else 0),
                                         disabled=not edit_mode)
                    device = st.selectbox("📱 Type d'appareil", sorted(encoders["DeviceType"].classes_),
                                          index=max(0, sorted(encoders["DeviceType"].classes_).index(def_val["DeviceType"])
                                                    if def_val["DeviceType"] in encoders["DeviceType"].classes_ else 0),
                                          disabled=not edit_mode)
                    en_ligne = st.selectbox("🛒 En ligne ?", ["Oui", "Non"],
                                            index=(0 if def_val["EnLigne"] == "Oui" else 1),
                                            disabled=not edit_mode)
                submit_eval = st.form_submit_button("🔍 Évaluer / Mettre à jour l'aperçu")

            # on évalue soit les valeurs d'origine (si pas d'édition), soit celles du formulaire
            vals = {
                "ClientID": client_id if edit_mode else def_val["ClientID"],
                "Amount": amount if edit_mode else def_val["Montant"],
                "Heure": heure if edit_mode else def_val["Heure"],
                "HeurePreferee": heure_pref if edit_mode else def_val["HeurePreferee"],
                "DeltaHeure": delta_heure if edit_mode else def_val["DeltaHeure"],
                "NbTransactions24h": nb_tx_24h if edit_mode else def_val["NbTransactions24h"],
                "Pays": encoders["Pays"].transform([pays if edit_mode else def_val["Pays"]])[0],
                "PaysResidence": encoders["PaysResidence"].transform([pays_res if edit_mode else def_val["PaysResidence"]])[0],
                "Carte": encoders["Carte"].transform([carte if edit_mode else def_val["Carte"]])[0],
                "DeviceType": encoders["DeviceType"].transform([device if edit_mode else def_val["DeviceType"]])[0],
                "EnLigne": encoders["EnLigne"].transform([en_ligne if edit_mode else def_val["EnLigne"]])[0],
            }
            dfi = pd.DataFrame([vals])
            proba = float(model.predict_proba(dfi)[0][1])
            pred_fraude = proba > seuil_conseiller
            reco = ("Aucune" if proba <= seuil_conseiller else
                    "Confirmation manuelle" if (amount if edit_mode else def_val["Montant"]) <= 500 else
                    "Demande SMS" if 100 < (amount if edit_mode else def_val["Montant"]) <= 1000 else
                    "Blocage et contact conseiller")

            c1, c2 = st.columns([1,1])
            with c1:
                if pred_fraude:
                    st.error(f"🚨 Probabilité de FRAUDE : **{proba:.2%}** (seuil {seuil_conseiller:.0%})")
                else:
                    st.success(f"✅ Transaction NORMALE : **{(1-proba):.2%}** (proba fraude {proba:.2%})")
                st.markdown("**Action recommandée (calculée)** : " + reco)

                # bloc validation
                st.markdown("#### Valider la recommandation")
                decision = st.radio("Décision", ["Valider","Rejeter","Contacter client","Bloquer temporairement"], horizontal=True)
                decideur = st.text_input("Décideur", value="Conseiller")
                if st.button("✅ Appliquer la décision sur cette transaction"):
                    statut = "Validée" if decision == "Valider" else "Traitée"
                    if maj_statut(last["ID"], statut, decision, decideur):
                        st.success(f"Décision appliquée (ID {last['ID']}).")
                        st.rerun()
                    else:
                        st.error("Échec de la mise à jour.")

            with c2:
                # Jauge + histogramme (réservés au conseiller)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba*100, number={'suffix': "%"},
                    gauge={'axis': {'range':[0,100]}, 'bar': {'thickness':0.25},
                           'steps':[{'range':[0,seuil_conseiller*100],'color':'rgba(34,197,94,0.5)'},
                                    {'range':[seuil_conseiller*100,100],'color':'rgba(239,68,68,0.5)'}],
                           'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': .75, 'value': seuil_conseiller*100}},
                    title={'text': "Proba fraude"}
                ))
                fig.update_layout(template=plotly_template(), height=250, margin=dict(l=10,r=10,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

                proba_df = pd.DataFrame({"Classe":["Normale","Fraude"], "Probabilité":[1-proba, proba]})
                fig2 = px.bar(proba_df, x="Classe", y="Probabilité", text="Probabilité", template=plotly_template(),
                              title="Probabilité de prédiction")
                fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig2.update_layout(yaxis_range=[0,1], height=350, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 2 : Analyses globales ---
    with tab2:
        st.subheader("Importance des variables")
        fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                         title="Poids des variables (Random Forest)", template=plotly_template())
        fig_imp.update_layout(height=540, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- Tab 3 : Historique & Validation (liste complète) ---
    with tab3:
        st.subheader("Historique des détections")
        histo = charger_historique()
        st.dataframe(histo, use_container_width=True, height=420)

        st.markdown("### Traiter une transaction en attente")
        en_attente = histo[histo["Statut"] == "En attente"]
        if len(en_attente) == 0:
            st.info("Aucun enregistrement en attente.")
        else:
            left, right = st.columns([2,3])
            with left:
                rec_id = st.selectbox("Sélectionner l'ID à traiter", en_attente["ID"].astype(str).tolist())
                decision = st.radio("Décision", ["Valider","Rejeter","Contacter client","Bloquer temporairement"], horizontal=True, key="dec2")
                decideur = st.text_input("Décideur", value="Conseiller", key="decideur2")
                if st.button("✅ Appliquer la décision", key="apply2"):
                    statut = "Validée" if decision == "Valider" else "Traitée"
                    if maj_statut(rec_id, statut, decision, decideur):
                        st.success(f"Décision appliquée sur ID {rec_id}"); st.rerun()
                    else:
                        st.error("Échec de la mise à jour.")
            with right:
                if 'rec_id' in locals() and rec_id:
                    details = histo[histo["ID"].astype(str) == str(rec_id)]
                    st.markdown("**Détails sélectionnés :**"); st.table(details)

        st.markdown("---")
        if len(histo):
            st.download_button("⬇️ Télécharger l'historique (CSV)",
                               data=histo.to_csv(index=False).encode("utf-8"),
                               file_name="historique_fraude.csv", mime="text/csv")

    st.markdown('<div class="footer">Espace Conseiller</div>', unsafe_allow_html=True)
