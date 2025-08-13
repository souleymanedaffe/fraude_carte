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
