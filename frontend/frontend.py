import streamlit as st
import requests
from PIL import Image
import os
from datetime import datetime
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Détection de Mélanome",
    page_icon="🩺",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .malignant {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .benign {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🩺 Détection de Mélanome")
st.markdown("""
Cette application utilise l'intelligence artificielle pour analyser les images de lésions cutanées 
et évaluer le risque de mélanome. Téléchargez une image pour obtenir une analyse préliminaire.
""")

# Avertissement médical
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Avertissement médical important :</strong><br>
    Cette application est un outil d'aide au diagnostic et ne remplace en aucun cas un avis médical professionnel. 
    Consultez toujours un dermatologue pour un diagnostic précis et un suivi approprié.
</div>
""", unsafe_allow_html=True)

# Configuration de l'URL de l'API
API_URL = os.getenv("API_URL", "http://backend:8000")

# Fonction pour afficher les résultats
def display_results(result):
    # Récupération des données de la réponse
    prediction = result.get("result", "inconnu")
    confidence = float(result.get("confidence", 0)) * 100
    prediction_id = result.get("prediction_id", "")
    
    # Affichage des résultats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Résultats de l'analyse")
        st.metric("Résultat", prediction.capitalize())
        st.metric("Confiance", f"{confidence:.2f}%")
    
    with col2:
        if prediction == "malin":
            st.error("⚠️ Lésion potentiellement maligne")
            st.warning("**Il est recommandé de consulter un dermatologue rapidement.**")
        else:
            st.success("✅ Lésion probablement bénigne")
            st.info("Pour une évaluation complète, consultez un professionnel de santé.")
    
    # Barre de progression
    st.progress(min(confidence / 100, 1.0))
    st.caption(f"Score de confiance: {confidence:.2f}%")
    
    # Explication des résultats
    with st.expander("🔍 Comprendre ces résultats"):
        st.markdown("""
        - **0-30%** : Faible probabilité de malignité
        - **30-70%** : Probabilité modérée, surveillance recommandée
        - **70-100%** : Probabilité élevée de malignité, consultation médicale recommandée
        
        Ces résultats sont basés sur une analyse automatisée et doivent être interprétés par un professionnel de santé qualifié.
        """)

# Section principale
st.markdown("### Télécharger une image de lésion cutanée")
st.markdown("""
Veuillez télécharger une image claire de la lésion cutanée à analyser. 
Pour de meilleurs résultats :
- Utilisez un fond neutre
- Assurez-vous que la lésion est bien éclairée et nette
- Évitez les reflets ou ombres excessives
""")

uploaded_file = st.file_uploader(
    "Sélectionnez une image (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Afficher l'image téléchargée
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        
        # Bouton pour effectuer la prédiction
        if st.button("Analyser l'image", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Envoi de l'image à l'API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        display_results(result)
                    else:
                        st.error(f"Erreur lors de l'analyse (code {response.status_code}): {response.text}")    
                        
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la communication avec le serveur: {str(e)}")
    except Exception as e:
        st.error("Erreur lors du chargement de l'image. Veuillez vérifier le format du fichier.")

# Section d'information
st.markdown("---")
with st.expander("ℹ️ À propos de cette application"):
    st.markdown("""
    ### Comment fonctionne cette application ?
    Cette application utilise un modèle d'apprentissage profond entraîné sur des milliers d'images de lésions cutanées 
    pour détecter les signes potentiels de mélanome. Le modèle analyse les caractéristiques visuelles de la lésion 
    et fournit une estimation du risque de malignité.
    
    ### Que faire des résultats ?
    - **Résultats inquiétants** : Prenez rendez-vous avec un dermatologue pour une évaluation professionnelle.
    - **Surveillance** : Même pour les lésions bénignes, il est recommandé de surveiller tout changement de taille, 
    de couleur ou de forme.
    - **Prévention** : Protégez-vous du soleil et effectuez des auto-examens réguliers de votre peau.
    
    ### Confidentialité
    Les images que vous téléchargez sont traitées de manière sécurisée et ne sont pas stockées de manière permanente.
    """)

# Pied de page
st.markdown("---")
st.caption("© 2024 Détection de Mélanome - Outil d'aide au diagnostic - Ne remplace pas une consultation médicale")