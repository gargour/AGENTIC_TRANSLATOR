import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Charger les variables du fichier .env
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Traducteur Intelligent LLM",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Traducteur Intelligent avec LangChain & Groq")
st.markdown("Traduisez instantanément vos textes d'une langue à une autre en utilisant les modèles ultra-rapides de Groq.")

# Récupération de la clé API depuis l'environnement (.env)
env_groq_key = os.environ.get("GROQ_API_KEY")

# Barre latérale pour la configuration
with st.sidebar:
    st.header("Configuration")
    if env_groq_key:
        st.success("✅ Clé API Groq détectée via .env")
    else:
        st.warning("⚠️ Aucune clé trouvée dans le .env")
        
    api_key_input = st.text_input(
        "Clé API Groq (Optionnel)",
        type="password",
        value=env_groq_key if env_groq_key else "",
        placeholder="gsk_...",
        help="La clé est automatiquement chargée depuis le fichier .env."
    )
    st.markdown("[Obtenir une clé API Groq](https://console.groq.com/keys)")

# Utilisation de la clé (priorité à l'input de la sidebar, sinon la variable d'environnement)
groq_api_key = api_key_input or env_groq_key

# Interface principale de saisie
col1, col2 = st.columns(2)
with col1:
    source_language = st.selectbox(
        "Langue source (Main Language)",
        ["Anglais", "Français", "Espagnol", "Allemand", "Italien", "Arabe", "Autre"]
    )

with col2:
    target_language = st.selectbox(
        "Langue cible (Target Language)",
        ["Espagnol", "Français", "Anglais", "Allemand", "Italien", "Arabe", "Autre"]
    )

text_to_translate = st.text_area(
    "Texte à traduire",
    placeholder="Écrivez ou collez votre texte ici...",
    height=150
)

# Logique de traduction
if st.button("Traduire le texte", type="primary"):
    if not groq_api_key:
        st.error("⚠️ Veuillez fournir une clé API Groq valide dans le fichier .env ou dans la barre latérale.")
    elif not text_to_translate.strip():
        st.warning("⚠️ Veuillez entrer du texte à traduire.")
    else:
        try:
            with st.spinner("Traduction ultra-rapide en cours..."):
                # Initialisation de ChatGroq avec un modèle valide et actif sur l'API Groq
                llm = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0.3,
                    groq_api_key=groq_api_key
                )
                
                # Création du template de prompt structuré
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "Tu es un traducteur expert. Traduis fidèlement le texte suivant de la langue source vers la langue cible demandée. Fournis uniquement le texte traduit, sans commentaires ni notes supplémentaires."),
                    ("human", "Langue source : {source_lang}\nLangue cible : {target_lang}\nTexte :\n{text}")
                ])
                
                # Chaîne d'exécution LangChain (Pipeline)
                chain = prompt_template | llm | StrOutputParser()
                
                # Exécution de la traduction
                translated_result = chain.invoke({
                    "source_lang": source_language,
                    "target_lang": target_language,
                    "text": text_to_translate
                })
                
                # Affichage du résultat
                st.subheader("Résultat de la traduction")
                st.success(translated_result)
                
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la communication avec l'API Groq : {e}")