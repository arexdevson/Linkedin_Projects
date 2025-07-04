import os
import re
import nltk
import string
import textstat
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from httpx import get
from parsel import Selector
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

# ⚠️ Setup seguro para NLTK no Streamlit Cloud
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("words", download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

common_words = set(words.words())

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

def lexical_difficulty(text):
    tokens = word_tokenize(preprocess(text))
    total_words = len(tokens)
    rare_words = [w for w in tokens if w not in common_words and w.isalpha()]
    return len(rare_words) / total_words if total_words > 0 else 0

def grammar_complexity(text):
    sentences = sent_tokenize(text)
    return sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def classify_song(text):
    rare_ratio = lexical_difficulty(text)
    avg_sent_len = grammar_complexity(text)
    readability = readability_score(text)
    if rare_ratio > 0.35 or avg_sent_len > 20 or readability < 30:
        level = "Difícil"
    elif rare_ratio > 0.2 or readability < 50:
        level = "Média"
    else:
        level = "Fácil"
    return level, round(rare_ratio, 2), round(avg_sent_len, 2), round(readability, 2)

def letra(url: str) -> str:
    response = get(url, timeout=20)
    s = Selector(response.text)
    texto = s.css("div#lyrics::text").getall()
    return "\n".join(t.strip() for t in texto if t.strip())

def faixas(url: str) -> list[tuple[str, str]]:
    response = get(url, timeout=20)
    s = Selector(response.text)
    nomes = s.css('a.nameMusic::text').getall()
    hrefs = s.css('a.nameMusic::attr(href)').getall()
    nomes = [n.strip() for n in nomes if n.strip()]
    return list(zip(nomes, hrefs))

# STREAMLIT UI
st.title("🎧 Classificador de Dificuldade de Letras de Música")

if "letras_extraidas" not in st.session_state:
    st.session_state.letras_extraidas = {}

url_artista = st.text_input("Cole a URL do artista no Vagalume (ex: https://www.vagalume.com.br/eminem/)")

if st.button("🔍 Buscar Letras") and url_artista:
    url_base = "https://www.vagalume.com.br"
    musicas = faixas(url_artista)
    with st.spinner("Extraindo letras..."):
        for titulo, link in musicas:
            url_musica = url_base + link
            conteudo = letra(url_musica)
            titulo_limpo = re.sub(r'[\\/*?:"<>|]', "", titulo)
            st.session_state.letras_extraidas[f"{titulo_limpo}.txt"] = conteudo
        st.success(f"{len(st.session_state.letras_extraidas)} letras extraídas com sucesso!")

if st.session_state.letras_extraidas:
    resultados = []
    textos_concatenados = ""
    for nome, content in st.session_state.letras_extraidas.items():
        textos_concatenados += " " + content
        nivel, rare, avg_len, readability = classify_song(content)
        resultados.append({
            "Nome da Música": nome,
            "Dificuldade": nivel,
            "% Palavras Raras": rare,
            "Média Palavras/Frase": avg_len,
            "Readability (Flesch)": readability
        })

    df = pd.DataFrame(resultados)
    st.dataframe(df)

    st.download_button("⬇️ Baixar CSV", df.to_csv(index=False).encode("utf-8"), "resultados.csv")

    st.markdown("### ☁️ Nuvem de Palavras")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(textos_concatenados)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("Aguardando letras para análise...")
