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
from nltk.corpus import words
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from wordcloud import STOPWORDS 

# ⚠️ Setup seguro para NLTK no Streamlit Cloud
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("words", download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)
nltk.download('punkt_tab')

# Estilo CSS personalizado
st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(to bottom, #ffffff, #f0f0f5);
            background-attachment: fixed;
            font-family: 'Courier New', monospace;
        }
        .css-1v3fvcr {
            background-color: #8e44ad !important;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Diretório local acessível no ambiente Streamlit Cloud
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")

# Faz o download diretamente para essa pasta
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("words", download_dir=nltk_data_dir)

# Adiciona ao caminho do NLTK para busca
nltk.data.path.append(nltk_data_dir)

common_words = set(words.words())

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

def lexical_difficulty(text):
    tokens = word_tokenize(preprocess(text))
    total_words = len(tokens)
    rare_words = [word for word in tokens if word not in common_words and word.isalpha()]
    rare_ratio = len(rare_words) / total_words if total_words > 0 else 0
    return rare_ratio

def grammar_complexity(text):
    sentences = sent_tokenize(text)
    avg_words_per_sentence = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
    return avg_words_per_sentence

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def classify_song(text):
    rare_ratio = lexical_difficulty(text)
    avg_sentence_len = grammar_complexity(text)
    readability = readability_score(text)
    if rare_ratio > 0.35 or avg_sentence_len > 20 or readability < 30:
        level = 'Difícil'
    elif rare_ratio > 0.2 or readability < 50:
        level = 'Média'
    else:
        level = 'Fácil'
    return level, round(rare_ratio, 2), round(avg_sentence_len, 2), round(readability, 2)

def letra(url: str) -> str:
    response = get(url, timeout=20)
    s = Selector(response.text)
    texto = s.css("div#lyrics::text").getall()
    texto_final = "\n".join(t.strip() for t in texto if t.strip())
    return texto_final

def faixas(url: str) -> list[tuple[str, str]]:
    response = get(url, timeout=20)
    s = Selector(response.text)
    nomes = s.css('a.nameMusic::text').getall()
    hrefs = s.css('a.nameMusic::attr(href)').getall()
    nomes = [n.strip() for n in nomes if n.strip()]
    return list(zip(nomes, hrefs))
# Função para destacar palavras raras e frequentes
def destacar_palavras(texto):
    palavras = re.findall(r'\b\w+\b', texto.lower())
    contador = Counter(palavras)
    mais_frequentes = {p for p, _ in contador.most_common(20)}

    def estilizar(p):
        if p in STOPWORDS:
            return p
        if p in mais_frequentes:
            return f"<span style='color:green'><b>{p}</b></span>"
        elif p not in common_words:
            return f"<span style='color:red'><b>{p}</b></span>"
        else:
            return p

    palavras_destacadas = [estilizar(p) for p in re.findall(r'\b\w+\b', texto)]
    texto_final = re.sub(r'\b\w+\b', lambda m: estilizar(m.group(0)), texto)
    return texto_final

# Interface com Streamlit
st.title("🎧 Classificador de Dificuldade de Letras de Música em Inglês")

aba = st.sidebar.radio("Escolha a funcionalidade:", ["📥 Extrair letras da web"])

if "letras_extraidas" not in st.session_state:
    st.session_state.letras_extraidas = {}

if aba == "📥 Extrair letras da web":
    st.markdown("### 📡 Extração de letras de um artista do Vagalume")
    url_artista = st.text_input("Cole a URL da página de músicas do artista (ex: https://www.vagalume.com.br/eminem/)")

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
        st.markdown("### 🎼 Resultado das Análises das Letras Extraídas")
        resultados = []
        textos_concatenados = ""

        for nome, content  in st.session_state.letras_extraidas.items():
            textos_concatenados += " " + content
            level, rare_ratio, avg_len, readability = classify_song(content)
            resultados.append({
                "Nome da Música": nome,
                "Dificuldade": level,
                "% Palavras Raras": rare_ratio,
                "Média Palavras/Frase": avg_len,
                "Readability (Flesch)": readability
            })

        df_resultado = pd.DataFrame(resultados)

        col1, col2 = st.columns([2, 1])
        with col1:
            filtro_nome = st.text_input("🔍 Filtrar por nome da música", key="filtro_web")
        with col2:
            filtro_dificuldade = st.selectbox("🎚 Filtrar por dificuldade", options=["Todas", "Fácil", "Média", "Difícil"], key="dificuldade_web")

        df_filtrado = df_resultado.copy()
        if filtro_nome:
            df_filtrado = df_filtrado[df_filtrado["Nome da Música"].str.contains(filtro_nome, case=False, na=False)]
        if filtro_dificuldade != "Todas":
            df_filtrado = df_filtrado[df_filtrado["Dificuldade"] == filtro_dificuldade]

        st.dataframe(df_filtrado, use_container_width=True)

        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Baixar resultados em CSV", csv, "resultado_musicas.csv", "text/csv")

        st.markdown("### ☁️ Nuvem de Palavras das Letras")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(textos_concatenados)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Aguardando envio dos arquivos `.txt`.")

# Visualização da letra da música selecionada
st.subheader("🎵 Clique em uma música para ver a letra com destaque")

if 'letras_extraidas' in st.session_state and st.session_state.letras_extraidas:
    nomes_musicas = list(st.session_state.letras_extraidas.keys())
    musica_escolhida = st.selectbox("Escolha a música:", nomes_musicas)

    if musica_escolhida:
        letra = st.session_state.letras_extraidas[musica_escolhida]
        letra_destacada = destacar_palavras(letra)

        st.markdown(f"""
        <div style='font-family: monospace; font-size: 15px;'>
        {letra_destacada}
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("Nenhuma letra extraída ainda. Busque um artista para começar.")
