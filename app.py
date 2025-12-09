import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import os

st.set_page_config(page_title="GranosBot Argentina 2025", layout="wide")
st.title("GranosBot Diario + Chat IA Argentina")
st.markdown("**Dashboard precios + Chatbot que sabe TODO del agro argentino porque leíste tus documentos**")

# === CARGA DE DOCUMENTOS ===
@st.cache_resource
def cargar_rag():
    with st.spinner("Leyendo todos los PDFs y Excel del agro argentino..."):
        loader = PyPDFDirectoryLoader("documentos_granos/")
        docs = loader.load()

        # Si tenés Excel de MAGyP en Excel también los lee
        for file in os.listdir("documentos_granos"):
            if file.endswith(".xlsx"):
                excel_loader = UnstructuredExcelLoader(f"documentos_granos/{file}")
                docs += excel_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=None,  # Groq no necesita embeddings locales
            collection_name="granos_arg",
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )

        template = """Sos el trader más capo de Rosario con 25 años en la Bolsa.
        Respondé siempre en español argentino, usando jerga del sector (pizarra, cupo, DJVE, blend, fijaciones, etc.).
        Si no sabés, decí que no está en los documentos.

        Contexto de los documentos:
        {context}

        Pregunta del usuario: {question}
        Respuesta:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

# === PRECIOS RÁPIDOS (mismo que antes) ===
import pandas as pd
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)
def precios_hoy():
    # Simulado – mañana lo reemplazás con tu CSV real o API MAGyP
    fechas = pd.date_range(end=datetime.today(), periods=15).strftime("%d/%m")
    return pd.DataFrame({
        "Fecha": fechas,
        "Soja": [378,379,382,380,377,381,379,383,381,378,382,380,379,381,380],
        "Maíz": [188,190,189,192,187,190,188,191,189,187,190,188,189,190,189],
        "Trigo": [208,210,207,212,209,211,208,213,210,207,211,209,210,212,210]
    })

df = precios_hoy()

col1, col2, col3 = st.columns(3)
ult = df.iloc[-1]
penult = df.iloc[-2]
col1.metric("Soja Rosario", f"${ult['Soja']}", f"{ult['Soja']-penult['Soja']:+.0f}")
col2.metric("Maíz", f"${ult['Maíz']}", f"{ult['Maíz']-penult['Maíz']:+.0f}")
col3.metric("Trigo", f"${ult['Trigo']}", f"{ult['Trigo']-penult['Trigo']:+.0f}")

st.line_chart(df.set_index("Fecha"))

# === CHATBOT ===
st.divider()
st.subheader("Chat IA – Preguntale lo que quieras al experto en granos")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Buen día! Tirame cualquier pregunta sobre precios, DJVE, retenciones, cupos, liquidaciones o lo que diga algún informe que me cargaste."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

try:
    chain = cargar_rag()
except:
    st.error("Falta la carpeta 'documentos_granos' con PDFs o la API key de Groq")
    st.stop()

if prompt := st.chat_input("Ej: ¿Cuánto soja se liquidó esta semana? / ¿Sigue el blend 80/20? / ¿Qué dice el último informe BCR?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando los documentos..."):
            respuesta = chain.invoke(prompt)
        st.write(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})