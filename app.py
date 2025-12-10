import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="GranosBot Argentina", layout="centered")
st.title("GranosBot Argentina 2025")
st.markdown("**El chatbot que leyó todos tus informes del agro y responde en 2 segundos**")

@st.cache_resource
def crear_chatbot():
    with st.spinner("Cargando PDFs y creando la IA… (solo la primera vez tarda 20-40 segundos)"):
        loader = PyPDFDirectoryLoader("documentos_granos/")
        docs = loader.load()
        
        if len(docs) == 0:
            st.warning("Todavía no hay PDFs en la carpeta documentos_granos")
            return None
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(chunks, collection_name="granos")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3)
        
        template = """Sos un trader senior de Rosario con 25 años de experiencia.
        Hablás en español argentino, usás jerga del agro (pizarra, DJVE, blend, cupo, fijaciones, etc.).
        Solo usás información que está en los documentos que te cargaron.
        
        Contexto:
        {context}
        
        Pregunta: {question}
        Respuesta útil y corta:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

chain = crear_chatbot()

st.divider()
st.subheader("Chat – Preguntale lo que quieras")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Buen día che! Tirame cualquier pregunta sobre granos, DJVE, retenciones, cupos, liquidaciones o lo que diga algún PDF que me hayas puesto."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if chain is None:
    st.info("Poné al menos un PDF en la carpeta documentos_granos y recargá la página")
else:
    if prompt := st.chat_input("Ej: ¿Cuánto soja queda por declarar DJVE? / ¿Sigue el dólar blend?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los documentos…"):
                respuesta = chain.invoke(prompt)
            st.write(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})