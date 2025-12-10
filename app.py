import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="GranosBot Argentina", layout="wide")
st.title("🚀 GranosBot Diario – IA para Granos Argentina 2025")
st.markdown("**Chat IA que responde TODO sobre precios, retenciones, DJVE y más. Desplegado en Streamlit!**")

# Config Groq (modelo actualizado: reemplazo oficial del deprecated)
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # ¡Nuevo! Reemplazo recomendado por Groq para 2025
        temperature=0.1,
        max_tokens=500
    )
except Exception as e:
    st.error(f"Error al conectar Groq: {e}. Chequeá tu API key en secrets.toml (sacala gratis en https://console.groq.com/keys)")
    st.stop()

# Template con conocimiento base de granos AR (para respuestas sin PDFs)
template = """Sos un trader experto de granos en Rosario, Argentina (diciembre 2025). 
Usás jerga local: pizarra, DJVE, blend dólar, retenciones (soja 24%, maíz 8.5%, trigo 7.5%), Up-River.
Datos clave: 
- Precios pizarra hoy (Rosario): Soja $380 USD/TN, Maíz $190, Trigo $210.
- Liquidaciones: ~US$30B en 2025, 9M TN soja pendientes.
- Dólar CCL: ~$1513. Blend exportador 80/20 vigente.
- Capacidad teórica: 80% Up-River, alertas por Niña débil.

Pregunta: {question}
Respuesta corta y útil:"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm

# Chat interface
st.divider()
st.subheader("💬 Chat IA – Preguntale lo que quieras")

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="¡Buen día! Soy tu asistente de granos. Preguntame por precios mañana, liquidaciones DJVE, retenciones o estrategias de trading.")
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

if prompt_text := st.chat_input("Ej: ¿Cuánto cobra un trader por 1000 TN soja mañana con blend?"):
    st.session_state.messages.append(HumanMessage(content=prompt_text))
    with st.chat_message("user"):
        st.write(prompt_text)
    
    with st.chat_message("assistant"):
        with st.spinner("Calculando como capo rosarino..."):
            try:
                response = chain.invoke({"question": prompt_text})
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error Groq: {e}. Probá con un prompt más corto (máx 100 palabras) o chequeá la API key.")