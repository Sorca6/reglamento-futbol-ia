import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Asistente Reglamentario", page_icon="⚽", layout="wide")

# --- CONTROL DE ACCESO CON CONTRASEÑA ---
def verificar_acceso():
    if st.session_state.get("autenticado", False):
        return True

    st.title("🔒 Acceso Restringido")
    st.markdown("Introduce la clave de acceso del equipo arbitral para continuar:")
    
    password_input = st.text_input("Contraseña", type="password")
    clave_correcta = st.secrets.get("APP_PASSWORD", "admin123")

    if st.button("Entrar"):
        if password_input == clave_correcta:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    
    return False

if not verificar_acceso():
    st.stop()

# --- APLICACIÓN PRINCIPAL ---
st.title("⚽ Asistente Oficial de Normativa y Reglas de Juego (Google Gemini)")
st.markdown("Consulta cualquier jugada técnica o disciplinaria basada en los documentos oficiales cargados.")

# 1. Obtener la API key de Google
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("No se encontró la GOOGLE_API_KEY configurada en los Secrets de Streamlit.")
    st.stop()

# Inyectarla en las variables de entorno del sistema
os.environ["GOOGLE_API_KEY"] = api_key

# 2. Carga e indexación de PDFs con embeddings de Google
@st.cache_resource(show_spinner="Procesando e indexando la documentación con Google Embeddings...")
def cargar_vectorstore_multiples_pdfs(carpeta_docs: str):
    archivos_pdf = glob.glob(os.path.join(carpeta_docs, "*.pdf"))
    
    if not archivos_pdf:
        return None, []
    
    todos_los_documentos = []
    for ruta_pdf in archivos_pdf:
        loader = PyPDFLoader(ruta_pdf)
        docs = loader.load()
        for doc in docs:
            doc.metadata["fuente_archivo"] = os.path.basename(ruta_pdf)
        todos_los_documentos.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "Regla ", "REGLA ", "Circular ", ". "]
    )
    docs_divididos = splitter.split_documents(todos_los_documentos)
    
    # Modelo universal compatible de Google para embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vectorstore = FAISS.from_documents(docs_divididos, embeddings)
    nombres_archivos = [os.path.basename(f) for f in archivos_pdf]
    return vectorstore, nombres_archivos

CARPETA_DOCUMENTOS = "documentos"
vectorstore, archivos_cargados = cargar_vectorstore_multiples_pdfs(CARPETA_DOCUMENTOS)

if vectorstore is None:
    st.error(f"No se encontraron archivos PDF dentro de la carpeta '{CARPETA_DOCUMENTOS}/'.")
    st.stop()

with st.sidebar:
    st.subheader("📚 Documentos cargados")
    for archivo in archivos_cargados:
        st.markdown(f"- `{archivo}`")
    st.divider()
    if st.button("Cerrar sesión"):
        st.session_state["autenticado"] = False
        st.rerun()

# 3. Configuración del prompt arbitral
system_prompt = (
    "Eres un instructor arbitral experto y riguroso. Tu labor es responder a la duda "
    "basándote exclusivamente en los fragmentos de la normativa y reglamentos oficiales proporcionados.\n\n"
    "Estructura tu respuesta siempre en este formato:\n"
    "1. **Decisión técnica:** (ej. tiro libre directo, indirecto, penalti, balón a tierra, saque de banda).\n"
    "2. **Decisión disciplinaria:** (ej. sin tarjeta, amonestación/tarjeta amarilla, expulsión/tarjeta roja).\n"
    "3. **Regla y fuente aplicable:** Cita el documento específico, la Regla/Artículo y el extracto normativo que justifica el fallo.\n\n"
    "Si la jugada descrita no tiene sustento en los fragmentos facilitados, indica honestamente que la normativa cargada no lo especifica.\n\n"
    "Contexto oficial recuperado:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 4. Configuración del modelo Gemini (LLM)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.0
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def formatear_documentos(docs):
    return "\n\n".join(
        f"[Fuente: {doc.metadata.get('fuente_archivo', 'PDF')} - Pág. {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )

cadena_rag = (
    {"context": retriever | formatear_documentos, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Historial de mensajes en el chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Cuadro de texto para hacer preguntas
pregunta = st.chat_input("Plantea aquí una jugada o duda reglamentaria...")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
        
    with st.chat_message("assistant"):
        with st.spinner("Analizando la jugada en la normativa con Gemini..."):
            docs_relevantes = retriever.invoke(pregunta)
            texto_respuesta = cadena_rag.invoke(pregunta)
            st.markdown(texto_respuesta)
            
            with st.expander("Ver fragmentos consultados en los PDFs"):
                for i, doc in enumerate(docs_relevantes):
                    origen = doc.metadata.get("fuente_archivo", "PDF")
                    pagina = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Referencia {i+1} — Archivo:** `{origen}` (Página {pagina})")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()
                    
    st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})
