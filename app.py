import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # Compatibilidad para versiones reorganizadas
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configuración de página
st.set_page_config(page_title="Asistente Reglamentario", page_icon="⚽", layout="wide")
st.title("⚽ Asistente Oficial de Normativa y Reglas de Juego")
st.markdown("Consulta cualquier jugada técnica o disciplinaria basada en los documentos oficiales cargados.")

# 1. Gestión de API Key (Secretos de Streamlit o input en barra lateral)
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Introduce tu OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Introduce tu API Key de OpenAI para activar el asistente.")
        st.stop()

# 2. Carga, troceado e indexación multi-documento con caché
@st.cache_resource(show_spinner="Procesando e indexando la documentación oficial...")
def cargar_vectorstore_multiples_pdfs(carpeta_docs: str):
    # Buscar todos los archivos .pdf dentro de la carpeta
    archivos_pdf = glob.glob(os.path.join(carpeta_docs, "*.pdf"))
    
    if not archivos_pdf:
        return None, []
    
    todos_los_documentos = []
    
    # Leer cada PDF preservando el nombre del archivo de origen
    for ruta_pdf in archivos_pdf:
        loader = PyPDFLoader(ruta_pdf)
        docs = loader.load()
        for doc in docs:
            # Añadir nombre limpio del archivo a los metadatos
            doc.metadata["fuente_archivo"] = os.path.basename(ruta_pdf)
        todos_los_documentos.extend(docs)
    
    # Trocear respetando títulos, reglas y párrafos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "Regla ", "REGLA ", "Circular ", ". "]
    )
    docs_divididos = splitter.split_documents(todos_los_documentos)
    
    # Crear índice vectorial conjunto
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs_divididos, embeddings)
    
    nombres_archivos = [os.path.basename(f) for f in archivos_pdf]
    return vectorstore, nombres_archivos

CARPETA_DOCUMENTOS = "documentos"
vectorstore, archivos_cargados = cargar_vectorstore_multiples_pdfs(CARPETA_DOCUMENTOS)

if vectorstore is None:
    st.error(f"No se encontraron archivos PDF dentro de la carpeta '{CARPETA_DOCUMENTOS}/'. Añade al menos uno.")
    st.stop()

# Mostrar en la barra lateral qué documentos están activos
with st.sidebar:
    st.subheader("📚 Documentos cargados")
    for archivo in archivos_cargados:
        st.markdown(f"- `{archivo}`")

# 3. Configuración del prompt arbitral estricto
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
    ("human", "{input}")
])

llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0.0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 4. Control de mensajes en sesión
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Interfaz de consulta
pregunta = st.chat_input("Ej: Un jugador en fuera de juego recibe el balón de un despeje intencionado de un defensa, ¿se sanciona?")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
        
    with st.chat_message("assistant"):
        with st.spinner("Buscando en todos los reglamentos..."):
            respuesta = rag_chain.invoke({"input": pregunta})
            texto_respuesta = respuesta["answer"]
            st.markdown(texto_respuesta)
            
            # Desplegable con los fragmentos y páginas exactas de cada documento
            with st.expander("Ver fuentes consultadas en los PDFs"):
                for i, doc in enumerate(respuesta["context"]):
                    origen = doc.metadata.get("fuente_archivo", "PDF")
                    pagina = doc.metadata.get("page", 0) + 1  # Base 0 a Base 1
                    st.markdown(f"**Referencia {i+1} — Archivo:** `{origen}` (Página {pagina})")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()
                    
    st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})
