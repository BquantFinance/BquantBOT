"""
BQuant ChatBot - RAG con Embeddings Reales
Búsqueda semántica sobre las cartas de Warren Buffett (1977-2024)

USO:
    pip install streamlit sentence-transformers faiss-cpu groq
    python build_index.py  # Solo la primera vez
    streamlit run app.py

Requiere en .streamlit/secrets.toml:
    GROQ_API_KEY = "tu-api-key"
"""

import streamlit as st
from groq import Groq
import pickle
import faiss-cpu
from sentence_transformers import SentenceTransformer
import re

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="BQuant · Buffett", page_icon="⚡", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================
# CSS - DISEÑO EDITORIAL MINIMALISTA
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;500;600&display=swap');
    
    :root {
        --bg: #0a0a0a;
        --card: #111;
        --border: #222;
        --accent: #d4a853;
        --text: #eee;
        --muted: #777;
    }
    
    * { font-family: 'DM Sans', -apple-system, sans-serif; }
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
        background: var(--bg) !important;
    }
    [data-testid="stBottomBlockContainer"] { background: transparent !important; }
    .block-container { padding: 2rem 1.5rem 6rem !important; max-width: 780px !important; }
    
    /* HEADER */
    .header {
        text-align: center;
        padding: 1.5rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .logo {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 2.2rem;
        color: var(--text);
        letter-spacing: -1px;
    }
    .logo span { color: var(--accent); }
    .tagline {
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .meta {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 1rem;
        font-size: 0.78rem;
        color: var(--muted);
    }
    .meta-dot {
        display: inline-block;
        width: 6px; height: 6px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 50% { opacity: 0.4; } }
    
    /* WELCOME */
    .welcome {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
    }
    .welcome h1 {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1.35;
        margin: 0 0 0.4rem;
    }
    .welcome p {
        color: var(--muted);
        font-size: 0.95rem;
    }
    
    /* PILLS */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
        border-radius: 50px !important;
        padding: 0.55rem 1.1rem !important;
        font-size: 0.82rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: rgba(212,168,83,0.08) !important;
    }
    
    /* CHAT */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 1rem 0 !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 !important;
    }
    [data-testid="stChatMessageContent"], [data-testid="stChatMessageContent"] p {
        color: var(--text) !important;
        font-size: 0.92rem !important;
        line-height: 1.7 !important;
    }
    
    /* SOURCES */
    .sources {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 0.8rem;
        padding-top: 0.6rem;
        border-top: 1px dashed var(--border);
    }
    .src {
        background: var(--card);
        border: 1px solid var(--border);
        color: var(--muted);
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .src b { color: var(--accent); }
    
    /* INPUT */
    [data-testid="stChatInput"] > div {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent) !important;
    }
    [data-testid="stChatInput"] input, [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        font-size: 0.92rem !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] input::placeholder { color: var(--muted) !important; }
    [data-testid="stChatInput"] button {
        background: var(--accent) !important;
        border: none !important;
        border-radius: 10px !important;
    }
    
    /* FOOTER */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 1.5rem;
        border-top: 1px solid var(--border);
        font-size: 0.72rem;
        color: var(--muted);
    }
    .footer a { color: var(--muted); text-decoration: none; }
    .footer a:hover { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD RESOURCES
# ============================================
@st.cache_resource
def load_index():
    try:
        return faiss.read_index(INDEX_FILE)
    except:
        return None

@st.cache_resource
def load_chunks():
    try:
        with open(CHUNKS_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)


# ============================================
# SEMANTIC SEARCH
# ============================================
def search(query: str, index, chunks: list, model, top_k: int = 5) -> list[dict]:
    """Búsqueda semántica con FAISS"""
    
    # Extraer año si se menciona
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    year_filter = year_match.group(1) if year_match else None
    
    # Embedding de la query
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    
    # Buscar (más resultados si hay filtro de año)
    k = top_k * 4 if year_filter else top_k
    scores, indices = index.search(q_emb.astype('float32'), min(k, len(chunks)))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        chunk['score'] = float(score)
        
        # Filtrar por año
        if year_filter and chunk['year'] != year_filter:
            continue
        
        results.append(chunk)
        if len(results) >= top_k:
            break
    
    return results


# ============================================
# GENERATE RESPONSE
# ============================================
def generate(query: str, results: list[dict], client) -> tuple[str, list]:
    """Genera respuesta con Groq"""
    
    if not results:
        system = """Eres el asistente de BQuant sobre las cartas de Warren Buffett (1977-2024).
Responde amigablemente y sugiere temas: inversión, adquisiciones, crisis, seguros, o años específicos."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    else:
        context = "\n\n---\n\n".join([f"[{r['year']}]\n{r['text']}" for r in results])
        
        system = """Eres experto en las cartas de Warren Buffett (1977-2024).

REGLAS:
- Responde SOLO con información del contexto
- Cita el año: "En [año], Buffett..."
- Sé directo y específico
- Si no está en el contexto, dilo
- Español, máximo 250 palabras"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXTO:\n{context}\n\n---\nPREGUNTA: {query}"}
        ]
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=600
        )
        return resp.choices[0].message.content, results
    except Exception as e:
        return f"Error: {e}", []


# ============================================
# UI HELPERS
# ============================================
def show_sources(results: list):
    if not results:
        return
    pills = ''.join([f'<span class="src"><b>{r["year"]}</b></span>' for r in results[:4]])
    st.markdown(f'<div class="sources">{pills}</div>', unsafe_allow_html=True)


# ============================================
# MAIN
# ============================================
def main():
    index = load_index()
    chunks = load_chunks()
    model = load_model()
    client = get_client()
    
    if not index or not chunks:
        st.error("⚠️ Ejecuta primero: `python build_index.py`")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    
    # HEADER
    st.markdown("""
    <div class="header">
        <div class="logo">⚡ <span>BQuant</span>ChatBot</div>
        <div class="tagline">Berkshire Letters AI</div>
        <div class="meta">
            <span><span class="meta-dot"></span>Online</span>
            <span>📚 48 cartas</span>
            <span>1977–2024</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # WELCOME
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
            <h1>Pregunta sobre las cartas<br/>de Warren Buffett</h1>
            <p>47 años de filosofía inversora</p>
        </div>
        """, unsafe_allow_html=True)
        
        suggestions = [
            ("Filosofía", "¿Cuál es la filosofía de inversión de Buffett?"),
            ("Crisis 2008", "¿Qué dijo Buffett sobre la crisis de 2008?"),
            ("Carta 1983", "Resume la carta de 1983"),
            ("Oro", "¿Qué opina Buffett del oro?"),
            ("Moats", "¿Qué son los moats?"),
            ("Seguros", "¿Por qué son importantes los seguros para Berkshire?"),
        ]
        
        cols = st.columns(3)
        for i, (label, q) in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(label, key=f"s{i}", use_container_width=True):
                    st.session_state.pending = q
                    st.rerun()
    
    # PENDING
    if st.session_state.pending:
        q = st.session_state.pending
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": q})
        
        results = search(q, index, chunks, model)
        response, used = generate(q, results, client)
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": used})
        st.rerun()
    
    # MESSAGES
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
            st.write(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"])
    
    # INPUT
    if prompt := st.chat_input("Pregunta sobre las cartas de Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Buscando..."):
                results = search(prompt, index, chunks, model)
                response, used = generate(prompt, results, client)
            st.write(response)
            show_sources(used)
        
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": used})
    
    # FOOTER
    st.markdown("""
    <div class="footer">
        <a href="https://bquantfinance.com">BQuant Finance</a> · 
        <a href="https://twitter.com/Gsnchez">@Gsnchez</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
