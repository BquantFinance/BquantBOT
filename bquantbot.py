import streamlit as st
from google import genai
import json
import re

# ============================================
# CONFIGURACIÓN
# ============================================
st.set_page_config(
    page_title="BQuant Chatbot",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# API Key
GEMINI_API_KEY = "AIzaSyBuxu0jsV6t0hVBVmksD6LBJhKPu8VjPOY"

# ============================================
# ESTILOS CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    #MainMenu, footer, header, .stDeployButton {display: none !important; visibility: hidden !important;}
    
    .stApp {
        background: #08080c;
        background-image: radial-gradient(ellipse at top, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
    }
    
    .block-container {
        padding: 1rem 1rem 0 1rem !important;
        max-width: 800px !important;
    }
    
    .header {
        text-align: center;
        padding: 0.75rem 0;
        margin-bottom: 0.5rem;
    }
    
    .logo {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.35);
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .badges {
        display: flex;
        justify-content: center;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.65rem;
    }
    
    .badge-online {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    
    .badge-data {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #818cf8;
    }
    
    .pulse-dot {
        width: 5px;
        height: 5px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    .welcome {
        text-align: center;
        padding: 1rem 0;
    }
    
    .welcome-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
    }
    
    .welcome-sub {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 20px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: white !important;
    }
    
    .stChatMessage {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stChatMessageContent"] {
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: rgba(20, 20, 30, 0.95) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 16px !important;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.08) !important;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(99, 102, 241, 0.4) !important;
    }
    
    [data-testid="stChatInput"] input {
        color: white !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stChatInput"] input::placeholder {
        color: rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border-radius: 12px !important;
    }
    
    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        margin-top: 0.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 0.75rem 0;
        color: rgba(255, 255, 255, 0.2);
        font-size: 0.6rem;
    }
    
    .footer a { color: rgba(99, 102, 241, 0.5); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR DATOS Y MODELO
# ============================================
@st.cache_data
def load_berkshire_letters():
    try:
        with open("berkshire_letters.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def init_gemini():
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client
    return None

# ============================================
# BÚSQUEDA
# ============================================
def search_letters(query: str, letters: dict, max_chunks: int = 5, chunk_size: int = 2000) -> list:
    query_lower = query.lower()
    
    stopwords = {'que', 'qué', 'como', 'cómo', 'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 
                 'a', 'por', 'para', 'con', 'sobre', 'es', 'son', 'fue', 'the', 'a', 'an', 'of', 'in', 
                 'to', 'for', 'on', 'with', 'about', 'is', 'are', 'what', 'how', 'buffett', 'warren', 
                 'berkshire', 'piensa', 'dijo', 'carta', 'letter', 'año', 'year', 'resume', 'hola', 
                 'hello', 'dice', 'buena', 'empresa'}
    
    keywords = [w for w in re.findall(r'\w+', query_lower) if w not in stopwords and len(w) > 2]
    
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    specific_year = year_match.group(1) if year_match else None
    
    results = []
    
    for year, letter_data in letters.items():
        if specific_year and year != specific_year:
            continue
            
        text = letter_data.get('text', '')
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(para_lower.count(kw) for kw in keywords)
            
            if score > 0 or specific_year:
                results.append({'year': year, 'text': para[:chunk_size], 'score': score or 0.1})
    
    if not results:
        for year in sorted(letters.keys(), reverse=True)[:2]:
            text = letters[year].get('text', '')
            paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 200][:2]
            for p in paras:
                results.append({'year': year, 'text': p[:chunk_size], 'score': 0})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_chunks]

# ============================================
# RESPUESTA
# ============================================
def get_response(query: str, letters: dict, client, history: list) -> tuple:
    chunks = search_letters(query, letters)
    
    context = ""
    sources = []
    for c in chunks:
        context += f"[{c['year']}]: {c['text']}\n\n"
        if c['year'] not in sources:
            sources.append(c['year'])
    
    prompt = f"""Eres el asistente de BQuant especializado en las cartas de Warren Buffett (1977-2024).

DATOS:
{context}

REGLAS:
- Responde basándote en los datos
- Cita el año: "En [año], Buffett..."
- Sé conciso (150-250 palabras)
- Idioma del usuario
- No inventes citas

Usuario: {query}

Respuesta:"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text, sources
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# INIT
# ============================================
letters = load_berkshire_letters()
client = init_gemini()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="header">
    <div class="logo">⚡ BQuant</div>
    <div class="tagline">Berkshire Letters AI</div>
    <div class="badges">
        <span class="badge badge-online"><span class="pulse-dot"></span> Online</span>
        <span class="badge badge-data">📚 48 cartas</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# WELCOME + SUGGESTIONS
# ============================================
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">¿Qué quieres saber?</div>
        <div class="welcome-sub">47 años de sabiduría de Buffett</div>
    </div>
    """, unsafe_allow_html=True)
    
    suggestions = [
        "Filosofía de inversión",
        "Crisis de 2008",
        "Sobre la inflación",
        "Carta de 2023",
        "Qué es una buena empresa",
        "Opinión del oro",
    ]
    
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"s{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": f"¿Qué dice Buffett sobre {s.lower()}?"})
                st.rerun()

# ============================================
# MENSAJES
# ============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown(f'<div class="source-tag">📚 {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

# ============================================
# INPUT
# ============================================
if letters and client:
    if prompt := st.chat_input("Pregunta sobre Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner(""):
                response, sources = get_response(prompt, letters, client, st.session_state.messages)
            st.write(response)
            if sources:
                st.markdown(f'<div class="source-tag">📚 {", ".join(sources)}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    No es asesoramiento financiero · <a href="https://bquantfinance.com">BQuant</a> · <a href="https://twitter.com/Gsnchez">@Gsnchez</a>
</div>
""", unsafe_allow_html=True)
